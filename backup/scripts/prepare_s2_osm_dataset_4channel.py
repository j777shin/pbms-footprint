from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _sample_point_in_circle(lat0: float, lon0: float, radius_m: float) -> Tuple[float, float]:
    from pyproj import CRS as PyCRS, Transformer
    from building_footprint_segmentation.geo.crs import utm_epsg_for_latlon

    epsg = utm_epsg_for_latlon(lat0, lon0)
    wgs84 = PyCRS.from_epsg(4326)
    utm = PyCRS.from_epsg(epsg)
    to_utm = Transformer.from_crs(wgs84, utm, always_xy=True)
    to_wgs = Transformer.from_crs(utm, wgs84, always_xy=True)

    x0, y0 = to_utm.transform(lon0, lat0)
    u = random.random()
    r = radius_m * math.sqrt(u)
    theta = 2 * math.pi * random.random()
    dx = r * math.cos(theta)
    dy = r * math.sin(theta)
    x = x0 + dx
    y = y0 + dy
    lon, lat = to_wgs.transform(x, y)
    return float(lat), float(lon)


def _rasterize_osm_buildings(
    geoms_projected: List["object"],
    bbox_projected: Tuple[float, float, float, float],
    out_size: int,
) -> "object":
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    minx, miny, maxx, maxy = bbox_projected
    transform = from_bounds(minx, miny, maxx, maxy, out_size, out_size)
    shapes = [(mapping(g), 1) for g in geoms_projected]
    mask = rasterize(
        shapes=shapes,
        out_shape=(out_size, out_size),
        fill=0,
        transform=transform,
        dtype=np.uint8,
        all_touched=False,
    )
    return mask


def _save_4channel_image(out_path: Path, rgb_uint8, osm_mask01) -> None:
    import cv2
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    osm_channel = (osm_mask01.astype(np.uint8) * 255)
    
    four_channel = np.dstack([rgb_uint8, osm_channel])
    
    cv2.imwrite(str(out_path), four_channel)


def _save_label(out_lbl: Path, osm_mask01) -> None:
    import cv2
    import numpy as np

    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl = (osm_mask01.astype(np.uint8) * 255)
    cv2.imwrite(str(out_lbl), lbl)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create 4-channel dataset (RGB + OSM mask) from Sentinel-2 + OSM."
    )
    p.add_argument("--dotenv", type=str, default=".env")
    p.add_argument("--center-lat", type=float, required=True)
    p.add_argument("--center-lon", type=float, required=True)
    p.add_argument("--sample-radius-m", type=float, default=3000.0)
    p.add_argument("--patch-px", type=int, default=384)
    p.add_argument("--resolution-m", type=float, default=10.0)
    p.add_argument("--time-start", type=str, default="2025-06-01")
    p.add_argument("--time-end", type=str, default="2025-10-01")
    p.add_argument("--mosaicking-order", type=str, default="leastCC", choices=["leastCC", "mostRecent"])
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-val", type=int, default=40)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sleep-s", type=float, default=1.0)
    p.add_argument("--skip-empty", action="store_true", default=False)
    p.add_argument("--overpass-url", type=str, default=None)
    p.add_argument("--overpass-timeout-s", type=int, default=180)
    p.add_argument("--overpass-retries", type=int, default=3)
    p.add_argument("--overpass-retry-sleep-s", type=float, default=3.0)
    p.add_argument("--fail-on-overpass-error", action="store_true", default=False)
    p.add_argument("--out-root", type=str, default="data/s2_osm_4ch/building_patches")
    return p


def main(argv: Optional[list] = None) -> int:
    _ensure_repo_on_path()
    args = build_parser().parse_args(argv)
    random.seed(int(args.seed))
    load_dotenv(Path(args.dotenv))

    from building_footprint_segmentation.geo.sentinel2 import fetch_sentinel2_rgb_patch_sentinelhub
    from building_footprint_segmentation.geo.osm import fetch_osm_buildings_within_radius

    out_root = Path(args.out_root)
    splits = [("train", int(args.n_train)), ("val", int(args.n_val)), ("test", int(args.n_test))]
    patch_half_m = (int(args.patch_px) * float(args.resolution_m)) / 2.0
    osm_query_radius_m = patch_half_m * 1.6

    counters = {"train": 0, "val": 0, "test": 0}
    for split, _ in splits:
        img_dir = out_root / split / "images"
        if img_dir.exists():
            counters[split] = len(list(img_dir.glob(f"s2_{split}_*.png")))
    idx = 0

    for split, n_needed in splits:
        while counters[split] < n_needed:
            idx += 1
            lat, lon = _sample_point_in_circle(args.center_lat, args.center_lon, float(args.sample_radius_m))

            s2 = fetch_sentinel2_rgb_patch_sentinelhub(
                lat=lat,
                lon=lon,
                patch_px=int(args.patch_px),
                resolution_m=float(args.resolution_m),
                time_interval=(args.time_start, args.time_end),
                mosaicking_order=args.mosaicking_order,
            )

            try:
                osm = fetch_osm_buildings_within_radius(
                    lat=lat,
                    lon=lon,
                    radius_m=osm_query_radius_m,
                    projected_epsg=s2.projected_epsg,
                    overpass_url=args.overpass_url,
                    requests_timeout=int(args.overpass_timeout_s),
                    max_retries=int(args.overpass_retries),
                    retry_sleep_s=float(args.overpass_retry_sleep_s),
                )
            except Exception as e:
                if args.fail_on_overpass_error:
                    raise
                print(f"[{split}] Overpass error (skipping): {e}")
                time.sleep(float(args.sleep_s))
                continue

            if args.skip_empty and len(osm.geoms_projected) == 0:
                time.sleep(float(args.sleep_s))
                continue

            mask01 = _rasterize_osm_buildings(osm.geoms_projected, s2.bbox_projected, out_size=int(args.patch_px))
            if args.skip_empty and mask01.sum() == 0:
                time.sleep(float(args.sleep_s))
                continue

            name = f"s2_{split}_{counters[split]:06d}"
            out_img = out_root / split / "images" / f"{name}.png"
            out_lbl = out_root / split / "labels" / f"{name}.png"
            _save_4channel_image(out_img, s2.rgb, mask01)
            _save_label(out_lbl, mask01)

            counters[split] += 1
            print(f"[{split}] wrote {name} (lat={lat:.6f}, lon={lon:.6f})")
            time.sleep(float(args.sleep_s))

    print(f"Done. 4-channel dataset written to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

