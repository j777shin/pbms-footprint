from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
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


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data)}")
    return data


def save_rgb_png(path: Path, rgb) -> None:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_mask_png(path: Path, mask01) -> None:
    import cv2
    import numpy as np

    cv2.imwrite(str(path), (mask01.astype(np.uint8) * 255))


def save_overlay_png(path: Path, rgb, mask01, alpha: float = 0.35) -> None:
    import cv2
    import numpy as np

    mask3 = cv2.cvtColor((mask01.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(rgb, 1.0, mask3, float(alpha), 0.0)
    save_rgb_png(path, overlay)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Sentinel-2 -> ReFineNet -> (optional YOLO/OSM) -> PNG + GeoJSON outputs"
    )
    p.add_argument("--coords", type=str, default="coords.yaml", help="YAML file with lat/lon/radius_m")
    p.add_argument("--dotenv", type=str, default=".env", help="Optional .env path with SENTINELHUB_CLIENT_ID/SECRET")
    p.add_argument("--weights", type=str, required=True, help="Path to ReFineNet weights (.pt)")
    p.add_argument("--outdir", type=str, default="geo_outputs")

    p.add_argument("--use-osm", action="store_true", default=False)
    p.add_argument("--osm-filter", action="store_true", default=False)
    p.add_argument("--osm-building", type=str, default=None, help='Optional building tag, e.g. "industrial"')

    p.add_argument("--yolo-weights", type=str, default=None)
    p.add_argument("--fuse-mode", type=str, default="intersection", choices=["none", "union", "intersection", "refinenet_only", "yolo_only"])
    return p


def main(argv: Optional[list] = None) -> int:
    _ensure_repo_on_path()
    args = build_parser().parse_args(argv)

    load_dotenv(Path(args.dotenv))
    coords = read_yaml(Path(args.coords))

    lat = float(coords["lat"])
    lon = float(coords["lon"])
    radius_m = float(coords.get("radius_m", 100.0))

    osm_tags = None
    if args.osm_building is not None:
        osm_tags = {"building": args.osm_building}

    from building_footprint_segmentation.geo.pipeline import predict_buildings_near_coordinate

    res = predict_buildings_near_coordinate(
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        refinenet_weights_path=args.weights,
        use_osm=bool(args.use_osm),
        osm_filter_predictions=bool(args.osm_filter),
        osm_extra_tags=osm_tags,
        yolo_weights=args.yolo_weights,
        fuse_mode=args.fuse_mode,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_rgb_png(outdir / "s2_rgb.png", res.rgb_patch)
    save_mask_png(outdir / "mask_refinenet.png", res.refinenet.mask)
    if res.yolo_mask01 is not None:
        save_mask_png(outdir / "mask_yolo.png", res.yolo_mask01)
    save_mask_png(outdir / "mask_fused.png", res.fused_mask01)
    save_overlay_png(outdir / "overlay_fused.png", res.rgb_patch, res.fused_mask01)

    with open(outdir / "predicted_buildings.geojson", "w") as f:
        json.dump(res.geojson_predicted, f)
    if res.osm_geojson is not None:
        with open(outdir / "osm_buildings.geojson", "w") as f:
            json.dump(res.osm_geojson, f)

    print(f"Done. Outputs written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


