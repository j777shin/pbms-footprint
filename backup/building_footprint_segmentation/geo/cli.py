from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def _save_rgb_png(path: Path, rgb) -> None:
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _save_mask_png(path: Path, mask01) -> None:
    import cv2
    import numpy as np
    cv2.imwrite(str(path), (mask01.astype(np.uint8) * 255))


def _save_overlay_png(path: Path, rgb, mask01, alpha: float = 0.35) -> None:
    import cv2
    import numpy as np
    mask3 = cv2.cvtColor((mask01.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(rgb, 1.0, mask3, float(alpha), 0.0)
    _save_rgb_png(path, overlay)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict building footprints near a coordinate using Sentinel-2 + ReFineNet (+ optional YOLO/OSM).")
    p.add_argument("--coords", type=str, default=None, help="YAML file containing lat/lon (and optional radius_m).")
    p.add_argument("--lat", type=float, required=False)
    p.add_argument("--lon", type=float, required=False)
    p.add_argument("--radius-m", type=float, default=None)

    p.add_argument("--refinenet-weights", type=str, required=False, help="Path to ReFineNet .pt weights (best.pt)")
    p.add_argument("--refinenet-threshold", type=float, default=0.5)

    p.add_argument("--patch-px", type=int, default=384)
    p.add_argument("--resolution-m", type=float, default=10.0)
    p.add_argument("--time-start", type=str, default="2025-06-01")
    p.add_argument("--time-end", type=str, default="2025-10-01")
    p.add_argument("--mosaicking-order", type=str, default="leastCC", choices=["leastCC", "mostRecent"])

    p.add_argument("--outdir", type=str, default="geo_outputs")

    p.add_argument("--sentinelhub-client-id", type=str, default=None)
    p.add_argument("--sentinelhub-client-secret", type=str, default=None)

    p.add_argument("--use-osm", action="store_true", default=False, help="Fetch OSM buildings within radius.")
    p.add_argument("--osm-filter", action="store_true", default=False, help="Filter predicted polygons by intersection with OSM buildings.")
    p.add_argument("--osm-building", type=str, default=None, help='Optional extra tag constraint, e.g. "industrial" to query building=industrial.')

    p.add_argument("--yolo-weights", type=str, default=None, help="Optional YOLOv8 weights file. Needs to be trained for buildings.")
    p.add_argument("--yolo-device", type=str, default=None)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou", type=float, default=0.7)
    p.add_argument("--fuse-mode", type=str, default="intersection", choices=["none", "union", "intersection", "refinenet_only", "yolo_only"])

    p.add_argument("--min-area-m2", type=float, default=1.0)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    # Optional coords file (YAML): provides lat/lon/(radius_m)
    if args.coords is not None:
        import yaml

        with open(args.coords, "r") as f:
            c = yaml.safe_load(f) or {}
        if args.lat is None:
            args.lat = c.get("lat")
        if args.lon is None:
            args.lon = c.get("lon")
        if args.radius_m is None:
            # allow both radius_m and radius-m keys if user prefers
            args.radius_m = c.get("radius_m", c.get("radius-m"))

    if args.lat is None or args.lon is None:
        raise SystemExit("Missing coordinates. Provide --lat/--lon or --coords coords.yaml")
    if args.radius_m is None:
        args.radius_m = 100.0
    if not args.refinenet_weights:
        raise SystemExit("Missing --refinenet-weights (path to your ReFineNet .pt weights)")

    # Import heavy deps only when actually running (not when showing --help)
    from .pipeline import predict_buildings_near_coordinate

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    osm_tags = None
    if args.osm_building is not None:
        osm_tags = {"building": args.osm_building}

    res = predict_buildings_near_coordinate(
        lat=args.lat,
        lon=args.lon,
        radius_m=args.radius_m,
        patch_px=args.patch_px,
        resolution_m=args.resolution_m,
        time_interval=(args.time_start, args.time_end),
        mosaicking_order=args.mosaicking_order,
        sentinelhub_client_id=args.sentinelhub_client_id,
        sentinelhub_client_secret=args.sentinelhub_client_secret,
        refinenet_weights_path=args.refinenet_weights,
        refinenet_threshold=args.refinenet_threshold,
        use_osm=bool(args.use_osm),
        osm_extra_tags=osm_tags,
        osm_filter_predictions=bool(args.osm_filter),
        yolo_weights=args.yolo_weights,
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        fuse_mode=args.fuse_mode,
        min_area_m2=args.min_area_m2,
    )

    _save_rgb_png(outdir / "s2_rgb.png", res.rgb_patch)
    _save_mask_png(outdir / "mask_refinenet.png", res.refinenet.mask)
    if res.yolo_mask01 is not None:
        _save_mask_png(outdir / "mask_yolo.png", res.yolo_mask01)
    _save_mask_png(outdir / "mask_fused.png", res.fused_mask01)
    _save_overlay_png(outdir / "overlay_fused.png", res.rgb_patch, res.fused_mask01)

    with open(outdir / "predicted_buildings.geojson", "w") as f:
        json.dump(res.geojson_predicted, f)
    if res.osm_geojson is not None:
        with open(outdir / "osm_buildings.geojson", "w") as f:
            json.dump(res.osm_geojson, f)

    print(f"Wrote outputs to: {outdir}")
    print(f"Predicted polygons: {len(res.geojson_predicted.get('features', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


