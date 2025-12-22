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


def save_overlay_png(path: Path, rgb, mask01, alpha: float = 0.5, use_outline: bool = True) -> None:
    """Create overlay with satellite imagery and predictions (red).
    
    Args:
        path: Output path
        rgb: Satellite imagery (HxWx3 uint8 RGB)
        mask01: Binary mask (HxW uint8 {0,1})
        alpha: Transparency for filled overlay (0-1)
        use_outline: If True, draw colored outlines; if False, use filled overlay
    """
    import cv2
    import numpy as np

    # Ensure mask and RGB have the same dimensions
    h, w = rgb.shape[:2]
    if mask01.shape[:2] != (h, w):
        mask01 = cv2.resize(mask01.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    # Start with satellite imagery (RGB) - keep it fully visible
    overlay = rgb.copy()
    
    if use_outline:
        # Draw colored outlines on satellite imagery (better visibility)
        # Find contours
        contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw red outlines (thickness=2 for visibility)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=2)
        # Also add a semi-transparent filled overlay for better visibility
        mask3 = np.zeros_like(rgb)
        mask3[:, :, 0] = mask01 * 255  # Red channel
        overlay = cv2.addWeighted(overlay, 1.0, mask3, alpha * 0.3, 0.0)
    else:
        # Original filled overlay approach
        overlay = overlay.astype(np.float32)
        pred_mask3 = np.zeros_like(rgb, dtype=np.float32)
        pred_mask_bool = mask01.astype(bool)
        pred_mask3[pred_mask_bool, 0] = 255.0  # Red channel for predictions
        overlay[pred_mask_bool] = overlay[pred_mask_bool] * (1.0 - alpha) + pred_mask3[pred_mask_bool] * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    save_rgb_png(path, overlay)




def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Sentinel-2 -> ReFineNet -> (optional YOLO/Google Maps) -> PNG + GeoJSON outputs"
    )
    p.add_argument("--coords", type=str, default="coords.yaml", help="YAML file with lat/lon/radius_m")
    p.add_argument("--dotenv", type=str, default=".env", help="Optional .env path with SENTINELHUB_CLIENT_ID/SECRET and GOOGLE_MAPS_API_KEY")
    p.add_argument("--weights", type=str, default=None, help="Path to ReFineNet weights (.pt). Optional if using YOLO-only mode (--fuse-mode yolo_only)")
    p.add_argument("--outdir", type=str, default="geo_outputs")


    p.add_argument("--yolo-weights", type=str, default=None, help="Path to YOLO weights for fusion (optional)")
    p.add_argument("--fuse-mode", type=str, default="union", choices=["none", "union", "intersection", "refinenet_only", "yolo_only"], help="Fusion mode: union (default, better recall), intersection (cleaner)")
    
    p.add_argument("--use-sr", action="store_true", default=True, help="Enable super resolution using SRDR3 to enhance imagery (default: True)")
    p.add_argument("--no-sr", action="store_false", dest="use_sr", help="Disable super resolution")
    p.add_argument("--sr-target-res", type=float, default=1.0, help="Target resolution in meters (e.g., 1.0m from 10m = 10x, default: 1.0)")
    p.add_argument("--sr-method", type=str, default="srdr3", choices=["srdr3", "bicubic", "bilinear", "lanczos", "real_esrgan"], help="SR method: srdr3 (default deep model), or traditional methods")
    p.add_argument("--sr-model-path", type=str, default=None, help="Optional path to SRDR3 model weights file")
    p.add_argument("--sr-device", type=str, default=None, help="Device for SRDR3 (cuda, cpu, or None for auto)")
    
    p.add_argument("--threshold", type=float, default=0.4, help="Probability threshold for building detection (default: 0.4, lower = more sensitive)")
    p.add_argument("--min-area", type=float, default=0.5, help="Minimum building area in square meters (default: 0.5)")
    p.add_argument("--patch-size", type=int, default=512, help="Sentinel-2 patch size in pixels (default: 512, larger = more context)")
    
    # Post-processing options
    p.add_argument("--enable-postprocess", action="store_true", default=True, help="Enable enhanced post-processing (default: True)")
    p.add_argument("--no-postprocess", action="store_false", dest="enable_postprocess", help="Disable post-processing")
    p.add_argument("--postprocess-smooth", action="store_true", default=True, help="Smooth polygon boundaries (default: True)")
    p.add_argument("--postprocess-regularize", action="store_true", default=False, help="Regularize building shapes (rectangularize)")
    return p


def main(argv: Optional[list] = None) -> int:
    _ensure_repo_on_path()
    args = build_parser().parse_args(argv)

    load_dotenv(Path(args.dotenv))
    coords = read_yaml(Path(args.coords))

    lat = float(coords["lat"])
    lon = float(coords["lon"])
    radius_m = float(coords.get("radius_m", 100.0))

    from building_footprint_segmentation.geo.pipeline import predict_buildings_near_coordinate

    # Validate arguments for YOLO-only mode
    if args.fuse_mode == "yolo_only" and not args.yolo_weights:
        raise ValueError("--yolo-weights is required when --fuse-mode is 'yolo_only'")
    if not args.weights and not args.yolo_weights:
        raise ValueError("Either --weights (ReFineNet) or --yolo-weights must be provided")
    
    res = predict_buildings_near_coordinate(
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        patch_px=getattr(args, 'patch_size', 512),
        refinenet_weights_path=args.weights or "",
        refinenet_threshold=getattr(args, 'threshold', 0.4),
        yolo_weights=args.yolo_weights,
        fuse_mode=args.fuse_mode,
        use_super_resolution=bool(args.use_sr),
        sr_target_resolution_m=args.sr_target_res,
        sr_method=args.sr_method,
        sr_model_path=args.sr_model_path,
        sr_device=args.sr_device,
        enable_postprocessing=getattr(args, 'enable_postprocess', True),
        postprocess_smooth=getattr(args, 'postprocess_smooth', True),
        postprocess_regularize=getattr(args, 'postprocess_regularize', False),
        min_area_m2=getattr(args, 'min_area', 0.5),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Always save satellite imagery (should always be available)
    print(f"Saving satellite imagery: shape={res.rgb_patch.shape}, dtype={res.rgb_patch.dtype}, min={res.rgb_patch.min()}, max={res.rgb_patch.max()}")
    
    # Save the main RGB (SR-enhanced if SR was applied, otherwise original)
    save_rgb_png(outdir / "s2_rgb.png", res.rgb_patch)
    
    # If SR was applied, also save the original before SR
    if res.rgb_patch_original is not None:
        print(f"Saving original satellite imagery (before SR): shape={res.rgb_patch_original.shape}")
        save_rgb_png(outdir / "s2_rgb_original.png", res.rgb_patch_original)
        # Also save the SR-enhanced version with a clear name
        save_rgb_png(outdir / "s2_rgb_sr.png", res.rgb_patch)
    
    # Save masks
    save_mask_png(outdir / "mask_refinenet.png", res.refinenet.mask)
    if res.yolo_mask01 is not None:
        save_mask_png(outdir / "mask_yolo.png", res.yolo_mask01)
    save_mask_png(outdir / "mask_fused.png", res.fused_mask01)
    
    # Create overlay with predictions (satellite imagery + predictions)
    print(f"Creating overlay: RGB shape={res.rgb_patch.shape}, mask shape={res.fused_mask01.shape}")
    save_overlay_png(outdir / "overlay_fused.png", res.rgb_patch, res.fused_mask01)

    # Save predicted building footprints as GeoJSON (polygons)
    with open(outdir / "predicted_buildings.geojson", "w") as f:
        json.dump(res.geojson_predicted, f, indent=2)
    
    num_predicted = len(res.geojson_predicted.get("features", []))
    if num_predicted > 0:
        print(f"\n✓ Saved {num_predicted} building footprint polygon(s) to predicted_buildings.geojson")
        # Print polygon details
        for i, feature in enumerate(res.geojson_predicted.get("features", [])):
            geom_type = feature.get("geometry", {}).get("type", "unknown")
            coords = feature.get("geometry", {}).get("coordinates", [])
            if geom_type == "Polygon" and coords:
                num_vertices = len(coords[0]) if coords else 0
                print(f"  Polygon {i+1}: {geom_type} with {num_vertices} vertices")
            elif geom_type == "MultiPolygon":
                num_polygons = len(coords)
                print(f"  Polygon {i+1}: {geom_type} with {num_polygons} parts")
    else:
        print(f"\n⚠ No building footprints detected at coordinate ({lat:.6f}, {lon:.6f})")
        print(f"  This may mean:")
        print(f"  - No building exists at this location")
        print(f"  - Building is outside the 50m search radius")
        print(f"  - Detection threshold may be too high")
    
    print(f"\n{'='*60}")
    print(f"Building Footprint Detection Complete")
    print(f"{'='*60}")
    print(f"Location: ({lat:.6f}, {lon:.6f})")
    print(f"Radius: {radius_m}m")
    print(f"Predicted buildings: {num_predicted}")
    print(f"\nOutputs written to: {outdir}")
    print(f"  - s2_rgb.png: Satellite imagery")
    print(f"  - mask_refinenet.png: ReFineNet predictions")
    if res.yolo_mask01 is not None:
        print(f"  - mask_yolo.png: YOLO predictions")
    print(f"  - mask_fused.png: Fused predictions")
    print(f"  - overlay_fused.png: Overlay with predictions (red)")
    print(f"  - predicted_buildings.geojson: Building footprint polygons (GeoJSON format)")
    if num_predicted > 0:
        print(f"    → Contains {num_predicted} polygon feature(s) in WGS84 (lat/lon) coordinates")
        print(f"    → QGIS compatible: Yes (drag & drop into QGIS)")
        print(f"    → OSM compatible: Yes (includes building=yes tag)")
        print(f"    → Properties: building, source, area_m2, id")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


