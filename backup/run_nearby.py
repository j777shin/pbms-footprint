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


def save_overlay_with_osm_png(path: Path, rgb, pred_mask01, osm_mask01=None, pred_alpha: float = 0.5, osm_alpha: float = 0.3, use_outline: bool = True) -> None:
    """Create overlay with satellite imagery, predictions (red), and OSM buildings (green).
    
    Uses colored outlines and semi-transparent fills to keep satellite imagery visible.
    """
    import cv2
    import numpy as np

    # Ensure all masks and RGB have the same dimensions
    h, w = rgb.shape[:2]
    
    if pred_mask01.shape[:2] != (h, w):
        pred_mask01 = cv2.resize(pred_mask01.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    
    # Start with satellite imagery (RGB) - keep it fully visible
    overlay = rgb.copy()
    
    # Draw predictions in red
    if use_outline:
        # Draw red outlines for predictions
        contours_pred, _ = cv2.findContours(pred_mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_pred, -1, (255, 0, 0), thickness=2)  # Red outlines
        # Add semi-transparent red fill
        pred_mask3 = np.zeros_like(rgb)
        pred_mask3[:, :, 0] = pred_mask01 * 255  # Red channel
        overlay = cv2.addWeighted(overlay, 1.0, pred_mask3, pred_alpha * 0.3, 0.0)
    else:
        # Filled overlay approach
        overlay = overlay.astype(np.float32)
        pred_mask_bool = pred_mask01.astype(bool)
        pred_mask3 = np.zeros_like(rgb, dtype=np.float32)
        pred_mask3[pred_mask_bool, 0] = 255.0
        overlay[pred_mask_bool] = overlay[pred_mask_bool] * (1.0 - pred_alpha) + pred_mask3[pred_mask_bool] * pred_alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Add OSM buildings in green if available
    if osm_mask01 is not None:
        if osm_mask01.shape[:2] != (h, w):
            osm_mask01 = cv2.resize(osm_mask01.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        if use_outline:
            # Draw green outlines for OSM buildings
            contours_osm, _ = cv2.findContours(osm_mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours_osm, -1, (0, 255, 0), thickness=2)  # Green outlines
            # Add semi-transparent green fill
            osm_mask3 = np.zeros_like(rgb)
            osm_mask3[:, :, 1] = osm_mask01 * 255  # Green channel
            overlay = cv2.addWeighted(overlay, 1.0, osm_mask3, osm_alpha * 0.3, 0.0)
        else:
            # Filled overlay approach
            if overlay.dtype != np.float32:
                overlay = overlay.astype(np.float32)
            osm_mask_bool = osm_mask01.astype(bool)
            osm_mask3 = np.zeros_like(rgb, dtype=np.float32)
            osm_mask3[osm_mask_bool, 1] = 255.0
            overlay[osm_mask_bool] = overlay[osm_mask_bool] * (1.0 - osm_alpha) + osm_mask3[osm_mask_bool] * osm_alpha
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    if overlay.dtype != np.uint8:
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    save_rgb_png(path, overlay)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Sentinel-2 -> ReFineNet -> (optional YOLO/OSM) -> PNG + GeoJSON outputs"
    )
    p.add_argument("--coords", type=str, default="coords.yaml", help="YAML file with lat/lon/radius_m")
    p.add_argument("--dotenv", type=str, default=".env", help="Optional .env path with SENTINELHUB_CLIENT_ID/SECRET")
    p.add_argument("--weights", type=str, required=True, help="Path to ReFineNet weights (.pt)")
    p.add_argument("--outdir", type=str, default="geo_outputs")

    p.add_argument("--use-osm", action="store_true", default=True, help="Use OSM data for accurate building footprints (default: True)")
    p.add_argument("--no-osm", action="store_false", dest="use_osm", help="Disable OSM data")
    p.add_argument("--osm-filter", action="store_true", default=False, help="Filter predictions to only those intersecting OSM")
    p.add_argument("--osm-prefer-direct", action="store_true", default=True, help="Use OSM footprints directly (most accurate when OSM available, default: True)")
    p.add_argument("--osm-refine", action="store_true", default=False, help="Refine predictions by snapping to OSM boundaries")
    p.add_argument("--osm-building", type=str, default=None, help='Optional building tag, e.g. "industrial"')

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

    osm_tags = None
    if hasattr(args, 'osm_building') and args.osm_building is not None:
        osm_tags = {"building": args.osm_building}

    from building_footprint_segmentation.geo.pipeline import predict_buildings_near_coordinate

    res = predict_buildings_near_coordinate(
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        patch_px=getattr(args, 'patch_size', 512),
        refinenet_weights_path=args.weights,
        refinenet_threshold=getattr(args, 'threshold', 0.4),
        use_osm=bool(args.use_osm),
        osm_filter_predictions=bool(args.osm_filter),
        osm_prefer_direct=bool(args.osm_prefer_direct),
        osm_refine_predictions=bool(args.osm_refine),
        osm_extra_tags=osm_tags,
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

    save_rgb_png(outdir / "s2_rgb.png", res.rgb_patch)
    save_mask_png(outdir / "mask_refinenet.png", res.refinenet.mask)
    if res.yolo_mask01 is not None:
        save_mask_png(outdir / "mask_yolo.png", res.yolo_mask01)
    save_mask_png(outdir / "mask_fused.png", res.fused_mask01)
    
    # Create overlay with predictions
    save_overlay_png(outdir / "overlay_fused.png", res.rgb_patch, res.fused_mask01)
    
    # Create overlay with OSM buildings if available
    if res.osm_geojson is not None and len(res.osm_geojson.get("features", [])) > 0:
        try:
            from building_footprint_segmentation.geo.osm import rasterize_osm_buildings_to_mask
            from pyproj import Transformer
            from shapely.geometry import shape
            from shapely.ops import transform
            
            # Extract OSM geometries from GeoJSON (they're in WGS84)
            osm_geoms_wgs84 = [shape(feat["geometry"]) for feat in res.osm_geojson.get("features", [])]
            
            # Reproject OSM geometries from WGS84 to the same projected CRS as the result
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{res.projected_epsg}", always_xy=True)
            
            def transform_geom(geom):
                return transform(transformer.transform, geom)
            
            osm_geoms_projected = []
            for geom in osm_geoms_wgs84:
                try:
                    geom_proj = transform_geom(geom)
                    if not geom_proj.is_empty:
                        osm_geoms_projected.append(geom_proj)
                except Exception as e:
                    print(f"Warning: Failed to transform OSM geometry: {e}")
                    continue
            
            if osm_geoms_projected:
                # Rasterize OSM buildings to match RGB patch size
                h, w = res.rgb_patch.shape[:2]
                osm_mask01 = rasterize_osm_buildings_to_mask(
                    geoms_projected=osm_geoms_projected,
                    bbox_projected=res.bbox_projected,
                    out_size=max(h, w),  # Use max to ensure square, will resize if needed
                )
                
                # Resize OSM mask to match RGB exactly
                import cv2
                import numpy as np
                if osm_mask01.shape[:2] != (h, w):
                    osm_mask01 = cv2.resize(osm_mask01.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                
                # Save OSM mask
                save_mask_png(outdir / "mask_osm.png", osm_mask01)
                
                # Create combined overlay with satellite imagery, predictions (red), and OSM (green)
                save_overlay_with_osm_png(
                    outdir / "overlay_with_osm.png",
                    res.rgb_patch,
                    res.fused_mask01,
                    osm_mask01=osm_mask01,
                    pred_alpha=0.4,
                    osm_alpha=0.3
                )
        except Exception as e:
            print(f"Warning: Failed to create OSM overlay: {e}")
            import traceback
            traceback.print_exc()

    with open(outdir / "predicted_buildings.geojson", "w") as f:
        json.dump(res.geojson_predicted, f, indent=2)
    if res.osm_geojson is not None:
        with open(outdir / "osm_buildings.geojson", "w") as f:
            json.dump(res.osm_geojson, f, indent=2)
    
    # Print summary
    num_predicted = len(res.geojson_predicted.get("features", []))
    num_osm = len(res.osm_geojson.get("features", [])) if res.osm_geojson else 0
    
    print(f"\n{'='*60}")
    print(f"Building Footprint Detection Complete")
    print(f"{'='*60}")
    print(f"Location: ({lat:.6f}, {lon:.6f})")
    print(f"Radius: {radius_m}m")
    print(f"Predicted buildings: {num_predicted}")
    if res.osm_geojson:
        print(f"OSM buildings: {num_osm}")
    print(f"\nOutputs written to: {outdir}")
    print(f"  - s2_rgb.png: Satellite imagery")
    print(f"  - mask_refinenet.png: ReFineNet predictions")
    if res.yolo_mask01 is not None:
        print(f"  - mask_yolo.png: YOLO predictions")
    print(f"  - mask_fused.png: Fused predictions")
    print(f"  - overlay_fused.png: Overlay with predictions (red)")
    if res.osm_geojson and (outdir / "overlay_with_osm.png").exists():
        print(f"  - mask_osm.png: OSM building mask")
        print(f"  - overlay_with_osm.png: Overlay with predictions (red) and OSM (green)")
    print(f"  - predicted_buildings.geojson: Building footprints")
    if res.osm_geojson:
        print(f"  - osm_buildings.geojson: OSM reference buildings")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


