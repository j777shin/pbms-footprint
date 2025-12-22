#!/usr/bin/env python3
"""
Simple script to apply super resolution to Sentinel-2 satellite imagery
based on coordinates in coords.yaml.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_dotenv(path: Path) -> None:
    """Load environment variables from .env file."""
    if not path.exists():
        return
    import os
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
    """Read YAML file."""
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data)}")
    return data


def save_rgb_png(path: Path, rgb) -> None:
    """Save RGB image as PNG."""
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    print(f"Saved: {path}")


def main():
    _ensure_repo_on_path()
    
    parser = argparse.ArgumentParser(
        description="Apply super resolution to Sentinel-2 imagery from coordinates"
    )
    parser.add_argument("--coords", type=str, default="coords.yaml", help="Coordinates YAML file")
    parser.add_argument("--dotenv", type=str, default=".env", help=".env file with Sentinel Hub credentials")
    parser.add_argument("--outdir", type=str, default="sr_outputs", help="Output directory")
    parser.add_argument("--patch-size", type=int, default=256, help="Sentinel-2 patch size in pixels")
    parser.add_argument("--sr-target-res", type=float, default=1.0, help="Target resolution in meters (e.g., 1.0m from 10m = 10x)")
    parser.add_argument("--sr-method", type=str, default="srdr3", choices=["srdr3", "bicubic", "bilinear", "lanczos"], help="SR method")
    parser.add_argument("--sr-device", type=str, default=None, help="Device for SR (cuda, cpu, or None for auto)")
    parser.add_argument("--save-original", action="store_true", help="Also save original image before SR")
    
    args = parser.parse_args()
    
    # Load environment variables
    repo_root = Path(__file__).resolve().parent.parent
    dotenv_file = repo_root / args.dotenv
    load_dotenv(dotenv_file)
    
    # Read coordinates
    coords_file = repo_root / args.coords
    if not coords_file.exists():
        raise FileNotFoundError(f"Coordinates file not found: {coords_file}")
    
    coords = read_yaml(coords_file)
    lat = float(coords["lat"])
    lon = float(coords["lon"])
    radius_m = float(coords.get("radius_m", 100.0))
    
    print(f"Coordinates: ({lat:.6f}, {lon:.6f})")
    print(f"Radius: {radius_m}m")
    print(f"Patch size: {args.patch_size}px")
    print(f"SR target resolution: {args.sr_target_res}m")
    print(f"SR method: {args.sr_method}")
    print()
    
    # Fetch Sentinel-2 imagery
    from building_footprint_segmentation.geo.sentinel2 import fetch_sentinel2_rgb_patch_sentinelhub
    import os
    
    print("Fetching Sentinel-2 imagery...")
    s2 = fetch_sentinel2_rgb_patch_sentinelhub(
        lat=lat,
        lon=lon,
        patch_px=args.patch_size,
        resolution_m=10.0,
        client_id=os.getenv("SENTINELHUB_CLIENT_ID"),
        client_secret=os.getenv("SENTINELHUB_CLIENT_SECRET"),
    )
    
    print(f"Fetched image: shape={s2.rgb.shape}, dtype={s2.rgb.dtype}, min={s2.rgb.min()}, max={s2.rgb.max()}, mean={s2.rgb.mean():.2f}")
    
    # Save original if requested
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    if args.save_original:
        save_rgb_png(outdir / "s2_original.png", s2.rgb)
    
    # Apply super resolution
    from building_footprint_segmentation.geo.super_resolution import apply_super_resolution
    import numpy as np
    
    # Calculate scale factor
    scale_factor = int(np.ceil(10.0 / args.sr_target_res))
    actual_resolution = 10.0 / scale_factor
    
    print(f"\nApplying super resolution: 10.0m -> {args.sr_target_res}m target (scale: {scale_factor}x)...")
    sr_rgb = apply_super_resolution(
        s2.rgb,
        scale_factor=scale_factor,
        method=args.sr_method,
        use_deep_model=(args.sr_method == "srdr3"),
        device=args.sr_device,
    )
    
    print(f"Super resolution applied. Enhanced resolution: {actual_resolution:.2f}m (scale: {scale_factor}x)")
    print(f"SR image: shape={sr_rgb.shape}, dtype={sr_rgb.dtype}, min={sr_rgb.min()}, max={sr_rgb.max()}, mean={sr_rgb.mean():.2f}")
    
    # Check if image is too dark and apply enhancement if needed
    if sr_rgb.max() < 50 or sr_rgb.mean() < 10:
        print(f"Warning: SR image is very dark (max={sr_rgb.max()}, mean={sr_rgb.mean():.2f})")
        print("Applying contrast enhancement...")
        # Apply histogram equalization or contrast stretching
        import cv2
        # Convert to float for processing
        sr_float = sr_rgb.astype(np.float32) / 255.0
        # Apply gamma correction to brighten
        gamma = 0.5  # < 1.0 brightens
        sr_float = np.power(sr_float, gamma)
        # Stretch contrast
        sr_float = (sr_float - sr_float.min()) / (sr_float.max() - sr_float.min() + 1e-8)
        # Convert back to uint8
        sr_rgb = (sr_float * 255.0).astype(np.uint8)
        print(f"After enhancement: min={sr_rgb.min()}, max={sr_rgb.max()}, mean={sr_rgb.mean():.2f}")
    
    # Save super-resolved image
    save_rgb_png(outdir / "s2_sr.png", sr_rgb)
    
    print(f"\n{'='*60}")
    print(f"Super Resolution Complete")
    print(f"{'='*60}")
    print(f"Output directory: {outdir}")
    print(f"  - s2_sr.png: Super-resolved satellite imagery ({sr_rgb.shape[0]}x{sr_rgb.shape[1]})")
    if args.save_original:
        print(f"  - s2_original.png: Original satellite imagery ({s2.rgb.shape[0]}x{s2.rgb.shape[1]})")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

