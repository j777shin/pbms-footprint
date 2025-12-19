from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def apply_sr_to_dataset(
    source_root: Path,
    target_root: Path,
    scale_factor: int = 4,
    method: str = "bicubic",
    use_real_esrgan: bool = False,
    splits: Optional[list[str]] = None,
) -> None:
    import cv2
    import numpy as np
    
    from building_footprint_segmentation.geo.super_resolution import apply_super_resolution
    
    if splits is None:
        splits = ["train", "val", "test"]
    
    target_root.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    
    for split in splits:
        src_img_dir = source_root / split / "images"
        src_lbl_dir = source_root / split / "labels"
        
        if not src_img_dir.exists() or not src_lbl_dir.exists():
            print(f"[{split}] Skipping (missing directories)")
            continue
        
        tgt_img_dir = target_root / split / "images"
        tgt_lbl_dir = target_root / split / "labels"
        tgt_img_dir.mkdir(parents=True, exist_ok=True)
        tgt_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        img_files = sorted(src_img_dir.glob("*.png"))
        if not img_files:
            print(f"[{split}] No images found")
            continue
        
        processed = 0
        for img_file in img_files:
            lbl_file = src_lbl_dir / img_file.name
            
            if not lbl_file.exists():
                print(f"[{split}] Warning: Missing label for {img_file.name}")
                continue
            
            # Load image and label
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                print(f"[{split}] Warning: Failed to load {img_file.name}")
                continue
            
            lbl = cv2.imread(str(lbl_file), cv2.IMREAD_GRAYSCALE)
            if lbl is None:
                print(f"[{split}] Warning: Failed to load label {lbl_file.name}")
                continue
            
            # Convert BGR to RGB for super resolution
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply super resolution to image
            if use_real_esrgan:
                try:
                    sr_img_rgb = apply_super_resolution(
                        img_rgb,
                        scale_factor=scale_factor,
                        method=method,
                        use_real_esrgan=True,
                    )
                except Exception as e:
                    print(f"[{split}] Real-ESRGAN failed for {img_file.name}: {e}, falling back to {method}")
                    sr_img_rgb = apply_super_resolution(
                        img_rgb,
                        scale_factor=scale_factor,
                        method=method,
                        use_real_esrgan=False,
                    )
            else:
                sr_img_rgb = apply_super_resolution(
                    img_rgb,
                    scale_factor=scale_factor,
                    method=method,
                    use_real_esrgan=False,
                )
            
            # Convert back to BGR for saving
            sr_img_bgr = cv2.cvtColor(sr_img_rgb, cv2.COLOR_RGB2BGR)
            
            # Upscale label using nearest neighbor (preserve binary values)
            h, w = lbl.shape
            sr_lbl = cv2.resize(
                lbl,
                (w * scale_factor, h * scale_factor),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Save enhanced image and label
            tgt_img_path = tgt_img_dir / img_file.name
            tgt_lbl_path = tgt_lbl_dir / img_file.name
            
            cv2.imwrite(str(tgt_img_path), sr_img_bgr)
            cv2.imwrite(str(tgt_lbl_path), sr_lbl)
            
            processed += 1
        
        print(f"[{split}] Processed {processed} samples")
        total_processed += processed
    
    print(f"\n✓ Super resolution applied to {total_processed} total samples")
    print(f"  Source: {source_root}")
    print(f"  Target: {target_root}")
    print(f"  Scale factor: {scale_factor}x ({method})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply super resolution to an existing dataset (images and labels)."
    )
    p.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source dataset root (e.g., data/s2_osm/building_patches)",
    )
    p.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target dataset root (e.g., data/s2_osm_sr/building_patches)",
    )
    p.add_argument(
        "--scale-factor",
        type=int,
        default=4,
        help="Upscaling factor (default: 4 = 10m -> 2.5m)",
    )
    p.add_argument(
        "--method",
        type=str,
        default="bicubic",
        choices=["bicubic", "bilinear", "lanczos"],
        help="Interpolation method (default: bicubic)",
    )
    p.add_argument(
        "--use-real-esrgan",
        action="store_true",
        default=False,
        help="Use Real-ESRGAN model (requires: pip install realesrgan)",
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process (default: train val test)",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    _ensure_repo_on_path()
    args = build_parser().parse_args(argv)
    
    source_root = Path(args.source)
    if not source_root.exists():
        raise SystemExit(f"Source dataset not found: {source_root}")
    
    target_root = Path(args.target)
    
    apply_sr_to_dataset(
        source_root=source_root,
        target_root=target_root,
        scale_factor=int(args.scale_factor),
        method=args.method,
        use_real_esrgan=bool(args.use_real_esrgan),
        splits=args.splits,
    )
    
    print(f"\n✓ Dataset with super resolution ready at: {target_root}")
    print(f"\n  To train ReFineNet:")
    print(f"    python train_refinenet.py --config configs/train_s2_osm.yaml --data-root {target_root}")
    print(f"\n  To train YOLO (after conversion):")
    print(f"    python scripts/convert_to_yolo_format.py --source-root {target_root} --yolo-root data/yolo_buildings_sr")
    print(f"    python scripts/train_yolo.py --data data/yolo_buildings_sr")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

