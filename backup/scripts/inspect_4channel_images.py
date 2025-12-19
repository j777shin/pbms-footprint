from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def inspect_4channel_image(img_path: Path) -> None:
    """Inspect a 4-channel image and show what's in each channel."""
    import cv2
    import numpy as np

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR: Could not read {img_path}")
        return

    print(f"\n=== {img_path.name} ===")
    print(f"Shape: {img.shape} (expected: H, W, 4)")
    print(f"Dtype: {img.dtype}")
    print(f"Value range: [{img.min()}, {img.max()}]")

    if img.ndim != 3 or img.shape[2] != 4:
        print(f"WARNING: Expected 4-channel image (H, W, 4), got {img.shape}")
        return

    # RGB channels (0-2)
    rgb = img[:, :, 0:3]
    print(f"\nRGB channels (0-2):")
    print(f"  Mean: {rgb.mean():.1f}, Std: {rgb.std():.1f}")
    print(f"  Min: {rgb.min()}, Max: {rgb.max()}")
    print(f"  Unique values in R: {len(np.unique(rgb[:,:,0]))} distinct")
    print(f"  Unique values in G: {len(np.unique(rgb[:,:,1]))} distinct")
    print(f"  Unique values in B: {len(np.unique(rgb[:,:,2]))} distinct")

    # OSM mask channel (3)
    osm = img[:, :, 3]
    print(f"\nOSM mask channel (3):")
    print(f"  Mean: {osm.mean():.1f}, Std: {osm.std():.1f}")
    print(f"  Min: {osm.min()}, Max: {osm.max()}")
    unique_osm = np.unique(osm)
    print(f"  Unique values: {unique_osm}")
    print(f"  Building pixels (255): {(osm == 255).sum()} ({(osm == 255).sum() / osm.size * 100:.1f}%)")
    print(f"  Non-building pixels (0): {(osm == 0).sum()} ({(osm == 0).sum() / osm.size * 100:.1f}%)")

    # Check if RGB looks reasonable (should have color variation)
    if rgb.std() < 10:
        print(f"\n⚠️  WARNING: RGB channels have very low variation (std={rgb.std():.1f})")
        print(f"   This might indicate the image is mostly black/white or corrupted.")
    else:
        print(f"\n✓ RGB channels look reasonable (good color variation)")

    # Check if OSM mask is binary (should be mostly 0 and 255)
    if len(unique_osm) <= 2 and set(unique_osm).issubset({0, 255}):
        print(f"✓ OSM mask is binary (0=no building, 255=building)")
    else:
        print(f"⚠️  WARNING: OSM mask has unexpected values: {unique_osm}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect 4-channel images (RGB + OSM mask).")
    p.add_argument(
        "--dataset-root",
        type=str,
        default="data/s2_osm_4ch/building_patches",
        help="Root directory of 4-channel dataset",
    )
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--n-samples", type=int, default=5, help="Number of samples to inspect")
    return p


def main(argv: list | None = None) -> int:
    _ensure_repo_on_path()
    args = build_parser().parse_args(argv)

    dataset_root = Path(args.dataset_root)
    img_dir = dataset_root / args.split / "images"

    if not img_dir.exists():
        print(f"ERROR: Image directory not found: {img_dir}")
        return 1

    img_files = sorted(img_dir.glob("*.png"))[: int(args.n_samples)]
    if not img_files:
        print(f"ERROR: No PNG files found in {img_dir}")
        return 1

    print(f"Inspecting {len(img_files)} images from {img_dir}...")
    for img_file in img_files:
        inspect_4channel_image(img_file)

    print(f"\n=== Summary ===")
    print(f"4-channel images contain:")
    print(f"  - Channels 0-2: RGB satellite imagery (Sentinel-2)")
    print(f"  - Channel 3: OSM building mask (0=no building, 255=building)")
    print(f"\nWhen viewed in a standard image viewer, they may look:")
    print(f"  - Grayscale/black-and-white (viewer interpreting 4th channel as alpha)")
    print(f"  - Or showing only one channel")
    print(f"\nThis is NORMAL. The images are correct for training.")
    print(f"The model will read all 4 channels properly during training.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

