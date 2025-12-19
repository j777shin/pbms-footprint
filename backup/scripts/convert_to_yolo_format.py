from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _mask_to_yolo_polygons(mask01, class_id: int = 0) -> list[str]:
    import cv2
    import numpy as np

    h, w = mask01.shape
    if mask01.dtype != np.uint8:
        mask01 = (mask01 > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if len(cnt) < 3:
            continue

        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        coords = []
        for pt in approx:
            x_norm = float(pt[0][0]) / w
            y_norm = float(pt[0][1]) / h
            coords.append(f"{x_norm:.6f}")
            coords.append(f"{y_norm:.6f}")

        if len(coords) >= 6:
            line = f"{class_id} " + " ".join(coords)
            lines.append(line)

    return lines


def convert_dataset(
    source_root: Path,
    yolo_root: Path,
    class_id: int = 0,
    splits: Optional[list[str]] = None,
) -> None:
    import cv2
    import numpy as np

    if splits is None:
        splits = ["train", "val", "test"]

    for split in splits:
        src_img_dir = source_root / split / "images"
        src_lbl_dir = source_root / split / "labels"

        if not src_img_dir.exists() or not src_lbl_dir.exists():
            print(f"Skipping {split}: missing directories")
            continue

        yolo_img_dir = yolo_root / split / "images"
        yolo_lbl_dir = yolo_root / split / "labels"
        yolo_img_dir.mkdir(parents=True, exist_ok=True)
        yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(src_img_dir.glob("*.png"))
        converted = 0

        for img_file in img_files:
            lbl_file = src_lbl_dir / img_file.name
            if not lbl_file.exists():
                print(f"Warning: missing label for {img_file.name}")
                continue

            # Read mask
            mask = cv2.imread(str(lbl_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: could not read {lbl_file}")
                continue

            mask01 = (mask > 127).astype(np.uint8)

            yolo_lines = _mask_to_yolo_polygons(mask01, class_id=class_id)

            import shutil

            shutil.copy2(img_file, yolo_img_dir / img_file.name)

            lbl_txt = yolo_lbl_dir / (img_file.stem + ".txt")
            with open(lbl_txt, "w") as f:
                f.write("\n".join(yolo_lines))
                if yolo_lines:
                    f.write("\n")

            converted += 1

        print(f"[{split}] converted {converted} samples")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert mask-based dataset to YOLO segmentation format."
    )
    p.add_argument(
        "--source-root",
        type=str,
        default="data/s2_osm/building_patches",
        help="Source dataset root (with {split}/images and {split}/labels).",
    )
    p.add_argument(
        "--yolo-root",
        type=str,
        default="data/yolo_buildings",
        help="Output YOLO dataset root.",
    )
    p.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class ID for buildings (default: 0).",
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to convert.",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    _ensure_repo_on_path()
    args = build_parser().parse_args(argv)

    source_root = Path(args.source_root)
    yolo_root = Path(args.yolo_root)

    if not source_root.exists():
        raise SystemExit(f"Source dataset not found: {source_root}")

    convert_dataset(
        source_root=source_root,
        yolo_root=yolo_root,
        class_id=int(args.class_id),
        splits=args.splits,
    )

    print(f"\nDone. YOLO dataset written to: {yolo_root}")
    print(f"\nNext step: train YOLOv8 with:")
    print(f"  python scripts/train_yolo.py --data {yolo_root} --epochs 100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

