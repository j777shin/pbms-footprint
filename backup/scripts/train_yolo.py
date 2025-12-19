from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def train_yolo_segmentation(
    data_root: Path,
    epochs: int = 100,
    imgsz: int = 384,
    batch: int = 16,
    model_size: str = "n",  # n, s, m, l, x
    device: Optional[str] = None,
    project: str = "yolo_runs",
    name: str = "buildings",
    exist_ok: bool = True,
) -> Path:
    # Suppress harmless overflow warnings from ultralytics augmentation
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
    
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from e

    data_yaml = data_root / "dataset.yaml"
    if not data_yaml.exists():
        _create_yolo_data_yaml(data_root, data_yaml)

    model_name = f"yolov8{model_size}-seg.pt"
    model = YOLO(model_name)

    results = model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        project=project,
        name=name,
        exist_ok=exist_ok,
        verbose=True,
    )

    best_pt = Path(project) / name / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"Training completed but best.pt not found at {best_pt}")

    return best_pt


def _create_yolo_data_yaml(data_root: Path, out_yaml: Path) -> None:
    import yaml

    splits = ["train", "val", "test"]
    paths = {}
    for split in splits:
        img_dir = data_root / split / "images"
        if img_dir.exists():
            paths[split] = str(img_dir.resolve())

    if not paths:
        raise ValueError(f"No image directories found in {data_root}")

    config = {
        "path": str(data_root.resolve()),
        "train": paths.get("train", ""),
        "val": paths.get("val", paths.get("train", "")),
        "test": paths.get("test", ""),
        "nc": 1,  # number of classes
        "names": ["building"],  # class names
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created dataset config: {out_yaml}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train YOLOv8 segmentation model for buildings.")
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="YOLO dataset root (should contain train/val/test/images and labels).",
    )
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    p.add_argument("--imgsz", type=int, default=384, help="Image size (square).")
    p.add_argument("--batch", type=int, default=16, help="Batch size.")
    p.add_argument(
        "--model-size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size: n=nano, s=small, m=medium, l=large, x=xlarge.",
    )
    p.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, 0, 1, ...).")
    p.add_argument("--project", type=str, default="yolo_runs", help="Project directory.")
    p.add_argument("--name", type=str, default="buildings", help="Run name.")
    return p


def main(argv: Optional[list] = None) -> int:
    _ensure_repo_on_path()
    args = build_parser().parse_args(argv)

    data_root = Path(args.data)
    if not data_root.exists():
        raise SystemExit(f"Dataset not found: {data_root}")

    print(f"Training YOLOv8-seg on {data_root}")
    print(f"  epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, model={args.model_size}")

    best_pt = train_yolo_segmentation(
        data_root=data_root,
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        model_size=args.model_size,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    print(f"\n✓ Training complete!")
    print(f"  Best weights: {best_pt}")
    print(f"\n  To use with run_nearby.py:")
    print(f"    --yolo-weights {best_pt}")

    target = Path("weights") / "yolo_buildings.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(best_pt, target)
    print(f"  Also copied to: {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

