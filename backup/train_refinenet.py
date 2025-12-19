from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List, Union


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _parse_augmenters(items: Optional[list]) -> "object":
    import albumentations as A

    if not items:
        return A.Compose([])

    transforms: List[Any] = []
    for item in items:
        if isinstance(item, str):
            key = item.lower()
            if key in ("horizontal_flip", "hflip", "flip_h"):
                transforms.append(A.HorizontalFlip(p=0.5))
            elif key in ("vertical_flip", "vflip", "flip_v"):
                transforms.append(A.VerticalFlip(p=0.5))
            else:
                raise ValueError(f"Unsupported augmenter string: {item}")
        elif isinstance(item, dict):
            if "rotation" in item:
                limit = float(item["rotation"])
                transforms.append(A.Rotate(limit=limit, p=0.5))
            else:
                raise ValueError(f"Unsupported augmenter dict: {item}")
        else:
            raise ValueError(f"Unsupported augmenter type: {type(item)}")

    return A.Compose(transforms)


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(cfg)}")
    return cfg


def _latest_best_state(log_root: Path) -> Optional[Path]:
    candidates = list(log_root.rglob("state/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ReFineNet and export weights (best.pt) inside this repo.")
    p.add_argument("--config", type=str, default="configs/train_s2_osm.yaml", help="Training config YAML.")
    p.add_argument("--data-root", type=str, default=None, help="Override Loader.root in config.")
    p.add_argument("--log-root", type=str, default="logs", help="Base log directory (runs will create timestamped subfolders).")

    p.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=None, help="Override Loader.batch_size in config.")

    p.add_argument("--resume", type=str, default=None, help="Resume from a saved training state (state/default.pt).")

    p.add_argument("--export-dir", type=str, default="weights", help="Where to copy exported weights.")
    p.add_argument("--export-name", type=str, default="best.pt", help="Exported filename.")
    return p


def main(argv: Optional[list] = None) -> int:
    _ensure_repo_on_path()
    args = build_parser().parse_args(argv)

    from building_footprint_segmentation.segmentation import init_segmentation, read_trainer_config
    from building_footprint_segmentation.helpers.callbacks import CallbackList, load_callback
    from building_footprint_segmentation.trainer import Trainer

    cfg_path = Path(args.config)
    cfg = read_trainer_config(str(cfg_path))

    # overrides
    if args.data_root is not None:
        cfg["Loader"]["root"] = args.data_root
    if args.batch_size is not None:
        cfg["Loader"]["batch_size"] = int(args.batch_size)
    cfg["Callbacks"]["log_dir"] = str(Path(args.log_root).resolve())

    segmentation = init_segmentation(cfg["Segmentation"])

    model_kwargs: Dict[str, Any] = dict(cfg.get("Model", {}).get("param", {}) or {})
    if "backbone" in model_kwargs and "res_net_to_use" not in model_kwargs:
        model_kwargs["res_net_to_use"] = model_kwargs.pop("backbone")

    model = segmentation.load_model(name=cfg["Model"]["name"], **model_kwargs)
    criterion = segmentation.load_criterion(**cfg["Criterion"])

    augmenters = _parse_augmenters(cfg["Loader"].get("augmenters"))
    loader = segmentation.load_loader(
        cfg["Loader"]["root"],
        cfg["Loader"]["image_normalizer"],
        cfg["Loader"]["label_normalizer"],
        augmenters,
        cfg["Loader"]["batch_size"],
    )
    metrics = segmentation.load_metrics(data_metrics=cfg["Metrics"])
    optimizer = segmentation.load_optimizer(
        model, name=cfg["Optimizer"]["name"], lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    callbacks = CallbackList()
    for caller in cfg["Callbacks"]["callers"]:
        callbacks.append(load_callback(cfg["Callbacks"]["log_dir"], caller))

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loader=loader,
        metrics=metrics,
        callbacks=callbacks,
        scheduler=None,
    )

    callbacks.update_params({"loader": loader})

    if args.resume:
        trainer.resume(args.resume, new_end_epoch=int(args.epochs))
    else:
        trainer.train(start_epoch=0, end_epoch=int(args.epochs))

    log_root = Path(cfg["Callbacks"]["log_dir"])
    best_state = _latest_best_state(log_root)
    if best_state is None:
        raise SystemExit(f"No best.pt found under {log_root}. Training may not have produced state outputs.")

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / args.export_name
    shutil.copy2(best_state, export_path)
    print(f"Exported: {export_path} (copied from {best_state})")
    print("Use this file with inference, e.g.:")
    print(f"  python run_nearby.py --coords coords.yaml --weights {export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


