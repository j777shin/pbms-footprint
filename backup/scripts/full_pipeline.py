#!/usr/bin/env python3
"""
Full Pipeline Script: Sentinel-2 -> Super Resolution -> Dataset Creation -> Training -> Inference

This script automates the complete workflow:
1. Fetch Sentinel-2 imagery for coords.yaml area
2. Apply super resolution
3. Create s2_osm_sr dataset (3-channel)
4. Train/test/validate dataset
5. Train ReFineNet on 3-channel dataset
6. Train YOLO on 3-channel dataset
7. Run inference with run_nearby
8. Save all outputs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


def run_command(cmd: list[str], description: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and print status."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=check, capture_output=False)
    if result.returncode != 0 and check:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    
    return result


def read_coords(coords_file: Path) -> Dict[str, Any]:
    """Read coordinates from YAML file."""
    with open(coords_file, 'r') as f:
        coords = yaml.safe_load(f) or {}
    return coords


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: Sentinel-2 -> SR -> Dataset -> Training -> Inference (3-channel only)"
    )
    parser.add_argument("--coords", type=str, default="coords.yaml", help="Coordinates YAML file")
    parser.add_argument("--dotenv", type=str, default=".env", help=".env file with Sentinel Hub credentials")
    
    # Dataset parameters
    parser.add_argument("--n-train", type=int, default=200, help="Number of training samples")
    parser.add_argument("--n-val", type=int, default=40, help="Number of validation samples")
    parser.add_argument("--n-test", type=int, default=40, help="Number of test samples")
    parser.add_argument("--sample-radius-m", type=float, default=300, help="Sampling radius in meters")
    parser.add_argument("--sr-scale-factor", type=int, default=10, help="Super resolution scale factor (10 = 10m->1m target)")
    parser.add_argument("--sr-method", type=str, default="srdr3", help="SR method (srdr3, bicubic, etc.)")
    parser.add_argument("--sr-model-path", type=str, default=None, help="Path to SRDR3 model weights")
    
    # Training parameters
    parser.add_argument("--refinenet-epochs", type=int, default=10, help="ReFineNet training epochs")
    parser.add_argument("--refinenet-batch-size", type=int, default=4, help="ReFineNet batch size")
    parser.add_argument("--yolo-epochs", type=int, default=10, help="YOLO training epochs")
    parser.add_argument("--yolo-batch", type=int, default=4, help="YOLO batch size")
    parser.add_argument("--yolo-imgsz", type=int, default=3840, help="YOLO image size (for 10x SR: 384*10=3840)")
    
    # Output directories
    parser.add_argument("--data-root", type=str, default="data", help="Root directory for datasets")
    parser.add_argument("--weights-dir", type=str, default="weights", help="Directory for model weights")
    parser.add_argument("--output-dir", type=str, default="pipeline_outputs", help="Directory for inference outputs")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory for training logs")
    
    # Options
    parser.add_argument("--skip-dataset-creation", action="store_true", help="Skip dataset creation (use existing)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (use existing weights)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference")
    parser.add_argument("--use-existing-weights", action="store_true", help="Use existing weights instead of training")
    
    args = parser.parse_args()
    
    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent
    coords_file = repo_root / args.coords
    dotenv_file = repo_root / args.dotenv
    
    if not coords_file.exists():
        raise FileNotFoundError(f"Coordinates file not found: {coords_file}")
    
    coords = read_coords(coords_file)
    lat = float(coords["lat"])
    lon = float(coords["lon"])
    radius_m = float(coords.get("radius_m", 20.0))
    
    print(f"\n{'#'*80}")
    print(f"# FULL PIPELINE: Building Footprint Detection (3-channel)")
    print(f"{'#'*80}")
    print(f"Location: ({lat:.6f}, {lon:.6f})")
    print(f"Radius: {radius_m}m")
    print(f"Training samples: {args.n_train} train, {args.n_val} val, {args.n_test} test")
    print(f"Epochs: ReFineNet={args.refinenet_epochs}, YOLO={args.yolo_epochs}")
    print(f"{'#'*80}\n")
    
    # Define dataset paths
    data_root = Path(args.data_root)
    s2_osm_root = data_root / "s2_osm_sr" / "building_patches"
    yolo_s2_osm_root = data_root / "yolo_buildings_sr"
    
    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    
    # Create directories
    weights_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # STEP 1: Create 3-channel dataset (s2_osm_sr)
    # ============================================================================
    if not args.skip_dataset_creation:
        print(f"\n{'='*80}")
        print("STEP 1: Creating 3-channel dataset with super resolution")
        print(f"{'='*80}")
        
        cmd = [
            sys.executable, "scripts/prepare_s2_osm_dataset.py",
            "--center-lat", str(lat),
            "--center-lon", str(lon),
            "--sample-radius-m", str(args.sample_radius_m),
            "--n-train", str(args.n_train),
            "--n-val", str(args.n_val),
            "--n-test", str(args.n_test),
            "--out-root", str(s2_osm_root),
            "--use-sr",
            "--sr-scale-factor", str(args.sr_scale_factor),
            "--sr-method", args.sr_method,
        ]
        
        if args.sr_model_path:
            cmd.extend(["--sr-model-path", args.sr_model_path])
        if dotenv_file.exists():
            cmd.extend(["--dotenv", str(dotenv_file)])
        
        run_command(cmd, "Creating 3-channel dataset (s2_osm_sr)")
    else:
        print(f"\nSkipping dataset creation (using existing dataset)")
    
    # ============================================================================
    # STEP 2: Convert to YOLO format (3-channel)
    # ============================================================================
    if not args.skip_training:
        print(f"\n{'='*80}")
        print("STEP 2: Converting 3-channel dataset to YOLO format")
        print(f"{'='*80}")
        
        cmd = [
            sys.executable, "scripts/convert_to_yolo_format.py",
            "--source-root", str(s2_osm_root),
            "--yolo-root", str(yolo_s2_osm_root),
        ]
        
        run_command(cmd, "Converting to YOLO format (3-channel)")
    
    
    # ============================================================================
    # STEP 3: Train ReFineNet on 3-channel dataset
    # ============================================================================
    if not args.skip_training and not args.use_existing_weights:
        print(f"\n{'='*80}")
        print("STEP 3: Training ReFineNet on 3-channel dataset")
        print(f"{'='*80}")
        
        # Create config for 3-channel SR dataset
        config_3ch = {
            "Segmentation": "binary",
            "Model": {
                "name": "ReFineNet",
                "param": {
                    "backbone": "resnet50"
                }
            },
            "Criterion": {
                "name": "Dice",
                "dice_weights": 1.0
            },
            "Loader": {
                "root": str(s2_osm_root),
                "image_normalizer": "divide_by_255",
                "label_normalizer": "binary_label",
                "augmenters": [
                    "horizontal_flip",
                    "vertical_flip",
                    {"rotation": 15}
                ],
                "batch_size": args.refinenet_batch_size
            },
            "Metrics": ["accuracy", "precision", "f1", "recall", "iou"],
            "Optimizer": {
                "name": "AdamW",
                "param": {
                    "lr": 1e-4
                }
            },
            "Callbacks": {
                "log_dir": str(logs_dir),
                "callers": ["TimeCallback", "TrainStateCallback", "TrainChkCallback"]
            }
        }
        
        config_path_3ch = repo_root / "configs" / "train_s2_osm_sr_pipeline.yaml"
        config_path_3ch.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path_3ch, 'w') as f:
            yaml.dump(config_3ch, f, default_flow_style=False)
        
        cmd = [
            sys.executable, "train_refinenet.py",
            "--config", str(config_path_3ch),
            "--epochs", str(args.refinenet_epochs),
            "--batch-size", str(args.refinenet_batch_size),
            "--log-root", str(logs_dir),
            "--export-dir", str(weights_dir),
            "--export-name", "best_sr_3ch.pt",
        ]
        
        run_command(cmd, "Training ReFineNet (3-channel)")
    
    
    # ============================================================================
    # STEP 4: Train YOLO on 3-channel dataset
    # ============================================================================
    if not args.skip_training and not args.use_existing_weights:
        print(f"\n{'='*80}")
        print("STEP 4: Training YOLO on 3-channel dataset")
        print(f"{'='*80}")
        
        cmd = [
            sys.executable, "scripts/train_yolo.py",
            "--data", str(yolo_s2_osm_root),
            "--epochs", str(args.yolo_epochs),
            "--imgsz", str(args.yolo_imgsz),
            "--batch", str(args.yolo_batch),
            "--model-size", "n",
            "--project", str(logs_dir / "yolo_runs"),
            "--name", "buildings_sr_3ch",
        ]
        
        run_command(cmd, "Training YOLO (3-channel)")
        
        # Copy weights
        yolo_weights_3ch = logs_dir / "yolo_runs" / "buildings_sr_3ch" / "weights" / "best.pt"
        if yolo_weights_3ch.exists():
            import shutil
            target = weights_dir / "yolo_buildings_sr_3ch.pt"
            shutil.copy2(yolo_weights_3ch, target)
            print(f"Copied YOLO weights to: {target}")
    
    
    # ============================================================================
    # STEP 5: Run inference with 3-channel model
    # ============================================================================
    if not args.skip_inference:
        print(f"\n{'='*80}")
        print("STEP 5: Running inference with 3-channel model")
        print(f"{'='*80}")
        
        refinenet_weights_3ch = weights_dir / "best_sr_3ch.pt"
        yolo_weights_3ch = weights_dir / "yolo_buildings_sr_3ch.pt"
        output_3ch = output_dir / "output_3ch"
        
        cmd = [
            sys.executable, "run_nearby.py",
            "--coords", str(coords_file),
            "--weights", str(refinenet_weights_3ch),
            "--outdir", str(output_3ch),
        ]
        
        if yolo_weights_3ch.exists():
            cmd.extend(["--yolo-weights", str(yolo_weights_3ch)])
        
        if dotenv_file.exists():
            cmd.extend(["--dotenv", str(dotenv_file)])
        
        # Use defaults (OSM enabled, SR enabled, post-processing enabled, etc.)
        run_command(cmd, "Inference with 3-channel model", check=False)

    
    # ============================================================================
    # Summary
    # ============================================================================
    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLETE")
    print(f"{'#'*80}")
    print(f"\nDataset:")
    print(f"  - 3-channel: {s2_osm_root}")
    print(f"\nWeights:")
    print(f"  - ReFineNet: {weights_dir / 'best_sr_3ch.pt'}")
    print(f"  - YOLO: {weights_dir / 'yolo_buildings_sr_3ch.pt'}")
    print(f"\nOutputs:")
    print(f"  - Results: {output_dir / 'output_3ch'}")
    print(f"\n{'#'*80}\n")


if __name__ == "__main__":
    main()
