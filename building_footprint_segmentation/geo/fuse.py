from __future__ import annotations

from typing import Literal


FuseMode = Literal["none", "union", "intersection", "refinenet_only", "yolo_only"]


def fuse_binary_masks(
    refinenet_mask01: "object",  # np.ndarray HxW uint8 {0,1}
    yolo_mask01: "object",  # np.ndarray HxW uint8 {0,1}
    mode: FuseMode = "intersection",
) -> "object":
    import numpy as np

    if mode in ("none", "refinenet_only"):
        return refinenet_mask01.astype(np.uint8, copy=False)
    if mode == "yolo_only":
        return yolo_mask01.astype(np.uint8, copy=False)
    if mode == "union":
        return np.logical_or(refinenet_mask01 > 0, yolo_mask01 > 0).astype(np.uint8)
    if mode == "intersection":
        return np.logical_and(refinenet_mask01 > 0, yolo_mask01 > 0).astype(np.uint8)
    raise ValueError(f"Unknown fuse mode: {mode}")


