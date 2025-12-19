from __future__ import annotations

from typing import Literal


FuseMode = Literal["none", "union", "intersection", "refinenet_only", "yolo_only"]


def fuse_binary_masks(
    refinenet_mask01: "object",  # np.ndarray HxW uint8 {0,1}
    yolo_mask01: "object",  # np.ndarray HxW uint8 {0,1}
    mode: FuseMode = "intersection",
) -> "object":
    import numpy as np
    import cv2

    if mode in ("none", "refinenet_only"):
        return refinenet_mask01.astype(np.uint8, copy=False)
    if mode == "yolo_only":
        return yolo_mask01.astype(np.uint8, copy=False)
    
    # Ensure masks have the same size (resize smaller to match larger)
    if refinenet_mask01.shape != yolo_mask01.shape:
        target_h, target_w = refinenet_mask01.shape
        yolo_h, yolo_w = yolo_mask01.shape
        
        if yolo_h != target_h or yolo_w != target_w:
            # Resize YOLO mask to match ReFineNet mask size
            yolo_mask01 = cv2.resize(
                yolo_mask01.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
    
    if mode == "union":
        return np.logical_or(refinenet_mask01 > 0, yolo_mask01 > 0).astype(np.uint8)
    if mode == "intersection":
        return np.logical_and(refinenet_mask01 > 0, yolo_mask01 > 0).astype(np.uint8)
    raise ValueError(f"Unknown fuse mode: {mode}")


