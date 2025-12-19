from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class YOLOMask:
    mask01: "object"  # np.ndarray HxW uint8 {0,1}


def predict_yolov8_mask_on_rgb(
    rgb_uint8: "object",  # np.ndarray HxWx3 uint8 RGB
    *,
    yolo_weights: str,
    device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.7,
    classes: Optional[Sequence[int]] = None,
) -> YOLOMask:
    import numpy as np
    import cv2

    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "ultralytics is required for YOLOv8 integration. Install with: pip install ultralytics"
        ) from e

    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {getattr(rgb_uint8, 'shape', None)}")
    if rgb_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype={rgb_uint8.dtype}")

    h, w, _ = rgb_uint8.shape

    # Ultralytics commonly expects BGR when feeding numpy (OpenCV conventions).
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    model = YOLO(yolo_weights)
    results = model.predict(
        source=bgr,
        device=device,
        conf=float(conf),
        iou=float(iou),
        classes=list(classes) if classes is not None else None,
        verbose=False,
    )
    if not results:
        return YOLOMask(mask01=np.zeros((h, w), dtype=np.uint8))

    r0 = results[0]

    # Prefer segmentation masks if available
    if getattr(r0, "masks", None) is not None and getattr(r0.masks, "data", None) is not None:
        # masks.data: (n, h, w) in torch, values in {0,1}
        md = r0.masks.data
        md = md.detach().cpu().numpy()
        if md.size == 0:
            return YOLOMask(mask01=np.zeros((h, w), dtype=np.uint8))
        union = (md.sum(axis=0) > 0).astype(np.uint8)
        return YOLOMask(mask01=union)

    # Fallback: boxes only -> rasterize as filled rectangles
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = getattr(r0, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return YOLOMask(mask01=mask)

    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    for x1, y1, x2, y2 in xyxy:
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        mask[y1 : y2 + 1, x1 : x2 + 1] = 1

    return YOLOMask(mask01=mask)


