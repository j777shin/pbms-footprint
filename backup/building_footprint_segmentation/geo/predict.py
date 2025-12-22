from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import torch

from building_footprint_segmentation.seg.binary.models import ReFineNet
from building_footprint_segmentation.utils.py_network import to_input_image_tensor, add_extra_dimension
from building_footprint_segmentation.utils.operations import handle_image_size


@dataclass(frozen=True)
class Prediction:
    prob: np.ndarray  # HxW float32 in [0,1]
    mask: np.ndarray  # HxW uint8 {0,1}


def _normalize_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got dtype={rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}")
    return (rgb.astype(np.float32) / 255.0).astype(np.float32)


def load_refinenet(
    weights_path: str,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = False,
) -> torch.nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load state dict
    state = torch.load(weights_path, map_location=device, weights_only=False)
    
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    
    if not isinstance(state, dict):
        raise ValueError("Unsupported weights format. Expected a state_dict or a dict containing 'model'.")

    cleaned: Dict[str, Any] = {}
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    
    # Create 3-channel model
    model = ReFineNet().to(device)
    model.load_state_dict(cleaned, strict=strict)
    model.eval()
    return model


@torch.no_grad()
def predict_refinenet_on_rgb(
    model: torch.nn.Module,
    rgb_uint8: np.ndarray,
    threshold: float = 0.5,
    model_input_size: Tuple[int, int] = (384, 384),
) -> Prediction:
    rgb_uint8 = handle_image_size(rgb_uint8, model_input_size)
    img = _normalize_uint8_rgb(rgb_uint8)

    x = add_extra_dimension(to_input_image_tensor(img))
    device = next(model.parameters()).device
    x = x.to(device)

    logits = model(x)  # [1,1,H,W]
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    mask = (prob >= float(threshold)).astype(np.uint8)
    return Prediction(prob=prob, mask=mask)


