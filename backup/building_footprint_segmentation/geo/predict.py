from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import torch

from building_footprint_segmentation.seg.binary.models import ReFineNet, ReFineNet4Ch
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

    # Load state dict first to detect input channels
    state = torch.load(weights_path, map_location=device, weights_only=False)
    
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    
    if not isinstance(state, dict):
        raise ValueError("Unsupported weights format. Expected a state_dict or a dict containing 'model'.")

    cleaned: Dict[str, Any] = {}
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    
    # Detect input channels from layer0.0.weight (first conv layer)
    input_channels = 3  # default
    if "layer0.0.weight" in cleaned:
        weight_shape = cleaned["layer0.0.weight"].shape
        if len(weight_shape) == 4:
            input_channels = weight_shape[1]  # [out_channels, in_channels, H, W]
    
    # Create appropriate model based on detected channels
    if input_channels == 4:
        model = ReFineNet4Ch(input_channels=4).to(device)
    elif input_channels == 3:
        model = ReFineNet().to(device)
    else:
        raise ValueError(f"Unsupported input_channels={input_channels}. Expected 3 or 4.")

    model.load_state_dict(cleaned, strict=strict)
    model.eval()
    return model


def _normalize_uint8_4channel(rgb_uint8: np.ndarray, osm_mask01: np.ndarray) -> np.ndarray:
    """Normalize RGB+OSM 4-channel image to float32 in [0,1]."""
    if rgb_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got dtype={rgb_uint8.dtype}")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb_uint8.shape}")
    if osm_mask01.dtype != np.uint8:
        raise ValueError(f"Expected uint8 OSM mask, got dtype={osm_mask01.dtype}")
    if osm_mask01.ndim != 2:
        raise ValueError(f"Expected 2D OSM mask, got shape {osm_mask01.shape}")
    
    # Normalize RGB to [0,1]
    rgb_norm = (rgb_uint8.astype(np.float32) / 255.0).astype(np.float32)
    # Normalize OSM mask to [0,1] (0 or 1 values)
    osm_norm = osm_mask01.astype(np.float32)
    
    # Stack: HxWx4
    four_ch = np.dstack([rgb_norm, osm_norm])
    return four_ch


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


@torch.no_grad()
def predict_refinenet_on_rgb_osm(
    model: torch.nn.Module,
    rgb_uint8: np.ndarray,
    osm_mask01: np.ndarray,
    threshold: float = 0.5,
    model_input_size: Tuple[int, int] = (384, 384),
) -> Prediction:
    """
    Predict using 4-channel input (RGB + OSM mask).
    
    Args:
        model: ReFineNet4Ch model (4-channel input)
        rgb_uint8: HxWx3 uint8 RGB image
        osm_mask01: HxW uint8 binary mask {0,1}
        threshold: Probability threshold for binary mask
        model_input_size: Target input size for model
        
    Returns:
        Prediction with probability map and binary mask
    """
    rgb_uint8 = handle_image_size(rgb_uint8, model_input_size)
    # Handle OSM mask size to match RGB
    if osm_mask01.shape[:2] != rgb_uint8.shape[:2]:
        import cv2
        osm_mask01 = cv2.resize(
            osm_mask01.astype(np.uint8),
            (rgb_uint8.shape[1], rgb_uint8.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
    
    img_4ch = _normalize_uint8_4channel(rgb_uint8, osm_mask01)

    x = add_extra_dimension(to_input_image_tensor(img_4ch))
    device = next(model.parameters()).device
    x = x.to(device)

    logits = model(x)  # [1,1,H,W]
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    mask = (prob >= float(threshold)).astype(np.uint8)
    return Prediction(prob=prob, mask=mask)


