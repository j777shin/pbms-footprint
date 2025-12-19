from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def apply_super_resolution(
    rgb_uint8: np.ndarray,
    scale_factor: int = 2,
    method: str = "bicubic",
    use_real_esrgan: bool = False,
    real_esrgan_model: Optional[str] = None,
) -> np.ndarray:
    """
    Apply super resolution to enhance satellite imagery resolution.
    
    Args:
        rgb_uint8: HxWx3 uint8 RGB image
        scale_factor: Upscaling factor (2 = 2x, 4 = 4x)
        method: Interpolation method if not using Real-ESRGAN: "bicubic", "bilinear", "lanczos"
        use_real_esrgan: If True, use Real-ESRGAN model (requires realesrgan package)
        real_esrgan_model: Real-ESRGAN model name (e.g., "realesrgan-x4plus", "realesrgan-x4plus-anime")
        
    Returns:
        Enhanced RGB image with shape (H*scale_factor, W*scale_factor, 3) as uint8
    """
    if rgb_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got dtype={rgb_uint8.dtype}")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb_uint8.shape}")
    
    if use_real_esrgan:
        try:
            from realesrgan import RealESRGAN
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if real_esrgan_model is None:
                # Default to x4plus for general images
                real_esrgan_model = "realesrgan-x4plus"
            
            # Initialize model
            model = RealESRGAN(device, scale=scale_factor, model_name=real_esrgan_model)
            model.load_weights()
            
            # Convert to PIL Image for Real-ESRGAN
            from PIL import Image
            pil_img = Image.fromarray(rgb_uint8)
            
            # Apply super resolution
            sr_img = model.predict(pil_img)
            
            # Convert back to numpy array
            sr_array = np.array(sr_img).astype(np.uint8)
            
            return sr_array
            
        except ImportError:
            print("Warning: realesrgan not installed. Falling back to bicubic upsampling.")
            print("Install with: pip install realesrgan")
            use_real_esrgan = False
    
    if not use_real_esrgan:
        # Use OpenCV for upsampling
        import cv2
        
        h, w = rgb_uint8.shape[:2]
        new_h, new_w = h * scale_factor, w * scale_factor
        
        if method == "bicubic":
            interpolation = cv2.INTER_CUBIC
        elif method == "bilinear":
            interpolation = cv2.INTER_LINEAR
        elif method == "lanczos":
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_CUBIC
        
        # Upscale using selected interpolation
        upscaled = cv2.resize(
            rgb_uint8,
            (new_w, new_h),
            interpolation=interpolation
        )
        
        # Optional: Apply sharpening to enhance details
        # This helps recover some detail lost in upsampling
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        # Clip to valid range and convert back to uint8
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened


def apply_super_resolution_adaptive(
    rgb_uint8: np.ndarray,
    target_resolution_m: float = 2.5,  # Target resolution in meters (e.g., 2.5m from 10m)
    source_resolution_m: float = 10.0,
    method: str = "bicubic",
    use_real_esrgan: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Apply super resolution to achieve a target resolution.
    
    Args:
        rgb_uint8: HxWx3 uint8 RGB image
        target_resolution_m: Desired resolution in meters per pixel
        source_resolution_m: Current resolution in meters per pixel (e.g., 10.0 for Sentinel-2)
        method: Interpolation method
        use_real_esrgan: If True, use Real-ESRGAN model
        
    Returns:
        Tuple of (enhanced RGB image, actual achieved resolution in meters)
    """
    scale_factor = int(np.ceil(source_resolution_m / target_resolution_m))
    
    if scale_factor <= 1:
        # Already at or better than target resolution
        return rgb_uint8, source_resolution_m
    
    enhanced = apply_super_resolution(
        rgb_uint8,
        scale_factor=scale_factor,
        method=method,
        use_real_esrgan=use_real_esrgan,
    )
    
    actual_resolution = source_resolution_m / scale_factor
    
    return enhanced, actual_resolution

