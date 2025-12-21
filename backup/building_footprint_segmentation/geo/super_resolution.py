from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def _load_srdr3_model(scale_factor: int, device, model_path: Optional[str] = None):
    """
    Load SRDR3 (Super-Resolution Deep Residual Network 3) model.
    
    Args:
        scale_factor: Upscaling factor (2, 4, etc.)
        device: torch device (cuda or cpu)
        model_path: Optional path to model weights
        
    Returns:
        Loaded SRDR3 model
    """
    try:
        import torch
        import torch.nn as nn
        
        # Try to import SRDR3 from common repositories
        # If you have a specific SRDR3 implementation, adjust this
        try:
            # Option 1: Try importing from a package
            from srdr3 import SRDR3
            model = SRDR3(scale=scale_factor)
        except ImportError:
            try:
                # Option 2: Try importing from basicsr or similar
                from basicsr.archs.srdr3_arch import SRDR3
                model = SRDR3(scale=scale_factor)
            except ImportError:
                # Option 3: Use a simple SRDR3-like architecture
                # This is a basic implementation - replace with actual SRDR3 if available
                model = _create_srdr3_architecture(scale_factor)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        raise ImportError(f"Failed to load SRDR3 model: {e}. Please install SRDR3 or provide model weights.")


def _create_srdr3_architecture(scale_factor: int):
    """
    Create a basic SRDR3-like architecture if the actual model is not available.
    This is a placeholder - replace with actual SRDR3 implementation.
    """
    import torch
    import torch.nn as nn
    
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return self.relu(out)
    
    class SRDR3Basic(nn.Module):
        def __init__(self, scale=4, num_channels=3, num_residual_blocks=16):
            super().__init__()
            self.scale = scale
            
            # Initial feature extraction
            self.conv_input = nn.Conv2d(num_channels, 64, 9, padding=4)
            self.relu = nn.ReLU(inplace=True)
            
            # Residual blocks
            self.residual_layers = nn.Sequential(*[
                ResidualBlock(64) for _ in range(num_residual_blocks)
            ])
            
            # Upsampling
            self.conv_mid = nn.Conv2d(64, 64, 3, padding=1)
            self.bn_mid = nn.BatchNorm2d(64)
            upscale_layers = []
            remaining_scale = scale
            
            # Chain 2x upscales for powers of 2
            # For non-power-of-2 scales (e.g., 10x), we'll do 8x via model and 1.25x via interpolation
            while remaining_scale >= 2:
                upscale_layers.extend([
                    nn.Conv2d(64, 256, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True)
                ])
                remaining_scale /= 2
            
            # Store remainder for post-processing interpolation
            # (e.g., for 10x: 3x 2x upscales = 8x, remainder = 1.25x handled via interpolation)
            if remaining_scale > 1.0:
                # Remainder will be handled via interpolation after model inference
                pass
            
            if not upscale_layers:
                raise ValueError(f"Unsupported scale factor: {scale}")
            
            self.upscale = nn.Sequential(*upscale_layers)
            
            # Output
            self.conv_output = nn.Conv2d(64, num_channels, 9, padding=4)
        
        def forward(self, x):
            out = self.relu(self.conv_input(x))
            residual = out
            out = self.residual_layers(out)
            out = self.bn_mid(self.conv_mid(out))
            out += residual
            out = self.upscale(out)
            out = self.conv_output(out)
            return out
    
    return SRDR3Basic(scale=scale_factor)


def apply_super_resolution(
    rgb_uint8: np.ndarray,
    scale_factor: int = 2,
    method: str = "srdr3",
    use_deep_model: bool = True,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Apply super resolution using SRDR3 (Super-Resolution Deep Residual Network 3) model.
    
    Args:
        rgb_uint8: HxWx3 uint8 RGB image
        scale_factor: Upscaling factor (2 = 2x, 4 = 4x)
        method: Method to use - "srdr3" (default), "bicubic", "bilinear", "lanczos", "real_esrgan"
        use_deep_model: If True, use deep learning model (SRDR3 by default)
        model_path: Optional path to SRDR3 model weights
        device: Device to use ("cuda", "cpu", or None for auto)
        
    Returns:
        Enhanced RGB image with shape (H*scale_factor, W*scale_factor, 3) as uint8
    """
    if rgb_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got dtype={rgb_uint8.dtype}")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb_uint8.shape}")
    
    # Use SRDR3 by default if use_deep_model is True
    if use_deep_model and method in ("srdr3", "deep", "default"):
        try:
            import torch
            from PIL import Image
            
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device)
            
            # Load SRDR3 model
            model = _load_srdr3_model(scale_factor, device, model_path)
            
            # Convert image to tensor
            # Normalize to [0, 1] and convert to tensor
            img_float = rgb_uint8.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Apply super resolution with memory management
            with torch.no_grad():
                sr_tensor = model(img_tensor)
                sr_tensor = torch.clamp(sr_tensor, 0, 1)
            
            # Convert back to numpy and free GPU memory immediately
            sr_array = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Clear GPU cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Delete tensors to free memory
            del sr_tensor, img_tensor
            
            # Handle remainder scaling if needed (e.g., for 10x = 8x + 1.25x)
            h, w = rgb_uint8.shape[:2]
            target_h, target_w = int(h * scale_factor), int(w * scale_factor)
            actual_h, actual_w = sr_array.shape[:2]
            
            if actual_h != target_h or actual_w != target_w:
                # Apply final interpolation to reach exact target size
                import cv2
                sr_array = cv2.resize(
                    (sr_array * 255.0).astype(np.uint8),
                    (target_w, target_h),
                    interpolation=cv2.INTER_CUBIC
                )
            else:
                sr_array = (sr_array * 255.0).astype(np.uint8)
            
            return sr_array
            
        except Exception as e:
            print(f"Warning: SRDR3 model failed: {e}")
            print("Falling back to bicubic upsampling.")
            method = "bicubic"
            use_deep_model = False
    
    # Fallback to Real-ESRGAN if requested
    if method == "real_esrgan" or (use_deep_model and method not in ("srdr3", "deep", "default")):
        try:
            from realesrgan import RealESRGAN
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize model
            model = RealESRGAN(device, scale=scale_factor, model_name="realesrgan-x4plus")
            model.load_weights()
            
            # Convert to PIL Image
            from PIL import Image
            pil_img = Image.fromarray(rgb_uint8)
            
            # Apply super resolution
            sr_img = model.predict(pil_img)
            
            # Convert back to numpy array
            sr_array = np.array(sr_img).astype(np.uint8)
            
            return sr_array
            
        except ImportError:
            print("Warning: realesrgan not installed. Falling back to bicubic upsampling.")
            method = "bicubic"
    
    # Fallback to traditional interpolation methods
    if not use_deep_model or method in ("bicubic", "bilinear", "lanczos"):
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
    method: str = "srdr3",
    use_deep_model: bool = True,
    model_path: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """
    Apply super resolution using SRDR3 to achieve a target resolution.
    
    Args:
        rgb_uint8: HxWx3 uint8 RGB image
        target_resolution_m: Desired resolution in meters per pixel
        source_resolution_m: Current resolution in meters per pixel (e.g., 10.0 for Sentinel-2)
        method: Method to use - "srdr3" (default), "bicubic", "bilinear", "lanczos", "real_esrgan"
        use_deep_model: If True, use deep learning model (SRDR3 by default)
        model_path: Optional path to SRDR3 model weights
        
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
        use_deep_model=use_deep_model,
        model_path=model_path,
    )
    
    actual_resolution = source_resolution_m / scale_factor
    
    return enhanced, actual_resolution

