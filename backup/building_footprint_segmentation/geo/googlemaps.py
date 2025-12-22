"""
Fetch Google Maps satellite imagery for building footprint detection.

Uses Google Maps Static API to fetch high-resolution satellite imagery
as an additional input channel for the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os


@dataclass(frozen=True)
class GoogleMapsPatch:
    """Google Maps satellite imagery patch."""
    rgb: "object"  # np.ndarray (uint8 HxWx3)
    bbox_projected: Tuple[float, float, float, float]  # minx,miny,maxx,maxy in projected CRS
    projected_epsg: int
    center_projected: Tuple[float, float]  # x,y in projected CRS


def fetch_googlemaps_satellite_patch(
    *,
    lat: float,
    lon: float,
    patch_px: int = 384,
    resolution_m: float = 10.0,
    api_key: Optional[str] = None,
    maptype: str = "satellite",
    zoom: Optional[int] = None,
) -> GoogleMapsPatch:
    """
    Fetch Google Maps satellite imagery patch.
    
    Args:
        lat: Latitude in WGS84
        lon: Longitude in WGS84
        patch_px: Output image size in pixels (square)
        resolution_m: Desired resolution in meters per pixel
        api_key: Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)
        maptype: Map type ("satellite", "hybrid")
        zoom: Zoom level (auto-calculated from resolution_m if None)
        
    Returns:
        GoogleMapsPatch with RGB image and geospatial metadata
    """
    import numpy as np
    import requests
    from PIL import Image
    from io import BytesIO
    from pyproj import CRS as PyCRS, Transformer
    
    if patch_px <= 0:
        raise ValueError("patch_px must be positive")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    
    if api_key is None:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Google Maps API key. Set GOOGLE_MAPS_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    # Calculate bounding box in WGS84
    # Approximate: 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude ≈ 111,000 * cos(latitude) meters
    meters_per_degree_lat = 111000.0
    meters_per_degree_lon = 111000.0 * np.cos(np.radians(lat))
    
    # Calculate patch size in degrees
    patch_size_m = patch_px * resolution_m
    half_size_deg_lat = (patch_size_m / 2.0) / meters_per_degree_lat
    half_size_deg_lon = (patch_size_m / 2.0) / meters_per_degree_lon
    
    # Bounding box in WGS84
    bbox_wgs84 = (
        lon - half_size_deg_lon,
        lat - half_size_deg_lat,
        lon + half_size_deg_lon,
        lat + half_size_deg_lat,
    )
    
    # Project to UTM for accurate measurements
    from sentinelhub import CRS
    sh_crs = CRS.get_utm_from_wgs84(lon, lat)
    projected_epsg = int(getattr(sh_crs, "epsg", sh_crs.value))
    
    wgs84 = PyCRS.from_epsg(4326)
    proj = PyCRS.from_epsg(projected_epsg)
    to_proj = Transformer.from_crs(wgs84, proj, always_xy=True)
    
    cx, cy = to_proj.transform(lon, lat)
    half = (patch_px * float(resolution_m)) / 2.0
    bbox_proj = (cx - half, cy - half, cx + half, cy + half)
    
    # Calculate zoom level from resolution
    # Google Maps zoom levels: https://developers.google.com/maps/documentation/static-maps/overview
    # Approximate: zoom 20 = ~0.5m/pixel, zoom 19 = ~1m/pixel, zoom 18 = ~2m/pixel, etc.
    if zoom is None:
        # Estimate zoom from desired resolution
        # Formula: resolution_m ≈ 156543.03392 * cos(lat) / (2^zoom)
        # Solving for zoom: zoom ≈ log2(156543.03392 * cos(lat) / resolution_m)
        import math
        base_res = 156543.03392 * np.cos(np.radians(lat))  # meters per pixel at zoom 0
        zoom = int(np.clip(np.log2(base_res / resolution_m), 1, 21))
    
    # Fetch image from Google Maps Static API
    # Use center point and calculate scale/zoom
    url = "https://maps.googleapis.com/maps/api/staticmap"
    
    # Calculate scale parameter (1, 2, or 4 for higher resolution)
    # Scale 2 gives 2x resolution (e.g., 640x640 request gives 1280x1280 image)
    scale = 2 if patch_px > 640 else 1
    
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{patch_px}x{patch_px}",
        "maptype": maptype,
        "key": api_key,
        "scale": scale,  # Higher resolution
        "format": "png",
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    # Parse response
    if response.headers.get("content-type", "").startswith("application/json"):
        # Error response
        error_data = response.json()
        raise RuntimeError(f"Google Maps API error: {error_data}")
    
    # Load image
    img = Image.open(BytesIO(response.content))
    img_rgb = img.convert("RGB")
    
    # Convert to numpy array
    rgb = np.array(img_rgb, dtype=np.uint8)
    
    # Resize if needed (due to scale parameter)
    if scale > 1:
        actual_size = rgb.shape[0]
        if actual_size != patch_px:
            from PIL import Image as PILImage
            img_resized = PILImage.fromarray(rgb).resize((patch_px, patch_px), PILImage.Resampling.LANCZOS)
            rgb = np.array(img_resized, dtype=np.uint8)
    
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Unexpected Google Maps image shape: {rgb.shape}")
    
    return GoogleMapsPatch(
        rgb=rgb,
        bbox_projected=bbox_proj,
        projected_epsg=projected_epsg,
        center_projected=(cx, cy),
    )

