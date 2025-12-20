"""
Enhanced post-processing for building footprint accuracy.

Improves polygon quality through:
- Boundary smoothing
- Artifact removal
- Shape regularization
- Hole filling
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np


def smooth_polygon_boundary(
    polygon: "object",  # shapely Polygon
    tolerance_m: float = 0.5,
) -> "object":
    """
    Smooth polygon boundary using Douglas-Peucker simplification.
    
    Args:
        polygon: Shapely polygon
        tolerance_m: Simplification tolerance in meters
        
    Returns:
        Smoothed polygon
    """
    from shapely.geometry import Polygon
    
    if polygon.is_empty:
        return polygon
    
    # Simplify the exterior ring
    simplified_exterior = polygon.exterior.simplify(tolerance_m, preserve_topology=True)
    
    # Simplify interior rings (holes)
    simplified_interiors = [
        interior.simplify(tolerance_m, preserve_topology=True)
        for interior in polygon.interiors
    ]
    
    return Polygon(simplified_exterior, simplified_interiors)


def remove_small_holes(
    polygon: "object",  # shapely Polygon
    min_hole_area_m2: float = 5.0,
) -> "object":
    """
    Remove small holes from polygon.
    
    Args:
        polygon: Shapely polygon
        min_hole_area_m2: Minimum hole area to keep (in m²)
        
    Returns:
        Polygon with small holes removed
    """
    from shapely.geometry import Polygon
    
    if polygon.is_empty or len(polygon.interiors) == 0:
        return polygon
    
    large_holes = [
        interior for interior in polygon.interiors
        if Polygon(interior).area >= min_hole_area_m2
    ]
    
    return Polygon(polygon.exterior, large_holes)


def remove_small_artifacts(
    polygons: List["object"],
    min_area_m2: float = 1.0,
    min_perimeter_m: float = 3.0,
) -> List["object"]:
    """
    Remove small artifacts and noise polygons.
    
    Args:
        polygons: List of shapely polygons
        min_area_m2: Minimum area in m²
        min_perimeter_m: Minimum perimeter in meters
        
    Returns:
        Filtered list of polygons
    """
    filtered = []
    for poly in polygons:
        if poly.is_empty:
            continue
        
        area = poly.area
        perimeter = poly.length
        
        # Remove if too small
        if area < min_area_m2:
            continue
        
        # Remove if perimeter too small (likely noise)
        if perimeter < min_perimeter_m:
            continue
        
        # Remove if very elongated (likely artifact)
        if area > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            if compactness < 0.1:  # Very elongated
                continue
        
        filtered.append(poly)
    
    return filtered


def regularize_building_shape(
    polygon: "object",  # shapely Polygon
    method: str = "min_area_rect",
    angle_tolerance_deg: float = 5.0,
) -> "object":
    """
    Regularize building shape (e.g., make rectangular if close to rectangle).
    
    Args:
        polygon: Shapely polygon
        method: "min_area_rect" or "none"
        angle_tolerance_deg: Tolerance for detecting rectangular shapes
        
    Returns:
        Regularized polygon
    """
    if method == "none" or polygon.is_empty:
        return polygon
    
    from shapely.geometry import Polygon, box
    from shapely.affinity import rotate
    
    if method == "min_area_rect":
        # Get minimum area bounding rectangle
        min_rect = polygon.minimum_rotated_rectangle
        
        # Check if polygon is close to rectangular
        area_ratio = polygon.area / min_rect.area
        if area_ratio > 0.85:  # Close to rectangular
            # Use the minimum rotated rectangle
            return min_rect
    
    return polygon


def apply_morphological_operations(
    mask01: np.ndarray,
    operation: str = "closing",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply morphological operations to clean mask.
    
    Args:
        mask01: Binary mask (0/1)
        operation: "closing" (fill gaps), "opening" (remove noise), "none"
        kernel_size: Kernel size in pixels
        
    Returns:
        Cleaned mask
    """
    import cv2
    
    if operation == "none":
        return mask01
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == "closing":
        # Fill small gaps
        return cv2.morphologyEx(mask01.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.uint8)
    elif operation == "opening":
        # Remove small noise
        return cv2.morphologyEx(mask01.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(np.uint8)
    elif operation == "both":
        # Opening then closing
        opened = cv2.morphologyEx(mask01.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
    
    return mask01


def postprocess_polygons(
    polygons: List["object"],
    *,
    smooth: bool = True,
    smooth_tolerance_m: float = 0.5,
    remove_holes: bool = True,
    min_hole_area_m2: float = 5.0,
    regularize: bool = False,
    regularize_method: str = "min_area_rect",
    min_area_m2: float = 1.0,
    min_perimeter_m: float = 3.0,
) -> List["object"]:
    """
    Comprehensive post-processing pipeline for building polygons.
    
    Args:
        polygons: List of shapely polygons
        smooth: Apply boundary smoothing
        smooth_tolerance_m: Smoothing tolerance
        remove_holes: Remove small holes
        min_hole_area_m2: Minimum hole area to keep
        regularize: Regularize shapes (e.g., rectangularize)
        regularize_method: Regularization method
        min_area_m2: Minimum polygon area
        min_perimeter_m: Minimum polygon perimeter
        
    Returns:
        Post-processed polygons
    """
    if not polygons:
        return []
    
    processed = []
    
    for poly in polygons:
        if poly.is_empty:
            continue
        
        # Remove small holes
        if remove_holes:
            poly = remove_small_holes(poly, min_hole_area_m2)
        
        # Smooth boundary
        if smooth:
            poly = smooth_polygon_boundary(poly, smooth_tolerance_m)
        
        # Regularize shape
        if regularize:
            poly = regularize_building_shape(poly, regularize_method)
        
        if not poly.is_empty and poly.area >= min_area_m2:
            processed.append(poly)
    
    # Remove small artifacts
    processed = remove_small_artifacts(processed, min_area_m2, min_perimeter_m)
    
    return processed


def postprocess_mask(
    mask01: np.ndarray,
    *,
    morphological: str = "closing",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Post-process binary mask with morphological operations.
    
    Args:
        mask01: Binary mask (0/1)
        morphological: Operation ("closing", "opening", "both", "none")
        kernel_size: Kernel size in pixels
        
    Returns:
        Post-processed mask
    """
    return apply_morphological_operations(mask01, morphological, kernel_size)

