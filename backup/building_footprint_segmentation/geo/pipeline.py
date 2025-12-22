from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

from .sentinel2 import fetch_sentinel2_rgb_patch_sentinelhub
from .predict import load_refinenet, predict_refinenet_on_rgb, Prediction
from .vectorize import (
    mask_to_polygons_projected,
    clip_polygons_to_radius_m,
    reproject_geoms_to_wgs84,
    geoms_to_geojson_feature_collection,
)
from .yolo import predict_yolov8_mask_on_rgb
from .fuse import fuse_binary_masks, FuseMode
from .super_resolution import apply_super_resolution_adaptive
from .postprocess import postprocess_polygons, postprocess_mask


@dataclass(frozen=True)
class NearbyBuildingsResult:
    rgb_patch: np.ndarray  # HxWx3 uint8 RGB (SR-enhanced if SR was applied, otherwise original)
    rgb_patch_original: Optional[np.ndarray]  # HxWx3 uint8 RGB (original, before SR, None if SR not applied)
    projected_epsg: int
    bbox_projected: Tuple[float, float, float, float]
    center_projected: Tuple[float, float]

    refinenet: Prediction
    yolo_mask01: Optional[np.ndarray]
    fused_mask01: np.ndarray

    predicted_polygons_wgs84: list
    geojson_predicted: dict


def predict_buildings_near_coordinate(
    *,
    lat: float,
    lon: float,
    radius_m: float = 100.0,
    # Sentinel-2 (Sentinel Hub)
    patch_px: int = 384,
    resolution_m: float = 10.0,
    time_interval: Tuple[str, str] = ("2025-06-01", "2025-10-01"),
    mosaicking_order: str = "leastCC",
    sentinelhub_client_id: Optional[str] = None,
    sentinelhub_client_secret: Optional[str] = None,
    # ReFineNet
    refinenet_weights_path: str = "",
    refinenet_threshold: float = 0.5,
    # YOLO
    yolo_weights: Optional[str] = None,
    yolo_device: Optional[str] = None,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.7,
    yolo_classes: Optional[list] = None,
    fuse_mode: FuseMode = "intersection",
    # Super Resolution (SRDR3)
    use_super_resolution: bool = False,
    sr_target_resolution_m: float = 1.0,  # Target resolution (e.g., 1.0m from 10m = 10x)
    sr_method: str = "srdr3",  # "srdr3" (default), "bicubic", "bilinear", "lanczos", "real_esrgan"
    sr_model_path: Optional[str] = None,  # Optional path to SRDR3 model weights
    sr_device: Optional[str] = None,  # Device for SRDR3 ("cuda", "cpu", or None for auto)
    # Post-processing
    enable_postprocessing: bool = True,  # Enable enhanced post-processing
    postprocess_smooth: bool = True,  # Smooth polygon boundaries
    postprocess_smooth_tolerance_m: float = 0.5,  # Smoothing tolerance
    postprocess_remove_holes: bool = True,  # Remove small holes
    postprocess_min_hole_area_m2: float = 5.0,  # Minimum hole area to keep
    postprocess_regularize: bool = False,  # Regularize shapes (rectangularize)
    postprocess_morphological: str = "closing",  # Mask post-processing: "closing", "opening", "both", "none"
    # Vectorization
    min_area_m2: float = 1.0,
) -> NearbyBuildingsResult:
    """
    End-to-end:
    lat/lon -> Sentinel-2 patch -> ReFineNet mask (optional) -> optional YOLO fusion ->
    polygons -> post-processing -> clip to radius -> optional OSM intersection filter -> GeoJSON.
    
    Note: Either refinenet_weights_path or (yolo_weights with fuse_mode="yolo_only") must be provided.
    """
    # Validate that at least one model is provided
    if not refinenet_weights_path and (not yolo_weights or fuse_mode != "yolo_only"):
        raise ValueError(
            "Either refinenet_weights_path is required, or yolo_weights with fuse_mode='yolo_only' must be provided"
        )

    s2 = fetch_sentinel2_rgb_patch_sentinelhub(
        lat=lat,
        lon=lon,
        patch_px=patch_px,
        resolution_m=resolution_m,
        time_interval=time_interval,
        mosaicking_order=mosaicking_order,  # type: ignore[arg-type]
        client_id=sentinelhub_client_id,
        client_secret=sentinelhub_client_secret,
    )

    # Apply super resolution if enabled
    actual_resolution = resolution_m
    rgb_for_prediction = s2.rgb
    rgb_original = s2.rgb.copy()  # Keep original for saving (before SR)
    if use_super_resolution:
        print(f"Applying super resolution: {resolution_m}m -> {sr_target_resolution_m}m target...")
        rgb_for_prediction, actual_resolution = apply_super_resolution_adaptive(
            s2.rgb,
            target_resolution_m=sr_target_resolution_m,
            source_resolution_m=resolution_m,
            method=sr_method,
            use_deep_model=True,
            model_path=sr_model_path,
        )
        print(f"Super resolution applied. Enhanced resolution: {actual_resolution:.2f}m (scale: {resolution_m/actual_resolution:.1f}x)")
        # Update patch size for model input (it will be resized anyway, but this helps with OSM mask sizing)
        enhanced_patch_px = rgb_for_prediction.shape[0]
    else:
        enhanced_patch_px = patch_px
        rgb_original = None  # No need to save separately if no SR was applied

    # ReFineNet prediction (optional if using YOLO-only mode)
    pred = None
    if refinenet_weights_path:
        model = load_refinenet(refinenet_weights_path)
        pred = predict_refinenet_on_rgb(
            model, rgb_for_prediction, 
            threshold=refinenet_threshold, 
            model_input_size=(enhanced_patch_px, enhanced_patch_px)
        )

    # YOLO prediction
    yolo_mask01 = None
    if yolo_weights:
        ymask = predict_yolov8_mask_on_rgb(
            rgb_for_prediction,
            yolo_weights=yolo_weights,
            device=yolo_device,
            conf=yolo_conf,
            iou=yolo_iou,
            classes=yolo_classes,
        )
        yolo_mask01 = ymask.mask01
    
    # Fuse masks based on mode
    if fuse_mode == "yolo_only":
        if yolo_mask01 is None:
            raise ValueError("yolo_only mode requires yolo_weights to be provided")
        fused_mask01 = yolo_mask01
    elif pred is not None:
        if yolo_mask01 is not None:
            fused_mask01 = fuse_binary_masks(pred.mask, yolo_mask01, mode=fuse_mode)
        else:
            fused_mask01 = pred.mask
    else:
        raise ValueError("No predictions available: either refinenet_weights_path or yolo_weights must be provided")

    # Apply post-processing to mask before vectorization
    if enable_postprocessing and postprocess_morphological != "none":
        fused_mask01 = postprocess_mask(
            fused_mask01,
            morphological=postprocess_morphological,
            kernel_size=3,
        )

    vect = mask_to_polygons_projected(
        fused_mask01,
        bbox_projected=s2.bbox_projected,
        projected_epsg=s2.projected_epsg,
        min_area_m2=min_area_m2,
    )
    clipped = clip_polygons_to_radius_m(vect.polygons_projected, s2.center_projected, radius_m)
    
    # Filter to only the building at the exact coordinate
    from shapely.geometry import Point
    target_point = Point(s2.center_projected[0], s2.center_projected[1])
    closest_poly = None
    min_distance = float('inf')
    for poly in clipped:
        if poly.contains(target_point):
            clipped = [poly]
            break
        distance = poly.distance(target_point)
        if distance < min_distance and distance <= 50.0:  # 50m max distance
            min_distance = distance
            closest_poly = poly
    if closest_poly is not None and not any(p.contains(target_point) for p in clipped):
        clipped = [closest_poly]

    # Apply post-processing to polygons
    if enable_postprocessing:
        clipped = postprocess_polygons(
            clipped,
            smooth=postprocess_smooth,
            smooth_tolerance_m=postprocess_smooth_tolerance_m,
            remove_holes=postprocess_remove_holes,
            min_hole_area_m2=postprocess_min_hole_area_m2,
            regularize=postprocess_regularize,
            regularize_method="min_area_rect",
            min_area_m2=min_area_m2,
            min_perimeter_m=3.0,
        )


    # Calculate areas in projected CRS (meters) before reprojecting to WGS84
    # This gives accurate area measurements in square meters
    polygon_areas = [g.area for g in clipped]  # Area in square meters (projected CRS)

    pred_wgs84 = reproject_geoms_to_wgs84(clipped, from_epsg=s2.projected_epsg)
    
    # Create GeoJSON with accurate area measurements
    geojson_pred = geoms_to_geojson_feature_collection(
        pred_wgs84, 
        properties={"source": "model"},
        polygon_areas=polygon_areas,  # Pass areas for accurate calculation
    )

    # Use enhanced RGB if super resolution was applied, otherwise original
    output_rgb = rgb_for_prediction if use_super_resolution else s2.rgb
    
    # Create a dummy ReFineNet prediction if not available (for YOLO-only mode)
    if pred is None:
        from building_footprint_segmentation.geo.predict import Prediction
        import numpy as np
        # Create empty prediction with same shape as fused mask
        pred = Prediction(
            prob=np.zeros_like(fused_mask01, dtype=np.float32),
            mask=np.zeros_like(fused_mask01, dtype=np.uint8)
        )
    
    return NearbyBuildingsResult(
        rgb_patch=output_rgb,
        rgb_patch_original=rgb_original,  # Original before SR (None if SR not applied)
        projected_epsg=s2.projected_epsg,
        bbox_projected=s2.bbox_projected,
        center_projected=s2.center_projected,
        refinenet=pred,
        yolo_mask01=yolo_mask01,
        fused_mask01=fused_mask01,
        predicted_polygons_wgs84=pred_wgs84,
        geojson_predicted=geojson_pred,
    )

