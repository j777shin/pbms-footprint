from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

from .sentinel2 import fetch_sentinel2_rgb_patch_sentinelhub
from .predict import load_refinenet, predict_refinenet_on_rgb, predict_refinenet_on_rgb_osm, Prediction
from .vectorize import (
    mask_to_polygons_projected,
    clip_polygons_to_radius_m,
    reproject_geoms_to_wgs84,
    geoms_to_geojson_feature_collection,
)
from .osm import (
    fetch_osm_buildings_within_radius,
    filter_predicted_geoms_by_osm_intersection,
    prefer_osm_footprints,
    rasterize_osm_buildings_to_mask,
)
from .refine import refine_predictions_with_osm
from .yolo import predict_yolov8_mask_on_rgb
from .fuse import fuse_binary_masks, FuseMode
from .super_resolution import apply_super_resolution_adaptive


@dataclass(frozen=True)
class NearbyBuildingsResult:
    rgb_patch: np.ndarray  # HxWx3 uint8 RGB
    projected_epsg: int
    bbox_projected: Tuple[float, float, float, float]
    center_projected: Tuple[float, float]

    refinenet: Prediction
    yolo_mask01: Optional[np.ndarray]
    fused_mask01: np.ndarray

    predicted_polygons_wgs84: list
    geojson_predicted: dict

    osm_geojson: Optional[dict]


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
    # OSM
    use_osm: bool = True,
    osm_extra_tags: Optional[Dict[str, Any]] = None,
    osm_filter_predictions: bool = True,
    osm_prefer_direct: bool = False,  # If True, use OSM footprints directly (most accurate)
    osm_refine_predictions: bool = False,  # If True, refine predictions by snapping to OSM
    # YOLO
    yolo_weights: Optional[str] = None,
    yolo_device: Optional[str] = None,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.7,
    yolo_classes: Optional[list] = None,
    fuse_mode: FuseMode = "intersection",
    # Super Resolution
    use_super_resolution: bool = False,
    sr_target_resolution_m: float = 2.5,  # Target resolution (e.g., 2.5m from 10m = 4x)
    sr_method: str = "bicubic",  # "bicubic", "bilinear", "lanczos"
    sr_use_real_esrgan: bool = False,  # Use Real-ESRGAN if available
    # Vectorization
    min_area_m2: float = 1.0,
) -> NearbyBuildingsResult:
    """
    End-to-end:
    lat/lon -> Sentinel-2 patch -> ReFineNet mask -> optional YOLO fusion ->
    polygons -> clip to radius -> optional OSM intersection filter -> GeoJSON.
    """
    if not refinenet_weights_path:
        raise ValueError("refinenet_weights_path is required")

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
    if use_super_resolution:
        print(f"Applying super resolution: {resolution_m}m -> {sr_target_resolution_m}m target...")
        rgb_for_prediction, actual_resolution = apply_super_resolution_adaptive(
            s2.rgb,
            target_resolution_m=sr_target_resolution_m,
            source_resolution_m=resolution_m,
            method=sr_method,
            use_real_esrgan=sr_use_real_esrgan,
        )
        print(f"Super resolution applied. Enhanced resolution: {actual_resolution:.2f}m (scale: {resolution_m/actual_resolution:.1f}x)")
        # Update patch size for model input (it will be resized anyway, but this helps with OSM mask sizing)
        enhanced_patch_px = rgb_for_prediction.shape[0]
    else:
        enhanced_patch_px = patch_px

    model = load_refinenet(refinenet_weights_path)
    
    # Check if model is 4-channel (ReFineNet4Ch)
    is_4channel = hasattr(model, 'input_channels') and model.input_channels == 4
    
    # For 4-channel models, we need OSM mask as input
    osm_early = None
    if is_4channel:
        # Fetch OSM data early to create the mask
        osm_early = fetch_osm_buildings_within_radius(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            projected_epsg=s2.projected_epsg,
            extra_tags=osm_extra_tags,
        )
        osm_mask01 = rasterize_osm_buildings_to_mask(
            geoms_projected=osm_early.geoms_projected,
            bbox_projected=s2.bbox_projected,
            out_size=enhanced_patch_px,
        )
        pred = predict_refinenet_on_rgb_osm(
            model, rgb_for_prediction, osm_mask01,
            threshold=refinenet_threshold,
            model_input_size=(enhanced_patch_px, enhanced_patch_px)
        )
    else:
        pred = predict_refinenet_on_rgb(
            model, rgb_for_prediction, 
            threshold=refinenet_threshold, 
            model_input_size=(enhanced_patch_px, enhanced_patch_px)
        )

    yolo_mask01 = None
    fused_mask01 = pred.mask
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
        fused_mask01 = fuse_binary_masks(pred.mask, ymask.mask01, mode=fuse_mode)

    vect = mask_to_polygons_projected(
        fused_mask01,
        bbox_projected=s2.bbox_projected,
        projected_epsg=s2.projected_epsg,
        min_area_m2=min_area_m2,
    )
    clipped = clip_polygons_to_radius_m(vect.polygons_projected, s2.center_projected, radius_m)

    osm_geojson = None
    if use_osm:
        # Reuse OSM data if already fetched for 4-channel model
        osm = osm_early if osm_early is not None else fetch_osm_buildings_within_radius(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            projected_epsg=s2.projected_epsg,
            extra_tags=osm_extra_tags,
        )
        osm_wgs84 = reproject_geoms_to_wgs84(osm.geoms_projected, from_epsg=osm.projected_epsg)
        osm_geojson = geoms_to_geojson_feature_collection(osm_wgs84, properties={"source": "osm"})

        if osm_prefer_direct:
            # Use OSM footprints directly (most accurate option)
            clipped = prefer_osm_footprints(
                predicted_geoms_projected=clipped,
                osm_buildings_projected=osm.geoms_projected,
                fallback_to_predictions=not osm_filter_predictions,
            )
        elif osm_refine_predictions:
            # Refine predictions by snapping to OSM boundaries
            clipped = refine_predictions_with_osm(
                predicted_geoms_projected=clipped,
                osm_buildings_projected=osm.geoms_projected,
                prefer_osm_when_available=True,
            )
        elif osm_filter_predictions:
            # Filter: only keep predictions that intersect OSM
            clipped = filter_predicted_geoms_by_osm_intersection(
                predicted_geoms_projected=clipped,
                osm_buildings_projected=osm.geoms_projected,
            )

    pred_wgs84 = reproject_geoms_to_wgs84(clipped, from_epsg=s2.projected_epsg)
    geojson_pred = geoms_to_geojson_feature_collection(pred_wgs84, properties={"source": "model"})

    # Use enhanced RGB if super resolution was applied, otherwise original
    output_rgb = rgb_for_prediction if use_super_resolution else s2.rgb
    
    return NearbyBuildingsResult(
        rgb_patch=output_rgb,
        projected_epsg=s2.projected_epsg,
        bbox_projected=s2.bbox_projected,
        center_projected=s2.center_projected,
        refinenet=pred,
        yolo_mask01=yolo_mask01,
        fused_mask01=fused_mask01,
        predicted_polygons_wgs84=pred_wgs84,
        geojson_predicted=geojson_pred,
        osm_geojson=osm_geojson,
    )


