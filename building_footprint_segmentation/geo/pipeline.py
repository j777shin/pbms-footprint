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
from .osm import (
    fetch_osm_buildings_within_radius,
    filter_predicted_geoms_by_osm_intersection,
)
from .yolo import predict_yolov8_mask_on_rgb
from .fuse import fuse_binary_masks, FuseMode


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
    # YOLO
    yolo_weights: Optional[str] = None,
    yolo_device: Optional[str] = None,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.7,
    yolo_classes: Optional[list] = None,
    fuse_mode: FuseMode = "intersection",
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

    model = load_refinenet(refinenet_weights_path)
    pred = predict_refinenet_on_rgb(model, s2.rgb, threshold=refinenet_threshold, model_input_size=(patch_px, patch_px))

    yolo_mask01 = None
    fused_mask01 = pred.mask
    if yolo_weights:
        ymask = predict_yolov8_mask_on_rgb(
            s2.rgb,
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
        osm = fetch_osm_buildings_within_radius(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            projected_epsg=s2.projected_epsg,
            extra_tags=osm_extra_tags,
        )
        osm_wgs84 = reproject_geoms_to_wgs84(osm.geoms_projected, from_epsg=osm.projected_epsg)
        osm_geojson = geoms_to_geojson_feature_collection(osm_wgs84, properties={"source": "osm"})

        if osm_filter_predictions:
            clipped = filter_predicted_geoms_by_osm_intersection(
                predicted_geoms_projected=clipped,
                osm_buildings_projected=osm.geoms_projected,
            )

    pred_wgs84 = reproject_geoms_to_wgs84(clipped, from_epsg=s2.projected_epsg)
    geojson_pred = geoms_to_geojson_feature_collection(pred_wgs84, properties={"source": "model"})

    return NearbyBuildingsResult(
        rgb_patch=s2.rgb,
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


