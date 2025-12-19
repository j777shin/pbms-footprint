from __future__ import annotations

from typing import List, Tuple, Optional


def refine_predictions_with_osm(
    *,
    predicted_geoms_projected: List["object"],
    osm_buildings_projected: List["object"],
    snap_distance_m: float = 5.0,
    prefer_osm_when_available: bool = True,
    min_iou_for_snap: float = 0.3,
) -> List["object"]:
    """
    Refine predicted building polygons using OSM data for higher accuracy.

    Strategy:
    1. If `prefer_osm_when_available=True` and an OSM building intersects/overlaps
       significantly with a prediction, use the OSM polygon (highest accuracy).
    2. Otherwise, snap predicted polygon boundaries to nearby OSM boundaries
       if within `snap_distance_m`.

    Args:
        predicted_geoms_projected: Predicted building polygons in projected CRS
        osm_buildings_projected: OSM building polygons in projected CRS
        snap_distance_m: Maximum distance (meters) to snap boundaries
        prefer_osm_when_available: If True, use OSM polygon when IoU > min_iou_for_snap
        min_iou_for_snap: Minimum IoU threshold to prefer OSM over prediction

    Returns:
        Refined list of polygons
    """
    if not osm_buildings_projected or not predicted_geoms_projected:
        return predicted_geoms_projected if predicted_geoms_projected else []

    from shapely.ops import unary_union
    from shapely.geometry import Point

    refined = []

    for pred_poly in predicted_geoms_projected:
        best_match = None
        best_iou = 0.0

        # Find best matching OSM building by IoU
        for osm_poly in osm_buildings_projected:
            if not pred_poly.intersects(osm_poly):
                continue

            intersection = pred_poly.intersection(osm_poly)
            union = pred_poly.union(osm_poly)
            iou = float(intersection.area / union.area) if union.area > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_match = osm_poly

        if prefer_osm_when_available and best_match is not None and best_iou >= min_iou_for_snap:
            # Use OSM polygon directly (most accurate)
            refined.append(best_match)
        elif best_match is not None:
            # Snap prediction to OSM: use intersection area from prediction, but snap boundaries
            # Simple approach: use OSM if it overlaps significantly, otherwise use prediction
            intersection = pred_poly.intersection(best_match)
            if intersection.area > 0:
                # Use intersection as refined polygon (conservative)
                refined.append(intersection)
            else:
                refined.append(pred_poly)
        else:
            # No good OSM match, keep prediction as-is
            refined.append(pred_poly)

    return refined


def use_osm_directly_when_available(
    *,
    predicted_geoms_projected: List["object"],
    osm_buildings_projected: List["object"],
    fallback_to_predictions: bool = True,
) -> List["object"]:
    """
    Use OSM building footprints directly when available, fallback to predictions otherwise.

    This gives maximum accuracy for buildings present in OSM.

    Args:
        predicted_geoms_projected: Predicted polygons (used if no OSM match)
        osm_buildings_projected: OSM building polygons
        fallback_to_predictions: If True, include predictions not covered by OSM

    Returns:
        List of polygons (OSM preferred, predictions as fallback)
    """
    if not osm_buildings_projected:
        return predicted_geoms_projected if fallback_to_predictions else []

    from shapely.ops import unary_union

    result = list(osm_buildings_projected)

    if fallback_to_predictions and predicted_geoms_projected:
        osm_union = unary_union(osm_buildings_projected)
        for pred_poly in predicted_geoms_projected:
            # Only add prediction if it doesn't significantly overlap with OSM
            if not pred_poly.intersects(osm_union):
                result.append(pred_poly)
            else:
                # Check if prediction covers area not in OSM
                diff = pred_poly.difference(osm_union)
                if diff.area > pred_poly.area * 0.2:  # >20% not covered by OSM
                    result.append(diff)

    return result

