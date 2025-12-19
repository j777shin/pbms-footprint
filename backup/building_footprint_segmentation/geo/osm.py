from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass(frozen=True)
class OSMBuildings:
    geoms_projected: List["object"]  # shapely geometries in projected CRS
    projected_epsg: int


def fetch_osm_buildings_within_radius(
    *,
    lat: float,
    lon: float,
    radius_m: float,
    projected_epsg: int,
    extra_tags: Optional[Dict[str, Any]] = None,
    overpass_url: Optional[str] = None,
    requests_timeout: Optional[int] = None,
    max_retries: int = 3,
    retry_sleep_s: float = 3.0,
) -> OSMBuildings:
    import osmnx as ox

    tags: Dict[str, Any] = {"building": True}
    if extra_tags:
        # Allow users to override/extend tags; e.g. building="industrial"
        tags.update(extra_tags)

    # Configure osmnx Overpass settings (osmnx 2.x uses settings.overpass_url + settings.requests_timeout)
    if overpass_url is not None:
        ox.settings.overpass_url = overpass_url
    if requests_timeout is not None:
        ox.settings.requests_timeout = int(requests_timeout)

    last_err: Optional[Exception] = None
    for attempt in range(max(1, int(max_retries))):
        try:
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=float(radius_m))
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt == int(max_retries) - 1:
                break
            import time

            time.sleep(float(retry_sleep_s))

    if last_err is not None:
        raise last_err

    if len(gdf) == 0:
        return OSMBuildings(geoms_projected=[], projected_epsg=projected_epsg)

    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf.to_crs(epsg=int(projected_epsg))

    buildings = [
        g for g in gdf.geometry.values
        if getattr(g, "geom_type", None) in ("Polygon", "MultiPolygon")
    ]
    return OSMBuildings(geoms_projected=buildings, projected_epsg=projected_epsg)


def filter_predicted_geoms_by_osm_intersection(
    *,
    predicted_geoms_projected: List["object"],
    osm_buildings_projected: List["object"],
    min_intersection_area_m2: float = 1.0,
) -> List["object"]:
    from shapely.ops import unary_union

    if not osm_buildings_projected:
        return []
    osm_union = unary_union(osm_buildings_projected)

    kept = []
    for g in predicted_geoms_projected:
        inter = g.intersection(osm_union)
        if not inter.is_empty and float(inter.area) >= float(min_intersection_area_m2):
            kept.append(g)
    return kept


def rasterize_osm_buildings_to_mask(
    geoms_projected: List["object"],
    bbox_projected: Tuple[float, float, float, float],
    out_size: int,
) -> "object":
    """
    Rasterize projected OSM building geometries into a binary mask.
    
    Args:
        geoms_projected: List of shapely geometries in projected CRS
        bbox_projected: (minx, miny, maxx, maxy) bounding box in projected CRS
        out_size: Output mask size (H=W=out_size)
        
    Returns:
        np.ndarray: HxW uint8 binary mask {0,1}
    """
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    minx, miny, maxx, maxy = bbox_projected
    transform = from_bounds(minx, miny, maxx, maxy, out_size, out_size)
    shapes = [(mapping(g), 1) for g in geoms_projected]
    mask = rasterize(
        shapes=shapes,
        out_shape=(out_size, out_size),
        fill=0,
        transform=transform,
        dtype=np.uint8,
        all_touched=False,
    )
    return mask


def prefer_osm_footprints(
    *,
    predicted_geoms_projected: List["object"],
    osm_buildings_projected: List["object"],
    fallback_to_predictions: bool = True,
) -> List["object"]:
    """
    Use OSM footprints directly for maximum accuracy, with optional fallback to predictions.

    This is the most accurate option when OSM data is available.
    """
    from .refine import use_osm_directly_when_available

    return use_osm_directly_when_available(
        predicted_geoms_projected=predicted_geoms_projected,
        osm_buildings_projected=osm_buildings_projected,
        fallback_to_predictions=fallback_to_predictions,
    )


