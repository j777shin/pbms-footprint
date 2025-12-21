from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class VectorizedMask:
    polygons_projected: List["object"]  # shapely geometries
    projected_epsg: int


def mask_to_polygons_projected(
    mask01: "object",  # np.ndarray HxW uint8 {0,1}
    bbox_projected: Tuple[float, float, float, float],
    projected_epsg: int,
    min_area_m2: float = 1.0,
) -> VectorizedMask:
    import numpy as np
    import rasterio
    from rasterio.features import shapes as raster_shapes
    from rasterio.transform import from_bounds
    from shapely.geometry import shape as shapely_shape

    if not isinstance(mask01, np.ndarray):
        raise ValueError("mask01 must be a numpy array")
    if mask01.ndim != 2:
        raise ValueError(f"mask01 must be HxW, got shape={mask01.shape}")

    minx, miny, maxx, maxy = bbox_projected
    h, w = mask01.shape
    transform = from_bounds(minx, miny, maxx, maxy, w, h)

    polys = []
    for geom, val in raster_shapes(mask01.astype(np.uint8), transform=transform):
        if int(val) != 1:
            continue
        poly = shapely_shape(geom)
        if poly.is_empty:
            continue
        # since we're in projected meters (UTM), area is m^2
        if float(poly.area) < float(min_area_m2):
            continue
        polys.append(poly)

    return VectorizedMask(polygons_projected=polys, projected_epsg=projected_epsg)


def clip_polygons_to_radius_m(
    polygons_projected: List["object"],
    center_projected: Tuple[float, float],
    radius_m: float,
) -> List["object"]:
    from shapely.geometry import Point

    cx, cy = center_projected
    circle = Point(cx, cy).buffer(float(radius_m))
    out = []
    for p in polygons_projected:
        q = p.intersection(circle)
        if not q.is_empty and q.area > 0:
            out.append(q)
    return out


def reproject_geoms_to_wgs84(
    geometries_projected: List["object"],
    from_epsg: int,
) -> List["object"]:
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs(f"EPSG:{from_epsg}", "EPSG:4326", always_xy=True)

    def _xy(x, y, z=None):
        return transformer.transform(x, y)

    return [shapely_transform(_xy, g) for g in geometries_projected]


def geoms_to_geojson_feature_collection(
    geoms_wgs84: List["object"],
    properties: Optional[dict] = None,
    polygon_areas: Optional[List[float]] = None,  # Areas in square meters (from projected CRS)
) -> dict:
    from shapely.geometry import mapping

    props = properties or {}
    
    # Add OSM-compatible properties
    osm_props = {
        "building": "yes",  # OSM standard building tag
        "source": props.get("source", "model"),  # Keep original source
    }
    osm_props.update(props)  # Allow override
    
    features = []
    for i, g in enumerate(geoms_wgs84):
        # Use provided area if available (more accurate, from projected CRS)
        # Otherwise calculate approximate area from WGS84
        if polygon_areas is not None and i < len(polygon_areas):
            area_m2 = polygon_areas[i]
        else:
            # Approximate area calculation (less accurate in WGS84)
            # This is a rough estimate - proper calculation needs projected CRS
            area_m2 = g.area * 111000 * 111000  # Rough conversion (degrees to meters)
        
        feature_props = {
            **osm_props,
            "id": i + 1,
            "area_m2": round(area_m2, 2),
        }
        
        features.append({
            "type": "Feature",
            "properties": feature_props,
            "geometry": mapping(g)
        })
    
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"  # WGS84 (lon/lat order) - QGIS/OSM compatible
            }
        },
        "features": features,
    }


