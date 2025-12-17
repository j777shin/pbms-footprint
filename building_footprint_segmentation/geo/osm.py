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
) -> OSMBuildings:
    import osmnx as ox

    tags: Dict[str, Any] = {"building": True}
    if extra_tags:
        # Allow users to override/extend tags; e.g. building="industrial"
        tags.update(extra_tags)

    gdf = ox.features_from_point((lat, lon), tags=tags, dist=float(radius_m))
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


