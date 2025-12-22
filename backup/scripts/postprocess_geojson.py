#!/usr/bin/env python3
"""
Apply post-processing to building footprint GeoJSON files.

This script reads a GeoJSON file, applies post-processing (smoothing, hole removal, etc.),
and saves the processed result.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_geojson(geojson_path: Path) -> Dict[str, Any]:
    """Load GeoJSON file."""
    with open(geojson_path, 'r') as f:
        return json.load(f)


def save_geojson(geojson: Dict[str, Any], output_path: Path) -> None:
    """Save GeoJSON file."""
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    print(f"Saved processed GeoJSON to: {output_path}")


def main():
    _ensure_repo_on_path()
    
    parser = argparse.ArgumentParser(
        description="Apply post-processing to building footprint GeoJSON"
    )
    parser.add_argument(
        "input_geojson",
        type=str,
        help="Path to input GeoJSON file (e.g., predicted_buildings.geojson)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GeoJSON path (default: input_geojson with _postprocessed suffix)"
    )
    parser.add_argument(
        "--projected-epsg",
        type=int,
        default=32648,
        help="Projected EPSG code for area calculations (default: 32648 for UTM zone 48N)"
    )
    
    # Post-processing options
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=True,
        help="Smooth polygon boundaries (default: True)"
    )
    parser.add_argument(
        "--no-smooth",
        action="store_false",
        dest="smooth",
        help="Disable boundary smoothing"
    )
    parser.add_argument(
        "--smooth-tolerance",
        type=float,
        default=0.5,
        help="Smoothing tolerance in meters (default: 0.5)"
    )
    parser.add_argument(
        "--remove-holes",
        action="store_true",
        default=True,
        help="Remove small holes (default: True)"
    )
    parser.add_argument(
        "--no-remove-holes",
        action="store_false",
        dest="remove_holes",
        help="Disable hole removal"
    )
    parser.add_argument(
        "--min-hole-area",
        type=float,
        default=5.0,
        help="Minimum hole area to keep in m² (default: 5.0)"
    )
    parser.add_argument(
        "--regularize",
        action="store_true",
        default=False,
        help="Regularize shapes (rectangularize, default: False)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1.0,
        help="Minimum polygon area in m² (default: 1.0)"
    )
    parser.add_argument(
        "--min-perimeter",
        type=float,
        default=3.0,
        help="Minimum polygon perimeter in meters (default: 3.0)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input_geojson)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    if args.output:
        output_path = Path(args.output)
    else:
        # Add _postprocessed suffix
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_postprocessed.geojson"
    
    # Load GeoJSON
    print(f"Loading GeoJSON from: {input_path}")
    geojson = load_geojson(input_path)
    
    features = geojson.get("features", [])
    if not features:
        print("Warning: No features found in GeoJSON")
        return 1
    
    print(f"Found {len(features)} building(s)")
    
    # Import required modules
    from building_footprint_segmentation.geo.vectorize import reproject_geoms_to_wgs84
    from building_footprint_segmentation.geo.postprocess import postprocess_polygons
    from shapely.geometry import shape
    from pyproj import Transformer
    from shapely.ops import transform
    
    # Convert GeoJSON features to Shapely geometries (WGS84)
    geoms_wgs84 = []
    for feature in features:
        geom = shape(feature.get("geometry", {}))
        if not geom.is_empty:
            geoms_wgs84.append(geom)
    
    if not geoms_wgs84:
        print("Error: No valid geometries found")
        return 1
    
    # Reproject from WGS84 to projected CRS for post-processing
    print(f"Reprojecting to EPSG:{args.projected_epsg} for post-processing...")
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{args.projected_epsg}", always_xy=True)
    
    def transform_geom(geom):
        return transform(transformer.transform, geom)
    
    geoms_projected = []
    for geom in geoms_wgs84:
        try:
            geom_proj = transform_geom(geom)
            if not geom_proj.is_empty:
                geoms_projected.append(geom_proj)
        except Exception as e:
            print(f"Warning: Failed to reproject geometry: {e}")
            continue
    
    if not geoms_projected:
        print("Error: No geometries could be reprojected")
        return 1
    
    print(f"Applying post-processing to {len(geoms_projected)} polygon(s)...")
    
    # Apply post-processing
    processed = postprocess_polygons(
        geoms_projected,
        smooth=args.smooth,
        smooth_tolerance_m=args.smooth_tolerance,
        remove_holes=args.remove_holes,
        min_hole_area_m2=args.min_hole_area,
        regularize=args.regularize,
        regularize_method="min_area_rect",
        min_area_m2=args.min_area,
        min_perimeter_m=args.min_perimeter,
    )
    
    print(f"Post-processing complete: {len(processed)} polygon(s) remaining")
    
    # Calculate areas in projected CRS (accurate)
    polygon_areas = [g.area for g in processed]
    
    # Reproject back to WGS84
    print("Reprojecting back to WGS84...")
    transformer_back = Transformer.from_crs(f"EPSG:{args.projected_epsg}", "EPSG:4326", always_xy=True)
    
    def transform_back(geom):
        return transform(transformer_back.transform, geom)
    
    processed_wgs84 = []
    for geom in processed:
        try:
            geom_wgs84 = transform_back(geom)
            if not geom_wgs84.is_empty:
                processed_wgs84.append(geom_wgs84)
        except Exception as e:
            print(f"Warning: Failed to reproject geometry back: {e}")
            continue
    
    # Create new GeoJSON
    from building_footprint_segmentation.geo.vectorize import geoms_to_geojson_feature_collection
    
    processed_geojson = geoms_to_geojson_feature_collection(
        processed_wgs84,
        properties={"source": "model", "building": "yes"},
        polygon_areas=polygon_areas,
    )
    
    # Preserve CRS from original
    if "crs" in geojson:
        processed_geojson["crs"] = geojson["crs"]
    
    # Save result
    save_geojson(processed_geojson, output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Post-Processing Summary")
    print(f"{'='*60}")
    print(f"Input: {len(features)} building(s)")
    print(f"Output: {len(processed_wgs84)} building(s)")
    print(f"Total area: {sum(polygon_areas):,.2f} m² ({sum(polygon_areas)/10000:.4f} ha)")
    print(f"Average area: {sum(polygon_areas)/len(polygon_areas):,.2f} m²" if processed_wgs84 else "N/A")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

