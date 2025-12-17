from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal


@dataclass(frozen=True)
class Sentinel2Patch:
    rgb: "object"  # actually np.ndarray (uint8 HxWx3). Kept as object to avoid hard import.
    bbox_projected: Tuple[float, float, float, float]  # minx,miny,maxx,maxy in projected CRS
    projected_epsg: int
    center_projected: Tuple[float, float]  # x,y in projected CRS


def fetch_sentinel2_rgb_patch_sentinelhub(
    *,
    lat: float,
    lon: float,
    patch_px: int = 384,
    resolution_m: float = 10.0,
    time_interval: Tuple[str, str] = ("2025-06-01", "2025-10-01"),
    mosaicking_order: Literal["mostRecent", "leastCC"] = "leastCC",
    data_collection: str = "SENTINEL2_L2A",
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> Sentinel2Patch:
    import os
    import numpy as np
    from pyproj import CRS as PyCRS, Transformer

    from sentinelhub import (
        SHConfig,
        CRS,
        BBox,
        SentinelHubRequest,
        DataCollection,
        MimeType,
    )

    if patch_px <= 0:
        raise ValueError("patch_px must be positive")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")

    if client_id is None:
        client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    if client_secret is None:
        client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Sentinel Hub credentials. Set SENTINELHUB_CLIENT_ID / SENTINELHUB_CLIENT_SECRET "
            "or pass client_id/client_secret."
        )


    sh_crs = CRS.get_utm_from_wgs84(lon, lat)
    projected_epsg = int(getattr(sh_crs, "epsg", sh_crs.value))
    wgs84 = PyCRS.from_epsg(4326)
    proj = PyCRS.from_epsg(projected_epsg)
    to_proj = Transformer.from_crs(wgs84, proj, always_xy=True)

    cx, cy = to_proj.transform(lon, lat)
    half = (patch_px * float(resolution_m)) / 2.0
    bbox_proj = (cx - half, cy - half, cx + half, cy + half)

    cfg = SHConfig()
    cfg.sh_client_id = client_id
    cfg.sh_client_secret = client_secret


    evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02","B03","B04"],
    output: { bands: 3, sampleType: "UINT8" }
  };
}
function evaluatePixel(s) {
  return [
    Math.max(0, Math.min(255, s.B04 * 255)),
    Math.max(0, Math.min(255, s.B03 * 255)),
    Math.max(0, Math.min(255, s.B02 * 255))
  ];
}
"""

    dc = getattr(DataCollection, data_collection)
    bbox = BBox(bbox=bbox_proj, crs=sh_crs)
    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=dc,
                time_interval=time_interval,
                mosaicking_order=mosaicking_order,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=(patch_px, patch_px),
        config=cfg,
    )

    rgb = req.get_data()[0]
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Unexpected Sentinel Hub response shape: {getattr(rgb, 'shape', None)}")
    rgb = rgb.astype(np.uint8, copy=False)

    return Sentinel2Patch(
        rgb=rgb,
        bbox_projected=bbox_proj,
        projected_epsg=projected_epsg,
        center_projected=(cx, cy),
    )


