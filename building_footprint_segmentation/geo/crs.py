from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalCRS:

    epsg: int


def utm_epsg_for_latlon(lat: float, lon: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    if not (1 <= zone <= 60):
        raise ValueError(f"Invalid UTM zone computed for lon={lon}: zone={zone}")
    return (32600 + zone) if lat >= 0 else (32700 + zone)


