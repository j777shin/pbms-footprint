from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import predict_buildings_near_coordinate as predict_buildings_near_coordinate


def predict_buildings_near_coordinate(*args: Any, **kwargs: Any):
    from .pipeline import predict_buildings_near_coordinate as _predict

    return _predict(*args, **kwargs)


__all__ = ["predict_buildings_near_coordinate"]

