"""Depth backends producing metric depth on the CWM condition grid."""

from __future__ import annotations

from .base import DepthBackend, DepthResult
from .scale import ScaleSolution, solve_scale_from_cameras, solve_scale_shift

__all__ = [
    "DepthBackend",
    "DepthResult",
    "ScaleSolution",
    "solve_scale_from_cameras",
    "solve_scale_shift",
    "get_backend",
]


def get_backend(name: str, **kwargs) -> DepthBackend:
    """Resolve a backend by name, importing its heavy dependencies only on use."""
    if name == "mapanything":
        from .mapanything import MapAnythingBackend

        return MapAnythingBackend(**kwargs)
    if name == "depth_anything":
        from .depth_anything import DepthAnythingBackend

        return DepthAnythingBackend(**kwargs)
    if name == "depth_anything_v3":
        from .depth_anything_v3 import DepthAnythingV3Backend

        return DepthAnythingV3Backend(**kwargs)
    if name == "synthetic":
        from .synthetic import SyntheticDepthBackend

        return SyntheticDepthBackend(**kwargs)
    raise ValueError(f"unknown depth backend: {name}")
