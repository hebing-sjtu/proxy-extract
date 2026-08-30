"""A deterministic stand-in for a real depth model.

Exists so the full pipeline - decode, estimate, calibrate, stabilise, encode,
validate - can be exercised end to end on a machine with no GPU. It produces
depth with the right shape, range and rough structure (higher in the image means
further away), which is enough to catch contract and plumbing bugs. It is not a
depth estimator and must never be used for real data.
"""

from __future__ import annotations

import numpy as np

from ..cameras import CameraTrack
from .base import DepthResult


class SyntheticDepthBackend:
    name = "synthetic"

    def __init__(self, *, near: float = 1.5, far: float = 80.0) -> None:
        self.near = near
        self.far = far

    def estimate(self, frames: list[np.ndarray], *, cameras: CameraTrack | None = None) -> DepthResult:
        depths = []
        for frame in frames:
            height, width = frame.shape[:2]
            # Ground-plane-ish prior: depth grows toward the horizon.
            vertical = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
            base = self.near + (self.far - self.near) * vertical
            # Modulate with luminance so the map is not perfectly rank-one and
            # downstream smoothing/pooling has something to act on.
            luma = frame.astype(np.float32).mean(axis=2) / 255.0
            depths.append((base * (0.85 + 0.3 * luma)).astype(np.float32))

        stacked = np.stack(depths)
        return DepthResult(
            depth=stacked,
            metric=True,
            valid=np.ones_like(stacked, dtype=bool),
            meta={"backend": "synthetic", "warning": "placeholder depth, not a real estimate"},
        )
