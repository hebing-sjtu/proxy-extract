"""Backend-agnostic depth interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..cameras import CameraTrack


@dataclass
class DepthResult:
    """Per-frame depth from one backend, plus whatever it knows about itself.

    `metric` records whether `depth` is already in metres. A backend that only
    predicts up to scale sets it False and leaves calibration to `scale.py`;
    downstream refuses to encode non-metric depth, so this flag cannot be
    forgotten silently.
    """

    depth: np.ndarray  # (N, H, W) float32, metres if metric else arbitrary units
    metric: bool
    confidence: np.ndarray | None = None  # (N, H, W) float32, higher is better
    valid: np.ndarray | None = None  # (N, H, W) bool
    cam2world: np.ndarray | None = None  # (N, 4, 4) predicted poses, if any
    intrinsics: np.ndarray | None = None  # (N, 3, 3) predicted intrinsics, if any
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.depth.ndim != 3:
            raise ValueError(f"depth must be (N, H, W), got {self.depth.shape}")
        self.depth = np.asarray(self.depth, dtype=np.float32)

    @property
    def frames(self) -> int:
        return len(self.depth)

    def valid_mask(self) -> np.ndarray:
        """Finite, positive, and accepted by the backend's own mask."""
        mask = np.isfinite(self.depth) & (self.depth > 0.0)
        if self.valid is not None:
            mask &= self.valid
        return mask

    def scaled(self, scale: float, *, shift: float = 0.0) -> DepthResult:
        """Apply a calibration and mark the result metric."""
        depth = self.depth * np.float32(scale) + np.float32(shift)
        return DepthResult(
            depth=np.where(self.valid_mask(), depth, 0.0).astype(np.float32),
            metric=True,
            confidence=self.confidence,
            valid=self.valid,
            cam2world=self.cam2world,
            intrinsics=self.intrinsics,
            meta={**self.meta, "applied_scale": scale, "applied_shift": shift},
        )


@runtime_checkable
class DepthBackend(Protocol):
    name: str

    def estimate(
        self, frames: list[np.ndarray], *, cameras: CameraTrack | None = None
    ) -> DepthResult:
        """Predict depth for a whole clip at once.

        Whole-clip rather than per-frame on purpose: the multi-view models this
        wraps derive their temporal consistency from seeing the sequence
        jointly, and calling them frame by frame throws that away.
        """
        ...
