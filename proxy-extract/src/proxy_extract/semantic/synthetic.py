"""A deterministic stand-in for a real segmenter, for GPU-free pipeline runs.

Produces plausibly-shaped label maps using crude colour heuristics so the whole
chain can be exercised locally. It is not a segmenter and must never be used for
real data.
"""

from __future__ import annotations

import numpy as np

from ..taxonomy import BUILDING_STRUCTURE, HUMAN, SKY, TERRAIN, VEGETATION
from .base import SemanticResult


class SyntheticSemanticBackend:
    name = "synthetic"

    def segment(self, frames: list[np.ndarray]) -> SemanticResult:
        labels = []
        for frame in frames:
            rgb = frame.astype(np.float32)
            red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            height = frame.shape[0]

            out = np.full(frame.shape[:2], TERRAIN, dtype=np.uint8)
            rows = np.arange(height)[:, None]
            out[np.broadcast_to((blue > red + 20) & (rows < height // 2), out.shape)] = SKY
            out[(green > red + 15) & (green > blue + 15)] = VEGETATION
            out[(np.abs(red - green) < 12) & (np.abs(green - blue) < 12) & (red > 90)] = BUILDING_STRUCTURE

            centre = slice(height // 2, None), slice(frame.shape[1] // 2 - 20, frame.shape[1] // 2 + 20)
            out[centre] = HUMAN
            labels.append(out)

        return SemanticResult(
            labels=np.stack(labels),
            meta={"backend": "synthetic", "warning": "placeholder labels, not a real segmentation"},
        )
