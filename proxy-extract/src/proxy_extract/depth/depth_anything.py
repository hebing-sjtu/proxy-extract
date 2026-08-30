"""Depth Anything V2 (metric heads) as a single-view depth backend.

Not the intended production backend — it sees one frame at a time, so it has no
mechanism for temporal consistency and its scale drifts across a shot. It earns
its place for two other reasons: it is small enough to run on a laptop GPU,
which makes it the only way to look at real depth on these clips before the
H800 is available, and it emits metres directly, which makes it the reference
that turns the COLMAP world unit into a physical one.
"""

from __future__ import annotations

import numpy as np

from ..cameras import CameraTrack
from .base import DepthResult

# The outdoor head is trained on driving-scale scenes (up to 80 m); the indoor
# one saturates around 20 m and would clip the horizon in most of these clips.
OUTDOOR_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
INDOOR_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DepthAnythingBackend:
    name = "depth_anything_v2_metric"

    def __init__(
        self, *, checkpoint: str = OUTDOOR_CHECKPOINT, device: str | None = None, batch_size: int = 4
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.device = pick_device(self.device)
            self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint)
            self._model = self._model.to(self.device).eval()
        return self._model, self._processor

    def estimate(self, frames: list[np.ndarray], *, cameras: CameraTrack | None = None) -> DepthResult:
        import torch

        model, processor = self._load()
        height, width = frames[0].shape[:2]
        maps: list[np.ndarray] = []

        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = processor.post_process_depth_estimation(
                outputs, target_sizes=[(height, width)] * len(batch)
            )
            maps.extend(p["predicted_depth"].float().cpu().numpy() for p in predictions)

        return DepthResult(
            depth=np.stack(maps),
            metric=True,
            meta={
                "backend": self.name,
                "checkpoint": self.checkpoint,
                "single_view": True,
                "caveat": "per-frame prediction; scale is not tied across frames",
            },
        )
