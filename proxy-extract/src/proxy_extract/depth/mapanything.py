"""MapAnything depth backend.

Chosen over VGGT because it predicts metric scale natively and can ingest known
calibration, and over VGGT-Omega because that model's weights are gated behind an
automated approval that frequently refuses, and Meta flagged possible benchmark
contamination in the released 1B checkpoint on 2026-08-18.

Everything torch-related is imported inside methods so the rest of the package
stays importable on a machine without CUDA.
"""

from __future__ import annotations

import numpy as np

from ..cameras import CameraTrack
from .base import DepthResult

# CC-BY-NC weights; the research choice per the licensing decision for this
# dataset. Swap to "facebook/map-anything-apache" if it ever needs to ship.
DEFAULT_CHECKPOINT = "facebook/map-anything"
APACHE_CHECKPOINT = "facebook/map-anything-apache"


class MapAnythingBackend:
    name = "mapanything"

    def __init__(
        self,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        max_views: int = 124,
        amp_dtype: str = "bf16",
        memory_efficient: bool = True,
        feed_cameras: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.max_views = max_views
        self.amp_dtype = amp_dtype
        self.memory_efficient = memory_efficient
        # Off by default: solving scale from the GT track after the fact is
        # robust to preprocessing rescaling the intrinsics we hand over, and it
        # keeps the model's own pose estimate available as a cross-check.
        self.feed_cameras = feed_cameras
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from mapanything.models import MapAnything

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = MapAnything.from_pretrained(self.checkpoint).to(device).eval()
            self.device = device
        return self._model

    def _build_views(self, frames: list[np.ndarray], cameras: CameraTrack | None) -> list[dict]:
        import torch

        from mapanything.utils.image import preprocess_inputs

        views: list[dict] = []
        for index, frame in enumerate(frames):
            view: dict = {"img": torch.from_numpy(np.ascontiguousarray(frame))}
            if cameras is not None and self.feed_cameras:
                view["intrinsics"] = torch.from_numpy(cameras.intrinsics_for(index)).float()
                view["camera_poses"] = torch.from_numpy(cameras.cam2world[index]).float()
                view["is_metric_scale"] = torch.tensor([bool(cameras.metric)])
            views.append(view)
        return preprocess_inputs(views)

    def _infer_chunk(self, frames: list[np.ndarray], cameras: CameraTrack | None) -> list[dict]:
        import torch

        model = self._load()
        views = self._build_views(frames, cameras)
        with torch.no_grad():
            return model.infer(
                views,
                memory_efficient_inference=self.memory_efficient,
                use_amp=True,
                amp_dtype=self.amp_dtype,
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
            )

    def estimate(self, frames: list[np.ndarray], *, cameras: CameraTrack | None = None) -> DepthResult:
        if cameras is not None and len(cameras) != len(frames):
            raise ValueError(f"camera track has {len(cameras)} poses for {len(frames)} frames")
        if len(frames) > self.max_views:
            raise ValueError(
                f"{len(frames)} frames exceeds max_views={self.max_views}; split the clip into "
                "windows before calling, so that each chunk keeps a single consistent world frame"
            )

        predictions = self._infer_chunk(frames, cameras)

        depths, confidences, masks, poses, intrinsics = [], [], [], [], []
        for pred in predictions:
            depths.append(_to_hw(pred["depth_z"]))
            if "conf" in pred:
                confidences.append(_to_hw(pred["conf"]))
            if "mask" in pred:
                masks.append(_to_hw(pred["mask"]).astype(bool))
            if "camera_poses" in pred:
                poses.append(_to_numpy(pred["camera_poses"]).reshape(4, 4))
            if "intrinsics" in pred:
                intrinsics.append(_to_numpy(pred["intrinsics"]).reshape(3, 3))

        return DepthResult(
            depth=np.stack(depths).astype(np.float32),
            # MapAnything predicts metric scale, but a monocular video of an
            # unfamiliar scene is exactly where that estimate is weakest, so the
            # GT camera solve still overrides it when available.
            metric=True,
            confidence=np.stack(confidences).astype(np.float32) if confidences else None,
            valid=np.stack(masks) if masks else None,
            cam2world=np.stack(poses) if poses else None,
            intrinsics=np.stack(intrinsics) if intrinsics else None,
            meta={"checkpoint": self.checkpoint, "fed_cameras": bool(cameras and self.feed_cameras)},
        )


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    return np.asarray(value)


def _to_hw(value) -> np.ndarray:
    """Squeeze a MapAnything output down to a plain (H, W) array.

    Outputs arrive as (B, H, W, 1) with a batch axis of one per view; squeezing
    by name is not possible, so drop every unit axis and insist on 2D.
    """
    array = np.squeeze(_to_numpy(value))
    if array.ndim != 2:
        raise ValueError(f"expected a single (H, W) map, got shape {array.shape}")
    return array
