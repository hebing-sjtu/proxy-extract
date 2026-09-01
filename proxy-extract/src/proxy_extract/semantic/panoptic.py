"""Closed-set semantic segmentation, projected onto the CWM taxonomy.

This is the trunk of the semantic stage. A closed-set model is used rather than
an open-vocabulary one because 6 of the 12 CWM classes - sky, water, terrain,
road_paved, vegetation, building_structure - are "stuff": unbounded regions with
no instances. Concept detectors are built to find things, and stuff is exactly
where they are weakest, while ADE20K-trained models cover it densely.
"""

from __future__ import annotations

import numpy as np

from ..taxonomy import (
    ADE20K_TO_COARSE6,
    ADE20K_TO_CWM,
    ADE20K_TO_STANDARD11,
    C6_BACKGROUND,
    CITYSCAPES_TO_CWM,
    PROP,
    S11_PROP,
    VOID_UNKNOWN,
)
from .base import SemanticResult, resolve_label_lut

ADE20K_CHECKPOINT = "facebook/mask2former-swin-large-ade-semantic"
CITYSCAPES_CHECKPOINT = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

# ADE20K's long tail is furniture, appliances and clutter, all of which is prop
# under the CWM taxonomy, so falling through to prop is right. Cityscapes has
# only 19 deliberately chosen classes and no catch-all, so anything unmapped
# there is genuinely unknown.
_PROFILES = {
    "ade20k": (ADE20K_CHECKPOINT, ADE20K_TO_CWM, PROP),
    "cityscapes": (CITYSCAPES_CHECKPOINT, CITYSCAPES_TO_CWM, VOID_UNKNOWN),
    # The 6-class set maps from ADE20K directly rather than by folding the
    # 12-class result, because the two disagree about grass: with no terrain
    # class to land in, it belongs with the vegetation it looks like. Anything
    # unmapped is background, which is what that class is for.
    "coarse6": (ADE20K_CHECKPOINT, ADE20K_TO_COARSE6, C6_BACKGROUND),
    # The delivered 11-class schema. prop is the standard's own stated default
    # for anything the rules do not claim, so the fallback is not a compromise
    # here the way VOID_UNKNOWN is for cityscapes.
    "standard11": (ADE20K_CHECKPOINT, ADE20K_TO_STANDARD11, S11_PROP),
}


def _load_segmentation_model(checkpoint: str):
    """Load a checkpoint through whichever auto-class actually claims it.

    Mask2Former registers under `AutoModelForUniversalSegmentation` while
    SegFormer registers under `AutoModelForSemanticSegmentation`, and picking
    the wrong one fails at load time. Both expose the logits that
    `post_process_semantic_segmentation` needs, so try each.
    """
    from transformers import AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation

    errors = []
    for auto_class in (AutoModelForUniversalSegmentation, AutoModelForSemanticSegmentation):
        try:
            return auto_class.from_pretrained(checkpoint)
        except (ValueError, KeyError, OSError) as error:
            errors.append(f"{auto_class.__name__}: {error}")
    raise ValueError(f"no auto-class could load {checkpoint!r}:\n  " + "\n  ".join(errors))


class PanopticBackend:
    name = "panoptic"

    def __init__(
        self,
        *,
        profile: str = "ade20k",
        checkpoint: str | None = None,
        device: str | None = None,
        batch_size: int = 4,
    ) -> None:
        if profile not in _PROFILES:
            raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(_PROFILES)}")
        default_checkpoint, mapping, fallback = _PROFILES[profile]
        self.profile = profile
        self.checkpoint = checkpoint or default_checkpoint
        self.mapping = mapping
        self.fallback = fallback
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._lut: np.ndarray | None = None
        self._unmapped: list[str] = []

    def _load(self):
        if self._model is None:
            import torch

            from transformers import AutoImageProcessor

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            self._model = _load_segmentation_model(self.checkpoint).to(device).eval()
            self.device = device

            id2label = {int(k): str(v) for k, v in self._model.config.id2label.items()}
            self._lut, self._unmapped = resolve_label_lut(id2label, self.mapping, default=self.fallback)
        return self._model, self._processor

    def segment(self, frames: list[np.ndarray]) -> SemanticResult:
        import torch

        model, processor = self._load()
        assert self._lut is not None

        height, width = frames[0].shape[:2]
        target_sizes = [(height, width)]
        labels: list[np.ndarray] = []

        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            maps = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes * len(batch)
            )
            for source in maps:
                source_ids = source.cpu().numpy().astype(np.int32)
                labels.append(self._lut[np.clip(source_ids, 0, len(self._lut) - 1)])

        return SemanticResult(
            labels=np.stack(labels),
            meta={
                "backend": self.name,
                "profile": self.profile,
                "checkpoint": self.checkpoint,
                "unmapped_source_labels": self._unmapped,
            },
        )
