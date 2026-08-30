"""SAM 3 concept refinement on top of a closed-set base segmentation.

Purpose is narrow. ADE20K collapses every creature into a single "animal" class
and has no notion of the loose world-clutter CWM calls "prop"; Cityscapes has
neither class at all. SAM 3 is prompted with the specific noun phrases those
gaps correspond to, and its masks are painted over the base map by taxonomy
priority.

Two deployment notes that bite:

- SAM 3 needs Python >= 3.12 and torch >= 2.7, while code-world-model pins
  Python 3.10 / torch 2.9.1. They cannot share one environment; run this stage
  separately and hand over the written condition_root.
- The checkpoints are gated on Hugging Face and need `hf auth login`.

This uses the per-image API, whose output contract (`masks`, `boxes`, `scores`)
is fully specified upstream, and gets temporal stability from `temporal.py`
instead of SAM 3's tracker. The video predictor - `build_sam3_video_predictor`
with `start_session` / `add_prompt` requests - would supply real tracking and is
the natural upgrade, but the shape of its `outputs` payload is only documented
in the example notebook, so it is deliberately not guessed at here.
"""

from __future__ import annotations

import numpy as np

from ..taxonomy import SAM3_PROMPTS, ConceptPrompt, overlay
from .base import SemanticResult


class Sam3ConceptRefiner:
    name = "sam3"

    def __init__(
        self,
        *,
        prompts: tuple[ConceptPrompt, ...] = SAM3_PROMPTS,
        score_threshold: float = 0.5,
        min_area_fraction: float = 1.0e-4,
        device: str | None = None,
    ) -> None:
        self.prompts = prompts
        self.score_threshold = score_threshold
        # At 336x192 the condition grid has 64,512 pixels, so a detection under
        # roughly a tenth of a percent of the frame survives downsampling only
        # as noise. Dropping it here avoids single-pixel class flicker.
        self.min_area_fraction = min_area_fraction
        self.device = device
        self._processor = None

    def _load(self):
        if self._processor is None:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            model = build_sam3_image_model()
            if self.device is not None:
                model = model.to(self.device)
            self._processor = Sam3Processor(model)
        return self._processor

    def _concept_mask(self, state, phrase: str, shape: tuple[int, int]) -> np.ndarray:
        processor = self._load()
        output = processor.set_text_prompt(state=state, prompt=phrase)
        masks, scores = output["masks"], output["scores"]

        union = np.zeros(shape, dtype=bool)
        min_area = self.min_area_fraction * shape[0] * shape[1]
        for mask, score in zip(_to_numpy(masks), _to_numpy(scores).ravel()):
            if float(score) < self.score_threshold:
                continue
            binary = np.squeeze(mask) > 0.5
            if binary.shape != shape or binary.sum() < min_area:
                continue
            union |= binary
        return union

    def refine(self, frames: list[np.ndarray], base: SemanticResult) -> SemanticResult:
        """Paint concept masks over `base`, higher taxonomy priority winning."""
        from PIL import Image

        if base.frames != len(frames):
            raise ValueError(f"base has {base.frames} frames for {len(frames)} images")

        processor = self._load()
        labels = base.labels.copy()
        hits: dict[str, int] = {}

        for index, frame in enumerate(frames):
            shape = frame.shape[:2]
            state = processor.set_image(Image.fromarray(frame))
            for prompt in self.prompts:
                mask = self._concept_mask(state, prompt.phrase, shape)
                if not mask.any():
                    continue
                hits[prompt.phrase] = hits.get(prompt.phrase, 0) + 1
                labels[index] = overlay(labels[index], prompt.cwm_class, mask)

        return SemanticResult(
            labels=labels,
            confidence=base.confidence,
            meta={
                **base.meta,
                "sam3_prompts": [p.phrase for p in self.prompts],
                "sam3_frames_with_hits": hits,
            },
        )


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    return np.asarray(value)
