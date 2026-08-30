"""Backend-agnostic semantic interface, in CWM class space."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..taxonomy import NUM_CLASSES, VOID_UNKNOWN


@dataclass
class SemanticResult:
    """Per-frame class IDs already projected onto the 12 CWM classes."""

    labels: np.ndarray  # (N, H, W) uint8, values in [0, 12)
    confidence: np.ndarray | None = None  # (N, H, W) float32
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.labels.ndim != 3:
            raise ValueError(f"labels must be (N, H, W), got {self.labels.shape}")
        self.labels = np.asarray(self.labels, dtype=np.uint8)
        top = int(self.labels.max(initial=0))
        if top >= NUM_CLASSES:
            raise ValueError(f"label {top} is outside the {NUM_CLASSES}-class CWM taxonomy")

    @property
    def frames(self) -> int:
        return len(self.labels)

    def class_histogram(self) -> dict[int, float]:
        counts = np.bincount(self.labels.ravel(), minlength=NUM_CLASSES)
        return {cls: round(float(n) / self.labels.size, 6) for cls, n in enumerate(counts) if n}


@runtime_checkable
class SemanticBackend(Protocol):
    name: str

    def segment(self, frames: list[np.ndarray]) -> SemanticResult:
        ...


# ---------------------------------------------------------- label resolution


def _synonyms(label: str) -> list[str]:
    """Split a dataset label into its comparable synonyms.

    ADE20K labels ship as synonym lists - "building;edifice",
    "windowpane;window" - and which synonym a checkpoint reports varies. Trying
    each one is what lets a mapping be written against readable names.
    """
    parts = re.split(r"[;,]", label.lower())
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def resolve_label_lut(
    id2label: dict[int, str], mapping: dict[str, int], *, default: int = VOID_UNKNOWN
) -> tuple[np.ndarray, list[str]]:
    """Build an index-aligned LUT from a checkpoint's own label list.

    Reading `id2label` off the loaded model rather than hardcoding an index
    order means a checkpoint trained on a permuted label set cannot silently
    relabel the dataset. Returns the LUT and the labels that fell through to
    `default`, so a mapping gap is visible instead of quietly becoming void.
    """
    size = max(id2label) + 1
    lut = np.full(size, default, dtype=np.uint8)
    unmapped: list[str] = []
    for index, label in id2label.items():
        for synonym in _synonyms(label):
            if synonym in mapping:
                lut[index] = mapping[synonym]
                break
        else:
            unmapped.append(label)
    return lut, unmapped
