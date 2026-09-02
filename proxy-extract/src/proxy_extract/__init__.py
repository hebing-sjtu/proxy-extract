"""Predict depth and semantics for RGB-only video corpora.

Two deliverables. `delivery.py` writes the 1280x720 segments DATA_F.md
specifies, which is what a delivery run produces. `pipeline.py` writes the
336x192 `condition_root` that `code-world-model`'s `prepare` step consumes
unmodified; `contract.py` holds that format.
"""

from __future__ import annotations

from .contract import validate_condition_root, write_frame
from .delivery import DeliveryConfig, extract_scene
from .pipeline import ExtractionConfig, extract_clip, extract_dataset
from .taxonomy import CLASS_NAMES, NUM_CLASSES

__version__ = "0.1.0"

__all__ = [
    "CLASS_NAMES",
    "DeliveryConfig",
    "ExtractionConfig",
    "NUM_CLASSES",
    "extract_clip",
    "extract_dataset",
    "extract_scene",
    "validate_condition_root",
    "write_frame",
    "__version__",
]
