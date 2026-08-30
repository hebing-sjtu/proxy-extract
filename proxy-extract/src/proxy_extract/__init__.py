"""Extract Depth + Semantic-ID proxy conditions for the CWM v2v dataset.

The output of this package is a `condition_root` directory that
`code-world-model`'s `prepare` step consumes unmodified. See `contract.py` for
the exact format and `pipeline.py` for the stage ordering.
"""

from __future__ import annotations

from .contract import validate_condition_root, write_frame
from .pipeline import ExtractionConfig, extract_clip, extract_dataset
from .qc import score_dataset, score_pair
from .taxonomy import CLASS_NAMES, NUM_CLASSES

__version__ = "0.1.0"

__all__ = [
    "CLASS_NAMES",
    "ExtractionConfig",
    "NUM_CLASSES",
    "extract_clip",
    "extract_dataset",
    "score_dataset",
    "score_pair",
    "validate_condition_root",
    "write_frame",
    "__version__",
]
