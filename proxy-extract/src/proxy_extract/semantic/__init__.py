"""Semantic backends producing CWM 12-class label maps."""

from __future__ import annotations

from .base import SemanticBackend, SemanticResult, resolve_label_lut

__all__ = ["SemanticBackend", "SemanticResult", "resolve_label_lut", "get_backend", "get_refiner"]


def get_backend(name: str, **kwargs) -> SemanticBackend:
    """Resolve a base segmenter by name, importing heavy deps only on use."""
    if name in {"ade20k", "cityscapes", "coarse6", "standard11"}:
        from .panoptic import PanopticBackend

        return PanopticBackend(profile=name, **kwargs)
    if name == "synthetic":
        from .synthetic import SyntheticSemanticBackend

        return SyntheticSemanticBackend(**kwargs)
    raise ValueError(f"unknown semantic backend: {name}")


def get_refiner(name: str, **kwargs):
    if name in {"none", ""}:
        return None
    if name == "sam3":
        from .sam3 import Sam3ConceptRefiner

        return Sam3ConceptRefiner(**kwargs)
    raise ValueError(f"unknown semantic refiner: {name}")
