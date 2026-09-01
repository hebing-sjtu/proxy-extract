"""Readers for the two source corpora.

`gtaweb` ships engine ground truth for depth and semantics and is therefore the
scoring set. `abot` ships colour and a COLMAP sparse model and nothing else, so
it is the corpus the pipeline actually has to predict for.
"""

from __future__ import annotations

__all__ = ["abot", "gtaweb"]
