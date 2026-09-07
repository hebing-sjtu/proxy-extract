"""Timestamped, structured captions for the ABot SFT clips.

One `prompt.json` per clip, at `<clip>/annotations/prompt.json`, holding a
one-second timeline of what happens, the evidence it was checked against, and
the compiled text a trainer feeds the model. See `contract.py` for the format
and `README.md` for how to run it.
"""

from .contract import CONTRACT, PROMPT_NAME, VERSION, Caption

__all__ = ["CONTRACT", "PROMPT_NAME", "VERSION", "Caption"]
