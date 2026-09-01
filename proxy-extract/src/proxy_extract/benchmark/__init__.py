"""Scoring against gta-web's engine ground truth.

The handpick29 study could only measure whether the same extractor agreed with
itself across two renders of one scene. That is self-consistency, not accuracy,
and it cannot score a class the extractor never predicts — which is why `hero`
came out as NaN there.

gta-web ships the engine's own depth and semantics, so predictions can finally
be compared against an answer. Two things follow that were previously out of
reach: real error figures for the depth backend, and a way to fit the
player/ped split's parameters instead of asserting them.

Named `benchmark` rather than `eval` so nothing in here shadows the builtin.
"""

from __future__ import annotations

__all__ = ["metrics", "player_bench"]
