"""The player/ped splitter and the bench that scores it.

Synthetic clips throughout: a real one cannot isolate a single signal, and the
question here is whether each rule fires on the thing it claims to detect.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxy_extract.benchmark import player_bench
from proxy_extract.semantic import player
from proxy_extract.taxonomy import S11_PED, S11_PLAYER, S11_ROAD, S11_VEHICLE

FRAMES, HEIGHT, WIDTH = 24, 192, 336


def blank(fill=S11_ROAD) -> np.ndarray:
    return np.full((FRAMES, HEIGHT, WIDTH), fill, dtype=np.uint8)


def stamp(labels, frame, cx, cy, *, half=14, value=S11_PED):
    y0, y1 = max(int(cy - half), 0), min(int(cy + half), HEIGHT)
    x0, x1 = max(int(cx - half), 0), min(int(cx + half), WIDTH)
    labels[frame, y0:y1, x0:x1] = value


def centred_clip(*, anchor=(0.5, 0.55), half=14) -> np.ndarray:
    labels = blank()
    for frame in range(FRAMES):
        stamp(labels, frame, anchor[0] * WIDTH, anchor[1] * HEIGHT, half=half)
    return labels


# --------------------------------------------------------------- the signal


def test_a_character_held_at_the_anchor_is_the_player():
    result = player.split(centred_clip())
    assert result.resolved
    assert np.any(result.labels == S11_PLAYER)


def test_a_bystander_at_the_edge_is_left_alone():
    labels = blank()
    for frame in range(FRAMES):
        stamp(labels, frame, 0.08 * WIDTH, 0.5 * HEIGHT)
    result = player.split(labels)
    assert not result.resolved
    assert "anchor" in result.note
    assert not np.any(result.labels == S11_PLAYER)


def test_the_centred_one_wins_against_a_drifting_bystander():
    labels = centred_clip()
    for frame in range(FRAMES):
        stamp(labels, frame, 0.05 * WIDTH + frame * 5, 0.30 * HEIGHT, half=10)

    result = player.split(labels)
    assert result.resolved

    promoted = np.argwhere(result.labels[0] == S11_PLAYER)
    centre_x = promoted[:, 1].mean() / WIDTH
    assert centre_x == pytest.approx(0.5, abs=0.08)


def test_an_orbiting_look_shot_still_resolves():
    """The case that separates this from `hero.py`.

    A `look` shot swings the camera around a stationary character, so the
    player's pixels sweep across frame. Ranking by stillness would prefer any
    static bystander; ranking by centredness does not.
    """
    labels = blank()
    for frame in range(FRAMES):
        angle = 2 * np.pi * frame / FRAMES
        stamp(
            labels,
            frame,
            (0.5 + 0.05 * np.cos(angle)) * WIDTH,
            (0.55 + 0.05 * np.sin(angle)) * HEIGHT,
        )
    # A perfectly still bystander off to the side, which stillness would pick.
    for frame in range(FRAMES):
        stamp(labels, frame, 0.85 * WIDTH, 0.5 * HEIGHT, half=10)

    result = player.split(labels)
    assert result.resolved
    promoted = np.argwhere(result.labels[0] == S11_PLAYER)
    assert promoted[:, 1].mean() / WIDTH < 0.7


# ------------------------------------------------------------- declining


def test_an_empty_clip_declines_rather_than_inventing_a_player():
    result = player.split(blank())
    assert not result.resolved
    assert not np.any(result.labels == S11_PLAYER)


def test_two_equally_central_people_are_ambiguous():
    labels = blank()
    for frame in range(FRAMES):
        stamp(labels, frame, 0.44 * WIDTH, 0.55 * HEIGHT, half=12)
        stamp(labels, frame, 0.56 * WIDTH, 0.55 * HEIGHT, half=12)
    result = player.split(labels)
    assert not result.resolved
    assert "ambiguous" in result.note


def test_a_character_visible_only_briefly_declines():
    labels = blank()
    for frame in range(5):
        stamp(labels, frame, 0.5 * WIDTH, 0.55 * HEIGHT)
    result = player.split(labels)
    assert not result.resolved
    assert "frames" in result.note


def test_a_driving_segment_declines_for_a_stated_reason():
    """The protagonist is in the car, and ego shares an id with traffic."""
    labels = blank()
    labels[:, :, :] = S11_VEHICLE
    for frame in range(FRAMES):
        stamp(labels, frame, 0.5 * WIDTH, 0.55 * HEIGHT)
    result = player.split(labels)
    assert result.driving
    assert not result.resolved
    assert "driving" in result.note


def test_persistent_blob_merging_withdraws_the_answer():
    labels = centred_clip()
    for frame in range(FRAMES // 2, FRAMES):
        stamp(labels, frame, 0.5 * WIDTH, 0.55 * HEIGHT, half=30)
    result = player.split(labels)
    assert result.merged_frames > 0.25
    assert not result.resolved
    assert "merged" in result.note


def test_permanent_fusion_is_a_known_blind_spot():
    """Two people fused for the whole clip set their own baseline.

    Nothing here can see it: with no unfused frames the doubled area *is* the
    median. Pinned so the limitation cannot be quietly forgotten.
    """
    labels = blank()
    for frame in range(FRAMES):
        stamp(labels, frame, 0.5 * WIDTH, 0.55 * HEIGHT, half=28)
    result = player.split(labels)
    assert result.merged_frames == 0.0
    assert result.resolved


# ------------------------------------------------------------------ tuning


def test_the_anchor_is_a_parameter_not_a_constant():
    """A clip framed off-centre resolves only once the anchor is told where to look.

    Displaced horizontally on purpose: `anchor_distance` normalises by width in
    both axes, so a vertical offset of the same fraction is a smaller distance.
    That is deliberate — the measure is isotropic in pixels — but it makes a
    vertical test read as if the threshold were looser than it is.
    """
    off_centre = blank()
    for frame in range(FRAMES):
        stamp(off_centre, frame, 0.85 * WIDTH, 0.55 * HEIGHT)

    assert not player.split(off_centre).resolved
    tuned = player.with_anchor(player.PlayerPrior(), (0.85, 0.55))
    assert player.split(off_centre, tuned).resolved


def test_anchor_distance_is_resolution_independent():
    prior = player.PlayerPrior(anchor=(0.5, 0.5))
    track = player.PersonTrack(frames=[0], centres=[(0.6 * 100, 0.5 * 100)], areas=[10.0])
    assert player.anchor_distance(track, (100, 100), prior) == pytest.approx(0.1, abs=1e-6)
    track_big = player.PersonTrack(frames=[0], centres=[(0.6 * 800, 0.5 * 800)], areas=[10.0])
    assert player.anchor_distance(track_big, (800, 800), prior) == pytest.approx(0.1, abs=1e-6)


def test_the_splitter_rejects_a_single_frame():
    with pytest.raises(ValueError, match="frames, height, width"):
        player.split(np.zeros((HEIGHT, WIDTH), dtype=np.uint8))


# ------------------------------------------------------------------- bench


def test_merging_people_reproduces_a_segmenter_view():
    truth = centred_clip()
    truth[truth == S11_PED] = S11_PLAYER
    merged = player_bench.merge_people(truth)
    assert not np.any(merged == S11_PLAYER)
    assert np.any(merged == S11_PED)


def test_the_bench_scores_a_correct_split_as_correct():
    truth = centred_clip()
    truth[truth == S11_PED] = S11_PLAYER
    score = player_bench.score_clip(truth, tag="walk")
    assert score.player_present
    assert score.decided
    assert score.correct
    assert score.iou > 0.9


def test_declining_on_a_clip_with_no_player_counts_as_correct():
    score = player_bench.score_clip(blank(), tag="drive")
    assert not score.player_present
    assert not score.decided
    assert score.correct


def test_promoting_the_wrong_person_scores_as_wrong():
    """GT player at the edge, a decoy at the anchor. Centredness picks the decoy."""
    truth = blank()
    for frame in range(FRAMES):
        stamp(truth, frame, 0.5 * WIDTH, 0.55 * HEIGHT, value=S11_PED)
        stamp(truth, frame, 0.9 * WIDTH, 0.5 * HEIGHT, half=10, value=S11_PLAYER)
    score = player_bench.score_clip(truth, tag="walk")
    assert score.decided
    assert not score.correct
    assert score.iou < player_bench.CORRECT_IOU


def test_measure_anchor_recovers_where_the_player_actually_sits():
    truth = blank()
    for frame in range(FRAMES):
        stamp(truth, frame, 0.42 * WIDTH, 0.71 * HEIGHT, value=S11_PLAYER)

    measured = player_bench.measure_anchor([truth])
    assert measured["frames_with_player"] == FRAMES
    assert measured["anchor"][0] == pytest.approx(0.42, abs=0.02)
    assert measured["anchor"][1] == pytest.approx(0.71, abs=0.02)
    assert measured["radius_p95"] < 0.02


def test_measure_anchor_survives_a_corpus_with_no_visible_player():
    assert player_bench.measure_anchor([blank()])["frames_with_player"] == 0


def test_aggregate_reports_declines_beside_accuracy():
    """A splitter that answers less often is a different thing from one that is wrong."""
    good = centred_clip()
    good[good == S11_PED] = S11_PLAYER
    scores = [
        player_bench.score_clip(good, tag="walk"),
        player_bench.score_clip(blank(), tag="drive"),
    ]
    summary = player_bench.aggregate(scores)
    assert summary["overall"]["clips"] == 2
    assert summary["overall"]["decided_fraction"] == 0.5
    assert set(summary["by_tag"]) == {"walk", "drive"}
    assert summary["by_tag"]["walk"]["accuracy"] == 1.0


def test_the_anchor_sweep_finds_the_framing_it_was_given():
    truth = blank()
    for frame in range(FRAMES):
        stamp(truth, frame, 0.85 * WIDTH, 0.55 * HEIGHT, value=S11_PLAYER)

    swept = player_bench.sweep_anchor([(truth, "walk")], [(0.5, 0.55), (0.85, 0.55)])
    by_anchor = {row["anchor"]: row["accuracy"] for row in swept}
    assert by_anchor[(0.85, 0.55)] > by_anchor[(0.5, 0.55)]


def test_aggregate_of_nothing_is_not_an_error():
    assert player_bench.aggregate([]) == {"clips": 0}
