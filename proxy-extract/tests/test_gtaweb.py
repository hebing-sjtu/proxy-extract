"""gta-web GT decoding, and the metrics that score against it.

The decode tests matter more than they look. Both GT streams decode to
something plausible under the wrong settings, so these pin the properties that
tell a correct decode from a convincing one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from proxy_extract import contract
from proxy_extract.benchmark import metrics
from proxy_extract.datasets import gtaweb
from proxy_extract.taxonomy import NUM_STANDARD11, STANDARD11_NAMES


# ------------------------------------------------------------------ depth


def test_the_log_encoding_round_trips_within_one_grey_level():
    metres = np.array([0.12, 0.5, 1.0, 7.5, 42.0, 200.0], dtype=np.float64)
    recovered = gtaweb.gray_to_metres(gtaweb.metres_to_gray(metres))
    assert np.all(np.abs(recovered - metres) / metres < gtaweb.quantisation_step())


def test_the_endpoints_are_the_ones_data_f_states():
    assert gtaweb.gray_to_metres(np.uint8(255)) == pytest.approx(0.1, rel=1e-6)
    # gray 1 is the furthest *finite* code; 0 is reserved for sky.
    assert float(gtaweb.gray_to_metres(np.uint8(1))) == pytest.approx(248.24, abs=0.02)


def test_one_grey_level_is_about_three_percent_of_depth():
    """The corpus's precision ceiling, and the reason DUV is not the limit."""
    assert gtaweb.quantisation_step() == pytest.approx(0.0313, abs=0.0002)


def test_the_duv_format_is_far_finer_than_the_source():
    duv_step = math.expm1(
        (math.log(contract.DEPTH_FAR_METRES) - math.log(contract.DEPTH_NEAR_METRES)) / 65535
    )
    assert gtaweb.quantisation_step() / duv_step > 250


def test_sky_becomes_the_far_plane_not_zero():
    """Writing 0.0 would send sky through the log encoding as the nearest depth."""
    assert gtaweb.gray_to_metres(np.uint8(0)) == pytest.approx(gtaweb.SKY_METRES)
    assert contract.encode_depth_codes(np.array([[gtaweb.SKY_METRES]]))[0, 0] == 0


def test_a_seventh_of_the_code_space_is_inside_the_cwm_near_plane():
    """near=0.1 here against 0.3 in the contract: those codes have to be clamped."""
    grays = np.arange(256, dtype=np.uint8)
    metres = gtaweb.gray_to_metres(grays)
    too_near = int(np.count_nonzero((metres < contract.DEPTH_NEAR_METRES) & (grays > 0)))
    assert 30 <= too_near <= 40


def test_lossy_compression_shows_up_as_extra_distinct_values():
    lossless = gtaweb.gray_to_metres(np.random.default_rng(0).integers(0, 256, size=(4, 32, 32)))
    assert gtaweb.lossy_depth_suspicion(lossless)["lossless"]

    ringing = lossless + np.random.default_rng(1).normal(0, 0.01, lossless.shape)
    assert not gtaweb.lossy_depth_suspicion(ringing)["lossless"]


# --------------------------------------------------------------- semantic


def test_ids_stay_inside_the_standard():
    ids = np.arange(NUM_STANDARD11, dtype=np.uint8)
    assert ids.max() < NUM_STANDARD11


def test_the_decoders_refuse_a_file_that_is_not_there():
    with pytest.raises(FileNotFoundError):
        gtaweb.decode_depth("does-not-exist.mp4")
    with pytest.raises(FileNotFoundError):
        gtaweb.decode_semantic("does-not-exist.mp4")


def test_probe_reports_a_failure_instead_of_raising():
    report = gtaweb.probe_decode(depth_path="missing.mp4", semantic_path="missing.mp4")
    assert report["depth"]["ok"] is False
    assert report["semantic"]["ok"] is False


# ------------------------------------------------------------------ clips


def test_clips_json_parses_both_key_spellings(tmp_path):
    (tmp_path / "clips.json").write_text(
        '[{"tag": "walk", "frameStart": 0, "frameEnd": 124, "color": "color_p000.mp4"},'
        ' {"tag": "drive", "frame_start": 124, "frame_end": 248}]'
    )
    clips = gtaweb.load_clips(tmp_path)
    assert [c.tag for c in clips] == ["walk", "drive"]
    assert clips[0].frames == gtaweb.FRAMES_PER_CLIP
    assert clips[1].is_driving


def test_a_clip_is_one_cwm_window():
    """124 frames is exactly the window size, so every clip yields one and no remainder."""
    assert gtaweb.FRAMES_PER_CLIP == contract.WINDOW_FRAMES
    assert contract.window_count_for(gtaweb.FRAMES_PER_CLIP) == 1


# ---------------------------------------------------------------- metrics


def test_a_perfect_prediction_scores_zero_error():
    truth = np.full((4, 8, 8), 10.0, dtype=np.float32)
    score = metrics.score_depth(truth, truth.copy())
    assert score.abs_rel == pytest.approx(0.0, abs=1e-9)
    assert score.delta1 == pytest.approx(1.0)
    assert score.scale == pytest.approx(1.0)


def test_a_pure_scale_error_is_reported_as_one():
    """The distinction that decides whether to fix the model or add an anchor."""
    truth = np.linspace(1.0, 50.0, 4 * 8 * 8).reshape(4, 8, 8)
    score = metrics.score_depth(truth, truth * 0.5)
    assert score.abs_rel == pytest.approx(0.5, abs=1e-6)
    assert score.abs_rel_scaled == pytest.approx(0.0, abs=1e-6)
    assert score.scale == pytest.approx(2.0, rel=1e-6)
    assert score.scale_limited


def test_a_geometry_error_survives_rescaling():
    rng = np.random.default_rng(0)
    truth = rng.uniform(2.0, 40.0, size=(4, 8, 8))
    scrambled = rng.uniform(2.0, 40.0, size=(4, 8, 8))
    score = metrics.score_depth(truth, scrambled)
    assert not score.scale_limited


def test_per_frame_scale_drift_is_visible():
    """Each frame internally consistent, the clip breathing. Invisible per frame."""
    base = np.full((6, 8, 8), 10.0)
    drifting = base / np.linspace(0.5, 2.0, 6)[:, None, None]
    assert metrics.score_depth(base, drifting).scale_drift > 0.2
    assert metrics.score_depth(base, base / 2.0).scale_drift == pytest.approx(0.0, abs=1e-9)


def test_sky_is_excluded_rather_than_scored():
    truth = np.full((2, 4, 4), 256.0)
    truth[:, 0, 0] = 5.0
    prediction = np.full((2, 4, 4), 5.0)
    score = metrics.score_depth(truth, prediction)
    assert score.valid_fraction == pytest.approx(2 / 32)
    assert score.abs_rel == pytest.approx(0.0, abs=1e-9)


def test_a_class_absent_from_both_sides_scores_nan_not_zero():
    """How `hero` came out NaN before; the distinction is worth keeping visible."""
    truth = np.zeros((4, 4), dtype=np.uint8)
    matrix = metrics.confusion(truth, truth, NUM_STANDARD11)
    iou = metrics.iou_from_confusion(matrix)
    assert iou[0] == pytest.approx(1.0)
    assert np.isnan(iou[1:]).all()


def test_confusion_pairs_surface_the_road_ground_swap():
    truth = np.full((10, 10), 5, dtype=np.uint8)
    prediction = np.full((10, 10), 6, dtype=np.uint8)
    matrix = metrics.confusion(truth, prediction, NUM_STANDARD11)
    pairs = metrics.confusion_pairs(matrix, STANDARD11_NAMES)
    assert pairs[0]["truth"] == "road"
    assert pairs[0]["predicted"] == "ground"
    assert pairs[0]["pixels"] == 100


def test_semantic_scoring_reports_shares_alongside_iou():
    truth = np.array([[0, 0], [5, 5]], dtype=np.uint8)
    prediction = np.array([[0, 0], [5, 6]], dtype=np.uint8)
    score = metrics.score_semantic(truth, prediction, NUM_STANDARD11)
    assert score["pixel_accuracy"] == pytest.approx(0.75)
    assert score["per_class_iou"][0] == pytest.approx(1.0)
    assert score["truth_pixel_share"][0] == pytest.approx(0.5)
