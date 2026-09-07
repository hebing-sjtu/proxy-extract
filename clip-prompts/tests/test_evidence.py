import numpy as np
import pytest
from clip_prompts import evidence, timeline


def _duv_frame(height=192, width=336, *, peds=0, vehicle=False, player=True, depth_code=100):
    """A synthetic DUV frame using only palette values the encoder emits.

    Built rather than decoded because `measure_frames` takes frames, not a
    file: the checks below are about the arithmetic over a bin, and routing
    them through a video codec would test the codec instead.
    """
    frame = np.zeros((height, width, 3), np.uint8)
    frame[..., 0] = depth_code
    frame[: height // 3, :, 0] = evidence.SKY_CODE
    frame[: height // 3, :, 1] = 255
    frame[: height // 3, :, 2] = 255  # sky is (255, 255, 255)
    frame[2 * height // 3 :, :, 1] = 255
    frame[2 * height // 3 :, :, 2] = 255  # road is the same GB with R < 255
    if player:
        frame[100:150, 160:180, 1] = 0
        frame[100:150, 160:180, 2] = 255
    for index in range(peds):
        left = 20 + index * 40
        frame[100:140, left : left + 12, 1] = 0
        frame[100:140, left : left + 12, 2] = 128
    if vehicle:
        frame[120:160, 260:320, 1] = 64
        frame[120:160, 260:320, 2] = 0
    return frame


def test_depth_decoding_matches_the_documented_table():
    """The reverse of DATA_CLIPS.md section 3, to its stated 2.2% bound."""
    for metres, code in ((0.1, 0), (1.0, 52), (5.0, 88), (50.0, 140), (8000.0, 254)):
        assert evidence.code_to_metres(code) == pytest.approx(metres, rel=0.025)


def test_depth_is_logarithmic_not_linear():
    """Reading the code linearly is the documented way to be wildly wrong."""
    assert evidence.code_to_metres(127) == pytest.approx(28.3, rel=0.02)


def test_the_sky_sentinel_is_missing_data_not_a_distance():
    assert np.isnan(evidence.depth_metres(np.array([evidence.SKY_CODE]))).all()


def test_classes_separate_sky_from_road_by_the_red_channel():
    groups = evidence.classes(_duv_frame(player=False))
    assert (groups[:60] == evidence.GROUP_SKY).all()
    assert (groups[130:] == evidence.GROUP_ROAD).all()


def test_classes_find_the_player_and_the_bystanders():
    groups = evidence.classes(_duv_frame(peds=3))
    assert (groups == evidence.GROUP_PLAYER).sum() == 50 * 20
    assert (groups == evidence.GROUP_PED).sum() == 3 * 40 * 12


def test_measure_reports_one_row_per_bin():
    grid = timeline.plan(48, 24.0)
    facts = evidence.measure_frames((_duv_frame(peds=2) for _ in range(48)), grid)
    assert [row["bin"] for row in facts["per_bin"]] == [0, 1]
    assert all(row["player"] for row in facts["per_bin"])
    assert all(row["peds"] == 2 for row in facts["per_bin"])
    assert all(not row["vehicle"] for row in facts["per_bin"])


def test_measure_sees_a_vehicle_in_the_second_it_appears():
    grid = timeline.plan(48, 24.0)
    frames = (_duv_frame(vehicle=index >= 24) for index in range(48))
    facts = evidence.measure_frames(frames, grid)
    assert facts["per_bin"][0]["vehicle"] is False
    assert facts["per_bin"][1]["vehicle"] is True


def test_measure_sees_the_protagonist_leave():
    grid = timeline.plan(48, 24.0)
    frames = (_duv_frame(player=index < 24) for index in range(48))
    facts = evidence.measure_frames(frames, grid)
    assert facts["per_bin"][0]["player"] is True
    assert facts["per_bin"][1]["player"] is False


def test_a_still_world_and_a_moving_one_are_told_apart():
    grid = timeline.plan(24, 24.0)
    still = evidence.measure_frames((_duv_frame() for _ in range(24)), grid)
    moving = evidence.measure_frames(
        (_duv_frame(depth_code=100 + 3 * index) for index in range(24)), grid
    )
    assert still["per_bin"][0]["world_moving"] is False
    assert moving["per_bin"][0]["world_moving"] is True


def test_the_depth_percentile_ignores_the_sky():
    """Sky is a third of the frame; counting it as 8000 m would move the median."""
    grid = timeline.plan(24, 24.0)
    facts = evidence.measure_frames((_duv_frame(depth_code=140) for _ in range(24)), grid)
    assert facts["per_bin"][0]["depth_p50_m"] == pytest.approx(50.0, rel=0.02)


def test_a_short_stream_is_an_error_not_a_short_table():
    grid = timeline.plan(48, 24.0)
    with pytest.raises(ValueError, match="decoded 24 frames"):
        evidence.measure_frames((_duv_frame() for _ in range(24)), grid)


def test_the_last_bin_of_a_clip_gets_its_own_row():
    grid = timeline.plan(124, 24.0)
    facts = evidence.measure_frames((_duv_frame() for _ in range(124)), grid)
    assert len(facts["per_bin"]) == 5
    assert facts["per_bin"][-1]["t"] == [4.0, 5.167]


def test_the_briefing_names_every_bin_and_its_seconds():
    facts = {
        "hero_resolved": True,
        "per_bin": [
            {"bin": 0, "t": [0.0, 1.0], "player": True, "peds": 2, "vehicle": False,
             "ego_vehicle": False, "depth_p50_m": 18.4, "sky_frac": 0.21, "world_moving": True},
        ],
    }
    text = evidence.briefing(facts)
    assert "bin  0 (0-1s):" in text
    assert "2 other people" in text
    assert "no vehicle" in text


def test_the_briefing_says_when_the_protagonist_tracker_failed():
    assert "did not resolve" in evidence.briefing({"hero_resolved": False, "per_bin": []})
