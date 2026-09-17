"""The depth scale lock, which has to remove one thing and preserve another.

These are the tests that stand between this pipeline and the failure
`PROXY_DUV_SPEC.md` section 2 is entirely about: depth that has been
renormalised per clip passes every structural check, trains, scores, and
teaches nothing. The lock and that mistake are the same operation applied to
different frequencies, so the two halves are asserted separately - that the
jitter goes, and that the clip's own metric level and its real motion do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxy_extract.temporal import (
    MAX_LOG_CORRECTION,
    _local_linear_trend,
    lock_depth_scale,
)

FRAMES = 60
SHAPE = (24, 40)


def _stack(levels: np.ndarray) -> np.ndarray:
    """A depth stack that is flat in space and carries `levels` in time."""
    return np.repeat(np.asarray(levels, dtype=np.float32), SHAPE[0] * SHAPE[1]).reshape(
        len(levels), *SHAPE
    )


def _geometric_mean(depth: np.ndarray) -> float:
    valid = depth[depth > 0]
    return float(np.exp(np.log(valid).mean()))


# ------------------------------------------------------- what must be removed


def test_per_frame_jitter_is_removed():
    # Alternating +/-5%, which is the shape a per-frame metric head produces and
    # the shape a windowed median provably cannot fix: every pixel of the frame
    # is wrong by the same factor, so there is no dissenting neighbour.
    jitter = np.where(np.arange(FRAMES) % 2, 1.05, 0.95)
    locked, info = lock_depth_scale(_stack(10.0 * jitter), radius=12)

    assert info["scale_locked"] is True
    per_frame = locked[:, 0, 0]
    before = float(np.std(np.log(10.0 * jitter)))
    after = float(np.std(np.log(per_frame)))
    assert after < before / 10, f"jitter only fell from {before:.4f} to {after:.4f}"
    assert info["scale_jitter_removed_pct"] > 1.0


def test_the_report_says_how_much_it_took_out():
    """A run needs to be able to tell "this did nothing" from "this saved us"."""
    steady = lock_depth_scale(_stack(np.full(FRAMES, 10.0)), radius=12)[1]
    noisy = lock_depth_scale(
        _stack(10.0 * np.where(np.arange(FRAMES) % 2, 1.05, 0.95)), radius=12
    )[1]

    assert steady["scale_jitter_removed_pct"] == pytest.approx(0.0, abs=1e-6)
    assert noisy["scale_jitter_removed_pct"] > 4.0


# ------------------------------------------------------ what must be preserved


def test_the_clips_own_level_survives():
    """The guarantee: mean log correction is exactly zero, so the geometric
    mean of the clip's depths is untouched.

    This is the whole difference between this and a per-clip renormalisation.
    The encoder's codes are centred on the log of the depth, so preserving the
    geometric mean is precisely preserving where this clip sits in the corpus.
    """
    jitter = np.where(np.arange(FRAMES) % 2, 1.05, 0.95)
    original = _stack(10.0 * jitter)

    locked, info = lock_depth_scale(original, radius=12)

    assert info["scale_mean_log_correction"] == pytest.approx(0.0, abs=1e-9)
    assert _geometric_mean(locked) == pytest.approx(_geometric_mean(original), rel=1e-6)


def test_a_steady_approach_is_not_mistaken_for_jitter():
    """A camera walking towards something is real motion and must pass through.

    Constant speed in log depth is a straight level curve, and the trend is a
    straight-line fit, so this is exact rather than approximate - including for
    the first and last `radius` frames, where a moving average would have
    invented a correction out of its own window bias.
    """
    approach = 40.0 * np.exp(-np.linspace(0.0, 1.5, FRAMES))
    locked, _info = lock_depth_scale(_stack(approach), radius=12)

    assert locked[:, 0, 0] == pytest.approx(approach.astype(np.float32), rel=1e-4)


def test_the_ends_of_a_clip_are_not_bent():
    """Stated separately because this is where the bias would have lived.

    An edge-padded moving average biases the trend at the boundary, and the
    trend is subtracted, so every clip in the corpus would have had its first
    and last half-second scaled wrongly in the same direction. A per-frame
    check at the edges is the only thing that would have caught it.
    """
    approach = 40.0 * np.exp(-np.linspace(0.0, 1.5, FRAMES))
    locked, _info = lock_depth_scale(_stack(approach), radius=12)

    edges = [0, 1, 2, FRAMES - 3, FRAMES - 2, FRAMES - 1]
    for index in edges:
        assert locked[index, 0, 0] == pytest.approx(approach[index], rel=1e-4), (
            f"frame {index} was moved, so the trend is biased at the boundary"
        )


def test_invalid_pixels_stay_invalid():
    """Zero is the sky sentinel, and a scale factor must not turn it into a
    surface at 0 m - which would encode as the nearest possible depth and
    occlude the whole frame."""
    stack = _stack(10.0 * np.where(np.arange(FRAMES) % 2, 1.05, 0.95))
    stack[:, :4, :] = 0.0

    locked, _info = lock_depth_scale(stack, radius=12)

    assert np.all(locked[:, :4, :] == 0.0)
    assert np.all(locked[:, 4:, :] > 0.0)


# --------------------------------------------------------------- degeneracies


def test_a_scene_cut_does_not_produce_a_step():
    """Two halves that share no surfaces cannot have their ratio measured.

    The pair is skipped, which freezes the level curve across the gap. The
    alternative - believing the ratio - would read the cut as a single enormous
    scale error and apply its inverse to half the clip.
    """
    stack = _stack(np.full(FRAMES, 10.0))
    # The second half is almost entirely invalid, so the pair straddling the
    # boundary falls under the co-visible floor.
    stack[FRAMES // 2 :, :, :] = 0.0
    stack[FRAMES // 2 :, 0, :2] = 10.0

    locked, info = lock_depth_scale(stack, radius=12)

    assert info["scale_pairs_uninformative"] >= 1
    assert np.all(np.isfinite(locked))
    assert info["scale_jitter_removed_pct"] < 100.0


def test_a_correction_is_clamped():
    """The clamp and the zero mean have to hold together, not in turn.

    Clipping one bad frame leaves a mean behind, and taking that mean off every
    frame moves the clipped one further out than the clip allowed. The fix is
    to take it out of the frames that have room, and this asserts both halves
    at once because fixing either one alone reintroduces the other's failure.
    """
    stack = _stack(np.full(FRAMES, 10.0))
    # One frame at a tenth of the level: a 10x step is not jitter, and applying
    # its inverse would be a worse frame than leaving it alone.
    stack[FRAMES // 2] = 1.0

    locked, info = lock_depth_scale(stack, radius=12)

    assert info["scale_frames_clamped"] >= 1
    applied = np.log(locked[:, 0, 0] / stack[:, 0, 0])
    assert np.all(np.abs(applied) <= MAX_LOG_CORRECTION + 1e-6)
    assert info["scale_mean_log_correction"] == pytest.approx(0.0, abs=1e-9)


def test_a_short_stack_is_returned_untouched():
    stack = _stack(np.array([5.0, 7.0]))
    locked, info = lock_depth_scale(stack, radius=12)

    assert info["scale_locked"] is False
    assert np.array_equal(locked, stack)


def test_a_radius_below_one_is_refused():
    with pytest.raises(ValueError, match="radius"):
        lock_depth_scale(_stack(np.full(FRAMES, 10.0)), radius=0)


def test_a_guide_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="guide"):
        lock_depth_scale(
            _stack(np.full(FRAMES, 10.0)),
            guide_frames=[np.zeros((*SHAPE, 3), np.uint8)] * 3,
            radius=12,
        )


# ------------------------------------------------------------------ the trend


def test_the_trend_reproduces_a_straight_line_everywhere():
    """The property the whole edge behaviour rests on, asserted on its own."""
    line = 3.0 + 0.25 * np.arange(40)

    assert _local_linear_trend(line, radius=5) == pytest.approx(line)


def test_the_trend_ignores_zero_mean_wobble():
    line = 3.0 + 0.25 * np.arange(41)
    wobbly = line + np.where(np.arange(41) % 2, 0.1, -0.1)

    trend = _local_linear_trend(wobbly, radius=5)

    # Interior only: at the boundary the window is one-sided and an odd number
    # of alternating samples cannot cancel, which is a property of the signal
    # rather than of the fit.
    assert trend[5:-5] == pytest.approx(line[5:-5], abs=0.02)
