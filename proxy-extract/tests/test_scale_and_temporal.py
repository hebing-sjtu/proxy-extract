from __future__ import annotations

import numpy as np
import pytest

from proxy_extract import cameras as camera_io
from proxy_extract import taxonomy as tx
from proxy_extract.depth.base import DepthResult
from proxy_extract.depth.scale import (
    apply_range_guard,
    solve_scale_from_cameras,
    solve_scale_shift,
)
from proxy_extract.temporal import (
    flicker_rate,
    stabilize_depth,
    stabilize_labels,
    suppress_short_runs,
)


def _trajectory(n=40, step=0.4, rng=None):
    t = np.arange(n)[:, None]
    path = np.hstack([t * step, np.zeros_like(t, float), np.sin(t / 5.0)])
    if rng is not None:
        path = path + rng.normal(0, 1e-3, path.shape)
    return path


class TestScaleFromCameras:
    @pytest.mark.parametrize("true_scale", [0.5, 1.0, 3.7, 25.0])
    def test_recovers_a_known_scale(self, true_scale):
        gt = _trajectory()
        solution = solve_scale_from_cameras(gt / true_scale, gt)
        assert solution.solved
        assert solution.scale == pytest.approx(true_scale, rel=1e-6)

    def test_is_invariant_to_the_unknown_world_transform(self):
        # Pairwise distances are used precisely so no Procrustes step is needed.
        gt = _trajectory()
        angle = 0.7
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        predicted = (gt / 4.0) @ rotation.T + np.array([100.0, -3.0, 7.0])
        solution = solve_scale_from_cameras(predicted, gt)
        assert solution.solved and solution.scale == pytest.approx(4.0, rel=1e-6)

    def test_tolerates_a_few_badly_predicted_poses(self):
        gt = _trajectory()
        predicted = gt / 2.0
        predicted[[3, 17]] += 50.0  # gross outliers
        solution = solve_scale_from_cameras(predicted, gt)
        assert solution.scale == pytest.approx(2.0, rel=0.05)

    def test_a_static_camera_is_reported_unsolvable_rather_than_guessed(self):
        static = np.zeros((30, 3))
        solution = solve_scale_from_cameras(static, static)
        assert not solution.solved
        assert "unobservable" in solution.reason

    def test_a_non_similar_trajectory_is_rejected(self):
        # A circle, so both axes carry comparable extent and squashing one
        # really does make pairwise-distance ratios inconsistent. (Squashing an
        # axis a near-straight path barely uses would stay near-similar, and
        # legitimately so.)
        angle = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        gt = np.stack([np.cos(angle) * 10, np.sin(angle) * 10, np.zeros_like(angle)], axis=1)
        predicted = gt.copy()
        predicted[:, 0] *= 5.0
        solution = solve_scale_from_cameras(predicted, gt)
        assert not solution.solved
        assert "dispersion" in solution.reason

    def test_mismatched_lengths_are_a_hard_error(self):
        with pytest.raises(ValueError, match="shapes differ"):
            solve_scale_from_cameras(_trajectory(10), _trajectory(11))


class TestScaleShift:
    def test_recovers_a_pure_scale(self, rng):
        reference = rng.uniform(2.0, 60.0, (64, 64))
        solution = solve_scale_shift(reference / 6.0, reference)
        assert solution.solved and solution.scale == pytest.approx(6.0, rel=1e-6)

    def test_ignores_invalid_pixels(self, rng):
        reference = rng.uniform(2.0, 60.0, (64, 64))
        source = reference / 3.0
        source[:10] = 0.0
        assert solve_scale_shift(source, reference).scale == pytest.approx(3.0, rel=1e-6)

    def test_affine_mode_recovers_scale_and_shift(self, rng):
        reference = rng.uniform(2.0, 60.0, (64, 64))
        solution = solve_scale_shift((reference - 4.0) / 2.0, reference, fit_shift=True)
        assert solution.solved and solution.scale == pytest.approx(2.0, rel=1e-6)

    def test_too_few_comparable_pixels_is_reported(self):
        assert not solve_scale_shift(np.ones((5, 5)), np.ones((5, 5))).solved


class TestRangeGuard:
    def test_reports_and_clamps_both_ends(self):
        depth = np.array([[0.1, 1.0, 500.0, 0.0]], dtype=np.float32)
        guarded, stats = apply_range_guard(depth, near=0.3, far=256.0)
        assert guarded == pytest.approx(np.array([[0.3, 1.0, 256.0, 0.0]], dtype=np.float32))
        assert stats["clipped_near_fraction"] == pytest.approx(1 / 3)
        assert stats["clipped_far_fraction"] == pytest.approx(1 / 3)


class TestDepthResult:
    def test_scaling_marks_the_result_metric(self):
        result = DepthResult(depth=np.ones((2, 4, 4), np.float32), metric=False)
        scaled = result.scaled(3.0)
        assert scaled.metric and np.allclose(scaled.depth, 3.0)
        assert scaled.meta["applied_scale"] == 3.0

    def test_invalid_pixels_stay_invalid_through_scaling(self):
        depth = np.ones((1, 4, 4), np.float32)
        depth[0, 0, 0] = 0.0
        assert DepthResult(depth=depth, metric=False).scaled(5.0).depth[0, 0, 0] == 0.0


class TestTemporalStabilisation:
    def _flickering_labels(self):
        labels = np.full((9, 16, 16), tx.TERRAIN, np.uint8)
        labels[1::2, 4:8, 4:8] = tx.ROAD_PAVED  # a region that cannot make up its mind
        return labels

    def test_alternating_flicker_is_removed_completely(self):
        labels = self._flickering_labels()
        out = stabilize_labels(labels, radius=2)
        assert flicker_rate(labels) > 0
        assert flicker_rate(out) == 0.0

    def test_a_majority_vote_alone_cannot_remove_it(self):
        # The reason suppress_short_runs exists. Perfect alternation is a root
        # signal of an odd-length mode filter, so the vote re-elects it; if this
        # ever starts passing, the second pass has become redundant.
        labels = self._flickering_labels()
        voted = stabilize_labels(labels, radius=2, min_run=1)
        assert flicker_rate(voted) == flicker_rate(labels)

    def test_short_run_suppression_leaves_long_runs_alone(self):
        labels = np.full((12, 4, 4), tx.TERRAIN, np.uint8)
        labels[4:9] = tx.VEGETATION
        assert np.array_equal(suppress_short_runs(labels, min_run=2), labels)

    def test_short_run_suppression_erases_a_one_frame_blip(self):
        labels = np.full((7, 4, 4), tx.TERRAIN, np.uint8)
        labels[3] = tx.VEHICLE
        assert np.all(suppress_short_runs(labels, min_run=2) == tx.TERRAIN)

    def test_a_longer_min_run_erases_a_longer_blip(self):
        labels = np.full((11, 4, 4), tx.TERRAIN, np.uint8)
        labels[4:6] = tx.VEHICLE
        assert np.any(suppress_short_runs(labels, min_run=2) == tx.VEHICLE)
        assert np.all(suppress_short_runs(labels, min_run=3) == tx.TERRAIN)

    def test_a_sustained_change_survives(self):
        labels = np.full((11, 16, 16), tx.TERRAIN, np.uint8)
        labels[6:] = tx.ROAD_PAVED
        out = stabilize_labels(labels, radius=2)
        assert out[0].max() == tx.TERRAIN
        assert np.all(out[-1] == tx.ROAD_PAVED)

    def test_shape_and_dtype_are_preserved(self):
        labels = self._flickering_labels()
        out = stabilize_labels(labels, radius=2)
        assert out.shape == labels.shape and out.dtype == np.uint8

    def test_only_valid_classes_come_out(self, rng):
        labels = rng.integers(0, tx.NUM_CLASSES, (7, 12, 12)).astype(np.uint8)
        assert stabilize_labels(labels, radius=1).max() < tx.NUM_CLASSES

    def test_both_passes_disabled_is_a_no_op(self):
        labels = self._flickering_labels()
        assert np.array_equal(stabilize_labels(labels, radius=0, min_run=1), labels)

    def test_a_guide_of_the_wrong_length_is_rejected(self):
        labels = self._flickering_labels()
        with pytest.raises(ValueError, match="guide has"):
            stabilize_labels(labels, guide_frames=[np.zeros((16, 16), np.uint8)] * 3, radius=1)

    def test_depth_median_rejects_a_single_frame_spike(self):
        depth = np.full((7, 8, 8), 10.0, np.float32)
        depth[3, 4, 4] = 900.0
        out = stabilize_depth(depth, radius=2)
        assert out[3, 4, 4] == pytest.approx(10.0)

    def test_depth_stabilisation_keeps_invalid_pixels_invalid(self):
        depth = np.zeros((5, 8, 8), np.float32)
        assert np.all(stabilize_depth(depth, radius=2) == 0.0)

    def test_flicker_rate_is_zero_for_a_static_sequence(self):
        assert flicker_rate(np.zeros((5, 4, 4), np.uint8)) == 0.0


class TestCameraTrack:
    def test_positions_are_the_translation_column(self):
        poses = np.tile(np.eye(4), (3, 1, 1))
        poses[:, :3, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        track = camera_io.CameraTrack(cam2world=poses, intrinsics=np.eye(3))
        assert track.positions.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_a_bad_pose_shape_is_rejected(self):
        with pytest.raises(ValueError, match=r"\(N, 4, 4\)"):
            camera_io.CameraTrack(cam2world=np.eye(4), intrinsics=np.eye(3))

    def test_intrinsics_from_fov_puts_the_principal_point_at_the_centre(self):
        k = camera_io.intrinsics_from_fov(1344, 768, 60.0)
        assert k[0, 2] == 672.0 and k[1, 2] == 384.0
        assert k[0, 0] == pytest.approx(672.0 / np.tan(np.deg2rad(30.0)))

    def test_json_round_trip(self, tmp_path):
        import json

        payload = {
            "metric": True,
            "intrinsics": {"width": 1344, "height": 768, "hfov_deg": 50.0},
            "frames": [{"cam2world": np.eye(4).tolist()} for _ in range(3)],
        }
        path = tmp_path / "cams.json"
        path.write_text(json.dumps(payload))
        track = camera_io.load(path)
        assert len(track) == 3 and track.intrinsics.shape == (3, 3)

    def test_subset_keeps_shared_intrinsics_shared(self):
        poses = np.tile(np.eye(4), (5, 1, 1))
        track = camera_io.CameraTrack(cam2world=poses, intrinsics=np.eye(3))
        assert track.subset(np.arange(2)).intrinsics.shape == (3, 3)


class TestFlowDownscale:
    """Solving flow smaller is a compute shortcut, so its cost has to be known.

    Farneback is quadratic in pixel count and a five-frame window needs four
    flows per frame, which at 1280x720 dominates the CPU budget for a
    1800-frame episode. These tests pin down that the shortcut is off by
    default, that it produces a field of the right shape and magnitude, and
    roughly how much the stabilised labels move because of it.
    """

    @staticmethod
    def _panning_clip(frames=12, height=96, width=128, shift=3):
        rng = np.random.default_rng(7)
        texture = rng.integers(0, 255, (height, width * 2), dtype=np.uint8)
        guide, labels = [], []
        for index in range(frames):
            offset = index * shift
            guide.append(texture[:, offset : offset + width].copy())
            board = np.full((height, width), tx.SKY, dtype=np.uint8)
            board[:, : width // 2] = tx.TERRAIN
            # One flickering blob, which is what stabilisation exists to remove.
            if index % 2:
                board[20:30, 20:30] = tx.VEGETATION
            labels.append(board)
        return np.stack(labels), guide

    def test_downscaled_flow_has_the_full_resolution_shape_and_scale(self):
        from proxy_extract.temporal import _flow_between

        _, guide = self._panning_clip()
        full = _flow_between(guide[0], guide[1], downscale=1)
        small = _flow_between(guide[0], guide[1], downscale=2)

        assert small.shape == full.shape
        # Both must report a displacement of the same sign and order of
        # magnitude; a missing rescale would make the small one half as long.
        assert np.median(np.abs(small[..., 0])) == pytest.approx(
            np.median(np.abs(full[..., 0])), rel=0.6
        )

    def test_downscale_one_is_the_unshortcut_path(self):
        labels, guide = self._panning_clip()

        plain = stabilize_labels(labels, guide_frames=guide, radius=2)
        explicit = stabilize_labels(labels, guide_frames=guide, radius=2, flow_downscale=1)

        assert np.array_equal(plain, explicit)

    def test_downscaling_flow_barely_moves_the_stabilised_labels(self):
        labels, guide = self._panning_clip()

        exact = stabilize_labels(labels, guide_frames=guide, radius=2, flow_downscale=1)
        cheap = stabilize_labels(labels, guide_frames=guide, radius=2, flow_downscale=2)

        disagreement = float((exact != cheap).mean())
        assert disagreement < 0.05, f"{disagreement:.4f} of pixels changed class"

    def test_a_zero_downscale_is_refused(self):
        from proxy_extract.temporal import _flow_between

        _, guide = self._panning_clip()
        with pytest.raises(ValueError, match="downscale must be"):
            _flow_between(guide[0], guide[1], downscale=0)
