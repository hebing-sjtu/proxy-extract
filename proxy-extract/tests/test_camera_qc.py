"""The epipolar machinery decides which clips are usable, so it gets checked
against geometry whose answer is known in closed form rather than against a
recorded output."""

from __future__ import annotations

import json

import numpy as np
import pytest

from proxy_extract import camera_qc, cameras


def look_along_z(position: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = position
    return pose


@pytest.fixture
def sideways_pair() -> cameras.CameraTrack:
    """Two identically oriented cameras, one metre apart along +X."""
    poses = np.stack([look_along_z(np.zeros(3)), look_along_z(np.array([1.0, 0.0, 0.0]))])
    intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    return cameras.CameraTrack(cam2world=poses, intrinsics=intrinsics, metric=True)


class TestPoseAlgebra:
    def test_world_to_camera_inverts_the_poses(self, sideways_pair):
        product = camera_qc.np.einsum(
            "nij,njk->nik", cameras.world_to_camera(sideways_pair), sideways_pair.cam2world
        )
        assert product == pytest.approx(np.stack([np.eye(4)] * 2), abs=1e-12)

    def test_relative_pose_of_a_pure_translation(self, sideways_pair):
        rotation, translation = cameras.relative_pose(sideways_pair, 0, 1)
        assert rotation == pytest.approx(np.eye(3), abs=1e-12)
        # Moving the camera +1 along X moves the world -1 along X in its frame.
        assert translation == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)

    def test_a_frame_relative_to_itself_is_the_identity(self, sideways_pair):
        rotation, translation = cameras.relative_pose(sideways_pair, 1, 1)
        assert rotation == pytest.approx(np.eye(3), abs=1e-12)
        assert translation == pytest.approx(np.zeros(3), abs=1e-12)


class TestEpipolarGeometry:
    def project(self, track, index, points_world):
        w2c = cameras.world_to_camera(track)[index]
        local = points_world @ w2c[:3, :3].T + w2c[:3, 3]
        pixels = local @ track.intrinsics.T
        return pixels[:, :2] / pixels[:, 2:3], local[:, 2]

    def test_true_correspondences_land_on_the_epipolar_line(self, sideways_pair):
        rng = np.random.default_rng(0)
        world = np.column_stack(
            [rng.uniform(-5, 5, 200), rng.uniform(-5, 5, 200), rng.uniform(4, 30, 200)]
        )
        points_a, _ = self.project(sideways_pair, 0, world)
        points_b, _ = self.project(sideways_pair, 1, world)

        rotation, translation = cameras.relative_pose(sideways_pair, 0, 1)
        fundamental = camera_qc.fundamental_from_pose(sideways_pair.intrinsics, rotation, translation)
        assert camera_qc.sampson_distance(points_a, points_b, fundamental).max() < 1e-6

    def test_a_displaced_match_is_penalised_by_roughly_its_offset(self, sideways_pair):
        rng = np.random.default_rng(1)
        world = np.column_stack(
            [rng.uniform(-5, 5, 50), rng.uniform(-5, 5, 50), rng.uniform(4, 30, 50)]
        )
        points_a, _ = self.project(sideways_pair, 0, world)
        points_b, _ = self.project(sideways_pair, 1, world)

        rotation, translation = cameras.relative_pose(sideways_pair, 0, 1)
        fundamental = camera_qc.fundamental_from_pose(sideways_pair.intrinsics, rotation, translation)

        # The baseline is along X, so the epipolar lines are horizontal and a
        # vertical push of 3 px is a 3 px violation. Sampson splits the
        # correction over both images, so it reports 3/sqrt(2) rather than 3 —
        # worth pinning down, because it means the pixel thresholds in
        # `classify` sit at ~1.4x that much total displacement.
        nudged = points_b + np.array([0.0, 3.0])
        errors = camera_qc.sampson_distance(points_a, nudged, fundamental)
        assert errors == pytest.approx(np.full(len(errors), 3.0 / np.sqrt(2)), abs=0.05)

    def test_sliding_along_the_line_is_not_penalised(self, sideways_pair):
        rng = np.random.default_rng(2)
        world = np.column_stack(
            [rng.uniform(-5, 5, 50), rng.uniform(-5, 5, 50), rng.uniform(4, 30, 50)]
        )
        points_a, _ = self.project(sideways_pair, 0, world)
        points_b, _ = self.project(sideways_pair, 1, world)

        rotation, translation = cameras.relative_pose(sideways_pair, 0, 1)
        fundamental = camera_qc.fundamental_from_pose(sideways_pair.intrinsics, rotation, translation)

        # A horizontal shift is a depth change, which the epipolar constraint
        # cannot see. This is the blind spot the triangulation step covers.
        slid = points_b + np.array([7.0, 0.0])
        assert camera_qc.sampson_distance(points_a, slid, fundamental).max() < 1e-6

    def test_triangulation_recovers_the_depths_it_was_built_from(self, sideways_pair):
        rng = np.random.default_rng(3)
        world = np.column_stack(
            [rng.uniform(-4, 4, 100), rng.uniform(-4, 4, 100), rng.uniform(5, 40, 100)]
        )
        points_a, depths = self.project(sideways_pair, 0, world)
        points_b, _ = self.project(sideways_pair, 1, world)

        rotation, translation = cameras.relative_pose(sideways_pair, 0, 1)
        recovered = camera_qc.triangulate(
            points_a, points_b, sideways_pair.intrinsics, rotation, translation
        )
        assert recovered == pytest.approx(depths, rel=1e-6)

    def test_triangulated_depth_scales_with_the_baseline(self, sideways_pair):
        """A track scaled by k triangulates depths scaled by k.

        This is the property that makes the world unit unrecoverable from the
        cameras alone, and the reason scale has to come from somewhere else.
        """
        rng = np.random.default_rng(4)
        world = np.column_stack(
            [rng.uniform(-4, 4, 60), rng.uniform(-4, 4, 60), rng.uniform(5, 40, 60)]
        )
        points_a, _ = self.project(sideways_pair, 0, world)
        points_b, _ = self.project(sideways_pair, 1, world)

        rotation, translation = cameras.relative_pose(sideways_pair, 0, 1)
        near = camera_qc.triangulate(points_a, points_b, sideways_pair.intrinsics, rotation, translation)
        far = camera_qc.triangulate(
            points_a, points_b, sideways_pair.intrinsics, rotation, 3.0 * translation
        )
        assert far == pytest.approx(3.0 * near, rel=1e-6)


class TestClassification:
    @pytest.mark.parametrize(
        "sampson,inlier,expected",
        [
            (0.4, 0.95, "match"),
            (1.9, 0.40, "loose"),
            (2.5, 0.50, "loose"),
            (11.0, 0.20, "mismatch"),
            (float("nan"), float("nan"), "unscored"),
        ],
    )
    def test_tiers(self, sampson, inlier, expected):
        assert camera_qc.classify(sampson, inlier)[0] == expected

    @pytest.mark.parametrize(
        "sampson,expected",
        [(0.4, "keep"), (1.0, "keep"), (1.8, "review"), (3.0, "review"), (4.5, "drop")],
    )
    def test_fidelity_is_a_stricter_cut_than_pose_validity(self, sampson, expected):
        assert camera_qc.fidelity_tier(sampson) == expected

    def test_the_two_questions_can_disagree(self):
        # 1.73 px on a low-poly render: the poses are clearly the right ones,
        # yet the geometry has drifted too far to feed training unreviewed.
        assert camera_qc.classify(1.73, 0.62)[0] == "match"
        assert camera_qc.fidelity_tier(1.73) == "review"

    def test_a_low_inlier_rate_demotes_an_otherwise_good_median(self):
        # Half the scene drifting while half stays put can leave the median
        # small; the inlier fraction is what catches it.
        assert camera_qc.classify(0.5, 0.30)[0] == "loose"


class TestAbotLoader:
    def test_sniffs_the_handpick_json_and_marks_it_non_metric(self, tmp_path):
        payload = {
            "intrinsics": {"model": "PINHOLE", "fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0},
            "frames": [{"c2w": np.eye(4).tolist()}, {"c2w": np.eye(4).tolist()}],
        }
        path = tmp_path / "clip.json"
        path.write_text(json.dumps(payload))

        track = cameras.load(path)
        assert len(track) == 2
        assert track.metric is False
        assert track.intrinsics[0, 0] == 700.0
        assert track.intrinsics[1, 2] == 360.0

    def test_sniffs_the_handpick_npz(self, tmp_path):
        path = tmp_path / "clip.npz"
        np.savez(path, c2w=np.stack([np.eye(4)] * 3), intrinsics=np.array([700.0, 700, 640, 360, 1280, 720]))

        track = cameras.load(path)
        assert len(track) == 3
        assert track.metric is False
        assert track.intrinsics[1, 1] == 700.0

    def test_the_neutral_npz_layout_still_loads(self, tmp_path):
        path = tmp_path / "other.npz"
        np.savez(path, cam2world=np.stack([np.eye(4)] * 2), intrinsics=np.eye(3))

        track = cameras.load(path)
        assert track.metric is True

    def test_an_unsupported_camera_model_is_refused(self, tmp_path):
        path = tmp_path / "fisheye.json"
        path.write_text(
            json.dumps(
                {
                    "intrinsics": {"model": "OPENCV_FISHEYE", "fx": 1, "fy": 1, "cx": 0, "cy": 0},
                    "frames": [{"c2w": np.eye(4).tolist()}],
                }
            )
        )
        with pytest.raises(ValueError, match="OPENCV_FISHEYE"):
            cameras.load(path)


class TestVerifyTrack:
    def test_a_frame_count_mismatch_is_refused(self, sideways_pair):
        frames = [np.zeros((48, 64), np.uint8)] * 5
        with pytest.raises(ValueError, match="5 frames but 2 poses"):
            camera_qc.verify_track(frames, sideways_pair, "clip")

    def test_featureless_frames_are_unscored_rather_than_perfect(self):
        poses = np.stack([look_along_z(np.array([0.0, 0.0, i * 0.5])) for i in range(20)])
        intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        track = cameras.CameraTrack(poses, intrinsics, metric=True)

        verdict = camera_qc.verify_track([np.zeros((480, 640), np.uint8)] * 20, track, "blank")
        assert verdict.tier == "unscored"
        assert verdict.scored_pairs == []
