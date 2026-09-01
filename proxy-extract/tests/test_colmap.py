"""COLMAP text ingestion, which is what ABot 500h actually ships."""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pytest

from proxy_extract import cameras
from proxy_extract.datasets import abot

CAMERAS_TXT = """# Camera list
1 PINHOLE 1280 720 800.0 800.0 640.0 360.0
"""

# images.txt alternates a pose line with a correspondences line.
IMAGES_TXT = """# Image list
2 1 0 0 0 0 0 -5 1 frame_000002.png
100.0 200.0 3 150.0 250.0 -1
1 1 0 0 0 0 0 0 1 frame_000001.png
10.0 20.0 1 30.0 40.0 -1
"""

POINTS_TXT = """# 3D points
1 1.0 2.0 3.0 255 255 255 0.5 1 0 2 0
2 4.0 5.0 6.0 128 128 128 0.7 1 1 2 1
"""


@pytest.fixture
def sparse(tmp_path):
    (tmp_path / "cameras.txt").write_text(CAMERAS_TXT)
    (tmp_path / "images.txt").write_text(IMAGES_TXT)
    (tmp_path / "points3D.txt").write_text(POINTS_TXT)
    return tmp_path


def test_a_sparse_model_loads_as_a_camera_track(sparse):
    track = cameras.from_colmap_text(sparse)
    assert len(track) == 2
    assert track.intrinsics.shape == (3, 3)
    assert track.intrinsics[0, 0] == pytest.approx(800.0)
    assert track.intrinsics[0, 2] == pytest.approx(640.0)


def test_a_sparse_reconstruction_is_never_metric(sparse):
    """Bundle adjustment fixes the scene only up to a similarity.

    Load-bearing rather than pedantic: the 11-class standard wants metres, so
    anything that reads these poses has to know it still needs a scale from
    somewhere else.
    """
    assert cameras.from_colmap_text(sparse).metric is False


def test_images_are_ordered_by_name_not_by_file_order(sparse):
    """images.txt is unordered; capture order has to come from the filename."""
    names = cameras.colmap_registered_names(sparse)
    assert names == ["frame_000001.png", "frame_000002.png"]

    track = cameras.from_colmap_text(sparse)
    # frame 1 sits at the origin, frame 2 five units along +Z of the camera.
    assert np.allclose(track.positions[0], [0.0, 0.0, 0.0])
    assert np.allclose(track.positions[1], [0.0, 0.0, 5.0])


def test_the_pose_convention_is_cam2world(sparse):
    """COLMAP stores world-to-camera; everything downstream expects the inverse."""
    track = cameras.from_colmap_text(sparse)
    for pose in track.cam2world:
        assert np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-9)
        assert np.allclose(pose[3], [0, 0, 0, 1])


def test_a_simple_pinhole_shares_one_focal_length(tmp_path):
    (tmp_path / "cameras.txt").write_text("1 SIMPLE_PINHOLE 640 480 500.0 320.0 240.0\n")
    (tmp_path / "images.txt").write_text(IMAGES_TXT)
    track = cameras.from_colmap_text(tmp_path)
    assert track.intrinsics[0, 0] == track.intrinsics[1, 1] == pytest.approx(500.0)


def test_an_unknown_camera_model_is_refused_rather_than_guessed(tmp_path):
    (tmp_path / "cameras.txt").write_text("1 SOMETHING_NEW 640 480 1 2 3\n")
    (tmp_path / "images.txt").write_text(IMAGES_TXT)
    with pytest.raises(ValueError, match="unsupported COLMAP camera model"):
        cameras.from_colmap_text(tmp_path)


def test_a_reconstruction_with_no_registered_images_raises(tmp_path):
    (tmp_path / "cameras.txt").write_text(CAMERAS_TXT)
    (tmp_path / "images.txt").write_text("# nothing registered\n")
    with pytest.raises(ValueError, match="no registered images"):
        cameras.from_colmap_text(tmp_path)


def test_the_sparse_cloud_reads_as_points(sparse):
    points = cameras.read_colmap_points(sparse / "points3D.txt")
    assert points.shape == (2, 3)
    assert np.allclose(points[0], [1.0, 2.0, 3.0])


def test_a_quaternion_becomes_a_rotation():
    identity = cameras.quaternion_to_rotation(1, 0, 0, 0)
    assert np.allclose(identity, np.eye(3))

    half_turn = cameras.quaternion_to_rotation(0, 0, 1, 0)
    assert np.allclose(half_turn @ np.array([1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0])


def test_an_unnormalised_quaternion_is_normalised_not_rejected():
    scaled = cameras.quaternion_to_rotation(2, 0, 0, 0)
    assert np.allclose(scaled, np.eye(3))


def test_a_zero_quaternion_is_an_error():
    with pytest.raises(ValueError, match="zero-norm"):
        cameras.quaternion_to_rotation(0, 0, 0, 0)


# ------------------------------------------------------------------- ABot


def _episode_tar(path):
    with tarfile.open(path, "w") as archive:
        for name, payload in (
            ("action.json", json.dumps({"actions": [{"w": 1}, {"w": 0}]})),
            ("caption.json", json.dumps({"caption": "a street"})),
            ("sparse/0/cameras.txt", CAMERAS_TXT),
            ("sparse/0/images.txt", IMAGES_TXT),
            ("sparse/0/points3D.txt", POINTS_TXT),
        ):
            data = payload.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_an_episode_is_read_without_unpacking(tmp_path):
    """30k episodes of extracted COLMAP text would cost more inodes than it is worth."""
    annotations = _episode_tar(tmp_path / "annotations.tar")
    members = abot.read_members(annotations)
    assert set(members) >= {"action.json", "caption.json", "cameras.txt", "images.txt"}


def test_actions_come_back_as_a_per_frame_list(tmp_path):
    annotations = _episode_tar(tmp_path / "annotations.tar")
    assert abot.load_actions(annotations) == [{"w": 1}, {"w": 0}]


def test_an_episodes_cameras_load_through_the_tar(tmp_path):
    annotations = _episode_tar(tmp_path / "annotations.tar")
    track = abot.load_cameras(annotations)
    assert len(track) == 2
    assert track.metric is False


def test_an_episode_without_a_colmap_model_says_so(tmp_path):
    path = tmp_path / "annotations.tar"
    with tarfile.open(path, "w") as archive:
        data = b"{}"
        info = tarfile.TarInfo("caption.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with pytest.raises(KeyError, match="COLMAP"):
        abot.load_cameras(path)


def test_discovery_finds_episodes_under_a_snapshot(tmp_path):
    episode = tmp_path / "data" / "ab" / "sample123"
    episode.mkdir(parents=True)
    (episode / "video.mp4").write_bytes(b"")
    _episode_tar(episode / "annotations.tar")

    found = abot.discover(tmp_path)
    assert len(found) == 1
    assert found[0].sample_id == "sample123"
    assert found[0].exists()
