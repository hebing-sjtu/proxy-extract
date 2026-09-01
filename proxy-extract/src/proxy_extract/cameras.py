"""Ground-truth camera tracks.

The engine-side GT format is not fixed yet, so this module defines one neutral
in-memory representation plus loaders for a couple of plausible on-disk shapes.
Adding a new source format should mean writing one `from_*` function, not
touching anything downstream.

Convention throughout: OpenCV camera axes (+X right, +Y down, +Z forward) and
cam2world 4x4 extrinsics, matching what MapAnything consumes and returns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraTrack:
    """Per-frame cam2world poses plus a shared or per-frame intrinsic matrix."""

    cam2world: np.ndarray  # (N, 4, 4) float64
    intrinsics: np.ndarray  # (3, 3) or (N, 3, 3) float64
    metric: bool = True

    def __post_init__(self) -> None:
        if self.cam2world.ndim != 3 or self.cam2world.shape[1:] != (4, 4):
            raise ValueError(f"cam2world must be (N, 4, 4), got {self.cam2world.shape}")
        if self.intrinsics.shape not in {(3, 3), (len(self.cam2world), 3, 3)}:
            raise ValueError(f"intrinsics must be (3, 3) or (N, 3, 3), got {self.intrinsics.shape}")

    def __len__(self) -> int:
        return len(self.cam2world)

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) camera centres in world coordinates."""
        return self.cam2world[:, :3, 3]

    def intrinsics_for(self, index: int) -> np.ndarray:
        return self.intrinsics if self.intrinsics.ndim == 2 else self.intrinsics[index]

    def subset(self, indices: np.ndarray) -> CameraTrack:
        return CameraTrack(
            cam2world=self.cam2world[indices],
            intrinsics=self.intrinsics if self.intrinsics.ndim == 2 else self.intrinsics[indices],
            metric=self.metric,
        )


def intrinsics_from_fov(width: int, height: int, horizontal_fov_degrees: float) -> np.ndarray:
    """Pinhole intrinsics from an image size and a horizontal field of view.

    Game engines usually expose FOV rather than a focal length, so this is the
    likeliest bridge between an engine dump and a camera matrix.
    """
    focal = (width / 2.0) / np.tan(np.deg2rad(horizontal_fov_degrees) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def from_npz(path: Path) -> CameraTrack:
    """Load from an .npz holding `cam2world` and `intrinsics` arrays."""
    with np.load(Path(path)) as data:
        missing = {"cam2world", "intrinsics"} - set(data.files)
        if missing:
            raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
        return CameraTrack(
            cam2world=np.asarray(data["cam2world"], dtype=np.float64),
            intrinsics=np.asarray(data["intrinsics"], dtype=np.float64),
            metric=bool(data["metric"]) if "metric" in data.files else True,
        )


def from_json(path: Path) -> CameraTrack:
    """Load from a JSON document shaped like:

        {
          "metric": true,
          "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
          "frames": [{"cam2world": [[...4x4...]]}, ...]
        }

    `intrinsics` may instead be `{"width": w, "height": h, "hfov_deg": f}`.
    """
    payload = json.loads(Path(path).read_text())
    raw_intrinsics = payload["intrinsics"]
    if isinstance(raw_intrinsics, dict):
        intrinsics = intrinsics_from_fov(
            int(raw_intrinsics["width"]), int(raw_intrinsics["height"]), float(raw_intrinsics["hfov_deg"])
        )
    else:
        intrinsics = np.asarray(raw_intrinsics, dtype=np.float64)

    poses = np.asarray([frame["cam2world"] for frame in payload["frames"]], dtype=np.float64)
    return CameraTrack(cam2world=poses, intrinsics=intrinsics, metric=bool(payload.get("metric", True)))


def from_abot_json(path: Path) -> CameraTrack:
    """Load the handpick29 `camera/<clip>.json` format.

    These come from a COLMAP sparse reconstruction of the *source* clip, so the
    poses are self-consistent but the world unit is arbitrary — hence
    `metric=False`. Intrinsics are already rescaled to the 1280x720 delivery.
    """
    payload = json.loads(Path(path).read_text())
    raw = payload["intrinsics"]
    if raw.get("model") not in {None, "PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError(f"{path}: unsupported camera model {raw['model']!r}")

    intrinsics = np.array(
        [[raw["fx"], 0.0, raw["cx"]], [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    poses = np.asarray([frame["c2w"] for frame in payload["frames"]], dtype=np.float64)
    return CameraTrack(cam2world=poses, intrinsics=intrinsics, metric=False)


def from_abot_npz(path: Path) -> CameraTrack:
    """Load the handpick29 `camera/<clip>.npz` sidecar (same data as the JSON)."""
    with np.load(Path(path)) as data:
        fx, fy, cx, cy = np.asarray(data["intrinsics"], dtype=np.float64)[:4]
        return CameraTrack(
            cam2world=np.asarray(data["c2w"], dtype=np.float64),
            intrinsics=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
            metric=False,
        )


def load(path: Path) -> CameraTrack:
    """Load a camera track, sniffing which of the known layouts it uses."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            keys = set(data.files)
        return from_abot_npz(path) if "c2w" in keys else from_npz(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        intrinsics = payload.get("intrinsics")
        is_abot = isinstance(intrinsics, dict) and "fx" in intrinsics
        return from_abot_json(path) if is_abot else from_json(path)
    raise ValueError(f"unsupported camera track format: {path.suffix}")


# ------------------------------------------------------------ COLMAP text
# What ABot-World-Explorer-500h actually ships, inside each episode's
# annotations.tar as sparse/0/{cameras,images,points3D}.txt. The existing
# from_abot_* loaders read a preprocessed sidecar that only the handpick29
# subset had.


def _colmap_intrinsics(model: str, width: int, height: int, params: list[float]) -> np.ndarray:
    """Camera matrix from one `cameras.txt` row.

    Distortion terms are read and dropped: everything downstream here treats
    the camera as a pinhole, and silently ignoring a distorted model would be
    worse than saying so. SIMPLE_* models share one focal length across axes.
    """
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"}:
        focal, cx, cy = params[0], params[1], params[2]
        fx = fy = focal
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise ValueError(f"unsupported COLMAP camera model: {model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP stores Hamilton quaternions as (w, x, y, z), world-to-camera."""
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("zero-norm quaternion")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _strip(lines) -> list[str]:
    return [line for line in (raw.strip() for raw in lines) if line and not line.startswith("#")]


def from_colmap_text(sparse_dir: Path, *, order: str = "name") -> CameraTrack:
    """Load a COLMAP text sparse model as a CameraTrack.

    `metric=False`, always. A sparse reconstruction fixes the scene only up to
    a similarity: the poses are self-consistent but the world unit is whatever
    the bundle adjustment settled on. Anything that needs metres has to get the
    scale from somewhere else.

    COLMAP's `images.txt` is unordered and holds one row per *registered*
    image, so frames that failed to register are simply absent. Ordering by
    `name` recovers capture order from the filename; the track is therefore not
    guaranteed to be gap-free. Use `colmap_registered_names` to find out which
    frames made it before assuming index i is frame i.
    """
    sparse_dir = Path(sparse_dir)
    cameras: dict[int, np.ndarray] = {}
    for line in _strip((sparse_dir / "cameras.txt").read_text().splitlines()):
        fields = line.split()
        cameras[int(fields[0])] = _colmap_intrinsics(
            fields[1], int(fields[2]), int(fields[3]), [float(v) for v in fields[4:]]
        )

    # images.txt alternates: a pose line, then a line of 2D-3D correspondences
    # that this loader does not need.
    rows = _strip((sparse_dir / "images.txt").read_text().splitlines())
    entries = []
    for line in rows:
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            qw, qx, qy, qz = (float(v) for v in fields[1:5])
            tx, ty, tz = (float(v) for v in fields[5:8])
            camera_id = int(fields[8])
        except ValueError:
            continue
        entries.append((fields[9], camera_id, (qw, qx, qy, qz), (tx, ty, tz)))

    if not entries:
        raise ValueError(f"no registered images in {sparse_dir / 'images.txt'}")
    if order == "name":
        entries.sort(key=lambda row: row[0])

    poses = np.zeros((len(entries), 4, 4), dtype=np.float64)
    intrinsics = np.zeros((len(entries), 3, 3), dtype=np.float64)
    for index, (_, camera_id, quaternion, translation) in enumerate(entries):
        rotation = quaternion_to_rotation(*quaternion)
        # COLMAP stores world-to-camera; invert for the cam2world convention
        # used everywhere else in this package.
        poses[index, :3, :3] = rotation.T
        poses[index, :3, 3] = -rotation.T @ np.asarray(translation)
        poses[index, 3, 3] = 1.0
        intrinsics[index] = cameras[camera_id]

    shared = intrinsics[0] if np.allclose(intrinsics, intrinsics[0]) else intrinsics
    return CameraTrack(cam2world=poses, intrinsics=shared, metric=False)


def colmap_registered_names(sparse_dir: Path, *, order: str = "name") -> list[str]:
    """Image names in the same order `from_colmap_text` returns poses.

    Needed because a reconstruction that dropped frames still produces a dense
    pose array, and pairing pose i with video frame i would then be wrong from
    the first gap onwards.
    """
    names = []
    for line in _strip((Path(sparse_dir) / "images.txt").read_text().splitlines()):
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            float(fields[1])
            int(fields[8])
        except ValueError:
            continue
        names.append(fields[9])
    return sorted(names) if order == "name" else names


def read_colmap_points(path: Path) -> np.ndarray:
    """The (M, 3) sparse point cloud from `points3D.txt`.

    Comes free with the reconstruction and shares its arbitrary scale, so it
    cannot supply metres on its own. What it can do is anchor a *relative*
    depth prediction: reprojected into a frame, these points give sparse depth
    samples that a per-frame monocular estimate can be fitted against, which
    turns per-frame scale drift into one scale per clip.
    """
    points = []
    for line in _strip(Path(path).read_text().splitlines()):
        fields = line.split()
        if len(fields) >= 4:
            points.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def world_to_camera(track: CameraTrack) -> np.ndarray:
    """(N, 4, 4) inverses of the cam2world poses, via transpose rather than solve."""
    poses = track.cam2world
    rotations = poses[:, :3, :3]
    w2c = np.zeros_like(poses)
    w2c[:, :3, :3] = np.swapaxes(rotations, 1, 2)
    w2c[:, :3, 3] = -np.einsum("nij,nj->ni", w2c[:, :3, :3], poses[:, :3, 3])
    w2c[:, 3, 3] = 1.0
    return w2c


def relative_pose(track: CameraTrack, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation taking a point in camera `i` into camera `j`."""
    w2c = world_to_camera(track)
    transform = w2c[j] @ track.cam2world[i]
    return transform[:3, :3], transform[:3, 3]


def baseline_extent(positions: np.ndarray) -> float:
    """Spread of a camera trajectory, as the median distance from its centroid.

    Used to decide whether a track moved enough for its scale to be solvable.
    """
    positions = np.asarray(positions, dtype=np.float64)
    return float(np.median(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))
