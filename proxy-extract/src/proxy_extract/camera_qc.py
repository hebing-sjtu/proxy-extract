"""Do the supplied camera tracks actually describe the supplied video?

Everything downstream that uses GT cameras — metric scale calibration, depth
consistency checks, pose-conditioned depth models — is worthless if the poses
belong to a different clip, a different frame ordering, or a differently
cropped render. This module answers that question from the video alone, using
sparse correspondences and the epipolar constraint, which needs no depth and no
GPU.

The same machinery also triangulates sparse depth in the track's own world
unit, which is what lets us later ask "how many metres is one of these units?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .cameras import CameraTrack, relative_pose

# A pair one frame apart has almost no baseline, which makes the essential
# matrix degenerate and the epipolar test vacuous. Eight frames at 24fps is a
# third of a second: enough parallax, still enough overlap to track through.
DEFAULT_GAP = 8
MIN_TRACKED_POINTS = 40
# Below this the translation is dominated by pose noise and the epipolar
# geometry stops being informative, so such pairs are reported, not scored.
MIN_BASELINE_FRACTION = 0.02


@dataclass
class PairEvidence:
    """One frame pair's worth of epipolar and triangulation evidence."""

    index_a: int
    index_b: int
    baseline: float
    num_points: int
    sampson_median: float = float("nan")
    sampson_p90: float = float("nan")
    inlier_fraction: float = float("nan")
    depths: np.ndarray = field(default_factory=lambda: np.empty(0))
    pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    @property
    def degenerate(self) -> bool:
        return not np.isfinite(self.sampson_median)


@dataclass
class TrackVerdict:
    """Whether a camera track and a video agree, and by how much."""

    clip: str
    pairs: list[PairEvidence]
    sampson_median: float
    inlier_fraction: float
    tier: str
    note: str

    @property
    def scored_pairs(self) -> list[PairEvidence]:
        return [pair for pair in self.pairs if not pair.degenerate]


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_from_pose(intrinsics: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """F mapping a point in image A to its epipolar line in image B."""
    essential = skew(translation) @ rotation
    inverse = np.linalg.inv(intrinsics)
    return inverse.T @ essential @ inverse


def sampson_distance(points_a: np.ndarray, points_b: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    """First-order geometric distance to the epipolar line, in pixels.

    Sampson rather than raw algebraic error because the latter scales with the
    arbitrary norm of F and cannot be compared against a pixel threshold.
    """
    ones = np.ones((len(points_a), 1))
    homogeneous_a = np.hstack([points_a, ones])
    homogeneous_b = np.hstack([points_b, ones])

    line_b = homogeneous_a @ fundamental.T
    line_a = homogeneous_b @ fundamental
    numerator = np.einsum("ij,ij->i", homogeneous_b, line_b) ** 2
    denominator = line_b[:, 0] ** 2 + line_b[:, 1] ** 2 + line_a[:, 0] ** 2 + line_a[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.where(denominator > 0, numerator / denominator, np.inf))


def triangulate(
    points_a: np.ndarray,
    points_b: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Depth of each correspondence along camera A's optical axis, in world units."""
    projection_a = intrinsics @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = intrinsics @ np.hstack([rotation, translation.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(projection_a, projection_b, points_a.T, points_b.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        points = homogeneous[:3] / homogeneous[3]
    return points[2]


def track_points(
    frame_a: np.ndarray, frame_b: np.ndarray, max_points: int = 600
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-backward checked Lucas-Kanade correspondences between two frames."""
    corners = cv2.goodFeaturesToTrack(
        frame_a, maxCorners=max_points, qualityLevel=0.01, minDistance=12, blockSize=7
    )
    if corners is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        frame_a, frame_b, corners, None, winSize=(21, 21), maxLevel=4, criteria=criteria
    )
    backward, _, _ = cv2.calcOpticalFlowPyrLK(
        frame_b, frame_a, forward, None, winSize=(21, 21), maxLevel=4, criteria=criteria
    )

    # A point that does not track back to where it started was matched onto an
    # occluder or a repeated texture; those are the correspondences that would
    # otherwise dominate the epipolar error.
    round_trip = np.linalg.norm(corners - backward, axis=2).ravel()
    keep = (status.ravel() == 1) & (round_trip < 1.0)
    return corners[keep, 0], forward[keep, 0]


def evaluate_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    track: CameraTrack,
    index_a: int,
    index_b: int,
    scene_scale: float,
    inlier_threshold: float = 2.0,
) -> PairEvidence:
    rotation, translation = relative_pose(track, index_a, index_b)
    baseline = float(np.linalg.norm(translation))

    points_a, points_b = track_points(frame_a, frame_b)
    evidence = PairEvidence(index_a, index_b, baseline, len(points_a))
    if len(points_a) < MIN_TRACKED_POINTS or baseline < MIN_BASELINE_FRACTION * scene_scale:
        return evidence

    intrinsics = track.intrinsics_for(index_a)
    fundamental = fundamental_from_pose(intrinsics, rotation, translation)
    errors = sampson_distance(points_a, points_b, fundamental)
    finite = np.isfinite(errors)

    evidence.sampson_median = float(np.median(errors[finite]))
    evidence.sampson_p90 = float(np.percentile(errors[finite], 90))
    evidence.inlier_fraction = float(np.mean(errors[finite] < inlier_threshold))

    depths = triangulate(points_a, points_b, intrinsics, rotation, translation)
    # Points behind the camera or at absurd range are triangulation failures on
    # near-parallel rays; they carry no scale information.
    usable = finite & (errors < inlier_threshold) & (depths > 0) & (depths < 1e4)
    evidence.depths = depths[usable]
    evidence.pixels = points_a[usable]
    return evidence


# `classify` below answers "do these poses belong to this video at all", and so
# tolerates a couple of pixels of sparse-reconstruction noise. Choosing which
# low-poly renders may enter training is a different and stricter question -
# how faithfully the re-render preserved the original geometry - so it gets its
# own cut. Keeping them apart stops a clip that merely proves the poses are
# right from reading as a clip that is good enough to train on.
FIDELITY_KEEP_PX = 1.0
FIDELITY_DROP_PX = 3.0


def fidelity_tier(sampson_median: float) -> str:
    """How faithful a re-render's geometry is, given the poses are known good."""
    if not np.isfinite(sampson_median):
        return "unscored"
    if sampson_median <= FIDELITY_KEEP_PX:
        return "keep"
    if sampson_median <= FIDELITY_DROP_PX:
        return "review"
    return "drop"


def classify(sampson_median: float, inlier_fraction: float) -> tuple[str, str]:
    """Turn the two headline numbers into a decision.

    Two pixels at 1280x720 is roughly the spread a good sparse reconstruction
    leaves behind; ten pixels is more than lens or timing slop can explain and
    means the poses do not belong to these frames.
    """
    if not np.isfinite(sampson_median):
        return "unscored", "no frame pair had both enough baseline and enough tracked points"
    if sampson_median < 2.0 and inlier_fraction > 0.6:
        return "match", "poses reproject onto the video within sparse-reconstruction noise"
    if sampson_median < 10.0:
        return "loose", "poses broadly follow the video but with more residual than tracking noise explains"
    return "mismatch", "poses do not describe these frames"


def verify_track(
    frames: list[np.ndarray], track: CameraTrack, clip: str = "", gap: int = DEFAULT_GAP, stride: int = 8
) -> TrackVerdict:
    """Check a whole track against decoded grayscale frames.

    `frames` must be the full decoded clip in order; pairs are sampled from it
    so that a localised failure (a bad splice, say) still shows up in the
    per-pair list even when the median looks healthy.
    """
    if len(frames) != len(track):
        raise ValueError(f"{clip}: {len(frames)} frames but {len(track)} poses")

    scene_scale = float(np.linalg.norm(track.positions.max(axis=0) - track.positions.min(axis=0)))
    pairs = [
        evaluate_pair(frames[i], frames[i + gap], track, i, i + gap, scene_scale)
        for i in range(0, len(frames) - gap, stride)
    ]

    scored = [pair for pair in pairs if not pair.degenerate]
    if scored:
        sampson = float(np.median([pair.sampson_median for pair in scored]))
        inliers = float(np.median([pair.inlier_fraction for pair in scored]))
    else:
        sampson, inliers = float("nan"), float("nan")

    tier, note = classify(sampson, inliers)
    return TrackVerdict(clip, pairs, sampson, inliers, tier, note)
