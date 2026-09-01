"""Split the `person` class into `player` and `ped` for the 11-class standard.

No segmenter can do this — a protagonist and a bystander are the same pixels
under any category scheme. What separates them is where the camera puts them.
A third-person camera is rigged to the player, so the player projects near a
fixed screen anchor and stays there, while everyone else drifts through frame.

This is the same problem `hero.py` solves for coarse6, but on a different
signal. `hero.py` ranks tracks by *stillness* in image space. Stillness and
centredness agree while the character walks and disagree during an orbiting
`look` shot, where the camera swings around a stationary player: the player
stays centred but their pixels sweep across frame, so a stillness score ranks
them below a genuinely static bystander. Centredness is also the cheaper claim
to defend — it follows from how the camera is mounted, not from how the subject
happens to move.

The anchor and the thresholds below are deliberately parameters rather than
constants baked into the scoring. `eval/player_bench.py` fits them against the
engine's own player/ped labels on the gta-web corpus; hard-coding a guess here
and calling it a prior would be asserting the thing that needs measuring.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..taxonomy import S11_PED, S11_PLAYER, S11_VEHICLE


@dataclass(frozen=True)
class PlayerPrior:
    """Where the protagonist is expected to sit, and how sure we must be.

    Distances are fractions of image width so the same numbers apply at capture
    resolution and at the 336x192 condition grid.
    """

    # Normalised (x, y) of the screen anchor. Third-person rigs frame the
    # character low-centre — the camera looks over their head at what is ahead
    # — so the default sits below the geometric centre. Fitted, not assumed.
    anchor: tuple[float, float] = (0.5, 0.55)
    # A blob whose median centroid is further than this from the anchor is not
    # a plausible protagonist, however persistent it is.
    max_anchor_distance: float = 0.22
    # Below this a blob at condition resolution is a distant pedestrian.
    min_area_fraction: float = 0.0008
    # One body-width of travel between frames; beyond that it is someone else.
    max_link_distance: float = 0.10
    max_frame_gap: int = 2
    min_track_length: int = 3
    # The winner must lead by this factor, or the split is declined. A wrong
    # player is worse than no player: downstream cannot tell it was a guess.
    min_score_margin: float = 1.5
    # A player visible in fewer than this fraction of frames is being occluded
    # or is inside a vehicle. Either way the attribution stops being safe.
    min_presence: float = 0.5
    # Past this much blob merging the player mask is contaminated often enough
    # that promoting it would teach the dataset that peds are part of the
    # protagonist. See `_merged_fraction` for why it cannot exceed 0.5.
    max_merged_fraction: float = 0.25
    # Fraction of the anchor neighbourhood that must be vehicle before we
    # conclude the protagonist is driving and therefore not separately visible.
    driving_vehicle_cover: float = 0.5


@dataclass
class PersonTrack:
    frames: list[int] = field(default_factory=list)
    centres: list[tuple[float, float]] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def median_area(self) -> float:
        return float(np.median(self.areas))


@dataclass
class PlayerSplit:
    labels: np.ndarray
    player_track: PersonTrack | None
    tracks: list[PersonTrack]
    merged_frames: float
    multi_person_frames: float
    driving: bool
    note: str

    @property
    def resolved(self) -> bool:
        return self.player_track is not None


def _components(mask: np.ndarray, min_area: float):
    import cv2

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    for component in range(1, count):
        area = float(stats[component, cv2.CC_STAT_AREA])
        if area >= min_area:
            yield component, area, centroids[component]


def build_tracks(person_masks: np.ndarray, prior: PlayerPrior) -> list[PersonTrack]:
    """Link person blobs across frames by centroid proximity."""
    height, width = person_masks.shape[1:]
    min_area = prior.min_area_fraction * height * width
    max_distance = prior.max_link_distance * max(height, width)

    tracks: list[PersonTrack] = []
    for index, mask in enumerate(person_masks):
        for _, area, centre in _components(mask, min_area):
            candidates = [
                (float(np.linalg.norm(np.array(t.centres[-1]) - centre)), t)
                for t in tracks
                if t.frames[-1] != index and index - t.frames[-1] <= prior.max_frame_gap
            ]
            best = min(candidates, default=None, key=lambda pair: pair[0])

            if best is not None and best[0] < max_distance:
                track = best[1]
            else:
                track = PersonTrack()
                tracks.append(track)
            track.frames.append(index)
            track.centres.append((float(centre[0]), float(centre[1])))
            track.areas.append(area)
    return tracks


def anchor_distance(track: PersonTrack, shape: tuple[int, int], prior: PlayerPrior) -> float:
    """Median distance from the track's centroid to the screen anchor.

    Normalised by image width, so the threshold is resolution-independent.
    Median rather than mean because a single frame where the blob merges with a
    passer-by drags the centroid a long way and should not decide the clip.
    """
    height, width = shape
    ax, ay = prior.anchor[0] * width, prior.anchor[1] * height
    offsets = np.array(track.centres) - np.array([ax, ay])
    return float(np.median(np.linalg.norm(offsets, axis=1)) / width)


def score_track(
    track: PersonTrack, shape: tuple[int, int], total_frames: int, prior: PlayerPrior
) -> float:
    """Rank a track as a protagonist candidate.

    Three terms, multiplied so that any one of them being near zero disqualifies
    the track outright:

    - presence, because the player never leaves for long;
    - centredness, the discriminating term;
    - size, which keeps a distant figure that happens to sit on the anchor from
      beating the character in the foreground.
    """
    height, width = shape
    presence = len(track) / max(total_frames, 1)
    centredness = 1.0 / (1.0 + anchor_distance(track, shape, prior) / max(prior.max_anchor_distance, 1e-6))
    size = np.sqrt(track.median_area / (height * width))
    return float(presence * centredness * size)


MERGE_AREA_RATIO = 1.6


def _merged_fraction(track: PersonTrack | None, prior: PlayerPrior) -> float:
    """Fraction of a track's frames where its blob swelled to hold someone else.

    Detected by area rather than by counting blobs, because a merge removes a
    blob rather than adding one: the two people become one region and the second
    track simply stops, which is indistinguishable from that person walking out
    of frame. What it cannot look like is the region staying the same size.

    Blind to two people fused for the whole clip — with no unfused frames to set
    the baseline, the doubled area *is* the median. That is also why
    `max_merged_fraction` must stay below 0.5.
    """
    if track is None or len(track) < prior.min_track_length:
        return 0.0
    areas = np.array(track.areas)
    return float(np.mean(areas > MERGE_AREA_RATIO * np.median(areas)))


def _is_driving(labels: np.ndarray, prior: PlayerPrior) -> bool:
    """Is the anchor region mostly vehicle?

    In a driving segment the protagonist is inside the car and contributes few
    or no person pixels, and the standard gives the ego vehicle the same id as
    traffic, so there is nothing to promote. Detecting it lets the split decline
    for a stated reason instead of picking whichever pedestrian wandered past.
    """
    frames, height, width = labels.shape
    half = int(0.15 * width)
    cx, cy = int(prior.anchor[0] * width), int(prior.anchor[1] * height)
    y0, y1 = max(cy - half, 0), min(cy + half, height)
    x0, x1 = max(cx - half, 0), min(cx + half, width)
    patch = labels[:, y0:y1, x0:x1]
    if patch.size == 0:
        return False
    return bool(np.mean(patch == S11_VEHICLE) > prior.driving_vehicle_cover)


def pick_player(
    tracks: list[PersonTrack],
    shape: tuple[int, int],
    total_frames: int,
    prior: PlayerPrior,
) -> tuple[PersonTrack | None, str]:
    """Choose the track the camera is rigged to, or decline to.

    Declining is the point. This is the one class the dataset cares most about,
    and an unmarked guess is worse than an admitted gap.
    """
    usable = [t for t in tracks if len(t) >= prior.min_track_length]
    if not usable:
        return None, "no person track survived the minimum length"

    ranked = sorted(usable, key=lambda t: score_track(t, shape, total_frames, prior), reverse=True)
    best = ranked[0]

    distance = anchor_distance(best, shape, prior)
    if distance > prior.max_anchor_distance:
        return None, (
            f"the best candidate sits {distance:.2f} of a frame width from the anchor "
            f"(limit {prior.max_anchor_distance:.2f}); nobody is where the protagonist should be"
        )
    if len(best) < prior.min_presence * total_frames:
        return None, (
            f"the best candidate appears in {len(best) / max(total_frames, 1):.0%} of frames, "
            f"below the {prior.min_presence:.0%} a camera-rigged character should hold"
        )

    if len(ranked) > 1:
        top = score_track(best, shape, total_frames, prior)
        runner_up = score_track(ranked[1], shape, total_frames, prior)
        if top < prior.min_score_margin * runner_up:
            return None, (
                f"top two person tracks score within {prior.min_score_margin}x "
                f"({top:.4f} vs {runner_up:.4f}); which is the protagonist is ambiguous"
            )

    return best, (
        f"{len(usable)} person track(s), one anchored at {distance:.2f} frame widths from centre"
    )


def split(labels: np.ndarray, prior: PlayerPrior | None = None) -> PlayerSplit:
    """Relabel the protagonist's pixels from `ped` to `player` across a clip.

    Input and output both use the 11-class standard, where every person arrives
    as `ped`.
    """
    prior = prior or PlayerPrior()
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.ndim != 3:
        raise ValueError(f"expected a (frames, height, width) label stack, got {labels.shape}")

    shape = (labels.shape[1], labels.shape[2])
    person_masks = labels == S11_PED
    tracks = build_tracks(person_masks, prior)
    driving = _is_driving(labels, prior)

    if driving:
        player, note = None, (
            "the frame anchor is covered by vehicle: this is a driving segment, "
            "the protagonist is inside the car and the standard gives ego and traffic the same id"
        )
    else:
        player, note = pick_player(tracks, shape, len(labels), prior)

    min_area = prior.min_area_fraction * shape[0] * shape[1]
    occupied = [sum(1 for _ in _components(mask, min_area)) for mask in person_masks]
    multi_person = float(np.mean([count > 1 for count in occupied])) if occupied else 0.0
    merged = _merged_fraction(player, prior)

    if player is not None and merged > prior.max_merged_fraction:
        player, note = None, (
            f"people are merged into one blob in {merged * 100:.0f}% of frames; "
            "the tracks are unions of people, so no protagonist can be attributed"
        )

    out = labels.copy()
    if player is not None:
        import cv2

        for slot, index in enumerate(player.frames):
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                person_masks[index].astype(np.uint8), connectivity=8
            )
            centre = np.array(player.centres[slot])
            distances = [
                (float(np.linalg.norm(np.array(_centroid(stats, c)) - centre)), c)
                for c in range(1, count)
            ]
            if distances:
                _, chosen = min(distances)
                out[index][components == chosen] = S11_PLAYER

    return PlayerSplit(
        labels=out,
        player_track=player,
        tracks=tracks,
        merged_frames=merged,
        multi_person_frames=multi_person,
        driving=driving,
        note=note,
    )


def with_anchor(prior: PlayerPrior, anchor: tuple[float, float]) -> PlayerPrior:
    """A copy of `prior` with a different screen anchor, for sweeps."""
    return replace(prior, anchor=anchor)


def _centroid(stats, component: int) -> tuple[float, float]:
    import cv2

    x = stats[component, cv2.CC_STAT_LEFT] + stats[component, cv2.CC_STAT_WIDTH] / 2
    y = stats[component, cv2.CC_STAT_TOP] + stats[component, cv2.CC_STAT_HEIGHT] / 2
    return float(x), float(y)
