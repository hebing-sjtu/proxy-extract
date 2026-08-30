"""Split the `person` class into `hero` and `npc`.

No segmenter can do this: a protagonist and a bystander are the same pixels
under any category scheme. What distinguishes them is not appearance but the
camera's relationship to them — a third-person camera is bolted to the
protagonist, so their silhouette stays put in image space and never leaves
frame, while everyone else drifts through. This turns the label problem into a
tracking problem and scores tracks on that behaviour.

Two limits worth stating up front, because both are real in this footage:

  - Semantic masks merge people who touch. When an npc walks in front of the
    hero they become one blob and no rule over that blob can be right. Such
    frames are reported, not guessed at.
  - On measured sample footage roughly four fifths of clips never show a second
    person at all, so most of the time this stage has nothing to decide. It
    earns its place on the remaining fifth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..taxonomy import C6_HERO, C6_NPC

# Below this a blob at condition resolution is a distant pedestrian, too small
# to matter and too noisy to track.
MIN_AREA_FRACTION = 0.0008
# One body-width of travel between sampled frames. Further than that and it is
# a different person, not the same one having moved.
MAX_LINK_DISTANCE_FRACTION = 0.10
MAX_FRAME_GAP = 2
MIN_TRACK_LENGTH = 3
# Past this much blob merging the hero mask is contaminated often enough that
# promoting it would teach the dataset that npcs are part of the protagonist.
# Necessarily below 0.5: the detector calls a frame merged by comparing it to
# the track's median area, so once merging is the majority it becomes the
# baseline and stops registering. That ceiling is the same blind spot described
# in `_merged_fraction`, seen from the threshold side.
MAX_MERGED_FRACTION = 0.25


@dataclass
class PersonTrack:
    frames: list[int] = field(default_factory=list)
    centres: list[tuple[float, float]] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def persistence(self) -> float:
        return len(self.frames)

    @property
    def median_area(self) -> float:
        return float(np.median(self.areas))

    @property
    def wander(self) -> float:
        """Mean per-axis standard deviation of the centroid, in pixels.

        The discriminating statistic: a camera-followed protagonist barely
        moves in image space even while the world sweeps past behind them.
        """
        if len(self.centres) < 2:
            return float("inf")
        return float(np.mean(np.std(np.array(self.centres), axis=0)))


@dataclass
class HeroSplit:
    labels: np.ndarray
    hero_track: PersonTrack | None
    tracks: list[PersonTrack]
    merged_frames: float
    multi_person_frames: float
    note: str


def _components(mask: np.ndarray, min_area: float):
    import cv2

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    for component in range(1, count):
        area = float(stats[component, cv2.CC_STAT_AREA])
        if area >= min_area:
            yield component, area, centroids[component], stats[component]


def build_tracks(person_masks: np.ndarray) -> list[PersonTrack]:
    """Link person blobs across frames by centroid proximity."""
    height, width = person_masks.shape[1:]
    min_area = MIN_AREA_FRACTION * height * width
    max_distance = MAX_LINK_DISTANCE_FRACTION * max(height, width)

    tracks: list[PersonTrack] = []
    for index, mask in enumerate(person_masks):
        for _, area, centre, _ in _components(mask, min_area):
            candidates = [
                (float(np.linalg.norm(np.array(t.centres[-1]) - centre)), t)
                for t in tracks
                if t.frames[-1] != index and index - t.frames[-1] <= MAX_FRAME_GAP
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


def pick_hero(tracks: list[PersonTrack], total_frames: int) -> tuple[PersonTrack | None, str]:
    """Choose the track the camera is following, or decline to.

    Declining matters more than choosing. Promoting the wrong track mislabels
    the one class the dataset cares most about, and a wrong hero is worse than
    no hero, because downstream cannot tell it was a guess.
    """
    usable = [t for t in tracks if len(t) >= MIN_TRACK_LENGTH]
    if not usable:
        return None, "no person track survived the minimum length"
    if len(usable) == 1:
        track = usable[0]
        if track.persistence < 0.5 * total_frames:
            return None, "the only person track covers less than half the clip"
        return track, "single persistent person track"

    # Persistent, large and still. Wander is the discriminating term; the other
    # two keep a briefly-glimpsed distant figure from winning on stillness.
    def score(track: PersonTrack) -> float:
        return (track.persistence / total_frames) * np.sqrt(track.median_area) / (1.0 + track.wander)

    ranked = sorted(usable, key=score, reverse=True)
    best, runner_up = ranked[0], ranked[1]
    if score(best) < 1.5 * score(runner_up):
        return None, (
            f"top two person tracks score within 1.5x "
            f"({score(best):.3f} vs {score(runner_up):.3f}); which is the protagonist is ambiguous"
        )
    return best, f"{len(usable)} person tracks, one clearly camera-followed"


def split(labels: np.ndarray) -> HeroSplit:
    """Relabel the hero's pixels from `npc` to `hero` across a clip.

    Input and output both use the coarse 6-class set, where every person
    arrives as `npc`.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    person_masks = labels == C6_NPC
    tracks = build_tracks(person_masks)
    hero, note = pick_hero(tracks, len(labels))

    # How often two people are one blob, which is where this approach is blind
    # rather than merely uncertain. Checked after picking so the reason for
    # declining is recorded, and checked at all because a track built from
    # merged blobs still looks perfectly well-behaved from the outside.
    height, width = labels.shape[1:]
    min_area = MIN_AREA_FRACTION * height * width
    occupied = [sum(1 for _ in _components(mask, min_area)) for mask in person_masks]
    multi_person = float(np.mean([count > 1 for count in occupied])) if occupied else 0.0
    merged = _merged_fraction(hero)

    if hero is not None and merged > MAX_MERGED_FRACTION:
        hero, note = None, (
            f"people are merged into one blob in {merged * 100:.0f}% of frames; "
            "the tracks are unions of people, so no hero can be attributed"
        )

    out = labels.copy()
    if hero is not None:
        import cv2

        for slot, index in enumerate(hero.frames):
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                person_masks[index].astype(np.uint8), connectivity=8
            )
            centre = np.array(hero.centres[slot])
            distances = [
                (np.linalg.norm(np.array(_centroid(stats, c)) - centre), c) for c in range(1, count)
            ]
            if distances:
                _, chosen = min(distances)
                out[index][components == chosen] = C6_HERO

    return HeroSplit(
        labels=out,
        hero_track=hero,
        tracks=tracks,
        merged_frames=merged,
        multi_person_frames=multi_person,
        note=note,
    )


MERGE_AREA_RATIO = 1.6


def _merged_fraction(track: PersonTrack | None) -> float:
    """Fraction of a track's frames where its blob swelled to hold someone else.

    Detected by area rather than by counting blobs, because a merge removes a
    blob rather than adding one — the two people become one region and the
    second track simply stops, which looks identical to that person leaving.
    What it cannot look like is the region staying the same size.

    This sees transient merges, which is what a passer-by produces. It is blind
    to two people who are fused for the entire clip: with no unfused frames to
    set the baseline, the doubled area *is* the median. Separating those needs
    instance segmentation, and `test_permanent_fusion_is_a_known_blind_spot`
    pins the limitation so it cannot be forgotten.
    """
    if track is None or len(track) < MIN_TRACK_LENGTH:
        return 0.0
    areas = np.array(track.areas)
    return float(np.mean(areas > MERGE_AREA_RATIO * np.median(areas)))


def _centroid(stats, component: int) -> tuple[float, float]:
    import cv2

    x = stats[component, cv2.CC_STAT_LEFT] + stats[component, cv2.CC_STAT_WIDTH] / 2
    y = stats[component, cv2.CC_STAT_TOP] + stats[component, cv2.CC_STAT_HEIGHT] / 2
    return float(x), float(y)
