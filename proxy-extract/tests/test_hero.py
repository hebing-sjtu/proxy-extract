"""Hero/npc separation. The behaviour that matters most is declining to guess."""

from __future__ import annotations

import numpy as np
import pytest

from proxy_extract.semantic import hero
from proxy_extract.taxonomy import C6_BACKGROUND, C6_HERO, C6_NPC

HEIGHT, WIDTH = 192, 336
FRAMES = 40


def blank() -> np.ndarray:
    return np.full((FRAMES, HEIGHT, WIDTH), C6_BACKGROUND, dtype=np.uint8)


def paint(labels: np.ndarray, index: int, centre: tuple[float, float], size: int = 22) -> None:
    x, y = int(centre[0]), int(centre[1])
    half = size // 2
    top, bottom = max(0, y - size), min(HEIGHT, y + size)
    left, right = max(0, x - half), min(WIDTH, x + half)
    labels[index, top:bottom, left:right] = C6_NPC


def still_person(labels: np.ndarray, centre=(168, 110)) -> None:
    for index in range(FRAMES):
        paint(labels, index, centre)


def walking_person(labels: np.ndarray, start=40, speed=5.0, y=100, size=22) -> None:
    for index in range(FRAMES):
        paint(labels, index, (start + speed * index, y), size)


class TestTracking:
    def test_a_stationary_person_forms_one_track(self):
        labels = blank()
        still_person(labels)
        tracks = hero.build_tracks(labels == C6_NPC)
        assert len(tracks) == 1
        assert len(tracks[0]) == FRAMES

    def test_two_separated_people_form_two_tracks(self):
        labels = blank()
        still_person(labels, centre=(80, 110))
        still_person(labels, centre=(260, 110))
        tracks = hero.build_tracks(labels == C6_NPC)
        assert len(tracks) == 2

    def test_a_stationary_person_wanders_less_than_a_walking_one(self):
        still, walking = blank(), blank()
        still_person(still)
        walking_person(walking)
        assert hero.build_tracks(still == C6_NPC)[0].wander < 1.0
        assert hero.build_tracks(walking == C6_NPC)[0].wander > 20.0

    def test_blobs_below_the_area_floor_are_ignored(self):
        labels = blank()
        for index in range(FRAMES):
            labels[index, 10:12, 10:12] = C6_NPC
        assert hero.build_tracks(labels == C6_NPC) == []


class TestPickHero:
    def test_the_camera_followed_track_wins_over_a_passer_by(self):
        labels = blank()
        still_person(labels, centre=(168, 60))
        walking_person(labels, start=20, speed=6.0, y=150)

        result = hero.split(labels)
        assert result.hero_track is not None
        assert result.hero_track.wander < 1.0
        assert (result.labels == C6_HERO).any()

    def test_the_hero_pixels_are_the_still_ones(self):
        labels = blank()
        still_person(labels, centre=(168, 60))
        walking_person(labels, start=20, speed=6.0, y=150)

        result = hero.split(labels)
        hero_rows = np.where((result.labels[0] == C6_HERO).any(axis=1))[0]
        assert hero_rows.mean() < 100

    def test_a_lone_person_needs_no_disambiguation(self):
        labels = blank()
        still_person(labels)
        result = hero.split(labels)
        assert result.hero_track is not None
        assert "single persistent" in result.note

    def test_an_empty_clip_yields_no_hero(self):
        result = hero.split(blank())
        assert result.hero_track is None
        assert not (result.labels == C6_HERO).any()

    def test_two_equally_still_people_are_declined_rather_than_guessed(self):
        """A wrong hero is worse than no hero: downstream cannot tell it was a guess."""
        labels = blank()
        still_person(labels, centre=(110, 110))
        still_person(labels, centre=(230, 110))

        result = hero.split(labels)
        assert result.hero_track is None
        assert "ambiguous" in result.note
        assert not (result.labels == C6_HERO).any()

    def test_a_person_glimpsed_briefly_is_not_promoted(self):
        labels = blank()
        for index in range(3):
            paint(labels, index, (168, 110))
        result = hero.split(labels)
        assert result.hero_track is None
        assert "less than half" in result.note


class TestReportedLimits:
    def test_a_brief_crossing_is_reported_but_still_resolved(self):
        labels = blank()
        still_person(labels, centre=(300, 110))
        # Someone crosses through the protagonist for part of the clip.
        walking_person(labels, start=10, speed=8.0, y=110)

        result = hero.split(labels)
        assert 0.0 < result.merged_frames <= hero.MAX_MERGED_FRACTION
        assert result.hero_track is not None

    def test_the_merge_detector_reads_area_spikes(self):
        track = hero.PersonTrack(
            frames=list(range(10)),
            centres=[(100.0, 100.0)] * 10,
            areas=[1000.0] * 7 + [2000.0] * 3,
        )
        assert hero._merged_fraction(track) == pytest.approx(0.3)

    def test_pervasive_merging_makes_the_split_decline(self):
        labels = blank()
        still_person(labels, centre=(160, 110))
        # Someone stands touching the protagonist for the first 30% of the
        # clip, then leaves: enough clean frames to set the baseline, enough
        # fused ones that promoting the blob would label them as hero too.
        for index in range(12):
            paint(labels, index, (180, 110))

        result = hero.split(labels)
        assert result.merged_frames > hero.MAX_MERGED_FRACTION
        assert result.hero_track is None
        assert "merged into one blob" in result.note
        assert not (result.labels == C6_HERO).any()

    def test_permanent_fusion_is_a_known_blind_spot(self):
        """Two people fused for the whole clip read as one person, undetectably.

        With no unfused frame to set a baseline, the doubled blob *is* the
        median area, so nothing looks anomalous. This is pinned rather than
        fixed: escaping it needs instance segmentation, not a better rule over
        semantic masks. If a future change makes this detectable, this test
        should fail and be replaced.
        """
        labels = blank()
        for index in range(FRAMES):
            paint(labels, index, (160, 110), size=22)
            paint(labels, index, (176, 110), size=22)

        result = hero.split(labels)
        assert result.merged_frames == 0.0
        assert len(result.tracks) == 1
        assert result.hero_track is not None

    def test_a_clip_with_one_person_reports_no_merging(self):
        labels = blank()
        still_person(labels)
        assert hero.split(labels).merged_frames == 0.0

    def test_multi_person_frames_and_merged_frames_measure_different_things(self):
        """Two people apart raise one counter, two people fused raise the other."""
        apart = blank()
        still_person(apart, centre=(80, 60))
        still_person(apart, centre=(260, 150))
        result = hero.split(apart)
        assert result.multi_person_frames == 1.0
        assert result.merged_frames == 0.0

    def test_the_split_only_ever_relabels_person_pixels(self):
        labels = blank()
        still_person(labels)
        labels[:, :20, :] = C6_BACKGROUND
        result = hero.split(labels)

        changed = result.labels != labels
        assert np.all(labels[changed] == C6_NPC)
        assert np.all(result.labels[changed] == C6_HERO)

    @pytest.mark.parametrize("size", [22, 40])
    def test_output_stays_inside_the_six_class_range(self, size):
        labels = blank()
        walking_person(labels, size=size)
        assert hero.split(labels).labels.max() <= C6_HERO
