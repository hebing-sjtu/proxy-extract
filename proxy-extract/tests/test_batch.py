"""Sharding, resume and the coarse taxonomy — the pieces batch runs depend on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from proxy_extract import contract, pipeline, taxonomy
from proxy_extract.cli import resolve_videos


class TestResolveVideos:
    @pytest.fixture
    def clip_dir(self, tmp_path) -> Path:
        directory = tmp_path / "clips"
        directory.mkdir()
        for name in ("b.mp4", "a.mp4", "c.MP4", "notes.txt", "cams.json"):
            (directory / name).touch()
        return directory

    def test_a_directory_expands_to_its_clips(self, clip_dir):
        assert [p.name for p in resolve_videos([clip_dir])] == ["a.mp4", "b.mp4", "c.MP4"]

    def test_non_video_files_are_left_out(self, clip_dir):
        assert not any(p.suffix == ".txt" for p in resolve_videos([clip_dir]))

    def test_the_order_is_stable_across_calls(self, clip_dir):
        # Every shard derives its slice by position, so two workers disagreeing
        # on the order would double-process some clips and drop others.
        assert resolve_videos([clip_dir]) == resolve_videos([clip_dir])

    def test_explicit_files_pass_through_unchanged(self, clip_dir):
        chosen = [clip_dir / "b.mp4", clip_dir / "a.mp4"]
        assert resolve_videos(chosen) == chosen

    def test_files_and_directories_can_be_mixed(self, clip_dir, tmp_path):
        extra = tmp_path / "extra.mp4"
        extra.touch()
        assert len(resolve_videos([clip_dir, extra])) == 4

    def test_a_missing_path_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="no such video"):
            resolve_videos([tmp_path / "ghost.mp4"])

    def test_an_empty_directory_fails_loudly(self, tmp_path):
        # Silently extracting nothing would look like a finished run.
        (tmp_path / "empty").mkdir()
        with pytest.raises(SystemExit, match="no .* files in"):
            resolve_videos([tmp_path / "empty"])


class TestShard:
    @pytest.fixture
    def clips(self) -> list[Path]:
        return [Path(f"clip_{i:03d}.mp4") for i in range(10)]

    def test_every_clip_lands_in_exactly_one_shard(self, clips):
        collected = [clip for index in range(4) for clip in pipeline.shard(clips, index, 4)]
        assert sorted(collected) == sorted(clips)
        assert len(collected) == len(set(collected))

    def test_shards_differ_by_at_most_one_clip(self, clips):
        sizes = [len(pipeline.shard(clips, index, 4)) for index in range(4)]
        assert max(sizes) - min(sizes) <= 1

    def test_striding_survives_a_cost_gradient(self, clips):
        # If cost rose along the list and shards were contiguous, the last
        # worker would carry the whole tail. Strided, each worker gets a spread.
        for index in range(4):
            positions = [clips.index(c) for c in pipeline.shard(clips, index, 4)]
            assert min(positions) < 4 and max(positions) > 5

    def test_a_single_worker_owns_everything(self, clips):
        assert pipeline.shard(clips, 0, 1) == clips

    @pytest.mark.parametrize("index,count", [(4, 4), (-1, 4), (0, 0)])
    def test_an_out_of_range_shard_is_refused(self, clips, index, count):
        with pytest.raises(ValueError, match="out of range"):
            pipeline.shard(clips, index, count)


class TestResume:
    def write_condition(self, root: Path, frames: int) -> None:
        rng = np.random.default_rng(0)
        for ordinal in range(frames):
            depth = rng.uniform(2.0, 40.0, (contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH))
            labels = np.zeros((contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH), np.uint8)
            contract.write_frame(root, ordinal, depth.astype(np.float32), labels)

    def test_a_complete_root_is_recognised(self, tmp_path):
        clip = Path("01/clip.mp4")
        root = pipeline.condition_dir_for(tmp_path, clip)
        self.write_condition(root, contract.WINDOW_FRAMES)
        (root / "extraction_report.json").write_text("{}")

        assert pipeline.already_done(tmp_path, clip) is True

    def test_an_untouched_output_is_not_done(self, tmp_path):
        assert pipeline.already_done(tmp_path, Path("01/clip.mp4")) is False

    def test_a_root_without_its_report_is_redone(self, tmp_path):
        """The report is written last, so its absence means the run was cut short."""
        clip = Path("01/clip.mp4")
        self.write_condition(pipeline.condition_dir_for(tmp_path, clip), contract.WINDOW_FRAMES)

        assert pipeline.already_done(tmp_path, clip) is False

    def test_a_truncated_root_is_redone_rather_than_trusted(self, tmp_path):
        clip = Path("01/clip.mp4")
        root = pipeline.condition_dir_for(tmp_path, clip)
        self.write_condition(root, contract.WINDOW_FRAMES)
        (root / "extraction_report.json").write_text("{}")
        next(root.glob("*.depth.f32")).unlink()

        assert pipeline.already_done(tmp_path, clip) is False

    def test_a_corrupt_file_is_redone_rather_than_trusted(self, tmp_path):
        clip = Path("01/clip.mp4")
        root = pipeline.condition_dir_for(tmp_path, clip)
        self.write_condition(root, contract.WINDOW_FRAMES)
        (root / "extraction_report.json").write_text("{}")
        next(root.glob("*.depth.f32")).write_bytes(b"truncated")

        assert pipeline.already_done(tmp_path, clip) is False

    def test_same_named_clips_resume_independently(self, tmp_path):
        first, second = Path("00/clip.mp4"), Path("01/clip.mp4")
        root = pipeline.condition_dir_for(tmp_path, first)
        self.write_condition(root, contract.WINDOW_FRAMES)
        (root / "extraction_report.json").write_text("{}")

        assert pipeline.already_done(tmp_path, first) is True
        assert pipeline.already_done(tmp_path, second) is False


class TestCoarseTaxonomy:
    def test_background_is_the_lowest_priority_id(self):
        # Importance ranking puts background third; paint order must not. If
        # background ever outranked another class it would erase it wholesale.
        assert taxonomy.C6_BACKGROUND == 0
        assert min(taxonomy.COARSE6_PRIORITY.values()) == taxonomy.COARSE6_PRIORITY[taxonomy.C6_BACKGROUND]
        assert max(taxonomy.COARSE6_PRIORITY.values()) == taxonomy.COARSE6_PRIORITY[taxonomy.C6_HERO]

    def test_people_outrank_the_surfaces_they_stand_on(self):
        for surface in (taxonomy.C6_ROAD, taxonomy.C6_VEGETATION, taxonomy.C6_BACKGROUND):
            assert taxonomy.COARSE6_PRIORITY[taxonomy.C6_NPC] > taxonomy.COARSE6_PRIORITY[surface]
            assert taxonomy.COARSE6_PRIORITY[taxonomy.C6_HERO] > taxonomy.COARSE6_PRIORITY[surface]

    def test_the_ade20k_map_covers_every_coarse_class_a_segmenter_can_emit(self):
        emitted = set(taxonomy.ADE20K_TO_COARSE6.values())
        # hero is not in there by design: no segmenter predicts it.
        assert emitted == {
            taxonomy.C6_ROAD,
            taxonomy.C6_VEGETATION,
            taxonomy.C6_VEHICLE,
            taxonomy.C6_NPC,
        }
        assert taxonomy.C6_HERO not in emitted

    def test_grass_is_vegetation_here_but_terrain_in_the_12_class_set(self):
        """The two taxonomies genuinely disagree, and it is not an oversight.

        With no terrain class to fall into, grass has to go somewhere, and it
        looks like the vegetation it is.
        """
        assert taxonomy.ADE20K_TO_COARSE6["grass"] == taxonomy.C6_VEGETATION
        assert taxonomy.ADE20K_TO_CWM["grass"] == taxonomy.TERRAIN

    def test_the_lut_builds_and_covers_all_of_ade20k(self):
        lut = taxonomy.coarse6_lut()
        assert lut.shape == (len(taxonomy.ADE20K_CLASSES),)
        assert lut.max() < taxonomy.NUM_COARSE6
        assert lut[taxonomy.ADE20K_CLASSES.index("person")] == taxonomy.C6_NPC
        assert lut[taxonomy.ADE20K_CLASSES.index("wall")] == taxonomy.C6_BACKGROUND

    def test_collapsing_the_12_class_set_keeps_the_classes_that_matter(self):
        source = np.array([[taxonomy.SKY, taxonomy.ROAD_PAVED], [taxonomy.HUMAN, taxonomy.VEGETATION]])
        collapsed = taxonomy.to_coarse6(source)
        assert collapsed.tolist() == [
            [taxonomy.C6_BACKGROUND, taxonomy.C6_ROAD],
            [taxonomy.C6_NPC, taxonomy.C6_VEGETATION],
        ]

    def test_persons_collapse_to_npc_because_hero_needs_tracking(self):
        assert taxonomy.to_coarse6(np.array([[taxonomy.HUMAN]]))[0, 0] == taxonomy.C6_NPC
        assert taxonomy.C6_PERSON_UNSPLIT == taxonomy.C6_NPC
