from __future__ import annotations

import json

import numpy as np
import pytest

from proxy_extract import contract, preview, taxonomy as tx


def write_condition_root(root, labels, *, taxonomy: str | None):
    """A minimal on-disk condition_root, optionally with a run report."""
    depth = np.full((contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH), 10.0, np.float32)
    for ordinal, frame in enumerate(labels):
        contract.write_frame(root, ordinal, depth, frame)
    if taxonomy is not None:
        (root / "extraction_report.json").write_text(
            json.dumps({"clip": "t", "semantic": {"taxonomy": taxonomy}})
        )
    return root


def flat(value, frames=2):
    return np.full(
        (frames, contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH), value, np.uint8
    )


class TestPalettes:
    @pytest.mark.parametrize("palette", [preview.CWM12, preview.COARSE6])
    def test_every_class_has_its_own_colour(self, palette):
        # A duplicated colour silently merges two classes in every review frame.
        colours = [palette.colors[cls] for cls in range(palette.size)]
        assert len(set(colours)) == palette.size

    @pytest.mark.parametrize("palette", [preview.CWM12, preview.COARSE6])
    def test_the_palette_covers_exactly_its_taxonomy(self, palette):
        assert sorted(palette.colors) == list(range(palette.size))

    def test_the_two_taxonomies_are_the_sizes_they_claim(self):
        assert preview.CWM12.size == tx.NUM_CLASSES
        assert preview.COARSE6.size == tx.NUM_COARSE6

    def test_hero_and_npc_are_far_apart_in_colour(self):
        # The whole point of the split is visible at a glance in review.
        hero = np.array(preview.COARSE6.colors[tx.C6_HERO], float)
        npc = np.array(preview.COARSE6.colors[tx.C6_NPC], float)
        assert np.linalg.norm(hero - npc) > 100

    def test_out_of_range_ids_clamp_instead_of_crashing(self):
        out = preview.colorize_semantic(np.array([[200]], np.uint8), preview.COARSE6)
        assert out.shape == (1, 1, 3)


class TestPaletteResolution:
    def test_a_coarse6_run_is_recognised(self, tmp_path):
        root = write_condition_root(tmp_path / "c6", flat(tx.C6_HERO), taxonomy="coarse6")
        assert preview.palette_for(root) is preview.COARSE6

    def test_a_cwm12_run_is_recognised(self, tmp_path):
        root = write_condition_root(tmp_path / "c12", flat(tx.HUMAN), taxonomy="cwm12")
        assert preview.palette_for(root) is preview.CWM12

    def test_a_run_with_no_report_falls_back_to_cwm12(self, tmp_path):
        root = write_condition_root(tmp_path / "bare", flat(tx.SKY), taxonomy=None)
        assert preview.palette_for(root) is preview.CWM12

    def test_a_truncated_report_falls_back_instead_of_raising(self, tmp_path):
        root = write_condition_root(tmp_path / "torn", flat(tx.SKY), taxonomy="coarse6")
        (root / "extraction_report.json").write_text('{"semantic": {"taxo')
        assert preview.palette_for(root) is preview.CWM12

    def test_an_unknown_taxonomy_name_falls_back(self, tmp_path):
        root = write_condition_root(tmp_path / "future", flat(tx.SKY), taxonomy="coarse42")
        assert preview.palette_for(root) is preview.CWM12

    def test_the_id_range_alone_does_not_decide(self, tmp_path):
        # A 12-class clip showing only sky and road uses IDs a 6-class clip
        # could also produce; only the report distinguishes them.
        labels = flat(tx.SKY)
        labels[1] = tx.ROAD_PAVED
        root = write_condition_root(tmp_path / "sparse", labels, taxonomy="cwm12")
        assert preview.palette_for(root) is preview.CWM12


class TestRenderPreview:
    def test_it_writes_a_playable_file(self, tmp_path):
        import cv2

        root = write_condition_root(tmp_path / "c6", flat(tx.C6_HERO, frames=4), taxonomy="coarse6")
        out = preview.render_preview(root, tmp_path / "preview.mp4", fps=8, scale=1)

        assert out.exists() and out.stat().st_size > 0
        capture = cv2.VideoCapture(str(out))
        try:
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        finally:
            capture.release()

    def test_the_same_ids_render_differently_under_each_taxonomy(self, tmp_path):
        # Guards the bug this palette plumbing exists to prevent: hero (id 5)
        # painted with the 12-class colour for vegetation.
        ids = np.full((8, 8), tx.C6_HERO, np.uint8)
        assert not np.array_equal(
            preview.colorize_semantic(ids, preview.COARSE6),
            preview.colorize_semantic(ids, preview.CWM12),
        )

    def test_an_empty_root_says_so(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no condition frames"):
            preview.render_preview(tmp_path / "empty", tmp_path / "out.mp4")
