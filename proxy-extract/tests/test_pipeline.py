"""End-to-end runs on a real clip, using the synthetic backends.

The model backends need a GPU, but everything around them - decode, frame-count
arithmetic, calibration, downsampling, stabilisation, encoding, validation - does
not, and that is where contract bugs actually live. Substituting deterministic
stand-ins exercises the whole path on the workstation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from proxy_extract import contract
from proxy_extract.pipeline import ExtractionConfig, extract_clip
from proxy_extract.video import probe, read_frames

SYNTHETIC = ExtractionConfig(
    depth_backend="synthetic",
    semantic_backend="synthetic",
    temporal_radius=1,
    flow_compensate=False,
)


@pytest.fixture(scope="module")
def extracted(sample_clip, tmp_path_factory):
    out = tmp_path_factory.mktemp("condition")
    return extract_clip(sample_clip, out, config=SYNTHETIC), out


class TestDecode:
    def test_frames_are_rgb_uint8_at_the_requested_size(self, sample_clip):
        frames = read_frames(sample_clip, size=(336, 192), limit=3)
        assert len(frames) == 3
        assert frames[0].shape == (192, 336, 3) and frames[0].dtype == np.uint8


class TestFrameBudget:
    @pytest.mark.parametrize("decoded,expected", [(124, 124), (200, 124), (214, 214), (300, 214)])
    def test_only_whole_windows_are_kept(self, decoded, expected):
        assert SYNTHETIC.usable_frame_count(decoded) == expected

    def test_a_clip_shorter_than_one_window_is_refused(self):
        with pytest.raises(ValueError, match="124"):
            SYNTHETIC.usable_frame_count(100)


class TestEndToEnd:
    def test_writes_exactly_one_window(self, extracted):
        report, _ = extracted
        assert report["frames"] == contract.WINDOW_FRAMES
        assert report["validation"]["windows"] == 1

    def test_output_passes_our_own_validator(self, extracted):
        _, out = extracted
        contract.validate_condition_root(out, expected_frames=contract.WINDOW_FRAMES)

    def test_output_is_readable_by_the_real_cwm_loader(self, cwm_duv, extracted):
        _, out = extracted
        window = cwm_duv.load_duv_window(out, 0, for_qwen=False)
        assert window.shape == (contract.WINDOW_FRAMES, 192, 336, 3)
        assert np.all(np.isfinite(window))

    def test_qwen_view_of_the_same_window_also_loads(self, cwm_duv, extracted):
        _, out = extracted
        window = cwm_duv.load_duv_window(out, 0, for_qwen=True)
        assert window.dtype == np.uint8
        assert len(window) == len(cwm_duv.QWEN_OFFSETS)

    def test_every_ordinal_is_present_and_contiguous(self, extracted):
        _, out = extracted
        names = sorted(p.name for p in out.glob("*.depth.f32"))
        assert names[0] == "000000.depth.f32"
        assert names[-1] == f"{contract.WINDOW_FRAMES - 1:06d}.depth.f32"
        assert len(names) == contract.WINDOW_FRAMES

    def test_depth_lands_inside_the_encodable_range(self, extracted):
        report, _ = extracted
        validation = report["validation"]
        assert validation["depth_min_metres"] >= contract.DEPTH_NEAR_METRES
        assert validation["depth_max_metres"] <= contract.DEPTH_FAR_METRES

    def test_report_records_where_the_metric_scale_came_from(self, extracted):
        report, _ = extracted
        assert report["depth"]["metric_source"] == "backend_native"

    def test_report_is_written_next_to_the_frames_and_is_valid_json(self, extracted):
        _, out = extracted
        assert json.loads((out / "extraction_report.json").read_text())["frames"] == 124

    def test_stabilisation_does_not_increase_flicker(self, extracted):
        report, _ = extracted
        semantic = report["semantic"]
        assert semantic["flicker_after"] <= semantic["flicker_before"]

    def test_class_fractions_sum_to_one(self, extracted):
        report, _ = extracted
        assert sum(report["semantic"]["class_fractions"].values()) == pytest.approx(1.0, abs=1e-4)


class TestOutputPaths:
    def test_the_source_directory_is_part_of_the_path(self, tmp_path):
        from proxy_extract.pipeline import condition_dir_for

        assert condition_dir_for(tmp_path, Path("data/00/clip_a.mp4")) == tmp_path / "00" / "clip_a"

    def test_same_named_clips_in_different_directories_do_not_collide(self, tmp_path):
        from proxy_extract.pipeline import condition_dir_for

        # ABot names every episode `video.mp4` and distinguishes them by the
        # directory above, so collapsing that would silently collapse the set.
        first = condition_dir_for(tmp_path, Path("data/00/sample_a/video.mp4"))
        second = condition_dir_for(tmp_path, Path("data/00/sample_b/video.mp4"))
        assert first != second


class TestPreview:
    def test_preview_renders_a_playable_file(self, extracted, tmp_path):
        from proxy_extract.preview import render_preview

        _, out = extracted
        path = render_preview(out, tmp_path / "preview.mp4")
        assert path.stat().st_size > 0
        assert probe(path).frames == contract.WINDOW_FRAMES
