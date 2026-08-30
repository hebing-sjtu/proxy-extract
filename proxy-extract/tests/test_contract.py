from __future__ import annotations

import numpy as np
import pytest

from proxy_extract import contract


def _depth_field(rng, shape=(contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH)):
    return rng.uniform(0.5, 120.0, size=shape).astype(np.float32)


def _labels(rng, shape=(contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH)):
    return rng.integers(0, contract.NUM_SEMANTIC_CLASSES, size=shape, dtype=np.uint8)


class TestDepthCoding:
    def test_endpoints_hit_the_full_uint16_range(self):
        codes = contract.encode_depth_codes(np.array([0.3, 256.0]))
        assert codes.tolist() == [65535, 0]

    def test_values_outside_the_range_clamp_rather_than_wrap(self):
        codes = contract.encode_depth_codes(np.array([0.01, 10_000.0]))
        assert codes.tolist() == [65535, 0]

    def test_invalid_depth_encodes_to_zero(self):
        codes = contract.encode_depth_codes(np.array([0.0, 1e-4, contract.DEPTH_VALID_EPSILON_METRES]))
        assert codes.tolist() == [0, 0, 0]

    def test_round_trip_is_accurate_to_the_quantiser(self, rng):
        original = _depth_field(rng)
        recovered = contract.decode_depth_codes(contract.encode_depth_codes(original))
        assert np.allclose(recovered, original, rtol=1e-3)

    def test_encoding_is_monotonically_decreasing_in_distance(self):
        codes = contract.encode_depth_codes(np.array([1.0, 2.0, 4.0, 8.0, 64.0]))
        assert np.all(np.diff(codes.astype(int)) < 0)

    def test_scaling_depth_shifts_every_code_by_one_constant(self, rng):
        # The property the calibration budget relies on: because the encoding is
        # logarithmic, residual scale error is a uniform offset, not a warp.
        depth = _depth_field(rng)
        scale = 2.0
        delta = contract.encode_depth_codes(depth * scale).astype(int) - contract.encode_depth_codes(
            depth
        ).astype(int)
        assert np.ptp(delta) <= 1
        assert delta.mean() == pytest.approx(contract.depth_code_shift_for_scale(scale), abs=1.0)


class TestResampling:
    def test_depth_block_median_picks_a_depth_that_occurred(self):
        source = np.zeros((contract.CONDITION_HEIGHT * 2, contract.CONDITION_WIDTH * 2), dtype=np.float32)
        source[:, : contract.CONDITION_WIDTH * 2 // 2] = 2.0
        source[:, contract.CONDITION_WIDTH * 2 // 2 :] = 50.0
        out = contract.downsample_depth(source)
        assert set(np.unique(out).tolist()) <= {2.0, 50.0}

    def test_depth_downsample_ignores_invalid_pixels(self):
        source = np.full((contract.CONDITION_HEIGHT * 2, contract.CONDITION_WIDTH * 2), 8.0, np.float32)
        source[::2, ::2] = 0.0
        out = contract.downsample_depth(source)
        assert np.allclose(out, 8.0)

    def test_depth_downsample_keeps_fully_invalid_blocks_invalid(self):
        source = np.zeros((contract.CONDITION_HEIGHT * 2, contract.CONDITION_WIDTH * 2), np.float32)
        assert np.all(contract.downsample_depth(source) == 0.0)

    def test_semantic_downsample_takes_the_majority_not_a_sample(self):
        source = np.full((contract.CONDITION_HEIGHT * 4, contract.CONDITION_WIDTH * 4), 5, np.uint8)
        source[::4, ::4] = 8  # one pixel of 16 per block
        assert np.all(contract.downsample_semantic(source) == 5)

    def test_semantic_downsample_never_invents_a_class(self, rng):
        source = rng.choice([3, 6], size=(contract.CONDITION_HEIGHT * 4, contract.CONDITION_WIDTH * 4))
        assert set(np.unique(contract.downsample_semantic(source)).tolist()) <= {3, 6}

    def test_non_integer_ratios_still_reach_the_target_grid(self, rng):
        out = contract.downsample_depth(rng.uniform(1, 50, size=(500, 913)).astype(np.float32))
        assert out.shape == (contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH)


class TestFrameIO:
    def test_written_depth_is_exactly_the_documented_byte_count(self, tmp_path, rng):
        contract.write_frame(tmp_path, 0, _depth_field(rng), _labels(rng))
        depth_path, semantic_path = contract.frame_paths(tmp_path, 0)
        assert depth_path.stat().st_size == 258_048 == contract.DEPTH_BYTES
        assert semantic_path.exists()

    def test_semantic_png_is_grayscale_at_the_documented_size(self, tmp_path, rng):
        from PIL import Image

        contract.write_frame(tmp_path, 7, _depth_field(rng), _labels(rng))
        with Image.open(contract.frame_paths(tmp_path, 7)[1]) as image:
            assert image.mode == "L"
            assert image.size == (contract.CONDITION_WIDTH, contract.CONDITION_HEIGHT)

    def test_read_back_matches_what_was_written(self, tmp_path, rng):
        depth, labels = _depth_field(rng), _labels(rng)
        contract.write_frame(tmp_path, 3, depth, labels)
        got_depth, got_labels = contract.read_frame(tmp_path, 3)
        assert np.allclose(got_depth, depth)
        assert np.array_equal(got_labels, labels)

    def test_out_of_range_class_is_rejected_at_write_time(self, tmp_path, rng):
        labels = _labels(rng)
        labels[0, 0] = 12
        with pytest.raises(ValueError, match="exceed"):
            contract.write_frame(tmp_path, 0, _depth_field(rng), labels)

    def test_non_finite_depth_is_rejected_at_write_time(self, tmp_path, rng):
        depth = _depth_field(rng)
        depth[5, 5] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            contract.write_frame(tmp_path, 0, depth, _labels(rng))

    def test_no_temporary_files_survive_a_write(self, tmp_path, rng):
        contract.write_frame(tmp_path, 0, _depth_field(rng), _labels(rng))
        assert not list(tmp_path.glob("*.tmp"))


class TestValidation:
    def _write_run(self, root, rng, frames):
        for ordinal in range(frames):
            contract.write_frame(root, ordinal, _depth_field(rng), _labels(rng))

    def test_a_full_window_validates(self, tmp_path, rng):
        self._write_run(tmp_path, rng, contract.WINDOW_FRAMES)
        summary = contract.validate_condition_root(tmp_path, expected_frames=contract.WINDOW_FRAMES)
        assert summary["frames"] == contract.WINDOW_FRAMES
        assert summary["windows"] == 1

    def test_a_gap_in_the_ordinals_is_caught(self, tmp_path, rng):
        self._write_run(tmp_path, rng, 4)
        contract.frame_paths(tmp_path, 2)[0].unlink()
        with pytest.raises(contract.ContractError, match="contiguous"):
            contract.validate_condition_root(tmp_path)

    def test_a_truncated_depth_file_is_caught(self, tmp_path, rng):
        self._write_run(tmp_path, rng, 2)
        contract.frame_paths(tmp_path, 1)[0].write_bytes(b"\x00" * 16)
        with pytest.raises(ValueError, match="byte count"):
            contract.validate_condition_root(tmp_path)

    def test_a_wrong_frame_count_is_caught(self, tmp_path, rng):
        self._write_run(tmp_path, rng, 3)
        with pytest.raises(contract.ContractError, match="expected 124"):
            contract.validate_condition_root(tmp_path, expected_frames=124)


class TestWindowArithmetic:
    @pytest.mark.parametrize(
        "frames,expected", [(123, 0), (124, 1), (213, 1), (214, 2), (304, 3)]
    )
    def test_window_count(self, frames, expected):
        assert contract.window_count_for(frames) == expected

    @pytest.mark.parametrize("windows,expected", [(1, 124), (2, 214), (3, 304)])
    def test_frames_for_windows(self, windows, expected):
        assert contract.frames_for_windows(windows) == expected
