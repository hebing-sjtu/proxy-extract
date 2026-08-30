"""Check the contract against its actual consumer, not against our own docs.

Everything else in this package is downstream of the assumption that what we
write is what `code-world-model` reads. These tests load the real loader and
make it read our output.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxy_extract import contract


class TestConstantsAgree:
    def test_condition_grid(self, cwm_constants):
        assert contract.CONDITION_WIDTH == cwm_constants.CONDITION_WIDTH
        assert contract.CONDITION_HEIGHT == cwm_constants.CONDITION_HEIGHT

    def test_depth_range(self, cwm_constants):
        assert contract.DEPTH_NEAR_METRES == cwm_constants.DEPTH_NEAR_METRES
        assert contract.DEPTH_FAR_METRES == cwm_constants.DEPTH_FAR_METRES
        assert contract.DEPTH_VALID_EPSILON_METRES == cwm_constants.DEPTH_VALID_EPSILON_METRES

    def test_window_geometry(self, cwm_constants):
        assert contract.WINDOW_FRAMES == cwm_constants.WINDOW_FRAMES
        assert contract.STRIDE_FRAMES == cwm_constants.STRIDE_FRAMES

    def test_semantic_palette_spans_our_class_count(self, cwm_constants):
        # The DUV encoder addresses classes as (id % 4, id // 4), so the palette
        # dimensions are what actually bound the legal ID range.
        assert len(cwm_constants.SEMANTIC_U) * len(cwm_constants.SEMANTIC_V) == contract.NUM_SEMANTIC_CLASSES


class TestRealLoaderAcceptsOurOutput:
    @pytest.fixture
    def written(self, tmp_path, rng):
        depth = rng.uniform(0.4, 200.0, (contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH)).astype(
            np.float32
        )
        depth[0, :] = 0.0  # an invalid row, as sky or a failed estimate would be
        labels = rng.integers(
            0, contract.NUM_SEMANTIC_CLASSES, (contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH)
        ).astype(np.uint8)
        contract.write_frame(tmp_path, 0, depth, labels)
        return tmp_path, depth, labels

    def test_loader_reads_our_frame_without_complaint(self, cwm_duv, written):
        root, _, _ = written
        frame = cwm_duv.load_duv_frame(root, 0, for_qwen=False)
        assert frame.shape == (contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH, 3)
        assert frame.dtype == np.float32

    def test_our_depth_codes_match_the_loaders(self, cwm_duv, written):
        root, depth, _ = written
        frame = cwm_duv.load_duv_frame(root, 0, for_qwen=False)
        loader_codes = np.rint(frame[..., 0] * 65535.0).astype(np.uint16)
        assert np.array_equal(loader_codes, contract.encode_depth_codes(depth))

    def test_our_labels_survive_the_palette_round_trip(self, cwm_duv, cwm_constants, written):
        root, _, labels = written
        frame = cwm_duv.load_duv_frame(root, 0, for_qwen=False)
        u = np.rint(frame[..., 1] * 255.0).astype(np.uint8)
        v = np.rint(frame[..., 2] * 255.0).astype(np.uint8)
        u_index = np.searchsorted(np.asarray(cwm_constants.SEMANTIC_U), u)
        v_index = np.searchsorted(np.asarray(cwm_constants.SEMANTIC_V), v)
        assert np.array_equal(v_index * 4 + u_index, labels)

    def test_qwen_view_is_also_readable(self, cwm_duv, written):
        root, _, _ = written
        frame = cwm_duv.load_duv_frame(root, 0, for_qwen=True)
        assert frame.dtype == np.uint8
        assert frame.shape == (contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH, 3)

    def test_loader_rejects_a_class_above_eleven(self, cwm_duv, tmp_path, rng):
        from PIL import Image

        depth = np.full((contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH), 5.0, np.float32)
        labels = np.zeros((contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH), np.uint8)
        contract.write_frame(tmp_path, 0, depth, labels)
        # Bypass our own writer to prove the downstream check is real.
        bad = labels.copy()
        bad[0, 0] = 12
        Image.fromarray(bad, mode="L").save(contract.frame_paths(tmp_path, 0)[1])
        with pytest.raises(ValueError, match=r"\[0,11\]"):
            cwm_duv.load_duv_frame(tmp_path, 0, for_qwen=False)


class TestFullWindow:
    def test_the_loader_can_read_a_whole_124_frame_window(self, cwm_duv, tmp_path, rng):
        for ordinal in range(contract.WINDOW_FRAMES):
            depth = rng.uniform(1.0, 60.0, (contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH))
            labels = rng.integers(
                0, 12, (contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH)
            ).astype(np.uint8)
            contract.write_frame(tmp_path, ordinal, depth.astype(np.float32), labels)

        window = cwm_duv.load_duv_window(tmp_path, 0, for_qwen=False)
        assert window.shape == (
            contract.WINDOW_FRAMES,
            contract.CONDITION_HEIGHT,
            contract.CONDITION_WIDTH,
            3,
        )
