from __future__ import annotations

import numpy as np
import pytest

from proxy_extract import taxonomy as tx
from proxy_extract.semantic.base import resolve_label_lut


class TestClassSet:
    def test_the_contract_and_the_taxonomy_agree_on_the_class_count(self):
        # contract.py restates the format standalone so it does not depend on
        # taxonomy.py; this is the seam where they could silently diverge.
        from proxy_extract import contract

        assert contract.NUM_SEMANTIC_CLASSES == tx.NUM_CLASSES == len(tx.CLASS_NAMES)

    def test_names_match_the_cwm_ordering(self):
        assert tx.CLASS_NAMES == (
            "void_unknown", "sky", "water", "terrain", "road_paved", "vegetation",
            "building_structure", "infrastructure", "human", "animal", "vehicle", "prop",
        )

    def test_every_class_has_a_priority(self):
        assert sorted(tx.PRIORITY) == list(range(tx.NUM_CLASSES))

    def test_void_never_wins_and_human_always_does(self):
        assert tx.PRIORITY[tx.VOID_UNKNOWN] == min(tx.PRIORITY.values())
        assert tx.PRIORITY[tx.HUMAN] == max(tx.PRIORITY.values())


class TestSourceMappings:
    def test_ade20k_has_exactly_150_distinct_classes(self):
        assert len(tx.ADE20K_CLASSES) == 150
        assert len(set(tx.ADE20K_CLASSES)) == 150

    def test_cityscapes_has_exactly_19_distinct_classes(self):
        assert len(tx.CITYSCAPES_CLASSES) == 19
        assert len(set(tx.CITYSCAPES_CLASSES)) == 19

    def test_mappings_only_reference_real_source_classes(self):
        tx.ade20k_lut()
        tx.cityscapes_lut()

    def test_a_typo_in_a_mapping_fails_loudly(self):
        with pytest.raises(ValueError, match="absent from the source label set"):
            tx.build_lut(("road", "sky"), {"rodd": tx.ROAD_PAVED}, default=tx.VOID_UNKNOWN)

    def test_luts_only_emit_valid_cwm_classes(self):
        for lut in (tx.ade20k_lut(), tx.cityscapes_lut()):
            assert lut.max() < tx.NUM_CLASSES

    @pytest.mark.parametrize(
        "source,expected",
        [("sky", tx.SKY), ("road", tx.ROAD_PAVED), ("tree", tx.VEGETATION),
         ("person", tx.HUMAN), ("car", tx.VEHICLE), ("animal", tx.ANIMAL)],
    )
    def test_ade20k_landmark_classes(self, source, expected):
        assert tx.ade20k_lut()[tx.ADE20K_CLASSES.index(source)] == expected

    def test_unpaved_ground_is_terrain_not_road(self):
        lut = tx.ade20k_lut()
        assert lut[tx.ADE20K_CLASSES.index("dirt track")] == tx.TERRAIN
        assert lut[tx.ADE20K_CLASSES.index("road")] == tx.ROAD_PAVED

    def test_cityscapes_covers_all_19_with_no_fallthrough(self):
        # Cityscapes has no catch-all class, so an unmapped entry would be a gap.
        assert tx.VOID_UNKNOWN not in tx.cityscapes_lut().tolist()

    def test_riders_count_as_humans(self):
        assert tx.cityscapes_lut()[tx.CITYSCAPES_CLASSES.index("rider")] == tx.HUMAN


class TestPriorityOverlay:
    def test_a_person_paints_over_a_road(self):
        base = np.full((4, 4), tx.ROAD_PAVED, np.uint8)
        mask = np.zeros((4, 4), bool)
        mask[1:3, 1:3] = True
        out = tx.overlay(base, tx.HUMAN, mask)
        assert out[1, 1] == tx.HUMAN and out[0, 0] == tx.ROAD_PAVED

    def test_a_road_does_not_paint_over_a_person(self):
        base = np.full((4, 4), tx.HUMAN, np.uint8)
        out = tx.overlay(base, tx.ROAD_PAVED, np.ones((4, 4), bool))
        assert np.all(out == tx.HUMAN)

    def test_anything_paints_over_void(self):
        base = np.full((4, 4), tx.VOID_UNKNOWN, np.uint8)
        out = tx.overlay(base, tx.SKY, np.ones((4, 4), bool))
        assert np.all(out == tx.SKY)

    def test_an_empty_mask_changes_nothing(self):
        base = np.full((4, 4), tx.TERRAIN, np.uint8)
        assert np.array_equal(tx.overlay(base, tx.HUMAN, np.zeros((4, 4), bool)), base)


class TestApplyLut:
    def test_ignore_index_becomes_void(self):
        labels = np.array([[0, 255]], dtype=np.int32)
        out = tx.apply_lut(labels, tx.cityscapes_lut(), ignore_index=255)
        assert out[0, 0] == tx.ROAD_PAVED and out[0, 1] == tx.VOID_UNKNOWN

    def test_out_of_range_source_labels_become_void(self):
        out = tx.apply_lut(np.array([[999]]), tx.cityscapes_lut())
        assert out[0, 0] == tx.VOID_UNKNOWN


class TestCheckpointLabelResolution:
    def test_synonym_lists_resolve_to_the_first_known_name(self):
        lut, unmapped = resolve_label_lut(
            {0: "building;edifice", 1: "sky", 2: "windowpane;window"}, tx.ADE20K_TO_CWM
        )
        assert lut.tolist() == [tx.BUILDING_STRUCTURE, tx.SKY, tx.BUILDING_STRUCTURE]
        assert unmapped == []

    def test_case_and_spacing_differences_still_resolve(self):
        lut, _ = resolve_label_lut({0: "  TRAFFIC   LIGHT "}, tx.CITYSCAPES_TO_CWM)
        assert lut[0] == tx.INFRASTRUCTURE

    def test_unknown_labels_are_reported_not_silently_defaulted(self):
        lut, unmapped = resolve_label_lut({0: "sky", 1: "quantum sofa"}, tx.ADE20K_TO_CWM, default=tx.PROP)
        assert lut[1] == tx.PROP
        assert unmapped == ["quantum sofa"]

    def test_a_permuted_label_order_is_handled_by_reading_names(self):
        # The failure this guards against: a checkpoint whose label indices are
        # not the canonical ADE20K order.
        lut, _ = resolve_label_lut({0: "person", 1: "sky", 2: "road"}, tx.ADE20K_TO_CWM)
        assert lut.tolist() == [tx.HUMAN, tx.SKY, tx.ROAD_PAVED]


class TestSam3Prompts:
    def test_prompts_only_target_valid_classes(self):
        assert all(0 <= p.cwm_class < tx.NUM_CLASSES for p in tx.SAM3_PROMPTS)

    def test_prompts_cover_the_classes_ade20k_cannot_express(self):
        # animal and prop are precisely why the refiner exists.
        targeted = {p.cwm_class for p in tx.SAM3_PROMPTS}
        assert {tx.ANIMAL, tx.PROP, tx.INFRASTRUCTURE} <= targeted
