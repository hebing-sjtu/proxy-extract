"""The 11-class standard, and the two places it can silently diverge."""

from __future__ import annotations

import numpy as np
import pytest

from proxy_extract import contract, taxonomy
from proxy_extract.preview import PALETTES, STANDARD11


def test_the_standard_has_the_eleven_ids_data_f_defines():
    assert taxonomy.NUM_STANDARD11 == 11
    assert taxonomy.STANDARD11_NAMES[taxonomy.S11_SKY] == "sky"
    assert taxonomy.STANDARD11_NAMES[taxonomy.S11_PLAYER] == "player"
    assert taxonomy.STANDARD11_NAMES[taxonomy.S11_PED] == "ped"
    assert taxonomy.STANDARD11_NAMES[taxonomy.S11_PROP] == "prop"


def test_every_id_fits_the_condition_format():
    """The DUV writer rejects ids at or above 12, so 11 classes must fit under it."""
    assert taxonomy.NUM_STANDARD11 <= contract.NUM_SEMANTIC_CLASSES


def test_the_ade20k_table_only_names_classes_ade20k_has():
    """`build_lut` raises on a stale name; this asserts the table is current."""
    lut = taxonomy.standard11_lut()
    assert len(lut) == len(taxonomy.ADE20K_CLASSES)
    assert lut.max() < taxonomy.NUM_STANDARD11


def test_unmapped_ade20k_classes_fall_to_prop():
    """The standard names prop as its own default, so the fallback is correct."""
    lut = taxonomy.standard11_lut()
    assert lut[taxonomy.ADE20K_CLASSES.index("sofa")] == taxonomy.S11_PROP
    assert lut[taxonomy.ADE20K_CLASSES.index("refrigerator")] == taxonomy.S11_PROP


def test_persons_arrive_undivided():
    """Nothing may map straight to player: that decision belongs to the splitter."""
    assert taxonomy.S11_PLAYER not in taxonomy.ADE20K_TO_STANDARD11.values()
    lut = taxonomy.standard11_lut()
    assert lut[taxonomy.ADE20K_CLASSES.index("person")] == taxonomy.S11_PED


def test_road_holds_only_the_driveable_surface():
    """Sidewalk must not land in road.

    The engine splits road from ground by material and a model can only split
    them by function, so the two definitions already disagree. Letting sidewalk
    into road would add a second, avoidable disagreement on top.
    """
    lut = taxonomy.standard11_lut()
    assert lut[taxonomy.ADE20K_CLASSES.index("road")] == taxonomy.S11_ROAD
    assert lut[taxonomy.ADE20K_CLASSES.index("sidewalk")] == taxonomy.S11_GROUND
    assert lut[taxonomy.ADE20K_CLASSES.index("path")] == taxonomy.S11_GROUND


def test_grass_is_vegetation_and_rock_is_terrain():
    """DATA_F.md puts 草 in vegetation and 岩石 in terrain; ADE20K separates them."""
    lut = taxonomy.standard11_lut()
    assert lut[taxonomy.ADE20K_CLASSES.index("grass")] == taxonomy.S11_VEGETATION
    assert lut[taxonomy.ADE20K_CLASSES.index("rock")] == taxonomy.S11_TERRAIN
    assert lut[taxonomy.ADE20K_CLASSES.index("mountain")] == taxonomy.S11_TERRAIN


def test_priority_is_a_total_order_with_the_protagonist_on_top():
    priority = taxonomy.STANDARD11_PRIORITY
    assert sorted(priority) == list(range(taxonomy.NUM_STANDARD11))
    assert len(set(priority.values())) == taxonomy.NUM_STANDARD11
    assert priority[taxonomy.S11_PLAYER] == max(priority.values())
    assert priority[taxonomy.S11_SKY] == min(priority.values())
    # A person standing on a road is a person.
    assert priority[taxonomy.S11_PED] > priority[taxonomy.S11_ROAD]


def test_the_cwm_projection_covers_every_source_class():
    assert sorted(taxonomy.CWM_TO_STANDARD11) == list(range(taxonomy.NUM_CLASSES))
    projected = taxonomy.to_standard11(np.arange(taxonomy.NUM_CLASSES, dtype=np.uint8))
    assert projected.max() < taxonomy.NUM_STANDARD11


def test_the_preview_palette_covers_every_class():
    """A missing colour renders as black, which reads as a real class."""
    assert STANDARD11.size == taxonomy.NUM_STANDARD11
    assert sorted(STANDARD11.colors) == list(range(taxonomy.NUM_STANDARD11))
    assert PALETTES["standard11"] is STANDARD11


def test_road_and_ground_are_visually_distinct():
    """The pair most likely to be confused must not look alike in a preview."""
    road = np.array(STANDARD11.colors[taxonomy.S11_ROAD], dtype=int)
    ground = np.array(STANDARD11.colors[taxonomy.S11_GROUND], dtype=int)
    assert np.abs(road - ground).sum() > 90


def test_the_functional_mismatch_is_written_down():
    """Prose, but load-bearing: mixing the two sources silently corrupts these two classes."""
    assert "not equivalent" in taxonomy.ROAD_GROUND_IS_FUNCTIONAL


@pytest.mark.parametrize("backend", ["standard11"])
def test_the_backend_name_is_wired_through(backend):
    from proxy_extract.cli import SEMANTIC_BACKENDS
    from proxy_extract.pipeline import ExtractionConfig
    from proxy_extract.semantic.panoptic import _PROFILES

    assert backend in SEMANTIC_BACKENDS
    assert backend in _PROFILES
    assert ExtractionConfig(semantic_backend=backend).taxonomy == "standard11"


def test_a_mount_lands_in_vehicle_rather_than_prop():
    """The standard has no animal class, and a ridden horse is not clutter.

    Filing it as prop would tell a world model that a large moving agent is
    static scenery, which is the more damaging of the two available errors.
    """
    assert taxonomy.ADE20K_TO_STANDARD11["animal"] == taxonomy.S11_VEHICLE


def test_the_mount_mapping_claims_all_wildlife_too():
    """ADE20K offers one `animal` label, so this is a package deal.

    Pinned so the cost stays visible: separating mounts from deer and birds
    needs instance masks, and nothing in this table can do it.
    """
    animal_classes = [c for c in taxonomy.ADE20K_CLASSES if "animal" in c]
    assert animal_classes == ["animal"]
