"""The 12-class CWM semantic taxonomy and projections onto it.

No off-the-shelf segmenter predicts these 12 classes, so every backend needs a
mapping. Mappings are written by source-class *name* rather than index: index
order varies between checkpoints and a silent off-by-one here would poison the
whole dataset, whereas a renamed class fails loudly in `build_lut`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOID_UNKNOWN = 0
SKY = 1
WATER = 2
TERRAIN = 3
ROAD_PAVED = 4
VEGETATION = 5
BUILDING_STRUCTURE = 6
INFRASTRUCTURE = 7
HUMAN = 8
ANIMAL = 9
VEHICLE = 10
PROP = 11

CLASS_NAMES = (
    "void_unknown",
    "sky",
    "water",
    "terrain",
    "road_paved",
    "vegetation",
    "building_structure",
    "infrastructure",
    "human",
    "animal",
    "vehicle",
    "prop",
)

NUM_CLASSES = len(CLASS_NAMES)

# Used when two backends claim the same pixel. Higher wins. Things beat stuff
# because a person standing on a road is a person; void loses to everything.
PRIORITY: dict[int, int] = {
    VOID_UNKNOWN: 0,
    SKY: 1,
    TERRAIN: 2,
    ROAD_PAVED: 3,
    WATER: 3,
    VEGETATION: 4,
    BUILDING_STRUCTURE: 5,
    INFRASTRUCTURE: 6,
    PROP: 7,
    VEHICLE: 8,
    ANIMAL: 9,
    HUMAN: 10,
}


# --------------------------------------------------------------- ADE20K-150

ADE20K_CLASSES = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain",
    "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house",
    "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock",
    "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column",
    "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper",
    "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar",
    "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier",
    "awning", "streetlight", "booth", "television", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent",
    "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket",
    "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag",
)

ADE20K_TO_CWM: dict[str, int] = {
    "sky": SKY,
    # Water. GTA has plenty of coastline and rivers.
    "water": WATER, "sea": WATER, "river": WATER, "lake": WATER,
    "swimming pool": WATER, "waterfall": WATER, "fountain": WATER,
    # Natural ground. "dirt track" is unpaved, so it is terrain, not road_paved.
    "grass": TERRAIN, "earth": TERRAIN, "mountain": TERRAIN, "field": TERRAIN,
    "rock": TERRAIN, "sand": TERRAIN, "hill": TERRAIN, "land": TERRAIN,
    "dirt track": TERRAIN,
    # Built ground surfaces.
    "road": ROAD_PAVED, "sidewalk": ROAD_PAVED, "path": ROAD_PAVED,
    "runway": ROAD_PAVED, "floor": ROAD_PAVED,
    "tree": VEGETATION, "plant": VEGETATION, "flower": VEGETATION, "palm": VEGETATION,
    # Building shells and their openings.
    "building": BUILDING_STRUCTURE, "house": BUILDING_STRUCTURE,
    "skyscraper": BUILDING_STRUCTURE, "hovel": BUILDING_STRUCTURE,
    "tower": BUILDING_STRUCTURE, "bridge": BUILDING_STRUCTURE,
    "wall": BUILDING_STRUCTURE, "ceiling": BUILDING_STRUCTURE,
    "windowpane": BUILDING_STRUCTURE, "door": BUILDING_STRUCTURE,
    "awning": BUILDING_STRUCTURE, "canopy": BUILDING_STRUCTURE,
    "column": BUILDING_STRUCTURE, "stage": BUILDING_STRUCTURE,
    "booth": BUILDING_STRUCTURE, "screen door": BUILDING_STRUCTURE,
    "grandstand": BUILDING_STRUCTURE, "pier": BUILDING_STRUCTURE,
    # Street furniture and signage: small fixed structures attached to the world.
    "fence": INFRASTRUCTURE, "railing": INFRASTRUCTURE, "bannister": INFRASTRUCTURE,
    "pole": INFRASTRUCTURE, "streetlight": INFRASTRUCTURE, "traffic light": INFRASTRUCTURE,
    "signboard": INFRASTRUCTURE, "trade name": INFRASTRUCTURE, "poster": INFRASTRUCTURE,
    "bulletin board": INFRASTRUCTURE, "stairs": INFRASTRUCTURE, "stairway": INFRASTRUCTURE,
    "step": INFRASTRUCTURE, "escalator": INFRASTRUCTURE, "conveyer belt": INFRASTRUCTURE,
    "light": INFRASTRUCTURE, "lamp": INFRASTRUCTURE, "sconce": INFRASTRUCTURE,
    "chandelier": INFRASTRUCTURE, "flag": INFRASTRUCTURE,
    "person": HUMAN,
    "animal": ANIMAL,
    "car": VEHICLE, "bus": VEHICLE, "truck": VEHICLE, "van": VEHICLE,
    "boat": VEHICLE, "ship": VEHICLE, "airplane": VEHICLE, "minibike": VEHICLE,
    "bicycle": VEHICLE, "tank": VEHICLE,
}

# Everything ADE20K knows that is none of the above is a movable object: indoor
# furniture, appliances, clutter. All of it collapses to prop.
_ADE20K_PROP_FALLBACK = True


# ------------------------------------------------------------- Cityscapes-19
# Near-perfect fit for the urban clips: its label set was designed for exactly
# this kind of street-level scene.

CITYSCAPES_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)

CITYSCAPES_TO_CWM: dict[str, int] = {
    "road": ROAD_PAVED, "sidewalk": ROAD_PAVED,
    "building": BUILDING_STRUCTURE, "wall": BUILDING_STRUCTURE,
    "fence": INFRASTRUCTURE, "pole": INFRASTRUCTURE,
    "traffic light": INFRASTRUCTURE, "traffic sign": INFRASTRUCTURE,
    "vegetation": VEGETATION, "terrain": TERRAIN, "sky": SKY,
    "person": HUMAN, "rider": HUMAN,
    "car": VEHICLE, "truck": VEHICLE, "bus": VEHICLE, "train": VEHICLE,
    "motorcycle": VEHICLE, "bicycle": VEHICLE,
}


# --------------------------------------------------- coarse 6-class taxonomy
# What the current dataset build actually asks for. IDs are ordered by overlay
# priority, not by how much each class matters: `background` is the fallback
# every unclaimed pixel lands in, so it must lose every conflict, and a person
# standing on a road has to beat the road. Ranking the classes by importance
# and then reusing that ranking as a paint order would let background erase
# almost everything, which is why the two orders are kept separate here.

C6_BACKGROUND = 0
C6_ROAD = 1
C6_VEGETATION = 2
C6_VEHICLE = 3
C6_NPC = 4
C6_HERO = 5

COARSE6_NAMES = ("background", "road", "vegetation", "vehicle", "npc", "hero")
NUM_COARSE6 = len(COARSE6_NAMES)

# Persons arrive from any segmenter as one undivided class; splitting hero from
# npc is a tracking problem, not a labelling one, so it happens after this map.
C6_PERSON_UNSPLIT = C6_NPC

CWM_TO_COARSE6: dict[int, int] = {
    VOID_UNKNOWN: C6_BACKGROUND,
    SKY: C6_BACKGROUND,
    WATER: C6_BACKGROUND,
    TERRAIN: C6_BACKGROUND,
    ROAD_PAVED: C6_ROAD,
    VEGETATION: C6_VEGETATION,
    BUILDING_STRUCTURE: C6_BACKGROUND,
    INFRASTRUCTURE: C6_BACKGROUND,
    HUMAN: C6_PERSON_UNSPLIT,
    ANIMAL: C6_VEHICLE,  # horses are transport in the RDR2 clips
    VEHICLE: C6_VEHICLE,
    PROP: C6_BACKGROUND,
}

# Built straight from ADE20K rather than by folding the 12-class map, because
# the two taxonomies disagree about ground cover: with no terrain class to fall
# into, grass belongs with the vegetation it looks like, not with background.
ADE20K_TO_COARSE6: dict[str, int] = {
    "road": C6_ROAD, "sidewalk": C6_ROAD, "path": C6_ROAD, "runway": C6_ROAD,
    "floor": C6_ROAD, "dirt track": C6_ROAD,
    "tree": C6_VEGETATION, "plant": C6_VEGETATION, "flower": C6_VEGETATION,
    "palm": C6_VEGETATION, "grass": C6_VEGETATION, "field": C6_VEGETATION,
    "car": C6_VEHICLE, "bus": C6_VEHICLE, "truck": C6_VEHICLE, "van": C6_VEHICLE,
    "boat": C6_VEHICLE, "ship": C6_VEHICLE, "airplane": C6_VEHICLE,
    "minibike": C6_VEHICLE, "bicycle": C6_VEHICLE, "tank": C6_VEHICLE,
    "animal": C6_VEHICLE,
    "person": C6_PERSON_UNSPLIT,
}

COARSE6_PRIORITY: dict[int, int] = {cls: cls for cls in range(NUM_COARSE6)}


def coarse6_lut() -> np.ndarray:
    return build_lut(ADE20K_CLASSES, ADE20K_TO_COARSE6, default=C6_BACKGROUND)


def to_coarse6(cwm_labels: np.ndarray) -> np.ndarray:
    """Collapse 12-class CWM labels onto the 6-class set."""
    lut = np.array([CWM_TO_COARSE6[c] for c in range(NUM_CLASSES)], dtype=np.uint8)
    return lut[np.clip(np.asarray(cwm_labels), 0, NUM_CLASSES - 1)]


# ------------------------------------------------------- open-vocab prompting


@dataclass(frozen=True)
class ConceptPrompt:
    """One text prompt for an open-vocabulary detector, and where it lands."""

    phrase: str
    cwm_class: int


# Classes the closed-set checkpoints genuinely cannot deliver. ADE20K collapses
# every creature into one "animal" class and has no notion of the loose
# world-clutter that CWM calls "prop"; Cityscapes has neither. These prompts are
# what SAM 3 is actually needed for.
SAM3_PROMPTS: tuple[ConceptPrompt, ...] = (
    ConceptPrompt("horse", ANIMAL),
    ConceptPrompt("dog", ANIMAL),
    ConceptPrompt("cow", ANIMAL),
    ConceptPrompt("deer", ANIMAL),
    ConceptPrompt("bird", ANIMAL),
    ConceptPrompt("trash can", PROP),
    ConceptPrompt("wooden crate", PROP),
    ConceptPrompt("barrel", PROP),
    ConceptPrompt("cardboard box", PROP),
    ConceptPrompt("bench", PROP),
    ConceptPrompt("suitcase", PROP),
    ConceptPrompt("backpack", PROP),
    ConceptPrompt("fire hydrant", INFRASTRUCTURE),
    ConceptPrompt("street sign", INFRASTRUCTURE),
    ConceptPrompt("mailbox", INFRASTRUCTURE),
    ConceptPrompt("utility pole", INFRASTRUCTURE),
    ConceptPrompt("parking meter", INFRASTRUCTURE),
    ConceptPrompt("person", HUMAN),
)


# ------------------------------------------------------------------ LUT build


def build_lut(source_classes: tuple[str, ...], mapping: dict[str, int], *, default: int) -> np.ndarray:
    """Index-aligned lookup table from a source label set to CWM class IDs.

    Raises if `mapping` mentions a class the source label set does not have,
    which is the failure mode that matters: a typo or a checkpoint whose label
    set drifted would otherwise silently route pixels to `default`.
    """
    unknown = set(mapping) - set(source_classes)
    if unknown:
        raise ValueError(f"mapping refers to classes absent from the source label set: {sorted(unknown)}")
    return np.array([mapping.get(name, default) for name in source_classes], dtype=np.uint8)


def ade20k_lut() -> np.ndarray:
    return build_lut(
        ADE20K_CLASSES, ADE20K_TO_CWM, default=PROP if _ADE20K_PROP_FALLBACK else VOID_UNKNOWN
    )


def cityscapes_lut() -> np.ndarray:
    return build_lut(CITYSCAPES_CLASSES, CITYSCAPES_TO_CWM, default=VOID_UNKNOWN)


def apply_lut(labels: np.ndarray, lut: np.ndarray, *, ignore_index: int = 255) -> np.ndarray:
    """Map a source-label image through `lut`, sending `ignore_index` to void."""
    labels = np.asarray(labels)
    out = np.full(labels.shape, VOID_UNKNOWN, dtype=np.uint8)
    known = (labels >= 0) & (labels < len(lut)) & (labels != ignore_index)
    out[known] = lut[labels[known]]
    return out


def overlay(base: np.ndarray, patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Write `patch` into `base` where `mask` holds and the patch outranks base."""
    base = np.asarray(base, dtype=np.uint8).copy()
    priority = np.array([PRIORITY[c] for c in range(NUM_CLASSES)], dtype=np.uint8)
    patch_arr = np.broadcast_to(np.asarray(patch, dtype=np.uint8), base.shape)
    wins = mask & (priority[patch_arr] > priority[base])
    base[wins] = patch_arr[wins]
    return base
