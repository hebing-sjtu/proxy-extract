"""Hold the caption against the evidence, and record where they disagree.

`contract.problems` asks whether a caption is well formed. This asks whether it
is true, as far as anything cheap can tell - and cheap matters, because the
alternative to an automatic check on ten thousand clips is no check on ten
thousand clips.

Nothing here rejects a clip. It classifies:

`fail` is a claim the DUV contradicts outright - a car in a second with no
vehicle pixels, a protagonist in a second with no player pixels. These are
hallucinations, and a training set that keeps them teaches the model to
generate objects the conditioning does not ask for.

`warn` is a disagreement with an innocent explanation. A bystander count off
by one is a blob under the size threshold or two people overlapping; a still
subject in a second where the depth field is moving is a follow camera doing
the moving. Worth counting across a corpus, not worth dropping a clip for.

The split is the point. `captions-audit` sums both, and which one grows tells
you different things: rising `fail` means the prompt is losing its grip on the
evidence, rising `warn` usually means a threshold here needs moving.
"""

from __future__ import annotations

import re

from .contract import Caption

# Words that only make sense if the frame contains a vehicle. Matched against
# the phrase text, which is where a hallucinated object shows up - the `kind`
# field is constrained enough that a model rarely invents one there.
VEHICLE_WORDS = re.compile(
    r"\b(car|cars|truck|trucks|bus|buses|van|vans|taxi|taxis|motorcycle|"
    r"motorbike|scooter|traffic|driving|drives|vehicle|vehicles)\b",
    re.IGNORECASE,
)

CROWD_WORDS = re.compile(
    r"\b(crowd|crowds|pedestrian|pedestrians|passer|passers|bystander|bystanders|"
    r"people|shopper|shoppers)\b",
    re.IGNORECASE,
)

# A bystander count may differ by this much before it is worth reporting. One,
# because the blob threshold and simple occlusion each cost about one person.
PED_TOLERANCE = 1

# Below this the captioner has already said it is guessing, so a disagreement
# is the confidence doing its job rather than a defect.
CHECKED_ABOVE = 0.5

STILL_VERBS = {"stand", "sit", "kneel", "crouch"}
STILL_CAMERA = {"hold"}


def _rows(caption: Caption) -> dict[int, dict]:
    return {int(row["bin"]): row for row in caption.evidence.get("per_bin") or ()}


def check(caption: Caption) -> dict:
    """Every disagreement between a caption and its own evidence."""
    rows = _rows(caption)
    if not rows:
        return {"passed": True, "checked": False, "fail": [], "warn": []}

    hero_known = bool(caption.evidence.get("hero_resolved", True))
    fail: list[str] = []
    warn: list[str] = []

    protagonist = caption.protagonist
    if protagonist is not None and hero_known:
        for index in protagonist.bins:
            row = rows.get(index)
            if row is not None and not row["player"]:
                fail.append(
                    f"entity {protagonist.id!r} is placed in bin {index}, "
                    "where the segmentation finds no protagonist pixels"
                )

    for entity in caption.entities:
        if entity.protagonist or entity.kind not in {"car", "truck", "bus", "motorcycle", "vehicle"}:
            continue
        for index in entity.bins:
            row = rows.get(index)
            if row is not None and not (row["vehicle"] or row["ego_vehicle"]):
                fail.append(
                    f"entity {entity.id!r} is a {entity.kind} in bin {index}, "
                    "where the segmentation finds no vehicle pixels"
                )

    for event in caption.events:
        if event.confidence < CHECKED_ABOVE:
            continue
        text = f"{event.phrase} {event.phrase_rich}"
        if VEHICLE_WORDS.search(text):
            missing = [
                index
                for index in event.bins
                if index in rows and not (rows[index]["vehicle"] or rows[index]["ego_vehicle"])
            ]
            if len(missing) == len(event.bins):
                fail.append(
                    f"event {event.id!r} describes a vehicle, but no bin it covers "
                    f"({', '.join(map(str, event.bins))}) has vehicle pixels"
                )
        if CROWD_WORDS.search(text):
            empty = [index for index in event.bins if index in rows and rows[index]["peds"] == 0]
            if len(empty) == len(event.bins):
                warn.append(
                    f"event {event.id!r} describes other people, but the segmentation "
                    f"finds none in bins {', '.join(map(str, empty))}"
                )

    for index, row in rows.items():
        named = sum(
            1
            for entity in caption.entities
            if index in entity.bins and not entity.protagonist and entity.kind
            in {"man", "woman", "person", "child", "crowd"}
        )
        if abs(named - row["peds"]) > PED_TOLERANCE:
            warn.append(
                f"bin {index} names {named} bystanders, the segmentation finds {row['peds']}"
            )

        subject_still = all(
            event.verb in STILL_VERBS
            for event in caption.in_bin(index, channel="subject")
        )
        camera_still = all(
            event.verb in STILL_CAMERA for event in caption.in_bin(index, channel="camera")
        )
        if subject_still and camera_still and row["world_moving"]:
            warn.append(
                f"bin {index} says nothing moves, but the depth field changes by "
                f"{row['depth_delta_code']:g} codes a frame"
            )

        if caption.scene.medium and "indoor" in caption.scene.medium.lower() and row["sky_frac"] > 0.1:
            warn.append(f"scene is called indoor, but bin {index} is {row['sky_frac']:.0%} sky")

    return {
        "passed": not fail,
        "checked": True,
        "fail": fail,
        "warn": warn,
    }


def score(caption: Caption) -> float:
    """One number for ranking a corpus: mean event confidence, docked for warnings.

    Not a probability and not calibrated. It exists so that `--min-score` can
    take the top of a corpus without anyone having to decide up front which of
    the individual checks matters most.
    """
    if not caption.events:
        return 0.0
    base = sum(event.confidence for event in caption.events) / len(caption.events)
    checks = caption.checks or {}
    if checks.get("fail"):
        return 0.0
    penalty = 0.05 * len(checks.get("warn") or ())
    return round(max(0.0, base - penalty), 3)
