"""What the control video means, said once for the corpus and once per clip.

H3 sees the DUV proxy alongside the text, and the text has to say what that
stream is: it is not footage, its red channel is a depth code and its green and
blue are a class palette. Without that, "follow the video" invites the model to
render the palette.

The split here is the whole point. **The encoding is one fact about the whole
corpus, so it is stored once, in this module, and each `prompt.json` keeps only
its id.** Writing the paragraph into 9,985 files would be 9,985 copies of one
sentence to keep in step, and - worse for training - a constant prefix on every
sample carries no information while costing tokens on all of them, then becomes
an incantation the finetuned model needs at inference and breaks without.

**What the clip's control video actually contains is not constant**, and that
part is compiled per clip, off the evidence table, at no extra cost. "Road,
vegetation and one protagonist, no vehicles" differs between clips, so it is
worth its tokens.

Both are kept out of `compiled.lean` and `compiled.rich` rather than glued onto
them, so a training run can include the conditioning line, drop it, or mix it
in at some rate without recompiling the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

# A group has to cover this much of the frame before it is worth naming. Below
# it the class is a few pixels at the horizon, and listing it would make every
# clip claim the same nine things.
PRESENT_FRACTION = 0.01


@dataclass(frozen=True)
class Card:
    """One control stream's meaning, versioned so a change is visible."""

    id: str
    stream: str
    text: str


DUV_ABOT_V1 = Card(
    id="duv.abot.v1",
    stream="proxy/duv.mp4",
    text=(
        "The control video is not photography. Its red channel is a logarithmic "
        "depth code running from 0.1 m to 8000 m, with 255 reserved for sky and "
        "for depth that could not be solved; its green and blue encode a class "
        "palette. One control pixel covers a 4x4 block of the target frame. "
        "Take layout, depth ordering and the position of every class from it, "
        "and take nothing from its colours or its flat fills."
    ),
)

CARDS = {card.id: card for card in (DUV_ABOT_V1,)}

DEFAULT_CARD = DUV_ABOT_V1.id

# Names as they read in a sentence, in the order a viewer would list them.
_GROUPS = (
    ("road_frac", "road surface"),
    ("vegetation_frac", "vegetation"),
    ("static_frac", "static structure"),
    ("sky_frac", "sky"),
)


def card(ident: str = DEFAULT_CARD) -> Card:
    if ident not in CARDS:
        raise KeyError(f"unknown conditioning card {ident!r}; have {sorted(CARDS)}")
    return CARDS[ident]


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def contents(evidence: dict) -> str:
    """One sentence naming what this clip's control video actually holds.

    Taken over the whole clip rather than per second: this describes the
    stream, and a stream whose contents changed halfway is still a stream that
    contains both things.
    """
    rows = evidence.get("per_bin") or ()
    if not rows:
        return ""

    named: list[str] = []
    for key, label in _GROUPS:
        share = max(float(row.get(key) or 0.0) for row in rows)
        if share >= PRESENT_FRACTION:
            named.append(label)

    people = max(int(row.get("peds") or 0) for row in rows)
    has_player = any(row.get("player") for row in rows)
    if has_player and evidence.get("hero_resolved", True):
        named.append("one protagonist")
    if people:
        named.append(_plural(people, "bystander"))

    if any(row.get("ego_vehicle") for row in rows):
        named.append("the vehicle the camera rides in")
    if any(row.get("vehicle") for row in rows):
        named.append("traffic")

    if not named:
        return ""
    listing = ", ".join(named[:-1]) + (" and " + named[-1] if len(named) > 1 else named[0])
    tail = "" if any(row.get("vehicle") or row.get("ego_vehicle") for row in rows) else ", and no vehicles"
    return f"In this clip the control video holds {listing}{tail}."


def block(evidence: dict, *, ident: str = DEFAULT_CARD) -> dict:
    """The `compiled.conditioning` block: a pointer, and the per-clip sentence."""
    return {"card": ident, "stream": card(ident).stream, "contents": contents(evidence)}


def full_text(compiled_block: dict) -> str:
    """Pointer plus stored sentence, resolved back into the paragraph to train on.

    The constant half is fetched from `CARDS` here rather than read out of the
    file, which is what keeps one edit to the wording from having to touch
    every clip in the corpus.
    """
    ident = compiled_block.get("card") or DEFAULT_CARD
    return " ".join(part for part in (card(ident).text, compiled_block.get("contents")) if part)
