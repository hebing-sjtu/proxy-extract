"""What the captioner is asked, and the shape of the reply it may give.

Four rules govern everything here, three of them bought by the H3 pipeline in
the next directory over.

**Code owns structure; the model owns observation.** The reply is a small JSON
document of facts. Headings, conjugation, ordering, the choice between a terse
and a long rendering, the timestamps themselves - all of that is
`render.py`'s. A model asked to produce finished prose produces different prose
each call, and a corpus captioned that way cannot be re-rendered when the
training format changes.

**The model chooses bin indices, never clock times.** It is asked to pick from
an enumerated list it can see, and the seconds are filled in afterwards from
the grid. Language models will emit `00:03.4` for anything you like; that is a
plausible number, not a measurement.

**Facing is named in screen space, from a closed list.** The single most
expensive error in the sibling pipeline was "facing forward" meaning opposite
things to the writer and the reader.

**The segmentation speaks first.** The per-second table from `evidence.py` is
in the prompt before the questions are. It fixes the counts the model would
otherwise fill in from expectation - a street implies parked cars whether or
not this street has any.
"""

from __future__ import annotations

from . import vocab
from .timeline import Grid

SYSTEM = """\
You annotate gameplay video for training a video generation model.
You return one JSON object and nothing else: no prose, no markdown fences.
Every claim you make must be something visible in the attached video.
"""

REPLY_SCHEMA = """\
{
  "scene": {
    "medium": "<=6 words, e.g. 'third-person open-world video game'",
    "environment": "<=12 words, the place",
    "environment_rich": "20-40 words, the place with what is actually in it",
    "lighting": "<=8 words",
    "lighting_rich": "15-30 words: time of day, sky, shadow direction and hardness"
  },
  "entities": [
    {
      "id": "char_0",
      "kind": "one of the KINDS list",
      "look": "<=15 words, no camera or action language",
      "look_rich": "25-45 words, wardrobe/build/hair/carried items",
      "protagonist": true,
      "bins": [0, 1, 2]
    }
  ],
  "events": [
    {
      "id": "e0",
      "entity": "char_0",
      "verb": "one of the SUBJECT VERBS list",
      "phrase": "<=10 words, the predicate WITHOUT the verb",
      "phrase_rich": "15-30 words, same action, more of what is around it",
      "facing": "one of the FACING list",
      "target": null,
      "bins": [0, 1],
      "confidence": 0.0-1.0
    }
  ],
  "camera": [
    {
      "id": "c0",
      "verb": "one of the CAMERA VERBS list",
      "phrase": "<=10 words, the predicate WITHOUT the verb",
      "phrase_rich": "15-30 words",
      "bins": [0, 1, 2],
      "confidence": 0.0-1.0
    }
  ],
  "quality": {"usable": true, "issues": []}
}
"""


def _bin_table(grid: Grid) -> str:
    lines = []
    for item in grid.bins:
        note = "" if abs(item.seconds - grid.bin_seconds) < 1e-6 else f"   ({item.seconds:g}s long)"
        lines.append(f"  bin {item.index:>2} = {item.start:g}s to {item.stop:g}s{note}")
    return "\n".join(lines)


def _list(values) -> str:
    return ", ".join(values)


def instruction(grid: Grid, briefing: str, *, sheet: bool) -> str:
    """The user turn: the clip, the grid, the evidence, and the questions."""
    sheet_line = (
        "\nA contact sheet is attached as well: one frame from each bin, stamped with "
        "that bin's index and time range. Use it to place events on the grid; use the "
        "video for how things move.\n"
        if sheet
        else ""
    )
    return f"""\
THE CLIP
{grid.frames} frames at {grid.fps:g} fps, {grid.duration:g} seconds of third-person
open-world gameplay footage. It is photographic game capture, not a segmentation map.
{sheet_line}
THE GRID
The clip is cut into {grid.count} one-second bins. Bin indices are the only way to
refer to time in your reply. Never write a clock time.

{_bin_table(grid)}

WHAT THE SEGMENTATION ALREADY KNOWS
These lines are measured from the clip's own per-pixel class and depth channels,
not guessed. They are more reliable than your reading of the frames for counting
people and vehicles, and for whether the protagonist is on screen.

{briefing}

Use them. If a line says there is no vehicle in a bin, do not describe one. If it
says two other people, do not describe five and do not describe none. If you can
see something the table does not report, you may still describe it, but give that
event a confidence at or below 0.5.

WHAT TO RETURN

scene - what holds for the whole clip. Two lengths of each field, written together
so they agree. The short form is for terse training captions and the long form for
detailed ones; the long form must not contradict the short one.

entities - every person, animal or vehicle the events refer to, plus the
protagonist. Exactly one entity may have "protagonist": true, and only if the
table reports the protagonist visible in at least one bin. "bins" is where the
entity is ON SCREEN, which is not the same as where it is doing something.
Give an entity a stable id and reuse it; do not create a second id for the same
person after they pass behind something.
KINDS: {_list(vocab.ENTITY_KINDS)}

events - what the entities do. One event per continuous action, spanning all the
bins that action covers. Do not emit one event per bin for an action that simply
continues; a walk across four seconds is one event with four bins, and the text
generator will phrase the continuation itself.
Start a new event when the action genuinely changes: a stop, a turn, a change of
pace or direction. Every bin must be covered by at least one event.
SUBJECT VERBS: {_list(sorted(vocab.SUBJECT_VERBS))}

camera - what the camera does, under the same rules. Every bin must be covered by
at least one camera event. If the camera is locked to the character's back and
neither is turning, that is "hold" or "follow", not "track".
CAMERA VERBS: {_list(sorted(vocab.CAMERA_VERBS))}

quality - set "usable" false and list the reasons if the clip is too dark, too
blurred, a menu or cutscene, or otherwise not something to train on.

HOW TO WRITE A PHRASE
"verb" is the bare lemma. "phrase" is the rest of the predicate with the verb
removed and with no subject. Correct: verb "walk", phrase "steadily along the
asphalt roadway toward an intersection". Wrong: "the man walks steadily along
the roadway", "walks steadily along the roadway".

FACING IS SCREEN-SPACE
Name where the character's body points relative to the camera, not where they are
going in the world. "back" means you are looking at the nape and they are moving
into the scene - the usual case behind a follow camera. "front" means the face is
toward the camera. Never write "forward": it reads as both.
FACING: {_list(vocab.FACING)}

CONFIDENCE
1.0 is "the frames show this". 0.5 is "this is the most likely reading". Below 0.3
is a guess, and a guess is worth less than an omission - leave it out instead.

Return one JSON object shaped exactly like this, and nothing else:
{REPLY_SCHEMA}
"""


def repair(problems: list[str]) -> str:
    """Follow-up turn naming everything that was wrong with the last reply.

    All of it at once: a repair loop that fixes one complaint per round trip
    costs as many calls as the corpus has defects, and the model has the whole
    reply in context either way.
    """
    listing = "\n".join(f"  - {line}" for line in problems)
    return f"""\
Your previous reply was rejected. Fix all of these and return the whole JSON
object again, in the same shape. Do not explain.

{listing}
"""
