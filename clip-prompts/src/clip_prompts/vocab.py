"""The closed vocabularies a caption is allowed to use.

Two of these earn their place by removing an ambiguity that costs real
training signal, and one by removing a defect in the old format.

**Facing is screen-space.** The sibling H3 pipeline records this as its most
expensive lesson: "facing forward" reads as *toward the camera* to a language
model and as *into the scene* to whoever wrote it, and the two are 180 degrees
apart. A caption that says "walks forward" while the frame shows a nape has
taught the model the opposite of what the pixels show. So facing is a token
from `FACING`, not a phrase, and `unknown` is available so that a captioner
that cannot tell is not forced to guess.

**Verbs are lemmas and phrases are complements.** The captioner returns
`verb="walk"` and `phrase="steadily along the asphalt roadway"`; this module
holds the conjugations and `render.py` assembles them. That split is what lets
a three-second walk read as "walks steadily...", "keeps walking steadily...",
"is still walking steadily..." instead of the same sentence three times, which
is what an event that merely listed the chunks it spanned used to produce. Per
-second text that is identical across seconds teaches a model that the
timestamp carries no information.

Unknown verbs are not fatal. `verb` falls back to `other`, whose conjugation is
generic, and the verifier records a warning - a corpus that keeps hitting the
fallback is telling you the list is too short, which is a thing to notice
rather than a thing to reject 10,000 clips over.
"""

from __future__ import annotations

from dataclasses import dataclass

# Screen-space subject orientation. `back` is the nape, i.e. the character is
# walking away into the scene, which is the common case for a third-person
# follow camera and the one most often mislabelled.
FACING = (
    "back",
    "front",
    "left_profile",
    "right_profile",
    "three_quarter_left",
    "three_quarter_right",
    "unknown",
)

CHANNELS = ("subject", "camera")


@dataclass(frozen=True)
class Verb:
    lemma: str
    third: str
    gerund: str


def _verbs(*rows: tuple[str, str, str]) -> dict[str, Verb]:
    return {lemma: Verb(lemma, third, gerund) for lemma, third, gerund in rows}


SUBJECT_VERBS = _verbs(
    ("stand", "stands", "standing"),
    ("walk", "walks", "walking"),
    ("run", "runs", "running"),
    ("sprint", "sprints", "sprinting"),
    ("jog", "jogs", "jogging"),
    ("turn", "turns", "turning"),
    ("stop", "stops", "stopping"),
    ("crouch", "crouches", "crouching"),
    ("kneel", "kneels", "kneeling"),
    ("sit", "sits", "sitting"),
    ("jump", "jumps", "jumping"),
    ("climb", "climbs", "climbing"),
    ("crawl", "crawls", "crawling"),
    ("swim", "swims", "swimming"),
    ("fall", "falls", "falling"),
    ("land", "lands", "landing"),
    ("drive", "drives", "driving"),
    ("ride", "rides", "riding"),
    ("enter", "enters", "entering"),
    ("exit", "exits", "exiting"),
    ("approach", "approaches", "approaching"),
    ("retreat", "retreats", "retreating"),
    ("follow", "follows", "following"),
    ("look", "looks", "looking"),
    ("aim", "aims", "aiming"),
    ("shoot", "shoots", "shooting"),
    ("swing", "swings", "swinging"),
    ("push", "pushes", "pushing"),
    ("pull", "pulls", "pulling"),
    ("carry", "carries", "carrying"),
    ("open", "opens", "opening"),
    ("close", "closes", "closing"),
    ("wave", "waves", "waving"),
    ("talk", "talks", "talking"),
    ("wander", "wanders", "wandering"),
    ("hover", "hovers", "hovering"),
    ("fly", "flies", "flying"),
    ("other", "moves", "moving"),
)

# Camera verbs name a motion class, not a shot name: "tracking shot" describes
# a whole take, and the point of a per-second grid is that the take can change.
CAMERA_VERBS = _verbs(
    ("hold", "holds", "holding"),
    ("track", "tracks", "tracking"),
    ("follow", "follows", "following"),
    ("orbit", "orbits", "orbiting"),
    ("pan", "pans", "panning"),
    ("tilt", "tilts", "tilting"),
    ("push_in", "pushes in", "pushing in"),
    ("pull_back", "pulls back", "pulling back"),
    ("crane", "cranes", "craning"),
    ("drift", "drifts", "drifting"),
    ("shake", "shakes", "shaking"),
    ("cut", "cuts", "cutting"),
    ("other", "moves", "moving"),
)

# Kinds a caption may give an entity. Deliberately coarse: the DUV can confirm
# these and cannot confirm "police officer", so this is the level at which a
# claim is checkable. Detail belongs in `look`, where it is understood to be
# the captioner's reading rather than a fact the pipeline vouches for.
ENTITY_KINDS = (
    "man",
    "woman",
    "person",
    "child",
    "crowd",
    "animal",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "boat",
    "aircraft",
    "vehicle",
    "other",
)

FALLBACK_VERB = "other"


def verb(lemma: str, *, channel: str = "subject") -> Verb:
    """Look up a conjugation, falling back to `other` for anything unlisted."""
    table = CAMERA_VERBS if channel == "camera" else SUBJECT_VERBS
    return table.get((lemma or "").strip().lower(), table[FALLBACK_VERB])


def known_verb(lemma: str, *, channel: str = "subject") -> bool:
    table = CAMERA_VERBS if channel == "camera" else SUBJECT_VERBS
    return (lemma or "").strip().lower() in table


def strip_leading_verb(phrase: str, lemma: str, *, channel: str = "subject") -> str:
    """Drop a verb the captioner left on the front of a complement.

    The instruction asks for the complement alone, and models return
    "walks steadily along the road" often enough that repairing it here is
    cheaper than a retry. Only an exact leading form is removed, so a phrase
    that genuinely begins with the word - "running water ahead" under the verb
    `walk` - is untouched.
    """
    text = (phrase or "").strip()
    if not text:
        return ""
    form = verb(lemma, channel=channel)
    lowered = text.lower()
    for candidate in (form.third, form.gerund, form.lemma):
        prefix = candidate.lower() + " "
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()
        if lowered == candidate.lower():
            return ""
    return text
