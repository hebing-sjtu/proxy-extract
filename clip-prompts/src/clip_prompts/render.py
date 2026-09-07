"""Structure to text: the compiler that turns a caption into training prompts.

Everything this writes is a pure function of the caption, so a change of
training format is `captions-recompile` over the corpus rather than ten
thousand more API calls. That is the whole reason the captioner returns
structure. `contract.COMPILER_VERSION` records which version of these rules a
file's text came from.

Three renderings come out, and they are not three verbosities of one string:

`lean` and `rich` are the same events at two levels of detail, for mixing short
and long conditioning in one training set. `timed` is the one this format
exists for - one line per second, each stamped with the interval it covers.

The interesting rule is `_form`. An event spanning four bins used to be
rendered as its sentence, four times, with nothing to say whether that was one
action continuing or four separate ones. Here the first bin of an event reads
"walks steadily along the roadway", the next "keeps walking steadily along the
roadway", and the second it ends on "walks the last of the way steadily along
the roadway". Same fact, marked with where in the action this second falls,
which is what a reader and a loss function both need. No extra tokens are
bought from the captioner to do it: the split between `verb` and `phrase` is
what makes it a conjugation problem.

An action that genuinely does not change for ten seconds still produces
repeated middle lines, and that is deliberate. The repetition is true, and the
timestamp on each line is what distinguishes them. Rotating synonyms to make
the text look varied would teach a model that the paraphrase is the signal.
"""

from __future__ import annotations

from . import conditioning
from .contract import Caption, Entity, Event
from .vocab import verb as conjugate

CAMERA = "The camera"


def _sentence(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _join(parts, separator=" ") -> str:
    return separator.join(part for part in parts if part)


def _article(kind: str) -> str:
    return "an" if kind[:1].lower() in "aeiou" else "a"


def full_name(entity: Entity, *, rich: bool) -> str:
    """First mention: the entity described. Falls back to its kind alone."""
    look = (entity.look_rich if rich else entity.look) or ""
    if look:
        return f"{_article(look)} {look}" if not look[:1].isupper() else look
    return f"{_article(entity.kind)} {entity.kind}"


def short_name(entity: Entity) -> str:
    """Later mentions. Definite, so coreference is unambiguous in the text."""
    return f"the {entity.kind}"


def _predicate(event: Event, *, rich: bool, form: str) -> str:
    """The verb phrase for one event, marked with where in the action it falls."""
    word = conjugate(event.verb, channel=event.channel)
    complement = (event.phrase_rich if rich else event.phrase) or ""
    if form == "start":
        return _join((word.third, complement))
    if form == "already":
        return _join(("is already", word.gerund, complement))
    if form == "again":
        return _join(("keeps", word.gerund, complement))
    if form == "last":
        return _join((word.third, "the last of the way", complement))
    return _join(("is still", word.gerund, complement))


def _form(event: Event, index: int, *, final_bin: int) -> str:
    """Where in the action this second falls, as far as this window can tell.

    The onset and closing forms are only used where the action really does
    begin or end inside the window. One that a slice cut through, or that runs
    to the clip's final bin, is under way at that edge, and phrasing it as a
    start or a finish puts something into the caption the pixels never show.
    """
    position = event.bins.index(index)
    if position == 0:
        return "already" if event.starts_before else "start"
    ends_here = (
        index == event.bins[-1] and not event.continues_after and event.bins[-1] != final_bin
    )
    if ends_here and len(event.bins) >= 3:
        return "last"
    if position == 1:
        return "again"
    return "continue"


def _subject(caption: Caption, event: Event, *, rich: bool, seen: set[str]) -> str:
    if event.channel == "camera":
        return CAMERA
    entity = caption.entity(event.entity)
    if entity is None:
        return "The subject"
    if entity.id in seen:
        return short_name(entity)
    seen.add(entity.id)
    name = full_name(entity, rich=rich)
    return name[0].upper() + name[1:]


def _line(caption: Caption, index: int, *, rich: bool, seen: set[str]) -> str:
    """One second of text: subjects first, then the camera."""
    final = caption.grid.bins[-1].index
    clauses = []
    for event in caption.in_bin(index, channel="subject"):
        subject = _subject(caption, event, rich=rich, seen=seen)
        form = _form(event, index, final_bin=final)
        clauses.append(_sentence(f"{subject} {_predicate(event, rich=rich, form=form)}"))
    for event in caption.in_bin(index, channel="camera"):
        form = _form(event, index, final_bin=final)
        clauses.append(_sentence(f"{CAMERA} {_predicate(event, rich=rich, form=form)}"))
    return _join(clauses)


def _global(caption: Caption, *, rich: bool) -> str:
    """The whole clip in one paragraph, in the order things happen.

    Kept because it is what a model conditioned on a single prompt consumes,
    and because it is the fallback for any trainer that does not want to deal
    with timestamps at all - the per-second lines should be droppable without
    leaving the clip uncaptioned.
    """
    scene = caption.scene
    head = _join(
        (
            _sentence(scene.medium),
            _sentence(scene.environment_rich if rich else scene.environment),
            _sentence(scene.lighting_rich if rich else scene.lighting),
        )
    )
    seen: set[str] = set()
    clauses = []
    # Subjects before the camera within a second: the paragraph reads as what
    # happened and then how it was shot, which is the order the H3 prompt
    # format puts them in as well.
    order = {"subject": 0, "camera": 1}
    for event in sorted(
        caption.events, key=lambda e: (e.bins[0] if e.bins else 0, order.get(e.channel, 2))
    ):
        subject = _subject(caption, event, rich=rich, seen=seen)
        word = conjugate(event.verb, channel=event.channel)
        complement = (event.phrase_rich if rich else event.phrase) or ""
        clauses.append(_join((subject.lower() if clauses else subject, word.third, complement)))
    body = _sentence(", then ".join(clauses)) if clauses else ""
    return _join((head, body))


def facing_line(caption: Caption) -> str:
    """The protagonist's screen-space facing where it first appears.

    Its own line rather than folded into a phrase because it is a token, not
    prose: a trainer can drop it, template it, or key on it, and none of that
    works if it is buried in a sentence.
    """
    protagonist = caption.protagonist
    if protagonist is None:
        return ""
    for event in sorted(caption.events, key=lambda e: (e.bins[0] if e.bins else 0)):
        if event.entity == protagonist.id and event.facing != "unknown":
            return f"Opening screen facing: {event.facing.replace('_', ' ')}."
    return ""


def compile_all(caption: Caption) -> dict:
    """Every rendering, as `prompt.json`'s `compiled` block."""
    out: dict = {}
    for name, rich in (("lean", False), ("rich", True)):
        seen: set[str] = set()
        lines = [_line(caption, item.index, rich=rich, seen=seen) for item in caption.grid.bins]
        out[name] = {"global": _global(caption, rich=rich), "bins": lines}

    seen = set()
    timed = []
    for item in caption.grid.bins:
        timed.append(
            {
                "bin": item.index,
                "t": [item.start, item.stop],
                "text": _line(caption, item.index, rich=False, seen=seen),
            }
        )
    offset = float(caption.window.get("t0") or 0.0)
    out["timed"] = {
        "global": out["lean"]["global"],
        "facing": facing_line(caption),
        "offset": offset,
        "bins": timed,
        "script": script(timed, offset=offset),
    }
    # Its own block, not appended to the others, so a training run can include
    # it, drop it, or mix it in at some rate without recompiling the corpus.
    out["conditioning"] = conditioning.block(caption.evidence)
    return out


# The marker the released H3 examples already use, down to the two decimals:
# `examples/config.multiwindow.example.json` prompts a 124-frame window at
# `[0.00s-5.17s]` and its continuation at `[3.75s-8.92s]`. Matching it exactly
# is what makes a per-second caption an extension of a convention the model was
# trained with rather than a new one it has to be taught from scratch.
UPSTREAM_MARKER = "[{start:.2f}s-{stop:.2f}s]"


def script(timed: list[dict], *, marker: str = UPSTREAM_MARKER, offset: float = 0.0) -> str:
    """The per-second lines as one block of text.

    `offset` shifts the stamps onto the output video's own clock, because the
    upstream markers are absolute: window 1 of a two-window run is labelled
    `[3.75s-8.92s]`, not `[0.00s-5.17s]` again. The structured `bins` stay
    clip-relative, so the offset is applied here and recorded beside the result.

    The marker is a parameter for the same reason the structure is stored: a
    different timestamp syntax for the finetune should be a formatting decision
    at training time, not a re-run of the corpus.
    """
    return "\n".join(
        f"{marker.format(start=row['t'][0] + offset, stop=row['t'][1] + offset)} {row['text']}"
        for row in timed
    )
