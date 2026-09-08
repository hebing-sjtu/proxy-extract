"""The `timeline` caption contract, version 4, and its `prompt.json` form.

This is the successor to the `scene` contract version 3. It keeps that
format's best idea - the captioner returns *structure* and code renders the
*text*, so a phrasing change is a recompile rather than 10,000 API calls - and
changes four things that made version 3 hard to train on.

**Time replaced chunk indices.** Version 3 events carried `chunks: [1, 2, 3]`
against a four-way split of whatever the clip happened to be long. Here every
event carries `bins`, and `timeline.Grid` fixes a bin at one second for every
clip in the corpus. `t` is written alongside for readers that want seconds,
but it is derived, never authored: asking a language model for float seconds
gets you plausible-looking numbers, whereas asking it to choose from an
enumerated list of one-second bins gets you an index it can point at.

**The camera became a channel.** Version 3 had no way to say the camera did
anything, which for a third-person open-world corpus leaves out the largest
motion in most frames. Camera events live in the same list under
`channel: "camera"` so they bin, slice and render by the same rules.

**Claims became checkable.** `evidence` holds what the DUV proxy says about
each second - who is on screen, how much sky, how far the world is, how fast
it is moving - computed with no model call at all. `checks` holds what
disagreed. Neither is prose for a trainer to read; they are what lets a corpus
be filtered instead of trusted.

**The record is closed under slicing.** `slice_to` cuts a caption down to a
sub-window and returns a valid caption of that window: bins re-based to zero,
events intersected and flagged `truncated` where they ran past the cut,
entities that never appear dropped. That is the property that lets one pass
over a 60 s episode supply the captions for every training window cut out of
it, instead of one VLM call per window.

On disk, one of these sits at `<clip>/annotations/prompt.json` - beside the
`action.json` and `caption.json` the corpus shipped, because it is an
annotation of that clip and belongs where a reader already looks for them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import timeline, vocab

CONTRACT = "timeline"
VERSION = 4

# Bumped when `render.py` would produce different text from the same structure.
# Stored so a corpus can be recompiled selectively: the compiler is cheap, the
# captioner is not, and without this there is no way to tell which clips hold
# text from which compiler.
# 2 adds `compiled.cwm`, the flat user sentence CWM feeds Qwen. The lean, rich
# and timed renderings are byte-identical to version 1, but the set of texts a
# consumer can expect changed, so the number moves. `captions-recompile`
# backfills it from structure already on disk; no clip needs the VLM again.
COMPILER_VERSION = 2

PROMPT_NAME = "prompt.json"


class ContractError(ValueError):
    pass


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _bins(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, int):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ContractError(f"bins must be a list of integers, got {raw!r}")
    out = sorted({int(v) for v in raw})
    if not out:
        raise ContractError("bins must not be empty")
    return tuple(out)


@dataclass(frozen=True)
class Scene:
    """What holds for the whole clip. Two lengths of each, authored together.

    Version 3 produced the short form first and a separate `enrichment` pass
    rewrote it long, which meant the two could disagree about what the scene
    was. Asking for both in one reply costs a few hundred tokens and removes
    that class of contradiction.
    """

    medium: str = ""
    environment: str = ""
    environment_rich: str = ""
    lighting: str = ""
    lighting_rich: str = ""

    def as_dict(self) -> dict:
        return {
            "medium": self.medium,
            "environment": self.environment,
            "environment_rich": self.environment_rich,
            "lighting": self.lighting,
            "lighting_rich": self.lighting_rich,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Scene:
        return cls(
            medium=_clean(data.get("medium")),
            environment=_clean(data.get("environment")),
            environment_rich=_clean(data.get("environment_rich")) or _clean(data.get("environment")),
            lighting=_clean(data.get("lighting")),
            lighting_rich=_clean(data.get("lighting_rich")) or _clean(data.get("lighting")),
        )


@dataclass(frozen=True)
class Entity:
    """Someone or something the events can refer to.

    `bins` is presence, not action: the seconds this entity is on screen at
    all. Slicing needs it - an entity that only appears in the last ten
    seconds of a minute must not survive into a caption of the first five -
    and it is the field the DUV can most directly contradict.
    """

    id: str
    kind: str = "other"
    look: str = ""
    look_rich: str = ""
    protagonist: bool = False
    bins: tuple[int, ...] = ()

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "look": self.look,
            "look_rich": self.look_rich,
            "protagonist": self.protagonist,
            "bins": list(self.bins),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Entity:
        ident = _clean(data.get("id"))
        if not ident:
            raise ContractError("entity is missing an id")
        look = _clean(data.get("look"))
        return cls(
            id=ident,
            kind=_clean(data.get("kind")).lower() or "other",
            look=look,
            look_rich=_clean(data.get("look_rich")) or look,
            protagonist=bool(data.get("protagonist")),
            bins=_bins(data.get("bins", [])) if data.get("bins") else (),
        )


@dataclass(frozen=True)
class Event:
    """One continuous action over a contiguous-ish set of one-second bins.

    `phrase` is the predicate with the verb removed - "steadily along the
    asphalt roadway", not "walks steadily along the asphalt roadway" - so that
    `render.py` can conjugate it differently for the second an action starts
    and the seconds it continues. See `vocab`.

    `t` mirrors `bins` in seconds and is filled in by `bind`, never by the
    captioner.
    """

    id: str
    channel: str
    verb: str
    phrase: str = ""
    phrase_rich: str = ""
    entity: str | None = None
    target: str | None = None
    facing: str = "unknown"
    bins: tuple[int, ...] = ()
    t: tuple[float, float] | None = None
    confidence: float = 1.0
    # Set by slicing, never by the captioner. Two flags rather than one
    # `truncated`, because the renderer needs to tell them apart: an action
    # already under way when the window opens must not be phrased as
    # beginning, and one still going when it closes must not be phrased as
    # finishing. A single flag suppresses both and gets one of them wrong.
    starts_before: bool = False
    continues_after: bool = False

    @property
    def truncated(self) -> bool:
        return self.starts_before or self.continues_after

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "channel": self.channel,
            "entity": self.entity,
            "verb": self.verb,
            "phrase": self.phrase,
            "phrase_rich": self.phrase_rich,
            "target": self.target,
            "facing": self.facing,
            "bins": list(self.bins),
            "t": list(self.t) if self.t else None,
            "confidence": self.confidence,
            "starts_before": self.starts_before,
            "continues_after": self.continues_after,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Event:
        ident = _clean(data.get("id"))
        if not ident:
            raise ContractError("event is missing an id")
        channel = _clean(data.get("channel")).lower() or "subject"
        verb = _clean(data.get("verb")).lower() or vocab.FALLBACK_VERB
        phrase = vocab.strip_leading_verb(_clean(data.get("phrase")), verb, channel=channel)
        rich = vocab.strip_leading_verb(_clean(data.get("phrase_rich")), verb, channel=channel)
        facing = _clean(data.get("facing")).lower() or "unknown"
        raw_t = data.get("t")
        return cls(
            id=ident,
            channel=channel,
            verb=verb,
            phrase=phrase,
            phrase_rich=rich or phrase,
            entity=_clean(data.get("entity")) or None,
            target=_clean(data.get("target")) or None,
            facing=facing if facing in vocab.FACING else "unknown",
            bins=_bins(data.get("bins", [])) if data.get("bins") else (),
            t=(float(raw_t[0]), float(raw_t[1])) if raw_t else None,
            confidence=float(data.get("confidence", 1.0)),
            starts_before=bool(data.get("starts_before")),
            continues_after=bool(data.get("continues_after")),
        )


@dataclass(frozen=True)
class Caption:
    """A whole `prompt.json`."""

    grid: timeline.Grid
    scene: Scene
    entities: tuple[Entity, ...]
    events: tuple[Event, ...]
    window: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    compiled: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    # -------------------------------------------------------------- queries

    def entity(self, ident: str | None) -> Entity | None:
        if ident is None:
            return None
        for item in self.entities:
            if item.id == ident:
                return item
        return None

    @property
    def protagonist(self) -> Entity | None:
        for item in self.entities:
            if item.protagonist:
                return item
        return None

    def channel(self, name: str) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.channel == name)

    def in_bin(self, index: int, *, channel: str | None = None) -> tuple[Event, ...]:
        return tuple(
            e
            for e in self.events
            if index in e.bins and (channel is None or e.channel == channel)
        )

    # ---------------------------------------------------------------- edits

    def bind(self) -> Caption:
        """Fill every event's `t` from its bins and the grid.

        Separate from construction because the grid can change under a
        caption - that is what slicing is - and the seconds have to follow.
        """
        events = tuple(
            replace(event, t=self.grid.span(event.bins)) if event.bins else event
            for event in self.events
        )
        return replace(self, events=events)

    def slice_to(self, start: float, stop: float) -> Caption:
        """The caption of the sub-window [start, stop), based at zero.

        `starts_before` and `continues_after` mark an event the cut ran
        through. A trainer that wants only self-contained actions can filter
        on them; the renderer uses them to avoid claiming that an action began
        or finished inside a window that only saw its middle. Losing those
        bits is how a sliced caption comes to assert something the window does
        not show.
        """
        cut = timeline.cut(self.grid, start, stop)
        keep = set(cut.kept)

        events: list[Event] = []
        for event in self.events:
            inside = tuple(b for b in event.bins if b in keep)
            if not inside:
                continue
            events.append(
                replace(
                    event,
                    bins=tuple(cut.rebase(b) for b in inside),
                    starts_before=event.starts_before or inside[0] != event.bins[0],
                    continues_after=event.continues_after or inside[-1] != event.bins[-1],
                    t=None,
                )
            )

        referenced = {e.entity for e in events} | {e.target for e in events}
        entities = []
        for item in self.entities:
            present = tuple(cut.rebase(b) for b in item.bins if b in keep)
            if not present and item.id not in referenced:
                continue
            entities.append(replace(item, bins=present))

        window = dict(self.window)
        window["t0"] = round(float(window.get("t0", 0.0)) + cut.offset, timeline.TIME_PLACES)
        window["sliced_from"] = window.get("clip")
        window["frames"] = cut.grid.frames
        window["duration"] = cut.grid.duration
        # The source ordinals named the whole clip's frames and now name the
        # wrong ones. Dropped rather than adjusted: the mapping is the report's
        # full per-frame list, not the two endpoints kept here, so `frame_offset`
        # is what a reader needs to go and index it correctly.
        window["frame_offset"] = (
            int(window.get("frame_offset", 0)) + self.grid.bins[cut.kept[0]].first_frame
        )
        window.pop("source_ordinals", None)

        evidence = _slice_evidence(self.evidence, cut)
        return replace(
            self,
            grid=cut.grid,
            entities=tuple(entities),
            events=tuple(events),
            window=window,
            evidence=evidence,
            checks={},
            compiled={},
        ).bind()

    # ----------------------------------------------------------- validation

    def problems(self) -> list[str]:
        """Everything structurally wrong with this caption.

        Structure only - whether the captioner told the truth is `verify`'s
        question. Returned as a list rather than raised one at a time because
        a repair prompt that fixes four things at once costs one round trip.
        """
        found: list[str] = []
        n = self.grid.count
        ids: set[str] = set()

        for item in self.entities:
            if item.id in ids:
                found.append(f"duplicate entity id {item.id!r}")
            ids.add(item.id)
            if item.kind not in vocab.ENTITY_KINDS:
                found.append(f"entity {item.id!r} has unknown kind {item.kind!r}")
            for b in item.bins:
                if not 0 <= b < n:
                    found.append(f"entity {item.id!r} names bin {b}, outside 0..{n - 1}")

        if sum(1 for item in self.entities if item.protagonist) > 1:
            found.append("more than one entity is marked protagonist")

        event_ids: set[str] = set()
        for event in self.events:
            if event.id in event_ids:
                found.append(f"duplicate event id {event.id!r}")
            event_ids.add(event.id)
            if event.channel not in vocab.CHANNELS:
                found.append(f"event {event.id!r} has unknown channel {event.channel!r}")
            if not event.bins:
                found.append(f"event {event.id!r} names no bins")
            for b in event.bins:
                if not 0 <= b < n:
                    found.append(f"event {event.id!r} names bin {b}, outside 0..{n - 1}")
            if event.channel == "subject":
                if event.entity is None:
                    found.append(f"subject event {event.id!r} has no entity")
                elif event.entity not in ids:
                    found.append(f"event {event.id!r} refers to unknown entity {event.entity!r}")
            if event.target is not None and event.target not in ids:
                found.append(f"event {event.id!r} targets unknown entity {event.target!r}")
            if not 0.0 <= event.confidence <= 1.0:
                found.append(f"event {event.id!r} has confidence {event.confidence}")

        for index in range(n):
            if not self.in_bin(index, channel="subject"):
                found.append(f"bin {index} has no subject event")
            if not self.in_bin(index, channel="camera"):
                found.append(f"bin {index} has no camera event")

        if not self.scene.medium:
            found.append("scene.medium is empty")
        return found

    def validate(self) -> Caption:
        found = self.problems()
        if found:
            raise ContractError("; ".join(found))
        return self

    # ------------------------------------------------------------ transport

    def as_dict(self) -> dict:
        return {
            "contract": CONTRACT,
            "version": VERSION,
            "compiler": COMPILER_VERSION,
            "window": self.window,
            "timeline": self.grid.as_dict(),
            "scene": self.scene.as_dict(),
            "entities": [e.as_dict() for e in self.entities],
            "events": [e.as_dict() for e in self.events],
            "evidence": self.evidence,
            "checks": self.checks,
            "compiled": self.compiled,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Caption:
        if data.get("contract") != CONTRACT:
            raise ContractError(f"not a {CONTRACT} contract: {data.get('contract')!r}")
        if int(data.get("version", 0)) != VERSION:
            raise ContractError(f"contract version {data.get('version')!r}, expected {VERSION}")
        return cls(
            grid=timeline.from_dict(data["timeline"]),
            scene=Scene.from_dict(data.get("scene") or {}),
            entities=tuple(Entity.from_dict(e) for e in data.get("entities") or ()),
            events=tuple(Event.from_dict(e) for e in data.get("events") or ()),
            window=dict(data.get("window") or {}),
            evidence=dict(data.get("evidence") or {}),
            checks=dict(data.get("checks") or {}),
            compiled=dict(data.get("compiled") or {}),
            provenance=dict(data.get("provenance") or {}),
        )

    def write(self, path: Path) -> Path:
        """Write atomically, so a killed run never leaves a half-parsed file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".partial")
        tmp.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp.replace(path)
        return path

    @classmethod
    def read(cls, path: Path) -> Caption:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _slice_evidence(evidence: dict, cut: timeline.Cut) -> dict:
    if not evidence:
        return {}
    keep = set(cut.kept)
    out = dict(evidence)
    rows = evidence.get("per_bin")
    if isinstance(rows, list):
        out["per_bin"] = [
            {**row, "bin": cut.rebase(int(row["bin"]))}
            for row in rows
            if int(row.get("bin", -1)) in keep
        ]
    return out
