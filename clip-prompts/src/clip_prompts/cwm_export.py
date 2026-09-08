"""Projecting a caption onto the one flat string CWM actually feeds Qwen.

Everything else in this package exists to keep the structure honest. This
module throws the structure away: what reaches the model is a window timestamp
and a paragraph of prose, and nothing else. `CWM_TEXT_EXPORT.md` is the
contract; this is its implementation.

Three things are deliberately *not* in the user sentence, because each of them
belongs to a different role and putting it here would spend the wrong tokens:
the `AWM_PROXY_CONTROL` system text, the conditioning card that explains what
the DUV proxy is, and any of the JSON. CWM carries the first two in its system
message, which FastVideo loads separately, and it never sees the third.
"""

from __future__ import annotations

from pathlib import Path

from .contract import Caption

# The same marker `render.UPSTREAM_MARKER` uses, and for the same reason: the
# released examples are written this way, down to the two decimals.
WINDOW_MARKER = "[{start:.2f}s-{stop:.2f}s]"

PROSE_STYLES = ("lean", "rich")
VARIANTS = ("window", "timed")


class ExportError(RuntimeError):
    pass


def system_id(caption: Caption) -> str:
    """`"w0"` for a take that starts at zero, `"wn"` for a Retake34 continuation.

    The trap this avoids is `window.window`, the clip's 0..4 ordinal within its
    episode. Those five clips are independent 124-frame takes, each with its own
    `anchor.png` as frame 0; none of them continues another. Choosing the
    continuation system prompt from that ordinal would mislabel four fifths of
    the corpus as continuations and contradict the pixels.

    A continuation is defined by the clock and only by the clock: `t0 > 0` means
    this window starts partway into an output video someone is already
    generating.
    """
    return "wn" if window_span(caption)[0] > 0 else "w0"


def window_span(caption: Caption) -> tuple[float, float]:
    """The window's `(start, stop)` in the output video's own seconds.

    `window.duration` is authoritative when present; the grid's last bin is the
    fallback, and being clip-relative it needs no offset removed. Both are true
    seconds - the two-decimal rounding happens once, at formatting time, so that
    `5.167` never becomes `5.17` twice by different routes.
    """
    start = float(caption.window.get("t0") or 0.0)
    duration = caption.window.get("duration")
    if duration is None:
        if not caption.grid.bins:
            raise ExportError("caption has neither a window duration nor any bins")
        duration = caption.grid.bins[-1].stop
    return start, start + float(duration)


def _prose(caption: Caption, style: str) -> str:
    if style not in PROSE_STYLES:
        raise ExportError(f"unknown prose style {style!r} (lean|rich)")
    compiled = caption.compiled or {}
    text = ((compiled.get(style) or {}).get("global") or "").strip()
    if not text:
        raise ExportError(
            f"the {style} global sentence is empty, so there is nothing to caption "
            "this window with. Recompile the caption before exporting; a bare "
            "timestamp is worse than no sample."
        )
    return text


def window_user(caption: Caption, *, prose: str = "lean") -> str:
    """The default user sentence: one line, one timestamp, one paragraph."""
    start, stop = window_span(caption)
    marker = WINDOW_MARKER.format(start=start, stop=stop)
    return f"{marker} {_prose(caption, prose)}"


def timed_user(caption: Caption) -> str:
    """The per-second variant, which is the same notation cut finer.

    Taken from `compiled.timed.script` rather than rebuilt, so there is exactly
    one place that decides how a second is worded.
    """
    compiled = caption.compiled or {}
    script = ((compiled.get("timed") or {}).get("script") or "").strip()
    if not script:
        raise ExportError("the timed script is empty; recompile the caption before exporting")
    return script


def canonical_caption(value: str) -> str:
    """Line endings exactly as `cwm_h3_inference.config._canonical_caption` makes them.

    The released Qwen caches were encoded from bytes whose internal newlines are
    CRLF. An editor that normalises them to LF changes the token ids, so the
    text that goes to disk is normalised here rather than trusted.
    """
    if not isinstance(value, str) or not value.strip():
        raise ExportError("caption must be a non-empty string")
    normalized = value.strip().replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", "\r\n")


def user_text(caption: Caption, *, variant: str = "window", prose: str = "lean") -> str:
    if variant == "window":
        return window_user(caption, prose=prose)
    if variant == "timed":
        return timed_user(caption)
    raise ExportError(f"unknown variant {variant!r} (window|timed)")


def compile_cwm(caption: Caption, *, variant: str = "window", prose: str = "lean") -> dict:
    """The `compiled.cwm` block.

    `user` and `timed` are stored with LF newlines. The CRLF form is produced on
    the way to disk: escaped CRLF inside JSON is both unreadable and the first
    thing a reformatter eats.
    """
    start, stop = window_span(caption)
    return {
        "variant": variant,
        "system": system_id(caption),
        "t": [start, stop],
        "user": user_text(caption, variant=variant, prose=prose),
        "timed": timed_user(caption),
    }


def has_failures(caption: Caption) -> bool:
    return bool((caption.checks or {}).get("fail"))


def should_write_txt(caption: Caption, *, keep_failed: bool = False) -> bool:
    """Whether this caption may become a training sample.

    A `fail` is the verifier having caught the model describing something the
    DUV says is not there. Writing that to `prompt.txt` is how a hallucination
    becomes a gradient, so the default is to leave the clip without a caption
    and let the audit count it.
    """
    return keep_failed or not has_failures(caption)


def write_prompt_txt(path: Path, user: str) -> Path:
    """Write the user sentence as CWM would read it: UTF-8, CRLF, no BOM, no trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_caption(user).encode("utf-8"))
    return path
