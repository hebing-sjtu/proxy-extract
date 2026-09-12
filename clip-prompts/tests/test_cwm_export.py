"""Tests for the projection onto CWM's user sentence.

The contract is `CWM_TEXT_EXPORT.md`. What makes this layer worth testing is
that every way of getting it wrong produces a file that still looks fine: a
timestamp off by a window, a newline that is LF instead of CRLF, a system
prompt that leaked into the user turn. None of those raise; they just quietly
train the model on the wrong tokens.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from clip_prompts import cwm_export, render, timeline
from clip_prompts.contract import COMPILER_VERSION, Caption

EXAMPLE = Path(__file__).resolve().parents[1] / "example" / "prompt.json"

# CWM_TEXT_EXPORT.md section 2.1, quoted rather than constructed. If the
# renderer's wording drifts, this must be updated deliberately, not silently.
GOLDEN = (
    "[0.00s-5.17s] Third-person open-world video game. City street. Bright daylight. "
    "A man in a dark green jacket stands still in the middle of the roadway, then the "
    "camera follows from behind at shoulder height, then the man walks forward along "
    "the road."
)


def _compiled(caption: Caption) -> Caption:
    return replace(caption, compiled=render.compile_all(caption))


@pytest.fixture
def example() -> Caption:
    return _compiled(Caption.read(EXAMPLE))


# --- section 2.1: the golden sentence -------------------------------------


def test_the_example_projects_to_the_golden_sentence_exactly(example):
    assert cwm_export.window_user(example) == GOLDEN


def test_the_golden_sentence_is_one_line_with_no_newline_anywhere(example):
    user = cwm_export.window_user(example)
    assert "\n" not in user and "\r" not in user
    assert user.count("[") == 1


def test_the_stamp_and_the_prose_are_separated_by_exactly_one_space(example):
    user = cwm_export.window_user(example)
    assert user.startswith("[0.00s-5.17s] Third-person")
    assert not user.startswith("[0.00s-5.17s]  ")


# --- section 3.1: the clock, and the ordinal that is not the clock ---------


def test_a_clip_that_starts_at_zero_is_w0_even_when_it_is_the_third_of_its_episode(example):
    """`window.window == 2` names the clip's place in its episode, not a continuation.

    The five clips cut from one episode are independent takes, each with its own
    anchor frame. Reading that ordinal as a CWM window index would label four
    fifths of this corpus as Retake34 continuations.
    """
    assert example.window["window"] == 2
    assert float(example.window["t0"]) == 0.0
    assert cwm_export.system_id(example) == "w0"


def test_only_a_nonzero_start_makes_it_a_continuation(long_caption):
    piece = _compiled(long_caption.slice_to(10.0, 15.0))
    assert cwm_export.system_id(piece) == "wn"


def test_a_sliced_window_is_stamped_on_the_output_videos_clock(long_caption):
    piece = _compiled(long_caption.slice_to(10.0, 15.0))
    assert cwm_export.window_span(piece) == (10.0, 15.0)
    assert cwm_export.window_user(piece).startswith("[10.00s-15.00s] ")


def test_the_per_second_lines_of_a_slice_share_that_clock(long_caption):
    piece = _compiled(long_caption.slice_to(10.0, 15.0))
    lines = cwm_export.timed_user(piece).splitlines()
    assert lines[0].startswith("[10.00s-11.00s]")
    assert lines[-1].startswith("[14.00s-15.00s]")


def test_the_timed_lines_span_exactly_the_window_they_came_from(example):
    """Section 2.2: the union of the fine lines is the window stamp."""
    start, stop = cwm_export.window_span(example)
    lines = cwm_export.timed_user(example).splitlines()
    assert lines[0].startswith(f"[{start:.2f}s-")
    assert lines[-1].startswith("[4.00s-")
    assert lines[-1].split("]")[0].endswith(f"-{stop:.2f}s")


def test_the_duration_is_rounded_once_at_the_very_end(example):
    """5.167 must reach the string as 5.17 by formatting, not by a prior round."""
    assert example.window["duration"] == 5.167
    assert cwm_export.window_span(example)[1] == 5.167
    assert "5.17s]" in cwm_export.window_user(example)


def test_a_window_with_no_recorded_duration_falls_back_to_its_grid(caption):
    caption = _compiled(caption)
    assert "duration" not in caption.window
    assert cwm_export.window_span(caption) == (0.0, caption.grid.bins[-1].stop)
    assert cwm_export.window_user(caption).startswith("[0.00s-5.17s] ")


# --- section 4: the bytes -------------------------------------------------


def test_line_endings_become_crlf_and_stay_there():
    once = cwm_export.canonical_caption("a\nb")
    assert once == "a\r\nb"
    assert cwm_export.canonical_caption(once) == once, "must be idempotent"
    assert cwm_export.canonical_caption("a\r\nb") == "a\r\nb"
    assert cwm_export.canonical_caption("a\rb") == "a\r\nb"


def test_surrounding_whitespace_is_stripped_rather_than_normalised():
    assert cwm_export.canonical_caption("  hi  \n") == "hi"


@pytest.mark.parametrize("value", ["", "   ", "\n", None, 7])
def test_an_empty_caption_is_refused(value):
    with pytest.raises(cwm_export.ExportError):
        cwm_export.canonical_caption(value)


def test_the_window_variant_reaches_disk_with_no_newline_byte(example, tmp_path):
    path = cwm_export.write_prompt_txt(tmp_path / "prompt.txt", cwm_export.window_user(example))
    data = path.read_bytes()
    assert data == cwm_export.canonical_caption(GOLDEN).encode("utf-8")
    assert b"\n" not in data, "the default variant is a single line"
    assert not data.startswith(b"\xef\xbb\xbf"), "no BOM"
    assert data == data.rstrip(), "no trailing whitespace"


def test_every_newline_in_the_timed_variant_is_preceded_by_a_carriage_return(example, tmp_path):
    path = cwm_export.write_prompt_txt(tmp_path / "prompt.txt", cwm_export.timed_user(example))
    data = path.read_bytes()
    assert b"\n" in data
    assert all(data[i - 1 : i] == b"\r" for i, byte in enumerate(data) if byte == 0x0A)
    assert not data.endswith(b"\r\n"), "no trailing line break"


# --- section 5.1: the compiled block --------------------------------------


def test_compile_all_carries_the_cwm_block(caption):
    compiled = render.compile_all(caption)
    block = compiled["cwm"]
    assert block["variant"] == "window"
    assert block["system"] == "w0"
    assert block["user"].startswith("[0.00s-5.17s] ")
    assert block["timed"] == compiled["timed"]["script"]


def test_the_block_stores_lf_so_that_json_stays_readable(caption):
    block = render.compile_all(caption)["cwm"]
    assert "\r" not in block["timed"]
    assert "\n" in block["timed"]


def test_the_compiler_version_moved_because_the_text_set_changed():
    assert COMPILER_VERSION == 2


def test_a_caption_written_now_records_the_new_compiler_version(caption, tmp_path):
    path = tmp_path / "prompt.json"
    _compiled(caption).write(path)
    assert Caption.read(path).compiled["cwm"]["user"].startswith("[0.00s-")
    import json

    assert json.loads(path.read_text())["compiler"] == 2


# --- section 2.3: what must not leak in -----------------------------------


def test_the_user_sentence_carries_no_system_text_and_no_conditioning_card(example):
    user = cwm_export.window_user(example)
    assert "AWM_PROXY_CONTROL" not in user
    assert "logarithmic depth code" not in user
    assert "Video 1" not in user
    # The card exists, and it stays where it is.
    assert example.compiled["conditioning"]


def test_the_facing_line_is_not_appended_by_default(example):
    assert "facing" not in cwm_export.window_user(example).lower()


def test_rich_prose_is_available_but_is_not_the_default(example):
    lean = cwm_export.window_user(example, prose="lean")
    rich = cwm_export.window_user(example, prose="rich")
    assert rich != lean
    assert rich.startswith("[0.00s-5.17s] ")
    assert cwm_export.compile_cwm(example)["user"] == lean


def test_a_bare_timestamp_is_refused_rather_than_written(caption):
    """An empty global would otherwise export as a stamp with nothing after it."""
    hollow = replace(caption, compiled={"lean": {"global": "   "}})
    with pytest.raises(cwm_export.ExportError, match="nothing to caption"):
        cwm_export.window_user(hollow)


def test_an_unknown_variant_or_prose_style_is_refused(example):
    with pytest.raises(cwm_export.ExportError):
        cwm_export.user_text(example, variant="freeform")
    with pytest.raises(cwm_export.ExportError):
        cwm_export.window_user(example, prose="terse")


# --- section 5.2: which clips may become samples --------------------------


def test_a_contradicted_caption_is_not_written_by_default(caption):
    bad = replace(caption, checks={"fail": ["claims a car; the DUV has no vehicle pixels"]})
    assert cwm_export.has_failures(bad)
    assert not cwm_export.should_write_txt(bad)
    assert cwm_export.should_write_txt(bad, keep_failed=True)


def test_a_warning_alone_does_not_withhold_a_caption(caption):
    warned = replace(caption, checks={"fail": [], "warn": ["pedestrian count differs by one"]})
    assert cwm_export.should_write_txt(warned)


def test_prompt_txt_sits_at_the_clip_root_where_fastvideo_looks(tmp_path):
    from clip_prompts import layout

    clip = layout.Clip(tmp_path / "clip_000000_0")
    assert clip.prompt_txt == clip.root / "prompt.txt"
    assert clip.prompt_txt.parent != clip.annotations


def test_discover_finds_gta_seg_directories_without_a_clip_report(tmp_path):
    from clip_prompts import layout

    seg = tmp_path / "seg_000000"
    (seg / "minimax_h3").mkdir(parents=True)
    (seg / "proxy").mkdir()
    (seg / "minimax_h3" / "output.mp4").write_bytes(b"x")
    (seg / "proxy" / "duv.mp4").write_bytes(b"x")
    found = layout.discover(tmp_path)
    assert [clip.name for clip in found] == ["seg_000000"]
    assert found[0].rgb == seg / "minimax_h3" / "output.mp4"
    assert found[0].duv == seg / "proxy" / "duv.mp4"


# --- the contract with FastVideo's manifest builder -----------------------


def _as_fastvideo_reads_it(path: Path) -> str:
    """Exactly what `clip_dir_to_encode_manifest.read_prompt` does to the file."""
    return path.read_text(encoding="utf-8").strip()


def test_the_exported_file_is_what_the_manifest_builder_will_read(example, tmp_path):
    """The consumer lives in another repository, so nothing here would fail if this drifted.

    `read_prompt` takes `<clip>/prompt.txt` in preference to the episode
    caption and rejects the clip if the text is empty, which makes both the
    path and the non-emptiness load-bearing.
    """
    clip = tmp_path / "clip_000414_2"
    clip.mkdir()
    user = cwm_export.window_user(example)
    cwm_export.write_prompt_txt(clip / "prompt.txt", user)

    got = _as_fastvideo_reads_it(clip / "prompt.txt")
    assert got == user
    assert got.startswith("[0.00s-5.17s] ")


def test_reading_the_timed_variant_as_text_silently_drops_its_carriage_returns(example, tmp_path):
    """A caveat worth pinning rather than discovering in a token diff.

    The bytes on disk are CRLF, as section 4 requires, but `read_text` applies
    universal newlines and hands back LF. The default single-line variant has no
    newline at all, so it is unaffected; the timed variant is only correct if
    the training side re-canonicalises before Qwen, which is section 10's third
    step and not something this package can do for it.
    """
    path = tmp_path / "prompt.txt"
    cwm_export.write_prompt_txt(path, cwm_export.timed_user(example))
    assert b"\r\n" in path.read_bytes()
    assert "\r" not in _as_fastvideo_reads_it(path)

    window = tmp_path / "window.txt"
    cwm_export.write_prompt_txt(window, cwm_export.window_user(example))
    assert b"\r" not in window.read_bytes(), "the default variant sidesteps this entirely"


# --- the notation must not fork -------------------------------------------


def test_the_export_marker_is_the_one_the_renderer_already_uses():
    assert cwm_export.WINDOW_MARKER == render.UPSTREAM_MARKER


def test_a_long_take_is_stamped_from_its_own_start(long_caption):
    whole = _compiled(long_caption)
    assert cwm_export.window_span(whole) == (0.0, timeline.plan(1440, 24.0).bins[-1].stop)
    assert cwm_export.system_id(whole) == "w0"
