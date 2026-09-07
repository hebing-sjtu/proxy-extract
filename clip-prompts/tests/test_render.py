from dataclasses import replace

from clip_prompts import render


def test_a_continuing_event_is_marked_as_continuing(caption):
    """The defect this format exists to fix.

    Version 3 rendered a four-chunk walk as one sentence four times, with
    nothing to say whether that was one action or four. Here the second an
    action starts reads differently from the seconds it continues.
    """
    lines = render.compile_all(caption)["lean"]["bins"]
    assert lines[1] != lines[2] != lines[3]


def test_an_unchanging_action_does_repeat_after_the_second_continuation(caption):
    """Deliberate: the repetition is true, and the timestamp separates the lines.

    Rotating synonyms here would look better and would teach a model that the
    paraphrase carries information it does not.
    """
    lines = render.compile_all(caption)["lean"]["bins"]
    assert lines[3] == lines[4]


def test_an_event_that_ends_inside_the_clip_reads_as_closing(caption):
    ending = replace(
        caption,
        events=(
            replace(caption.events[1], bins=(1, 2, 3)),
            replace(caption.events[0], id="e2", bins=(4,)),
            caption.events[2],
        ),
    )
    lines = render.compile_all(ending)["lean"]["bins"]
    assert "walks the last of the way forward along the road" in lines[3]


def test_an_event_cut_short_by_a_slice_never_reads_as_closing(long_caption):
    piece = long_caption.slice_to(10.0, 15.0)
    lines = render.compile_all(piece)["lean"]["bins"]
    assert "the last of the way" not in " ".join(lines)


def test_the_first_second_of_an_event_reads_as_an_onset(caption):
    compiled = render.compile_all(caption)
    assert "walks forward along the road" in compiled["lean"]["bins"][1]


def test_later_seconds_read_as_continuing(caption):
    compiled = render.compile_all(caption)
    assert "keeps walking" in compiled["lean"]["bins"][2]
    assert "is still walking" in compiled["lean"]["bins"][3]


def test_a_sliced_caption_still_compiles(long_caption):
    piece = long_caption.slice_to(20.0, 25.0)
    lines = render.compile_all(piece)["timed"]["bins"]
    assert [row["t"] for row in lines] == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
    assert all(row["text"] for row in lines)


def test_the_first_mention_describes_the_entity_and_later_ones_do_not(caption):
    compiled = render.compile_all(caption)
    assert "dark green jacket" in compiled["lean"]["bins"][0]
    assert "dark green jacket" not in compiled["lean"]["bins"][1]
    assert compiled["lean"]["bins"][1].startswith("The man walks")


def test_rich_uses_the_long_fields(caption):
    compiled = render.compile_all(caption)
    assert "asphalt roadway toward an intersection" in compiled["rich"]["bins"][1]
    assert "asphalt roadway toward an intersection" not in compiled["lean"]["bins"][1]


def test_every_bin_gets_a_line(caption):
    compiled = render.compile_all(caption)
    assert len(compiled["timed"]["bins"]) == caption.grid.count
    assert all(row["text"] for row in compiled["timed"]["bins"])


def test_the_script_uses_the_marker_the_released_examples_use(caption):
    """`[0.00s-5.17s]` is upstream's own window marker, two decimals and all."""
    lines = render.compile_all(caption)["timed"]["script"].splitlines()
    assert lines[0].startswith("[0.00s-1.00s]")
    assert lines[-1].startswith("[4.00s-5.17s]")


def test_the_stamps_are_absolute_when_the_window_starts_late(long_caption):

    piece = long_caption.slice_to(10.0, 15.0)
    compiled = render.compile_all(piece)
    assert compiled["timed"]["offset"] == 10.0
    assert compiled["timed"]["script"].splitlines()[0].startswith("[10.00s-11.00s]")
    assert compiled["timed"]["bins"][0]["t"] == [0.0, 1.0]


def test_the_global_paragraph_names_the_scene_and_the_actions(caption):
    text = render.compile_all(caption)["lean"]["global"]
    assert text.startswith("Third-person open-world video game.")
    assert "city street" in text.lower()
    assert "stands" in text and "walks" in text


def test_facing_is_reported_as_a_token(caption):
    assert render.compile_all(caption)["timed"]["facing"] == "Opening screen facing: back."


def test_the_camera_gets_its_own_clause(caption):
    line = render.compile_all(caption)["lean"]["bins"][0]
    assert "The camera follows from behind at shoulder height." in line


def test_an_unknown_verb_still_renders(caption):
    odd = replace(
        caption,
        events=tuple(
            replace(e, verb="parkours") if e.id == "e1" else e for e in caption.events
        ),
    )
    assert "moves forward along the road" in render.compile_all(odd)["lean"]["bins"][1]


def test_the_marker_format_is_a_parameter(caption):
    timed = render.compile_all(caption)["timed"]["bins"]
    assert render.script(timed, marker="<t={start:g}>").splitlines()[1].startswith("<t=1>")


def test_an_action_already_under_way_is_not_phrased_as_beginning(long_caption):
    piece = long_caption.slice_to(20.0, 25.0)
    assert piece.compiled == {}
    lines = render.compile_all(piece)["lean"]["bins"]
    assert lines[0].startswith("A man in a dark green jacket is already walking")
