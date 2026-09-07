from dataclasses import replace

import pytest
from clip_prompts.contract import Caption, ContractError, Entity, Event


def test_a_good_caption_has_no_problems(caption):
    assert caption.problems() == []


def test_bind_fills_seconds_from_bins(caption):
    walk = next(e for e in caption.events if e.id == "e1")
    assert walk.t == (1.0, 5.167)


def test_json_round_trip_is_lossless(caption):
    assert Caption.from_dict(caption.as_dict()).as_dict() == caption.as_dict()


def test_a_bin_with_no_camera_event_is_a_problem(caption):
    stripped = replace(caption, events=tuple(e for e in caption.events if e.channel != "camera"))
    assert any("no camera event" in line for line in stripped.problems())


def test_an_event_naming_a_bin_past_the_end_is_a_problem(caption):
    broken = replace(
        caption,
        events=tuple(
            replace(e, bins=(0, 9)) if e.id == "e0" else e for e in caption.events
        ),
    )
    assert any("outside 0..4" in line for line in broken.problems())


def test_an_event_pointing_at_an_unknown_entity_is_a_problem(caption):
    broken = replace(
        caption,
        events=tuple(
            replace(e, entity="ghost") if e.id == "e0" else e for e in caption.events
        ),
    )
    assert any("unknown entity" in line for line in broken.problems())


def test_validate_raises_with_every_problem_at_once(caption):
    broken = replace(caption, entities=(), events=caption.events)
    with pytest.raises(ContractError) as raised:
        broken.validate()
    assert "unknown entity" in str(raised.value)


def test_from_dict_strips_a_verb_the_model_left_on_the_phrase():
    event = Event.from_dict(
        {"id": "e0", "channel": "subject", "entity": "char_0", "verb": "walk",
         "phrase": "walks steadily along the roadway", "bins": [0]}
    )
    assert event.phrase == "steadily along the roadway"


def test_from_dict_keeps_a_phrase_that_only_looks_like_a_verb():
    event = Event.from_dict(
        {"id": "e0", "channel": "subject", "entity": "char_0", "verb": "walk",
         "phrase": "running water ahead of him", "bins": [0]}
    )
    assert event.phrase == "running water ahead of him"


def test_an_unknown_facing_becomes_unknown_rather_than_a_problem():
    event = Event.from_dict(
        {"id": "e0", "channel": "subject", "entity": "c", "verb": "walk",
         "facing": "forward", "bins": [0]}
    )
    assert event.facing == "unknown"


def test_rich_fields_fall_back_to_the_short_ones():
    entity = Entity.from_dict({"id": "c", "kind": "man", "look": "a man in green", "bins": [0]})
    assert entity.look_rich == "a man in green"


def test_a_wrong_version_is_refused(caption):
    data = caption.as_dict()
    data["version"] = 3
    with pytest.raises(ContractError):
        Caption.from_dict(data)


# ------------------------------------------------------------------ slicing


def test_slicing_rebases_bins_and_times(long_caption):
    piece = long_caption.slice_to(10.0, 15.0)
    assert piece.grid.count == 5
    assert piece.window["t0"] == 10.0
    walk = next(e for e in piece.events if e.id == "e1")
    assert walk.bins == (0, 1, 2, 3, 4)
    assert walk.t == (0.0, 5.0)


def test_slicing_marks_which_edge_it_cut(long_caption):
    piece = long_caption.slice_to(10.0, 15.0)
    walk = next(e for e in piece.events if e.id == "e1")
    assert walk.starts_before is True
    assert walk.continues_after is True
    assert walk.truncated is True


def test_a_window_that_ends_with_the_clip_does_not_claim_more_is_coming(long_caption):
    piece = long_caption.slice_to(55.0, 60.0)
    walk = next(e for e in piece.events if e.id == "e1")
    assert walk.starts_before is True
    assert walk.continues_after is False


def test_slicing_drops_an_event_the_window_never_shows(long_caption):
    piece = long_caption.slice_to(10.0, 15.0)
    assert {e.id for e in piece.events} == {"e1", "c0"}


def test_a_slice_is_itself_a_valid_caption(long_caption):
    for start in (0.0, 10.0, 55.0):
        piece = long_caption.slice_to(start, start + 5.0)
        assert piece.problems() == []


def test_slicing_carries_the_evidence_across(long_caption):
    rows = [{"bin": i, "player": True, "peds": 0} for i in range(60)]
    with_evidence = replace(long_caption, evidence={"source": "duv", "per_bin": rows})
    piece = with_evidence.slice_to(10.0, 13.0)
    assert [row["bin"] for row in piece.evidence["per_bin"]] == [0, 1, 2]


def test_slicing_an_entity_that_never_appears_removes_it(long_caption):
    extra = Entity(id="ped_0", kind="person", look="a woman", bins=(50, 51))
    with_extra = replace(long_caption, entities=(*long_caption.entities, extra))
    piece = with_extra.slice_to(10.0, 15.0)
    assert {e.id for e in piece.entities} == {"char_0"}


def test_slicing_reports_the_frame_offset_and_drops_the_stale_ordinals(long_caption):
    from dataclasses import replace

    based = replace(long_caption, window={"clip": "c", "t0": 0.0, "source_ordinals": [0, 1800]})
    piece = based.slice_to(10.0, 15.0)
    assert piece.window["frame_offset"] == 240
    assert "source_ordinals" not in piece.window
    again = piece.slice_to(1.0, 4.0)
    assert again.window["frame_offset"] == 264
    assert again.window["t0"] == 11.0
