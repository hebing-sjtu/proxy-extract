import pytest
from clip_prompts import conditioning, render


def _evidence(**overrides):
    row = {
        "bin": 0, "player": True, "peds": 0, "vehicle": False, "ego_vehicle": False,
        "road_frac": 0.11, "vegetation_frac": 0.30, "static_frac": 0.34, "sky_frac": 0.21,
    }
    row.update(overrides)
    return {"hero_resolved": True, "per_bin": [row]}


def test_the_encoding_is_stored_once_and_pointed_at(caption):
    """The constant half must not be duplicated into every clip's file.

    9,985 copies of one sentence is 9,985 places to edit, and a prefix that is
    identical in every training sample costs tokens on all of them while
    carrying no information about any of them.
    """
    from dataclasses import replace

    compiled = render.compile_all(replace(caption, evidence=_evidence()))
    stored = compiled["conditioning"]
    assert stored["card"] == "duv.abot.v1"
    assert "logarithmic depth code" not in str(stored)
    assert "logarithmic depth code" in conditioning.full_text(stored)


def test_the_per_clip_half_says_what_this_clip_holds():
    text = conditioning.contents(_evidence(peds=2))
    assert "road surface" in text
    assert "one protagonist" in text
    assert "2 bystanders" in text
    assert "no vehicles" in text


def test_two_clips_with_different_contents_get_different_sentences():
    on_foot = conditioning.contents(_evidence())
    driving = conditioning.contents(_evidence(ego_vehicle=True, vehicle=True))
    assert on_foot != driving
    assert "the vehicle the camera rides in" in driving
    assert "no vehicles" not in driving


def test_a_class_that_is_barely_there_is_not_named():
    assert "sky" not in conditioning.contents(_evidence(sky_frac=0.001))


def test_the_protagonist_is_not_claimed_when_the_tracker_failed():
    facts = _evidence()
    facts["hero_resolved"] = False
    assert "protagonist" not in conditioning.contents(facts)


def test_it_does_not_describe_how_the_control_video_looks():
    """NOTES.md #1: mixing the roles is what makes H3 render the palette."""
    text = conditioning.card().text
    assert "not photography" in text
    assert "take nothing from its colours" in text


def test_an_unknown_card_is_an_error_not_a_silent_default():
    with pytest.raises(KeyError):
        conditioning.card("duv.abot.v99")


def test_a_caption_with_no_evidence_gets_no_per_clip_sentence(caption):
    assert render.compile_all(caption)["conditioning"]["contents"] == ""
