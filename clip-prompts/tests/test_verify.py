from dataclasses import replace

from clip_prompts import verify
from clip_prompts.contract import Entity, Event


def _evidence(overrides=None, *, hero_resolved=True):
    rows = []
    for index in range(5):
        row = {
            "bin": index,
            "t": [float(index), float(index + 1)],
            "player": True,
            "peds": 0,
            "vehicle": False,
            "ego_vehicle": False,
            "sky_frac": 0.2,
            "depth_p50_m": 20.0,
            "depth_delta_code": 3.0,
            "world_moving": True,
        }
        row.update((overrides or {}).get(index, {}))
        rows.append(row)
    return {"source": "duv", "hero_resolved": hero_resolved, "per_bin": rows}


def test_a_caption_that_matches_its_evidence_passes(caption):
    checked = replace(caption, evidence=_evidence())
    result = verify.check(checked)
    assert result["passed"] is True
    assert result["fail"] == []


def test_a_protagonist_in_a_second_with_no_player_pixels_fails(caption):
    checked = replace(caption, evidence=_evidence({2: {"player": False}}))
    result = verify.check(checked)
    assert result["passed"] is False
    assert "bin 2" in result["fail"][0]


def test_that_check_is_switched_off_when_the_tracker_did_not_resolve(caption):
    facts = _evidence({2: {"player": False}}, hero_resolved=False)
    assert verify.check(replace(caption, evidence=facts))["passed"] is True


def test_an_invented_car_fails(caption):
    with_car = replace(
        caption,
        events=(
            *caption.events,
            Event(id="e2", channel="subject", entity="char_0", verb="look",
                  phrase="at a car parked by the kerb", bins=(0, 1)),
        ),
        evidence=_evidence(),
    )
    result = verify.check(with_car)
    assert result["passed"] is False
    assert "describes a vehicle" in result["fail"][0]


def test_a_car_that_the_segmentation_does_see_is_accepted(caption):
    with_car = replace(
        caption,
        events=(
            *caption.events,
            Event(id="e2", channel="subject", entity="char_0", verb="look",
                  phrase="at a car parked by the kerb", bins=(0, 1)),
        ),
        evidence=_evidence({0: {"vehicle": True}, 1: {"vehicle": True}}),
    )
    assert verify.check(with_car)["passed"] is True


def test_a_low_confidence_claim_is_not_held_to_the_evidence(caption):
    hedged = replace(
        caption,
        events=(
            *caption.events,
            Event(id="e2", channel="subject", entity="char_0", verb="look",
                  phrase="at a car parked by the kerb", bins=(0,), confidence=0.3),
        ),
        evidence=_evidence(),
    )
    assert verify.check(hedged)["passed"] is True


def test_a_bystander_count_off_by_three_is_a_warning(caption):
    crowded = replace(
        caption,
        entities=(
            *caption.entities,
            Entity(id="p0", kind="person", look="a woman", bins=(0,)),
            Entity(id="p1", kind="person", look="a man", bins=(0,)),
            Entity(id="p2", kind="person", look="a child", bins=(0,)),
        ),
        evidence=_evidence(),
    )
    result = verify.check(crowded)
    assert result["passed"] is True
    assert any("names 3 bystanders" in line for line in result["warn"])


def test_an_off_by_one_bystander_count_is_tolerated(caption):
    nearly = replace(
        caption,
        entities=(*caption.entities, Entity(id="p0", kind="person", look="a woman", bins=(0,))),
        evidence=_evidence(),
    )
    assert verify.check(nearly)["warn"] == []


def test_nothing_moving_while_the_depth_field_does_is_a_warning(caption):
    still = replace(
        caption,
        events=(
            replace(caption.events[0], bins=(0, 1, 2, 3, 4)),
            replace(caption.events[2], verb="hold"),
        ),
        evidence=_evidence(),
    )
    result = verify.check(still)
    assert result["passed"] is True
    assert len(result["warn"]) == 5


def test_score_is_zero_when_anything_failed(caption):
    scored = replace(caption, checks={"fail": ["something"], "warn": []})
    assert verify.score(scored) == 0.0


def test_score_is_docked_for_warnings(caption):
    clean = replace(caption, checks={"fail": [], "warn": []})
    noisy = replace(caption, checks={"fail": [], "warn": ["a", "b"]})
    assert verify.score(clean) == 1.0
    assert verify.score(noisy) == 0.9


def test_a_caption_with_no_evidence_reports_that_it_was_not_checked(caption):
    result = verify.check(caption)
    assert result["checked"] is False
    assert result["passed"] is True
