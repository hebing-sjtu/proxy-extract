import pytest
from clip_prompts import timeline
from clip_prompts.contract import Caption, Entity, Event, Scene


@pytest.fixture
def grid():
    return timeline.plan(124, 24.0)


def _caption(grid) -> Caption:
    return Caption(
        grid=grid,
        scene=Scene(
            medium="third-person open-world video game",
            environment="city street",
            environment_rich="a broad urban boulevard lined with stone buildings",
            lighting="bright daylight",
            lighting_rich="bright clear daylight with hard shadows on the pavement",
        ),
        entities=(
            Entity(
                id="char_0",
                kind="man",
                look="man in a dark green jacket",
                look_rich="man in a dark green bomber jacket, dark jeans and white sneakers",
                protagonist=True,
                bins=(0, 1, 2, 3, 4),
            ),
        ),
        events=(
            Event(
                id="e0",
                channel="subject",
                entity="char_0",
                verb="stand",
                phrase="still in the middle of the roadway",
                phrase_rich="motionless in the middle of the empty asphalt roadway",
                facing="back",
                bins=(0,),
            ),
            Event(
                id="e1",
                channel="subject",
                entity="char_0",
                verb="walk",
                phrase="forward along the road",
                phrase_rich="steadily along the asphalt roadway toward an intersection",
                facing="back",
                bins=(1, 2, 3, 4),
            ),
            Event(
                id="c0",
                channel="camera",
                verb="follow",
                phrase="from behind at shoulder height",
                phrase_rich="from behind at shoulder height, holding the man centred",
                bins=(0, 1, 2, 3, 4),
            ),
        ),
        window={"clip": "clip_000414_2", "t0": 0.0},
    ).bind()


@pytest.fixture
def caption(grid):
    return _caption(grid)


@pytest.fixture
def long_caption():
    grid = timeline.plan(1440, 24.0)
    base = _caption(grid)
    from dataclasses import replace

    entity = replace(base.entities[0], bins=tuple(range(60)))
    walker = replace(base.events[1], bins=tuple(range(1, 60)))
    camera = replace(base.events[2], bins=tuple(range(60)))
    return replace(base, entities=(entity,), events=(base.events[0], walker, camera)).bind()
