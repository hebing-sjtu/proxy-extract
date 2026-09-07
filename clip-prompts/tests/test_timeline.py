from clip_prompts import timeline


def test_clip_shape_gives_five_bins_and_a_long_tail():
    """124 frames at 24 fps is 5.1667 s: five bins, the last one 1.1667 s.

    The alternative - six bins whose last holds four frames - is the case the
    tail rule exists to prevent, so this is the test that fails if that rule
    is removed.
    """
    grid = timeline.plan(124, 24.0)
    assert grid.count == 5
    assert grid.duration == 5.167
    assert grid.bins[-1].start == 4.0
    assert grid.bins[-1].seconds == 1.167
    assert grid.bins[-1].frames == 28


def test_bins_partition_the_frames_exactly():
    for frames, fps in ((124, 24.0), (1800, 30.0), (1440, 24.0), (73, 24.0)):
        grid = timeline.plan(frames, fps)
        assert grid.bins[0].first_frame == 0
        assert grid.bins[-1].stop_frame == frames
        for left, right in zip(grid.bins, grid.bins[1:]):
            assert left.stop_frame == right.first_frame
            assert left.stop == right.start
        assert sum(b.frames for b in grid.bins) == frames


def test_a_minute_is_sixty_bins():
    grid = timeline.plan(1440, 24.0)
    assert grid.count == 60
    assert grid.bins[59].start == 59.0


def test_a_clip_shorter_than_one_bin_still_gets_one():
    grid = timeline.plan(6, 24.0)
    assert grid.count == 1
    assert grid.bins[0].frames == 6


def test_a_long_enough_tail_keeps_its_own_bin():
    grid = timeline.plan(132, 24.0)  # 5.5 s
    assert grid.count == 6
    assert grid.bins[-1].seconds == 0.5


def test_covering_is_half_open():
    grid = timeline.plan(124, 24.0)
    assert grid.covering(1.0, 2.0) == (1,)
    assert grid.covering(0.5, 2.5) == (0, 1, 2)


def test_cut_rebases_to_zero_and_keeps_whole_bins():
    grid = timeline.plan(1440, 24.0)
    piece = timeline.cut(grid, 10.0, 15.0)
    assert piece.kept == (10, 11, 12, 13, 14)
    assert piece.offset == 10.0
    assert piece.grid.bins[0].start == 0.0
    assert piece.grid.bins[0].first_frame == 0
    assert piece.grid.frames == 120
    assert piece.rebase(12) == 2


def test_cut_drops_a_bin_the_window_only_half_shows():
    grid = timeline.plan(1440, 24.0)
    piece = timeline.cut(grid, 10.5, 15.0)
    assert piece.kept == (11, 12, 13, 14)


def test_round_trip_through_json():
    grid = timeline.plan(124, 24.0)
    assert timeline.from_dict(grid.as_dict()) == grid
