"""The `proxy_duv` deliverable, checked against what PROXY_DUV_SPEC.md asks.

Two groups. The first is the format: byte counts, PNG mode, class range,
manifest keys - the things the consumer asserts on, so getting them wrong fails
loudly and cheaply. The second is the format's silent failures, which is where
the value is: a delivery with the wrong class table or per-segment depth
normalisation satisfies every assertion in the first group.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from proxy_extract import cli, contract, proxy_duv
from proxy_extract.taxonomy import ANIMAL, HUMAN, INFRASTRUCTURE, ROAD_PAVED, SKY

H, W = contract.CONDITION_HEIGHT, contract.CONDITION_WIDTH
FRAMES = 8


def _depth(metres: float, *, sky_rows: int = 20) -> np.ndarray:
    depth = np.full((H, W), metres, dtype=np.float32)
    depth[:sky_rows] = 0.0  # sky is zero, not a far surface
    return depth


def _standard11_labels() -> np.ndarray:
    """A frame in the delivered 11-class schema, with the interesting classes."""
    from proxy_extract.taxonomy import S11_PED, S11_PLAYER, S11_ROAD, S11_SKY

    labels = np.full((H, W), S11_ROAD, dtype=np.uint8)
    labels[:20] = S11_SKY
    labels[100:120, 40:80] = S11_PLAYER
    labels[100:120, 200:240] = S11_PED
    return labels


def _write_segment(root, name, *, metres=12.0, frames=FRAMES, taxonomy="standard11"):
    seg = root / name
    labels = _standard11_labels() if taxonomy == "standard11" else None
    for ordinal in range(frames):
        if labels is None:
            ids = np.full((H, W), SKY, dtype=np.uint8)
            ids[20:] = ROAD_PAVED
        else:
            ids = labels
        proxy_duv.write_frame(seg, ordinal, _depth(metres), ids, taxonomy=taxonomy)
    (seg / proxy_duv.TARGET_NAME).write_bytes(b"not really an mp4, but a file")
    return seg


# ------------------------------------------------------------------ the format


def test_a_frame_lands_where_the_consumer_opens_it(tmp_path):
    seg = _write_segment(tmp_path, "seg_000000", frames=1)

    depth_path = seg / proxy_duv.DUV_DIRNAME / "000000.depth.f32"
    semantic_path = seg / proxy_duv.DUV_DIRNAME / "000000.semantic_id.png"

    assert depth_path.stat().st_size == 258048, "the consumer asserts on this byte count"
    assert depth_path.stat().st_size == contract.DEPTH_BYTES
    with Image.open(semantic_path) as image:
        assert image.mode == "L", "RGB is DATA_F.md's semantic.mp4 convention, not this"
        assert image.size == (W, H) == (336, 192)


def test_one_root_semantic_json_records_the_cwm_classes_and_bit_exact_uv_codes(tmp_path):
    _write_segment(tmp_path, "seg_000000", frames=1)
    _write_segment(tmp_path, "seg_000001", frames=1)

    metadata = json.loads((tmp_path / "semantic.json").read_text())

    assert metadata["resolution"] == {"width": 336, "height": 192}
    assert metadata["encoding"]["semantic_png_mode"] == "L"
    assert metadata["encoding"]["valid_class_ids"] == [0, 11]
    assert metadata["encoding"]["u_channel"] == "G"
    assert metadata["encoding"]["v_channel"] == "B"
    assert metadata["classes"]["0"] == {"id": 0, "name": "void_unknown", "uv": [32, 43]}
    assert metadata["classes"]["3"] == {"id": 3, "name": "terrain", "uv": [224, 43]}
    assert metadata["classes"]["4"] == {"id": 4, "name": "road_paved", "uv": [32, 128]}
    assert metadata["classes"]["11"] == {"id": 11, "name": "prop", "uv": [224, 213]}
    assert len({tuple(item["uv"]) for item in metadata["classes"].values()}) == 12
    assert not list(tmp_path.glob("*/duv/semantic.json"))


def test_the_depth_is_little_endian_float32_metres(tmp_path):
    seg = _write_segment(tmp_path, "seg_000000", metres=7.5, frames=1)

    raw = (seg / proxy_duv.DUV_DIRNAME / "000000.depth.f32").read_bytes()
    depth = np.frombuffer(raw, "<f4").reshape(H, W)

    assert depth[100, 100] == pytest.approx(7.5)
    assert depth[0, 0] == 0.0, "sky must be the zero sentinel"
    assert np.all(np.isfinite(depth)) and np.all(depth >= 0)


def test_the_writer_forces_semantic_sky_depth_to_zero(tmp_path):
    labels = np.full((H, W), ROAD_PAVED, dtype=np.uint8)
    labels[:20] = SKY
    proxy_duv.write_frame(
        tmp_path / "seg_000000",
        0,
        np.full((H, W), 40.0, dtype=np.float32),
        labels,
        taxonomy="cwm12",
    )

    depth, semantic = contract.read_frame(tmp_path / "seg_000000" / proxy_duv.DUV_DIRNAME, 0)

    assert np.all(depth[semantic == SKY] == 0.0)
    assert np.all(depth[semantic != SKY] == 40.0)


# ------------------------------------------------------- the native grid

NATIVE_H, NATIVE_W = 720, 1280


def _write_native_segment(root, name, *, metres=12.0, frames=2):
    seg = root / name
    ids = np.full((NATIVE_H, NATIVE_W), ROAD_PAVED, dtype=np.uint8)
    ids[:40] = SKY
    for ordinal in range(frames):
        depth = np.full((NATIVE_H, NATIVE_W), metres, dtype=np.float32)
        proxy_duv.write_frame(seg, ordinal, depth, ids, taxonomy="cwm12", native=True)
    (seg / proxy_duv.TARGET_NAME).write_bytes(b"not really an mp4, but a file")
    return seg


def test_a_native_frame_keeps_the_targets_own_grid(tmp_path):
    seg = _write_native_segment(tmp_path, "seg_000000", frames=1)

    depth, ids = contract.read_frame(seg / proxy_duv.DUV_DIRNAME, 0)

    assert depth.shape == ids.shape == (NATIVE_H, NATIVE_W)
    assert (seg / proxy_duv.DUV_DIRNAME / "000000.depth.f32").stat().st_size == (
        NATIVE_H * NATIVE_W * 4
    )
    assert np.all(depth[ids == SKY] == 0.0), "sky is still the zero sentinel"
    assert np.all(depth[ids != SKY] == 12.0), "a native frame is never block-reduced"
    metadata = json.loads((tmp_path / proxy_duv.SEMANTIC_NAME).read_text())
    assert metadata["resolution"] == {"width": NATIVE_W, "height": NATIVE_H}


def test_one_root_refuses_a_second_grid(tmp_path):
    _write_native_segment(tmp_path, "seg_000000", frames=1)

    with pytest.raises(proxy_duv.ProxyDuvError, match="mix"):
        _write_segment(tmp_path, "seg_000001", frames=1)


def test_the_audit_accepts_a_native_corpus_and_reports_its_grid(tmp_path):
    for index in range(3):
        _write_native_segment(tmp_path, f"seg_{index:06d}", frames=2)

    report = proxy_duv.audit_root(tmp_path, frames=2)

    assert report["failed"] == 0
    assert report["resolutions"] == [f"{NATIVE_W}x{NATIVE_H}"]
    assert report["warnings"] == []
    assert report["segment_stats"][0]["size"] == [NATIVE_W, NATIVE_H]
    assert report["segment_stats"][0]["metres_p50"] == pytest.approx(12.0)


def test_the_audit_warns_when_segments_disagree_about_the_grid(tmp_path):
    _write_native_segment(tmp_path, "seg_000000", frames=FRAMES)
    other = tmp_path / "other"
    _write_segment(other, "seg_000001")
    (other / "seg_000001").rename(tmp_path / "seg_000001")

    report = proxy_duv.audit_root(tmp_path, frames=FRAMES)

    assert report["resolutions"] == ["336x192", f"{NATIVE_W}x{NATIVE_H}"]
    assert any("disagree about the DUV grid" in warning for warning in report["warnings"])


# ----------------------------------------------------- the silent failure mode


def test_the_delivered_eleven_classes_are_projected_onto_cwms_twelve(tmp_path):
    """The mistake the spec opens by warning about.

    An `standard11` id is a valid `cwm12` id - both are small integers in the
    same range - so nothing raises, no shape changes and the loss looks normal.
    Only the meaning differs. So the projection is asserted class by class
    against the spec's own table rather than "it produced ids under 12".
    """
    seg = _write_segment(tmp_path, "seg_000000", frames=1, taxonomy="standard11")

    _depth_out, ids = contract.read_frame(seg / proxy_duv.DUV_DIRNAME, 0)

    assert ids[0, 0] == SKY, "standard11 sky is 0 and CWM sky is 1"
    assert ids[150, 150] == ROAD_PAVED
    assert ids[110, 60] == HUMAN, "player -> human"
    # Deliberate in the spec: it keeps the slot assignment the existing
    # gta_v2_* caches were built with, so the two sets stay mixable.
    assert ids[110, 220] == ANIMAL, "ped -> animal, per the spec's table"


def test_a_taxonomy_nobody_has_a_table_for_is_refused(tmp_path):
    with pytest.raises(proxy_duv.ProxyDuvError, match="corpus-wide table"):
        proxy_duv.write_frame(
            tmp_path / "seg_000000",
            0,
            _depth(10.0),
            np.zeros((H, W), np.uint8),
            taxonomy="coarse6",
        )


def test_labels_declared_cwm12_are_range_checked(tmp_path):
    ids = np.zeros((H, W), np.uint8)
    ids[0, 0] = 12

    with pytest.raises(proxy_duv.ProxyDuvError, match="outside"):
        proxy_duv.write_frame(tmp_path / "seg_000000", 0, _depth(10.0), ids, taxonomy="cwm12")


# ---------------------------------------------------------------- the audit


def test_the_audit_reports_the_numbers_section_eight_asks_for(tmp_path):
    seg = _write_segment(tmp_path, "seg_000000", metres=12.0)

    stats = proxy_duv.audit_seg(seg, frames=FRAMES)

    assert stats.frames == FRAMES
    assert stats.percentiles["p50"] == pytest.approx(12.0, rel=1e-3)
    # 20 of 192 rows are sky, so a bit under 90% of pixels carry depth.
    assert 0.85 < stats.valid_fraction < 0.92
    assert stats.above_far_fraction == 0.0
    assert 0.09 < stats.sky_fraction < 0.12
    assert stats.sky_depth_valid_fraction == 0.0
    assert SKY in stats.classes_present


def test_the_audit_reads_each_frame_only_once(tmp_path, monkeypatch):
    seg = _write_segment(tmp_path, "seg_000000", metres=12.0)
    original = proxy_duv.contract.read_frame
    reads = 0

    def counted_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(proxy_duv.contract, "read_frame", counted_read)

    proxy_duv.audit_seg(seg, frames=FRAMES)

    assert reads == FRAMES


def test_a_segment_shorter_than_the_window_is_refused(tmp_path):
    seg = _write_segment(tmp_path, "seg_000000", frames=4)

    with pytest.raises(proxy_duv.ProxyDuvError, match="short"):
        proxy_duv.audit_seg(seg, frames=8)


def test_per_segment_normalisation_is_caught_by_comparing_segments(tmp_path):
    """The one fatal mistake, and the only check that can see it.

    Every segment here passes `audit_seg` on its own: finite, non-negative,
    in range, ids under 12. What gives it away is that segments of the same
    world disagree about the median by orders of magnitude, which is what
    normalising each clip to its own min and max produces.
    """
    _write_segment(tmp_path, "seg_000000", metres=2.0)
    _write_segment(tmp_path, "seg_000001", metres=8.0)
    _write_segment(tmp_path, "seg_000002", metres=180.0)

    summary = proxy_duv.audit_root(tmp_path, frames=FRAMES)

    assert summary["audited"] == 3
    assert summary["median_spread"] > proxy_duv.MAX_MEDIAN_SPREAD
    assert any("section 2" in line for line in summary["warnings"])


def test_a_consistent_corpus_raises_no_warning(tmp_path):
    for index, metres in enumerate((10.0, 12.0, 14.0, 11.0)):
        _write_segment(tmp_path, f"seg_{index:06d}", metres=metres)

    summary = proxy_duv.audit_root(tmp_path, frames=FRAMES)

    assert summary["warnings"] == []
    assert summary["median_spread"] < proxy_duv.MAX_MEDIAN_SPREAD


def test_the_audit_can_process_segments_in_parallel(tmp_path):
    for index, metres in enumerate((10.0, 12.0, 14.0, 11.0)):
        _write_segment(tmp_path, f"seg_{index:06d}", metres=metres)

    serial = proxy_duv.audit_root(tmp_path, frames=FRAMES)
    parallel = proxy_duv.audit_root(tmp_path, frames=FRAMES, workers=2)

    assert parallel == serial


def test_the_audit_reports_progress_without_changing_its_result(tmp_path):
    for index in range(2):
        _write_segment(tmp_path, f"seg_{index:06d}", metres=10.0 + index)
    events = []

    summary = proxy_duv.audit_root(
        tmp_path,
        frames=FRAMES,
        progress=lambda done, total, segment: events.append((done, total, segment)),
    )

    assert summary["audited"] == 2
    assert events == [
        (0, 2, ""),
        (1, 2, "seg_000000"),
        (2, 2, "seg_000001"),
    ]


def test_a_sky_written_as_a_surface_is_caught(tmp_path):
    """The audit checks semantic-sky pixels, not the whole-frame valid ratio."""
    seg = tmp_path / "seg_000000"
    labels = np.full((H, W), ROAD_PAVED, dtype=np.uint8)
    labels[:20] = SKY
    for ordinal in range(FRAMES):
        # Bypass proxy_duv.write_frame: the delivery writer now repairs this
        # inconsistency, while the audit must still catch old data.
        contract.write_frame(
            proxy_duv.duv_dir_for(seg),
            ordinal,
            np.full((H, W), 40.0, dtype=np.float32),
            labels,
        )
    (seg / proxy_duv.TARGET_NAME).write_bytes(b"x")

    summary = proxy_duv.audit_root(tmp_path, frames=FRAMES)

    assert any("semantic sky pixels" in line for line in summary["warnings"])


def test_a_fully_valid_frame_without_semantic_sky_is_not_called_a_sky_surface(tmp_path):
    seg = tmp_path / "seg_000000"
    for ordinal in range(FRAMES):
        contract.write_frame(
            proxy_duv.duv_dir_for(seg),
            ordinal,
            np.full((H, W), 40.0, dtype=np.float32),
            np.full((H, W), ROAD_PAVED, dtype=np.uint8),
        )
    (seg / proxy_duv.TARGET_NAME).write_bytes(b"x")
    _write_segment(tmp_path, "seg_000001")

    summary = proxy_duv.audit_root(tmp_path, frames=FRAMES)

    assert not any("carry valid depth" in line for line in summary["warnings"])
    assert any("no `sky` pixels" in line for line in summary["notices"])


def test_sky_repair_changes_only_semantic_sky_depth(tmp_path):
    seg = tmp_path / "seg_000000"
    labels = np.full((H, W), ROAD_PAVED, dtype=np.uint8)
    labels[:20] = SKY
    for ordinal in range(FRAMES):
        contract.write_frame(
            proxy_duv.duv_dir_for(seg),
            ordinal,
            np.full((H, W), 40.0, dtype=np.float32),
            labels,
        )

    dry_run = proxy_duv.repair_sky_root(tmp_path)
    assert dry_run["affected_segments"] == 1
    assert dry_run["changed_frames"] == FRAMES

    applied = proxy_duv.repair_sky_root(tmp_path, apply=True)
    assert applied["changed_pixels"] == FRAMES * 20 * W

    depth, semantic = contract.read_frame(proxy_duv.duv_dir_for(seg), 0)
    assert np.all(depth[semantic == SKY] == 0.0)
    assert np.all(depth[semantic != SKY] == 40.0)
    assert proxy_duv.repair_sky_root(tmp_path)["affected_segments"] == 0


def test_a_broken_segment_does_not_stop_the_audit(tmp_path):
    _write_segment(tmp_path, "seg_000000")
    broken = tmp_path / "seg_000001"
    proxy_duv.duv_dir_for(broken).mkdir(parents=True)
    (proxy_duv.duv_dir_for(broken) / "000000.depth.f32").write_bytes(b"too short")

    summary = proxy_duv.audit_root(tmp_path, frames=FRAMES)

    assert summary["audited"] == 1
    assert summary["failed"] == 1
    assert "seg_000001" in summary["failures"][0]["seg"]


# -------------------------------------------------------------- the manifest


def test_the_manifest_names_proxy_duv_and_never_the_video(tmp_path):
    """Section 7 makes the three proxy keys mutually exclusive.

    A clip cut by this pipeline also carries a composed `proxy/duv.mp4`, and
    that one is DATA_F.md's palette. Naming it here is the single way to get
    the consumer to read the wrong one of two files that both exist and are
    both internally valid.
    """
    seg = _write_segment(tmp_path, "seg_000000")
    (seg / "proxy").mkdir()
    (seg / "proxy" / "duv.mp4").write_bytes(b"the other palette")

    entries = proxy_duv.manifest_from_root(tmp_path)

    assert len(entries) == 1
    assert entries[0]["proxy_duv"] == "seg_000000/duv"
    assert "proxy_duv_video" not in entries[0]
    assert "proxy" not in entries[0]
    assert "anchor" not in entries[0], "omitting it makes the consumer use target frame 0"


def test_manifest_paths_are_relative_to_the_root(tmp_path):
    _write_segment(tmp_path, "seg_000000")

    entry = proxy_duv.manifest_from_root(tmp_path)[0]

    for key in ("target", "proxy_duv"):
        assert not entry[key].startswith("/"), f"{key} must be relative to --root"
        assert (tmp_path / entry[key]).exists()


def test_the_clip_routes_layout_is_found_too(tmp_path):
    """`clips.py` writes the target at `target/rgb.mp4`, which is also legal."""
    seg = _write_segment(tmp_path, "clip_000000_0")
    (seg / proxy_duv.TARGET_NAME).unlink()
    (seg / "target").mkdir()
    (seg / "target" / "rgb.mp4").write_bytes(b"x")

    entry = proxy_duv.manifest_from_root(tmp_path)[0]

    assert entry["name"] == "clip_000000_0"
    assert entry["target"] == "clip_000000_0/target/rgb.mp4"


def test_a_segment_without_a_target_is_left_out(tmp_path):
    _write_segment(tmp_path, "seg_000000")
    half = _write_segment(tmp_path, "seg_000001")
    (half / proxy_duv.TARGET_NAME).unlink()

    entries = proxy_duv.manifest_from_root(tmp_path)

    assert [entry["name"] for entry in entries] == ["seg_000000"]


def test_the_manifest_is_one_json_object_per_line(tmp_path):
    _write_segment(tmp_path, "seg_000000")
    _write_segment(tmp_path, "seg_000001")

    path = proxy_duv.write_manifest(
        tmp_path, proxy_duv.manifest_from_root(tmp_path, prompts={"seg_000000": "a street"})
    )

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["prompt"] == "a street"
    assert "prompt" not in parsed[1]
    assert not list(tmp_path.glob("*.tmp")), "the atomic write left scratch behind"


def test_a_segments_own_prompt_txt_is_picked_up(tmp_path):
    """What `clip-prompts captions-export --write-txt` leaves is what this reads.

    Without this the operator has to collect a hundred prompt.txt files into a
    prompts.json by hand, and that copy goes stale the first time a caption is
    recompiled.
    """
    _write_segment(tmp_path, "seg_000000")
    _write_segment(tmp_path, "seg_000001")
    (tmp_path / "seg_000000" / "prompt.txt").write_bytes(b"[0.00s-5.17s] A city street.")

    entries = proxy_duv.manifest_from_root(tmp_path)

    assert entries[0]["prompt"] == "[0.00s-5.17s] A city street."
    assert "prompt" not in entries[1]


def test_the_crlf_of_an_exported_prompt_survives_being_read(tmp_path):
    """CWM's caches were encoded from CRLF bytes, so LF is a different token stream.

    `read_text` would translate them and leave a file that looks identical.
    """
    _write_segment(tmp_path, "seg_000000")
    (tmp_path / "seg_000000" / "prompt.txt").write_bytes(b"[0.00s-1.00s] One.\r\n[1.00s-2.00s] Two.")

    entries = proxy_duv.manifest_from_root(tmp_path)

    assert "\r\n" in entries[0]["prompt"]


def test_an_explicit_mapping_beats_the_file_beside_the_segment(tmp_path):
    """A corpus whose text came from elsewhere has to stay captionable."""
    _write_segment(tmp_path, "seg_000000")
    (tmp_path / "seg_000000" / "prompt.txt").write_bytes(b"[0.00s-5.17s] From the file.")

    entries = proxy_duv.manifest_from_root(tmp_path, prompts={"seg_000000": "from the mapping"})

    assert entries[0]["prompt"] == "from the mapping"


# --------------------------------------------------------- frame arithmetic


@pytest.mark.parametrize("count", [124, 141, 158, 1790])
def test_deliverable_frame_counts_satisfy_the_vae_constraint(count):
    assert proxy_duv.frames_are_deliverable(count)
    assert count % 17 == 5


def test_the_cwm_window_stride_is_not_this_constraint():
    """124 + 90 is a valid code-world-model run and an invalid H3 sample.

    The two deliverables share their first window and nothing after it, which
    is why `proxy_duv` does its own arithmetic instead of reusing
    `contract.frames_for_windows`.
    """
    assert contract.frames_for_windows(2) == 214
    assert not proxy_duv.frames_are_deliverable(214)


@pytest.mark.parametrize(
    ("decoded", "expected"), [(124, 124), (130, 124), (141, 141), (200, 192)]
)
def test_the_longest_deliverable_segment_is_found(decoded, expected):
    assert proxy_duv.largest_deliverable_frame_count(decoded) == expected
    assert proxy_duv.frames_are_deliverable(expected)


def test_a_clip_short_of_the_window_is_refused_rather_than_padded():
    with pytest.raises(proxy_duv.ProxyDuvError, match="short of"):
        proxy_duv.largest_deliverable_frame_count(100)


# --------------------------------------------------------- the command line


def test_the_manifest_command_writes_the_file_and_reports_it(tmp_path, capsys):
    _write_segment(tmp_path, "seg_000000")
    _write_segment(tmp_path, "seg_000001")
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"seg_000000": "a street at dusk"}))

    code = cli.main(
        [
            "proxy-duv-manifest",
            "--root", str(tmp_path),
            "--prompts", str(prompts),
        ]
    )

    assert code == 0
    written = tmp_path / proxy_duv.MANIFEST_NAME
    assert written.exists()
    assert len(written.read_text().strip().splitlines()) == 2
    out = capsys.readouterr().out
    # The count of promptless segments is printed because forgetting --prompts
    # otherwise produces a manifest that loads and trains on nothing useful.
    assert "2 segments" in out
    assert "1 without a prompt" in out


def test_the_manifest_command_refuses_a_root_that_was_cut_without_the_flag(
    tmp_path, capsys
):
    """The likely operator mistake, and it has to name the fix.

    `clips` only writes `duv/` when asked with `--proxy-duv`, so a root cut
    without it has targets, has a composed `proxy/duv.mp4`, and has nothing
    this consumer can read. An empty manifest would be the wrong answer: it
    exits 0 and the next stage reports zero samples with no reason given.
    """
    seg = _write_segment(tmp_path, "seg_000000")
    import shutil

    shutil.rmtree(seg / proxy_duv.DUV_DIRNAME)

    code = cli.main(["proxy-duv-manifest", "--root", str(tmp_path)])

    assert code == 1
    assert not (tmp_path / proxy_duv.MANIFEST_NAME).exists()
    assert "--proxy-duv" in capsys.readouterr().err


def test_the_audit_command_exits_non_zero_on_the_fatal_mistake(tmp_path, capsys):
    """Exit code, not just output: this is what stands in a CI gate.

    Per-segment depth normalisation is reported as a warning rather than a
    failure, because every segment is individually valid. A warning that exits
    0 would wave the batch through, so the command treats it as a failure.

    Three segments rather than two because the spread is measured p90 over p10,
    which is intentionally blind to a single odd segment: on a two-element
    sample the interpolation pulls both ends towards the middle and nothing
    trips. A corpus of two is not the case this guards.
    """
    for index, metres in enumerate((2.0, 8.0, 180.0)):
        _write_segment(tmp_path, f"seg_{index:06d}", metres=metres)

    code = cli.main(["proxy-duv-audit", "--root", str(tmp_path), "--frames", "8"])

    captured = capsys.readouterr()
    assert code == 1
    assert "section 2" in captured.err
    summary = json.loads(captured.out)
    assert summary["failed"] == 0, "each segment is valid on its own; that is the trap"
    assert summary["audited"] == 3


def test_the_audit_command_passes_a_consistent_corpus(tmp_path, capsys):
    for index, metres in enumerate((10.0, 12.0, 11.0)):
        _write_segment(tmp_path, f"seg_{index:06d}", metres=metres)

    code = cli.main(["proxy-duv-audit", "--root", str(tmp_path), "--frames", "8"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert json.loads(captured.out)["audited"] == 3


def test_a_no_sky_notice_does_not_fail_the_audit_command(tmp_path, capsys):
    seg = tmp_path / "seg_000000"
    for ordinal in range(FRAMES):
        contract.write_frame(
            proxy_duv.duv_dir_for(seg),
            ordinal,
            np.full((H, W), 12.0, dtype=np.float32),
            np.full((H, W), ROAD_PAVED, dtype=np.uint8),
        )
    (seg / proxy_duv.TARGET_NAME).write_bytes(b"x")
    _write_segment(tmp_path, "seg_000001", metres=12.0)

    code = cli.main(["proxy-duv-audit", "--root", str(tmp_path), "--frames", "8"])

    captured = capsys.readouterr()
    assert code == 0
    assert "notice: seg_000000: no `sky` pixels" in captured.err
    assert json.loads(captured.out)["warnings"] == []


def test_the_audit_report_keeps_the_per_segment_detail(tmp_path):
    """The summary drops it to stay readable, and the file must not.

    Chasing which segment moved the median is the first thing anyone does with
    a failing audit, so a report without the per-segment rows sends them back
    to re-run it.
    """
    _write_segment(tmp_path, "seg_000000", metres=10.0)
    _write_segment(tmp_path, "seg_000001", metres=11.0)
    report = tmp_path / "audit.json"

    cli.main(
        [
            "proxy-duv-audit",
            "--root", str(tmp_path),
            "--frames", "8",
            "--report", str(report),
        ]
    )

    parsed = json.loads(report.read_text())
    assert len(parsed["segment_stats"]) == 2
    assert {item["seg"] for item in parsed["segment_stats"]} == {
        "seg_000000",
        "seg_000001",
    }


def test_the_sky_repair_command_requires_apply_before_mutating(tmp_path, capsys):
    seg = tmp_path / "seg_000000"
    labels = np.full((H, W), ROAD_PAVED, dtype=np.uint8)
    labels[:20] = SKY
    contract.write_frame(
        proxy_duv.duv_dir_for(seg),
        0,
        np.full((H, W), 40.0, dtype=np.float32),
        labels,
    )

    dry_code = cli.main(["proxy-duv-repair-sky", "--root", str(tmp_path)])
    dry_summary = json.loads(capsys.readouterr().out)
    assert dry_code == 1
    assert dry_summary["affected_segments"] == 1

    apply_code = cli.main(["proxy-duv-repair-sky", "--root", str(tmp_path), "--apply"])
    apply_summary = json.loads(capsys.readouterr().out)
    assert apply_code == 0
    assert apply_summary["changed_pixels"] == 20 * W
    assert proxy_duv.repair_sky_root(tmp_path)["affected_segments"] == 0


def test_the_new_backends_are_offered_by_the_command_line():
    """A backend that cannot be named on the command line does not exist.

    Both of these are the answer to a flickering delivery, so the names are
    part of the deliverable rather than an implementation detail.
    """
    from proxy_extract.cli import DEPTH_BACKENDS, REFINERS

    assert "moge3" in DEPTH_BACKENDS
    assert "sam2" in REFINERS


def test_the_projection_table_is_injective():
    """Two source classes sharing a CWM id would be indistinguishable in the
    delivered channel, and the spec's only hard requirements on the table are
    injectivity and corpus-wide stability."""
    from proxy_extract.taxonomy import NUM_STANDARD11, STANDARD11_TO_CWM

    assert len(set(STANDARD11_TO_CWM.values())) == NUM_STANDARD11
    assert STANDARD11_TO_CWM[6] == INFRASTRUCTURE, "ground -> infrastructure, not terrain"
