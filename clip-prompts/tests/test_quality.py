import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from clip_prompts import quality, workbench
from clip_prompts.layout import Clip
from clip_prompts.llm import ChatResponse

cv2 = pytest.importorskip("cv2")

GOOD = {
    "depth_temporal": 5,
    "depth_accuracy": 4,
    "semantic_temporal": 4,
    "semantic_accuracy": 5,
    "issues": [],
    "verdict": "accept",
    "summary": "stable and correct",
}


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 24.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
    writer.release()


def make_clip(root: Path, name: str = "clip_000001_0", frames: int = 12) -> Clip:
    clip_dir = root / name
    h, w = 36, 64
    rgb = [np.full((h, w, 3), 40 + 10 * t, np.uint8) for t in range(frames)]
    duv = []
    for t in range(frames):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[: h // 3, :, 0] = 255
        frame[h // 3 :, :, 0] = 100 + t
        duv.append(frame)
    _write_video(clip_dir / "target" / "rgb.mp4", rgb)
    _write_video(clip_dir / "proxy" / "duv.mp4", duv)
    (clip_dir / "duv").mkdir(parents=True)
    for t in range(frames):
        ids = np.full((h, w), 4, np.uint8)
        ids[: h // 3] = 1
        cv2.imwrite(str(clip_dir / "duv" / f"{t:06d}.semantic_id.png"), ids)
    (clip_dir / "clip_report.json").write_text(json.dumps({"frames": frames, "fps": 24.0}))
    return Clip(clip_dir)


class FakeClient:
    name = "fake"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return ChatResponse(text=self.replies.pop(0), usage={"totalTokenCount": 10}, raw={})


def vlm(**changes) -> dict:
    return quality.validate({**GOOD, **changes}, duration=5.0)


# ---------------------------------------------------------------- decision


def test_a_clean_review_is_accepted_with_its_mean_score():
    decision = quality.decide(vlm())
    assert decision["verdict"] == "accept"
    assert decision["reasons"] == []
    assert decision["score"] == pytest.approx(18 / 20)


def test_any_score_below_the_floor_rejects_and_says_which():
    decision = quality.decide(vlm(semantic_temporal=2))
    assert decision["verdict"] == "reject"
    assert decision["reasons"] == ["semantic_temporal < 3"]


def test_a_major_issue_rejects_even_with_good_scores():
    issue = {"track": "depth", "kind": "flicker", "severity": "major", "start": 1, "end": 2}
    assert quality.decide(vlm(issues=[issue]))["reasons"] == ["major depth flicker"]
    lenient = quality.Thresholds(reject_major=False)
    assert quality.decide(vlm(issues=[issue]), lenient)["verdict"] == "accept"


def test_the_vlm_verdict_counts_unless_told_not_to():
    assert quality.decide(vlm(verdict="reject"))["reasons"] == ["vlm verdict reject"]
    numbers_only = quality.Thresholds(trust_verdict=False)
    assert quality.decide(vlm(verdict="reject"), numbers_only)["verdict"] == "accept"


# ---------------------------------------------------------------- the reply


def test_a_reply_missing_a_score_or_verdict_is_refused_with_every_problem():
    with pytest.raises(quality.ReplyError) as error:
        quality.validate({**GOOD, "depth_accuracy": 7, "verdict": "maybe"}, duration=5.0)
    assert "depth_accuracy" in str(error.value) and "verdict" in str(error.value)


def test_issue_times_are_clamped_ordered_and_unknown_kinds_kept_as_other():
    issue = {"track": "Semantic", "kind": "sparkle", "severity": "MINOR", "start": 9, "end": -1}
    (item,) = vlm(issues=[issue])["issues"]
    assert item == {"track": "semantic", "kind": "other", "severity": "minor",
                    "start": 0.0, "end": 5.0, "note": ""}


# ---------------------------------------------------------------- measures


def _measure(codes, ids):
    temporal = quality._Temporal()
    for c, i in zip(codes, ids):
        temporal.add(c, i)
    return temporal.result()


def test_steady_motion_is_not_flicker():
    codes = [np.full((8, 8), 50 + 3 * t, np.uint8) for t in range(6)]
    ids = [np.full((8, 8), 4, np.uint8)] * 6
    result = _measure(codes, ids)
    assert result["depth_jitter"] == 0
    assert result["semantic_flicker"] == 0
    assert result["depth_median_step_max"] == 3


def test_a_one_frame_flash_is_measured_in_both_tracks():
    codes = [np.full((8, 8), 50, np.uint8) for _ in range(5)]
    codes[2] = np.full((8, 8), 70, np.uint8)
    ids = [np.full((8, 8), 4, np.uint8) for _ in range(5)]
    ids[2] = ids[2].copy()
    ids[2][:4] = 5
    result = _measure(codes, ids)
    assert result["depth_jitter_max"] == 40
    assert result["semantic_flicker_max"] == pytest.approx(0.5)


def test_sky_is_left_out_of_depth_jitter():
    codes = [np.full((8, 8), 50, np.uint8) for _ in range(4)]
    codes[1][:4] = 255
    result = _measure(codes, [np.zeros((8, 8), np.uint8)] * 4)
    assert result["depth_jitter"] == 0


def test_depth_colours_put_sky_at_black_and_near_warmer_than_far():
    rgb = quality.colour_depth(np.array([[0, 254, 255]], np.uint8))
    near, far, sky = rgb[0]
    assert tuple(sky) == (0, 0, 0)
    assert near[0] > near[2] and far[2] > far[0]


# ---------------------------------------------------------------- end to end


def test_judging_a_clip_writes_panel_and_verdict_and_then_reuses_it(tmp_path):
    clip = make_clip(tmp_path)
    client = FakeClient("not json at all", json.dumps(GOOD))
    row = quality.judge_clip(clip, client, model="m", backend="fake")
    assert row["status"] == "written" and row["verdict"] == "accept"

    panel = quality.panel_path(clip)
    capture = cv2.VideoCapture(str(panel))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 2 * quality.TILE_W
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    capture.release()

    saved = json.loads(quality.quality_path(clip).read_text())
    assert saved["prompt"] == quality.PROMPT_VERSION
    assert saved["attempts"] == 2 and saved["usage"]["totalTokenCount"] == 20
    assert saved["metrics"]["frames"] == 12
    assert saved["duration"] == pytest.approx(0.5)
    video = client.requests[0].messages[1].parts[1]
    assert video.path == panel and video.fps == quality.DEFAULT_SAMPLE_FPS

    again = quality.judge_clip(clip, FakeClient(), model="m", backend="fake")
    assert again["status"] == "reused"


def test_a_clip_without_semantic_ids_fails_loudly_and_leaves_nothing(tmp_path):
    clip = make_clip(tmp_path)
    for png in (clip.root / "duv").glob("*.png"):
        png.unlink()
    with pytest.raises(FileNotFoundError):
        quality.render_panel(clip)
    assert not list(clip.annotations.glob("*.mp4"))


def test_a_human_override_wins_in_the_audit(tmp_path):
    accepted = make_clip(tmp_path, "clip_000001_0")
    overruled = make_clip(tmp_path, "clip_000002_0")
    pending = make_clip(tmp_path, "clip_000003_0")
    (tmp_path / "clip_000004_0").mkdir()
    for clip in (accepted, overruled):
        quality.judge_clip(clip, FakeClient(json.dumps(GOOD)), model="m", backend="fake")
    quality.set_override(overruled, "reject", "road flickers at 3s")

    result = quality.audit(quality.scan(tmp_path))
    assert result["accepted"] == [accepted.name]
    assert result["rejected"] == [overruled.name]
    assert result["unjudged"] == [pending.name]
    assert result["overridden"] == [overruled.name]
    assert result["summary"]["clips"] == 4 and result["summary"]["complete"] == 3

    quality.set_override(overruled, None)
    assert quality.summarize(overruled.root)["final"] == "accept"


# ---------------------------------------------------------------- workbench


@pytest.fixture
def bench(tmp_path):
    clip = make_clip(tmp_path)
    quality.judge_clip(clip, FakeClient(json.dumps(GOOD)), model="m", backend="fake")
    index = workbench.Index(tmp_path)
    index.refresh()
    server = workbench.ThreadingHTTPServer(("127.0.0.1", 0), workbench.make_handler(index))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", clip
    server.shutdown()
    server.server_close()


def _get(url, **headers):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5)


def test_the_workbench_serves_the_page_and_the_rows(bench):
    base, clip = bench
    assert b"<title>" in _get(base + "/").read()
    data = json.loads(_get(base + "/api/clips").read())
    assert data["audit"]["accepted"] == 1
    assert data["rows"][0]["clip"] == clip.name and data["rows"][0]["score"] == pytest.approx(0.9)
    detail = json.loads(_get(f"{base}/api/clip/{clip.name}").read())
    assert detail["quality"]["vlm"]["verdict"] == "accept"


def test_the_workbench_serves_byte_ranges_so_videos_can_seek(bench):
    base, clip = bench
    size = quality.panel_path(clip).stat().st_size
    response = _get(f"{base}/media/{clip.name}/panel", Range="bytes=10-19")
    assert response.status == 206
    assert response.headers["Content-Range"] == f"bytes 10-19/{size}"
    assert response.read() == quality.panel_path(clip).read_bytes()[10:20]


def test_the_workbench_refuses_names_that_are_not_clips(bench):
    base, _ = bench
    for path in ("/media/..%2F..%2Fetc/panel", "/api/clip/secrets", "/media/clip_000001_0/report"):
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(base + path)
        assert error.value.code == 404


def test_an_override_posted_from_the_workbench_lands_on_disk(bench):
    base, clip = bench
    body = json.dumps({"verdict": "reject", "note": "sky flashes"}).encode()
    request = urllib.request.Request(f"{base}/api/override/{clip.name}", data=body, method="POST")
    row = json.loads(urllib.request.urlopen(request, timeout=5).read())["row"]
    assert row["final"] == "reject" and row["override"]["note"] == "sky flashes"
    assert json.loads(quality.override_path(clip).read_text())["verdict"] == "reject"

    bad = urllib.request.Request(f"{base}/api/override/{clip.name}",
                                 data=json.dumps({"verdict": "maybe"}).encode(), method="POST")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(bad, timeout=5)
    assert error.value.code == 400
