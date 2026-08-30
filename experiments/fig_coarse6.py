"""Two questions about the 6-class taxonomy, answered from one batch of inference.

1. Under `background / road / vegetation / vehicle / person`, does the low-poly
   render still support the same labelling as the high render? The 12-class
   answer was no, but most of what it lost was small classes that this taxonomy
   folds into background, so the coarse answer could differ.

2. Is `hero` vs `npc` separable at all? No segmenter predicts it — both are
   "person" — so it has to come from tracking. In a third-person game the
   camera is bolted to the protagonist, which should make their silhouette sit
   still in image space while everyone else drifts. This measures whether that
   signal is actually there, and whether it survives the low-poly render.

Per-frame label maps are cached so the analysis can be re-run without paying
for inference again.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proxy_extract import taxonomy, video
from proxy_extract.semantic.base import resolve_label_lut
from proxy_extract.semantic.panoptic import PanopticBackend

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "handpick29_high_low"
OUT = ROOT / "experiments" / "figures"
CACHE = ROOT / "experiments" / "coarse6_labels.npz"
STATS = ROOT / "experiments" / "coarse6.json"

SEG_SIZE = (640, 360)
FRAME_STEP = 8
MIN_PERSON_AREA = 120

CLIPS = [
    "11_john_marston_seg_0313",
    "26_trevor_seg_0004",
    "32_trevor_seg_0085",
    "46_franklin_seg_0093",
    "02_john_marston_seg_0017",
    "17_michael_seg_0042",
]


def coarse6_backend() -> PanopticBackend:
    """Mask2Former/ADE20K wired straight to the 6-class set."""
    backend = PanopticBackend(batch_size=2)
    backend.mapping = taxonomy.ADE20K_TO_COARSE6
    backend.fallback = taxonomy.C6_BACKGROUND
    return backend


def segment_all() -> dict[str, np.ndarray]:
    backend = coarse6_backend()
    backend._load()
    id2label = {int(k): str(v) for k, v in backend._model.config.id2label.items()}
    backend._lut, _ = resolve_label_lut(
        id2label, taxonomy.ADE20K_TO_COARSE6, default=taxonomy.C6_BACKGROUND
    )

    cache: dict[str, np.ndarray] = {}
    for stem in CLIPS:
        for kind in ("high", "low"):
            frames = video.read_frames(DATA / kind / f"{stem}.mp4", size=SEG_SIZE)
            chosen = frames[::FRAME_STEP]
            cache[f"{stem}|{kind}"] = backend.segment(chosen).labels.astype(np.uint8)
            print(f"{stem} [{kind}] {len(chosen)} frames segmented")
    return cache


def person_tracks(labels: np.ndarray) -> list[dict]:
    """Link person blobs across frames into tracks by centroid proximity.

    Deliberately crude: semantic segmentation merges touching people into one
    blob, so a track here is "a connected region of person pixels that persists",
    not a true instance. That limitation is part of what is being measured.
    """
    tracks: list[dict] = []
    for index, frame in enumerate(labels):
        mask = np.isin(frame, (taxonomy.C6_NPC, taxonomy.C6_HERO)).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < MIN_PERSON_AREA:
                continue
            centre = centroids[component]

            best, best_distance = None, np.inf
            for track in tracks:
                if track["frames"][-1] == index:
                    continue
                distance = float(np.linalg.norm(np.array(track["centres"][-1]) - centre))
                if distance < best_distance:
                    best, best_distance = track, distance

            # 60 px at 640x360 is roughly a body width per sampled step; beyond
            # that it is a different person, not the same one having moved.
            if best is not None and best_distance < 60 and index - best["frames"][-1] <= 2:
                best["frames"].append(index)
                best["centres"].append(centre.tolist())
                best["areas"].append(area)
            else:
                tracks.append({"frames": [index], "centres": [centre.tolist()], "areas": [area]})
    return tracks


def hero_evidence(labels: np.ndarray) -> dict:
    tracks = person_tracks(labels)
    total = len(labels)
    if not tracks:
        return {"tracks": 0, "hero_persistence": 0.0, "hero_jitter": None, "others_jitter": None}

    def score(track: dict) -> float:
        return len(track["frames"]) * float(np.median(track["areas"]))

    def jitter(track: dict) -> float:
        centres = np.array(track["centres"])
        return float(np.mean(np.std(centres, axis=0)))

    ranked = sorted(tracks, key=score, reverse=True)
    hero = ranked[0]
    others = [t for t in ranked[1:] if len(t["frames"]) >= 3]

    return {
        "tracks": len(tracks),
        "multi_person_frames": float(
            np.mean([sum(index in t["frames"] for t in tracks) > 1 for index in range(total)])
        ),
        "hero_persistence": len(hero["frames"]) / total,
        "hero_area_share": float(np.median(hero["areas"])) / (labels.shape[1] * labels.shape[2]),
        "hero_jitter": jitter(hero),
        "others_jitter": float(np.median([jitter(t) for t in others])) if others else None,
    }


def analyse(cache: dict[str, np.ndarray]) -> dict:
    per_clip = []
    intersection = np.zeros(taxonomy.NUM_COARSE6)
    union = np.zeros(taxonomy.NUM_COARSE6)
    share = np.zeros(taxonomy.NUM_COARSE6)

    for stem in CLIPS:
        high, low = cache[f"{stem}|high"], cache[f"{stem}|low"]
        clip_inter = np.zeros(taxonomy.NUM_COARSE6)
        clip_union = np.zeros(taxonomy.NUM_COARSE6)

        for a, b in zip(high, low):
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
            for cls in range(taxonomy.NUM_COARSE6):
                hit, miss = a == cls, b == cls
                clip_inter[cls] += np.sum(hit & miss)
                clip_union[cls] += np.sum(hit | miss)
                share[cls] += np.sum(hit)

        intersection += clip_inter
        union += clip_union
        with np.errstate(invalid="ignore"):
            iou = np.where(clip_union > 0, clip_inter / clip_union, np.nan)
        per_clip.append(
            {
                "clip": stem,
                "miou": float(np.nanmean(iou)),
                "high_hero": hero_evidence(high),
                "low_hero": hero_evidence(low),
            }
        )
        print(f"{stem:32} coarse6 mIoU {per_clip[-1]['miou'] * 100:5.1f}%")

    with np.errstate(invalid="ignore"):
        overall = np.where(union > 0, intersection / union, np.nan)
    return {
        "per_class_iou": overall.tolist(),
        "share": (share / share.sum()).tolist(),
        "per_clip": per_clip,
    }


def plot(data: dict, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.0), width_ratios=[1, 1.1])

    iou = np.array(data["per_class_iou"])
    share = np.array(data["share"])
    present = [c for c in range(taxonomy.NUM_COARSE6) if share[c] > 1e-4]
    x = np.arange(len(present))
    colours = ["#1f8a4c" if iou[c] > 0.6 else "#c8952b" if iou[c] > 0.35 else "#b3382c" for c in present]
    left.bar(x, iou[present] * 100, color=colours)
    for slot, cls in enumerate(present):
        left.text(slot, iou[cls] * 100 + 1.5, f"{share[cls] * 100:.0f}%", ha="center", fontsize=8, color="#555")
    left.set_xticks(x, [taxonomy.COARSE6_NAMES[c] for c in present], fontsize=9)
    left.set_ylim(0, 100)
    left.set_ylabel("IoU, low render vs high render (%)")
    left.set_title("6-class agreement\n(grey = share of high-render pixels)", fontsize=10)
    left.grid(axis="y", alpha=0.25)

    clips = data["per_clip"]
    y = np.arange(len(clips))
    multi = [c["high_hero"]["multi_person_frames"] * 100 for c in clips]
    right.barh(y, multi, color=["#c8952b" if m > 5 else "#8a8a8a" for m in multi], height=0.55)
    for slot, record in enumerate(clips):
        evidence = record["high_hero"]
        if evidence["others_jitter"] is None and evidence["multi_person_frames"] < 0.01:
            note = "only the protagonist on screen"
        elif evidence["others_jitter"] is None:
            note = "a second person appears too briefly to track"
        else:
            note = (
                f"hero wanders {evidence['hero_jitter']:.1f} px, others {evidence['others_jitter']:.1f} px"
            )
        right.text(max(multi[slot], 1) + 1.5, slot, note, va="center", fontsize=8, color="#444")

    right.set_yticks(y, [c["clip"].split("_seg_")[0] for c in clips], fontsize=8)
    right.set_xlim(0, 100)
    right.set_xlabel("frames containing more than one person blob (%)")
    right.set_title(
        "hero vs npc is barely exercised in this sample:\n"
        "five of six clips never show a second person, so it stays untested",
        fontsize=10,
    )
    right.grid(axis="x", alpha=0.25)

    fig.suptitle("The coarse taxonomy changes the semantic verdict; hero/npc needs a different lever", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        with np.load(CACHE) as stored:
            cache = {key: stored[key] for key in stored.files}
    else:
        cache = segment_all()
        np.savez_compressed(CACHE, **cache)

    data = analyse(cache)
    STATS.write_text(json.dumps(data, indent=1))
    plot(data, OUT / "coarse6.png")

    print("\nper-class IoU:")
    for cls in range(taxonomy.NUM_COARSE6):
        print(f"  {taxonomy.COARSE6_NAMES[cls]:12} {data['per_class_iou'][cls] * 100:5.1f}%  share {data['share'][cls] * 100:4.1f}%")
    print("\nhero evidence (high render):")
    for record in data["per_clip"]:
        h = record["high_hero"]
        others = f"{h['others_jitter']:.1f}" if h["others_jitter"] is not None else "  -"
        print(
            f"  {record['clip']:32} tracks {h['tracks']:3d}  multi-person {h['multi_person_frames'] * 100:3.0f}%"
            f"  persistence {h['hero_persistence']:.2f}  jitter {h['hero_jitter']:5.1f} vs others {others}"
        )
    print(f"\nwrote {OUT / 'coarse6.png'}")


if __name__ == "__main__":
    main()
