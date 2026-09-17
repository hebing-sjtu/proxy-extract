"""SAM 2 as a temporal consistency layer over a closed-set segmentation.

**SAM 2 is not a classifier and this module does not pretend otherwise.** It is
class-agnostic: it emits masks with no labels, so it cannot produce a CWM class
id for any pixel by itself. What it has that no per-frame segmenter has is a
memory over the video, and that is the thing worth taking.

So the two are composed rather than swapped. The closed-set trunk
(`panoptic.py`) says *what*, per frame, and flickers; SAM 2 says *which pixels
are the same surface as before*, across the clip, and does not. The join is:

1. **Seed.** On the first frame, every connected component of the trunk's own
   label map that is bigger than `min_area_fraction` becomes a mask prompt.
   Seeding from the trunk rather than from SAM 2's automatic mask generator is
   what keeps the two aligned - a masklet is born already corresponding to a
   region the trunk had an opinion about - and it avoids loading a second model.
2. **Propagate.** SAM 2 carries those masklets through the clip with its own
   memory attention.
3. **Re-seed.** Every `reseed_every` frames, whatever the trunk claims that no
   masklet already covers is seeded too. This is how a car that drives into
   frame at second three gets tracked at all, and the interval is the delay
   before it does.
4. **Vote once per masklet, over the whole clip.** Each masklet's class is the
   majority of the trunk's votes inside it, pooled across every frame it
   appears in - one decision per masklet, not one per frame.

Step 4 is the entire point, and it is why this is a different kind of fix from
`temporal.py`. The window vote and the run filter there *suppress* flicker after
the fact, and README's `test_a_majority_vote_alone_cannot_remove_it` shows the
limit of that: an odd-length window centred on a pixel always holds one extra
copy of that pixel's own class, so perfect alternation re-elects itself. Here a
masklet has a single label for the clip, so within it frame-to-frame alternation
is not suppressed, it is unrepresentable. The residual flicker lives at masklet
boundaries and in whatever no masklet covers, which is the part `temporal.py`
is still good at - so both stages run, and this one runs first.

What it cannot do is invent a class the trunk never predicts. A masklet's vote
is drawn from the trunk's labels, so if ADE20K has no word for something, the
masklet is stable and wrong rather than flickering and wrong. `sam3.py` is
the module for that gap; the two are independent and can both be on.

Two deployment notes:

- **The clip has to arrive in one call.** `refine()` gets whatever batch the
  caller hands it, and SAM 2's memory only spans that batch, so a run with
  `--chunk-frames 64` gets two independent tracking passes over a 124-frame
  window and a seam between them. The batch length is recorded in the report.
  For the clip route pass `--chunk-frames` >= the window, or leave it unset.
- SAM 2's video predictor reads frames from a directory of JPEGs, so each call
  stages its batch into a temporary one. That costs a lossy re-encode of the
  *conditioning* pixels only: the labels come from the trunk, which saw the
  original frames, and nothing SAM 2 outputs is stored except mask geometry.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..taxonomy import NUM_CLASSES
from .base import SemanticResult

# SAM 2.1 large. The video predictor's cost is dominated by the number of
# objects tracked rather than by the backbone, so the small checkpoints buy less
# here than they would for per-frame work.
CHECKPOINT = "facebook/sam2.1-hiera-large"

# Frames between re-seeding passes. One second at 24 fps. Lower picks up
# entering objects sooner and costs a component pass plus however many new
# masklets it starts; higher leaves new content to the per-frame trunk for
# longer, which is the pre-existing behaviour rather than a new failure.
DEFAULT_RESEED_EVERY = 24

# A component smaller than this share of the frame is not seeded. At the
# 1344x768 work size that is about 1000 pixels, which survives the reduction to
# 192x336 as roughly one pixel - and a one-pixel masklet costs a full object
# slot in SAM 2's memory for the whole clip.
DEFAULT_MIN_AREA_FRACTION = 1.0e-3

# Hard ceiling on tracked masklets. SAM 2's memory attention is per object, so
# this is the knob that decides whether a clip fits on the card at all. Chosen
# by area, largest first: the biggest regions are both the cheapest to be right
# about and the ones whose flicker moves the most pixels.
DEFAULT_MAX_OBJECTS = 48

# How much of a component must already belong to some masklet before re-seeding
# skips it. Below 1.0 on purpose - a masklet's boundary drifts a little from the
# trunk's, and requiring exact coverage would re-seed the same road every
# second until the object cap was full of copies of it.
DEFAULT_COVERAGE_THRESHOLD = 0.7

# `owner` maps a pixel to the masklet that holds it, and -1 to none. int16
# rather than a stack of boolean masks per object: one owner map is 2 bytes a
# pixel a frame, where 48 separate masks would be 48, and at 124 frames of
# 1344x768 that is the difference between 250 MB and 6 GB.
NO_OWNER = -1
OWNER_DTYPE = np.int16


class Sam2ConsistencyRefiner:
    name = "sam2"

    def __init__(
        self,
        *,
        checkpoint: str = CHECKPOINT,
        device: str | None = None,
        reseed_every: int = DEFAULT_RESEED_EVERY,
        min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
        max_objects: int = DEFAULT_MAX_OBJECTS,
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
        jpeg_quality: int = 95,
    ) -> None:
        if reseed_every < 1:
            raise ValueError(f"reseed_every must be >= 1, got {reseed_every}")
        if max_objects < 1:
            raise ValueError(f"max_objects must be >= 1, got {max_objects}")
        if not 0.0 < coverage_threshold <= 1.0:
            raise ValueError(
                f"coverage_threshold must be in (0, 1], got {coverage_threshold}"
            )
        self.checkpoint = checkpoint
        self.device = device
        self.reseed_every = reseed_every
        self.min_area_fraction = min_area_fraction
        self.max_objects = max_objects
        self.coverage_threshold = coverage_threshold
        self.jpeg_quality = jpeg_quality
        self._predictor = None

    # -------------------------------------------------------------- the model

    def _load(self):
        if self._predictor is None:
            from ..accel import pick_device

            try:
                from sam2.sam2_video_predictor import SAM2VideoPredictor
            except ImportError as exc:  # pragma: no cover - environment specific
                raise ImportError(
                    "sam2 is not installed, and it is not on PyPI:\n"
                    "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                    "The checkpoints are on the Hub and are not gated, unlike SAM 3's."
                ) from exc

            self.device = pick_device(self.device)
            self._predictor = SAM2VideoPredictor.from_pretrained(
                self.checkpoint, device=self.device
            )
        return self._predictor

    def _autocast(self):
        """bfloat16 on CUDA, nothing anywhere else.

        SAM 2's own README runs its video examples under this, and the output
        here is a mask boundary, so the only thing reduced precision can move is
        which side of an edge a pixel falls on.
        """
        import torch

        device_type = str(self.device).split(":")[0]
        return torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"
        )

    # --------------------------------------------------------------- staging

    def _stage_frames(self, frames: list[np.ndarray], directory: Path) -> None:
        """Write the batch as the numerically-named JPEGs SAM 2 expects.

        `sam2.utils.misc.load_video_frames` sorts a directory by
        `int(splitext(name)[0])`, so the names have to be bare integers - a
        `frame_00000.jpg` raises ValueError inside the loader, several seconds
        into a call, with a message about an int conversion.
        """
        import cv2

        for index, frame in enumerate(frames):
            path = directory / f"{index:05d}.jpg"
            ok = cv2.imwrite(
                str(path),
                frame[:, :, ::-1],
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:  # pragma: no cover - a full or unwritable scratch dir
                raise RuntimeError(f"failed to stage a SAM 2 conditioning frame at {path}")

    # ----------------------------------------------------------- seed finding

    def _components(self, labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Connected components of a label map as `(class, mask)`, largest first.

        Per class rather than over the whole map, because two classes that
        happen to touch are one component of "not background" and must not
        become one masklet: a car parked against a wall would then vote as
        whichever of the two owns more pixels, and the other would lose its
        identity for the rest of the clip.
        """
        import cv2

        floor = self.min_area_fraction * labels.size
        found: list[tuple[int, int, np.ndarray]] = []
        for cls in np.unique(labels):
            binary = (labels == cls).astype(np.uint8)
            count, tagged = cv2.connectedComponents(binary, connectivity=8)
            for tag in range(1, count):
                mask = tagged == tag
                area = int(mask.sum())
                if area >= floor:
                    found.append((area, int(cls), mask))
        found.sort(key=lambda item: item[0], reverse=True)
        return [(cls, mask) for _area, cls, mask in found]

    def _seeds_for(
        self,
        labels: np.ndarray,
        owner: np.ndarray,
        running: np.ndarray,
        budget: int,
    ) -> list[np.ndarray]:
        """Components of `labels` that no masklet of the same class already holds.

        "Of the same class" is the part that took a test to get right. Asking
        only whether a pixel is owned means a car that drives onto a road is
        never seeded: it appears inside the road masklet, which owns 100% of it,
        so it reads as already covered and spends the rest of the clip labelled
        road. Since the thing being avoided is a *duplicate* masklet, the
        question is whether the incumbent agrees - and a masklet that disagrees
        is not a duplicate, it is the region this object is moving across.

        `running` is the vote so far rather than the final one, because the
        final one does not exist until the clip is over. It is a good estimate
        for exactly the masklets this matters for: a region big enough to
        swallow a new object has been accumulating votes since it was seeded.
        """
        seeds = []
        for cls, mask in self._components(labels):
            if len(seeds) >= budget:
                break
            held = owner[mask]
            agreed = (held != NO_OWNER) & (running[np.maximum(held, 0)] == cls)
            if agreed.mean() < self.coverage_threshold:
                seeds.append(mask)
        return seeds

    @staticmethod
    def _running_classes(votes: list[np.ndarray]) -> np.ndarray:
        """Each masklet's leading class given the votes counted so far."""
        if not votes:
            return np.full(1, -1, dtype=np.int16)
        return np.array([int(np.argmax(vote)) for vote in votes], dtype=np.int16)

    # ---------------------------------------------------------------- public

    def refine(self, frames: list[np.ndarray], base: SemanticResult) -> SemanticResult:
        """Replace the trunk's per-frame labels with one label per masklet."""
        if base.frames != len(frames):
            raise ValueError(f"base has {base.frames} frames for {len(frames)} images")
        if not frames:
            raise ValueError("refine() needs at least one frame")

        height, width = frames[0].shape[:2]
        if base.labels.shape[1:] != (height, width):
            raise ValueError(
                f"base labels are {base.labels.shape[1:]} for {height}x{width} frames"
            )

        predictor = self._load()
        owner = np.full((len(frames), height, width), NO_OWNER, dtype=OWNER_DTYPE)
        # Votes pooled over the clip: one row per masklet, one column per CWM
        # class. This is the array that makes a masklet's label a property of
        # the clip rather than of a frame.
        votes: list[np.ndarray] = []
        seeded_at: list[int] = []

        import torch

        with tempfile.TemporaryDirectory(prefix="sam2-") as scratch:
            staging = Path(scratch)
            self._stage_frames(frames, staging)

            with torch.inference_mode(), self._autocast():
                state = predictor.init_state(
                    video_path=str(staging),
                    # The batch is already decoded in host memory; letting SAM 2
                    # hold a second copy of it on the card is what pushes a
                    # 124-frame window past a 24 GB budget once a depth model is
                    # resident too.
                    offload_video_to_cpu=True,
                )
                for start in range(0, len(frames), self.reseed_every):
                    budget = self.max_objects - len(votes)
                    if budget > 0:
                        for mask in self._seeds_for(
                            base.labels[start],
                            owner[start],
                            self._running_classes(votes),
                            budget,
                        ):
                            predictor.add_new_mask(
                                inference_state=state,
                                frame_idx=start,
                                obj_id=len(votes),
                                mask=mask,
                            )
                            votes.append(np.zeros(NUM_CLASSES, dtype=np.int64))
                            seeded_at.append(start)
                    if not votes:
                        # Nothing in this frame was big enough to seed, and
                        # nothing is being tracked yet, so there is nothing to
                        # propagate. Skip rather than call the predictor with an
                        # empty prompt set, which raises inside the preflight.
                        continue

                    # `propagate_in_video` is inclusive at both ends - it tracks
                    # `range(start, min(start + max, last) + 1)` - so each pass
                    # re-tracks one frame the previous pass already did. That
                    # overlap is what the next re-seed reads: the coverage test
                    # needs `owner[start]` to be filled, and it is only filled
                    # if the previous pass reached that frame.
                    for index, object_ids, logits in predictor.propagate_in_video(
                        inference_state=state,
                        start_frame_idx=start,
                        max_frame_num_to_track=self.reseed_every,
                    ):
                        self._record(
                            owner[index], base.labels[index], votes, object_ids, logits
                        )

        return self._compose(base, owner, votes, seeded_at)

    def _record(
        self,
        owner: np.ndarray,
        labels: np.ndarray,
        votes: list[np.ndarray],
        object_ids,
        logits,
    ) -> None:
        """Write one frame's masks into the owner map and tally their votes.

        Painted largest first, so the smallest masklet holding a pixel is the
        one that keeps it. A single owner per pixel is all the output needs -
        it is one class id per pixel in the end - and resolving the overlap here
        rather than keeping every mask is what bounds this to one int16 map a
        frame.

        Area order rather than taxonomy priority, because a masklet's class is
        not known yet: it is being voted on by this very loop. Area is the
        available proxy and a good one - a person in front of a building is a
        smaller region than the building - and the case it gets wrong, a large
        thing in front of a small one, does not arise in a single frame's
        components.
        """
        masks = []
        for position, object_id in enumerate(object_ids):
            mask = np.asarray(_to_numpy(logits[position]) > 0.0).reshape(
                owner.shape[-2:]
            )
            area = int(mask.sum())
            if area:
                masks.append((area, int(object_id), mask))
        masks.sort(key=lambda item: item[0], reverse=True)

        for _area, object_id, mask in masks:
            owner[mask] = object_id
            votes[object_id] += np.bincount(
                labels[mask].ravel(), minlength=NUM_CLASSES
            )[:NUM_CLASSES]

    def _compose(
        self,
        base: SemanticResult,
        owner: np.ndarray,
        votes: list[np.ndarray],
        seeded_at: list[int],
    ) -> SemanticResult:
        """Paint each masklet's voted class over the trunk's per-frame labels.

        Pixels no masklet reached keep what the trunk said for that frame.
        That is the pre-existing behaviour rather than a regression, and it is
        what `temporal.py` still has to clean up afterwards - which is why the
        covered fraction is in the report: it is the share of the frame for
        which flicker has been made structurally impossible.
        """
        labels = base.labels.copy()
        if votes:
            classes = np.array(
                [int(np.argmax(vote)) for vote in votes], dtype=np.uint8
            )
            held = owner != NO_OWNER
            labels[held] = classes[owner[held]]
        else:
            classes = np.zeros(0, dtype=np.uint8)

        return SemanticResult(
            labels=labels,
            confidence=base.confidence,
            meta={
                **base.meta,
                "sam2_checkpoint": self.checkpoint,
                "sam2_masklets": len(votes),
                "sam2_masklets_at_cap": len(votes) >= self.max_objects,
                "sam2_reseed_every": self.reseed_every,
                # Where the clip's masklets were born. All zeros means the first
                # frame explained the clip; a long tail means content kept
                # entering, and those objects were on the trunk's per-frame
                # labels until their seed frame.
                "sam2_seed_frames": sorted(set(seeded_at)),
                "sam2_covered_fraction": round(
                    float((owner != NO_OWNER).mean()), 6
                ),
                "sam2_masklet_classes": {
                    int(cls): int(count)
                    for cls, count in zip(*np.unique(classes, return_counts=True))
                },
                # SAM 2's memory spans one call, so a caller that batches has
                # split the clip into independent tracking passes. 124 here
                # means the window was tracked whole.
                "sam2_frames_in_call": len(base.labels),
            },
        )


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    return np.asarray(value)
