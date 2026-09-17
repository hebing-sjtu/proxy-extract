# proxy-extract

Predicts depth and semantics for RGB-only video corpora, and writes them in the
two formats downstream needs.

> First time here? Read [`../RUNBOOK.md`](../RUNBOOK.md) instead — it is the
> step-by-step version, in Chinese. This file is the design rationale: why each
> piece is the way it is.

## Two deliverables

They share every model and every post-processing stage; only the resolution and
the on-disk form differ. Confusing them is the most common source of trouble.

| | `scenes` → `seg_NNNNNN` | `extract` → `condition_root` |
| --- | --- | --- |
| Consumer | the delivered dataset, per `../DATA_F.md` | `code-world-model`'s `prepare`, unmodified |
| Resolution | 1280x720 | 336x192 |
| On disk | four mp4s under `proxy/` + `annotations.tar` | `NNNNNN.depth.f32` + `NNNNNN.semantic_id.png` |
| Length | the whole episode | truncated to `124 + 90k` to fit the window |
| Code | `delivery.py` | `pipeline.py` |

`scenes` deliberately does **not** apply the 336x192 reduction. It throws away
93% of the pixels and is cheap to redo from the delivered videos later, so doing
it at delivery time would only mean the grid could never be revisited.

### And a third, for `PROXY_DUV_SPEC.md`

`proxy_duv.py` writes the `condition_root` files *into* a cut clip, as
`<clip>/duv/`, alongside the clip's own target video. Same bytes as the
`extract` deliverable, different place, and it is turned on with `--proxy-duv`.

It exists as its own module rather than as a flag on `extract` because of one
trap that `../DATA_F.md` sets. A clip cut by this pipeline **already** carries a
composed `proxy/duv.mp4`, and that file is valid, loadable, and in the wrong
palette for this consumer — `DATA_F.md` scales the R channel over 0.1–8000 m and
puts semantics in G/B, where this spec wants metric view-z and a separate 8-bit
id plane. Naming `proxy_duv_video` in the manifest instead of `proxy_duv` is
therefore the one mistake that makes the consumer read the wrong one of two
files that both exist and are both internally consistent. Section 7's three
proxy keys are mutually exclusive for this reason, and
`test_the_manifest_names_proxy_duv_and_never_the_video` is what holds the line.

```bash
proxy-extract clips --out <dataset_root> --clips-out <clips> --proxy-duv
proxy-extract proxy-duv-manifest --root <clips> --prompts prompts.json
proxy-extract proxy-duv-audit --root <clips>
```

The audit is the part to actually run. Its per-segment checks only repeat what
the consumer's loader already asserts; the check that earns its keep is the
**cross-segment** depth median spread, because per-segment depth normalisation
passes every per-frame test there is.

### The condition_root contract

`code-world-model` does not leave this format open. Per source-frame ordinal:

| File | Format |
| --- | --- |
| `NNNNNN.depth.f32` | headerless C-order little-endian float32, **192x336**, metres, exactly 258,048 bytes |
| `NNNNNN.semantic_id.png` | 8-bit grayscale PNG, **336x192**, values in `[0, 11]` |

Ordinals must be contiguous from `000000`, and a run must be `124 + 90 * (n - 1)`
frames long. The 12 classes are `void_unknown, sky, water, terrain, road_paved,
vegetation, building_structure, infrastructure, human, animal, vehicle, prop`.

Depth is encoded logarithmically downstream:

```
code = (ln(256) - ln(d)) / (ln(256) - ln(0.3)) * 65535
```

That has a useful consequence. A **global scale error is a uniform code offset**,
not a geometric distortion — a 2x error costs 10.3% of full range. So one
well-estimated scale per clip is enough, and chasing per-frame absolute metric
accuracy is not worth it.

## Install

```bash
pip install -e .                 # contract, taxonomy, encoding, preview — no GPU
pip install -e ".[depth]"        # model backends
pip install -e ".[semantic]"     # transformers segmentation
pip install -e ".[sam3]"         # SAM 3 concept refinement
```

The two anti-flicker backends are git-only, so they are deliberately not extras
here: naming a package that is not on PyPI would make the extra uninstallable
rather than optional, which is the same reason `mapanything` is absent.

```bash
pip install git+https://github.com/facebookresearch/sam2.git    # --refiner sam2
pip install git+https://github.com/microsoft/MoGe.git           # --depth-backend moge3
```

MoGe-3 does not install on macOS at all: FlexGEMM builds on Triton, and Triton
publishes no macOS wheels. `../scripts/doctor.py` reports that as unavailable
rather than missing, because there is no install to suggest.

Model backends are imported lazily, so the contract, taxonomy and encoding layers
work on a laptop. In practice use `../scripts/setup_venv.sh`, which installs the
pinned `../requirements.txt` these measurements were taken on.

**The stages cannot share one environment.** SAM 3 needs Python >= 3.12 and
torch >= 2.7; `code-world-model` pins Python 3.10 and torch 2.9.1. Run extraction
separately and hand over what it wrote.

## Use

```bash
# The delivery run. See RUNBOOK section 3; ../scripts/run_scenes.sh wraps this
# with sharding, resume and a preflight.
proxy-extract scenes --video <corpus>/data --recursive --out <dataset_root> \
    --semantic-backend standard11 --depth-backend depth_anything_v3 \
    --resume --keep-going
proxy-extract scenes-audit --out <dataset_root>
proxy-extract scenes-preview --scene <dataset_root>/seg_000000 --out sheet.png

# The condition_root run.
proxy-extract extract --video <corpus>/data --recursive --out <out> \
    --semantic-backend coarse6 --depth-backend depth_anything --chunk-frames 124
proxy-extract preview  --condition-root <out>/<clip> --out duv.mp4
proxy-extract validate --condition-root <out>/<clip> --expect-frames 124
```

## Stages

1. **Decode** to 1344x768 (`extract`) or 1280x720 (`scenes`). The former is an
   exact 4x multiple of the condition grid, which lets the reducers use clean
   block reductions instead of resampling; the latter is exactly 2/3 of ABot's
   1920x1080, so it introduces no aspect distortion.
2. **Depth** — a metric backend, per batch of frames.
3. **Calibrate** — scale from a GT camera baseline where one exists, otherwise
   the backend's own metric estimate. Recorded in the report either way.
4. **Semantic** — a closed-set ADE20K or Cityscapes model projected onto the
   target taxonomy, optionally refined by SAM 3 for classes those sets cannot
   express.
5. **Reduce** to 336x192 (`extract` only). Depth by block median, labels by block
   majority vote — averaging depth across a discontinuity invents surfaces, and
   nearest-neighbour sampling of labels keeps or drops thin structures at random.
6. **Stabilise** temporally, flow-compensated.
7. **Split the protagonist** out of the person class, then **encode**.

## Backend choices

**Depth.** The binding constraint is not accuracy: it is whether the weights can
be obtained on the node at all, and whether the backend will declare itself
metric. `scenes` refuses up-to-scale depth outright, because the delivery videos
encode absolute metres and a COLMAP sparse model — the only geometry the corpus
ships — is itself defined only up to a similarity. `depth_anything_v3` is the
default because it is the one that carries its DINOv2 backbone inside its own
checkpoint; `mapanything` pulls that backbone through `torch.hub` from a host
most egress allowlists do not cover, and hangs silently where it is blocked.
`moge3` is the one to reach for when the delivery flickers — see below. RUNBOOK
section 5 has the full comparison.

**Semantic: closed-set trunk, SAM 3 for the gaps.** Six of the twelve CWM classes
are "stuff" — unbounded regions with no instances — which is where concept
detectors are weakest and ADE20K models are strongest. SAM 3 is prompted only for
`animal` and `prop`, which no closed-set label set expresses, and for extra
`infrastructure` detail.

SAM 2 cannot do that job at all — it is class-agnostic and produces masks with no
labels — but it is the answer to a different question, which is the next section.

Mappings are written by source-class **name**, not index, and resolved against
each checkpoint's own `id2label`. A checkpoint with a permuted label order would
otherwise silently relabel the whole dataset.

## Flicker, and why swapping the checkpoint does not fix it

**A vote cannot remove alternation, and a per-pixel filter cannot see a global
error.** Those are two separate limits, and between them they are the whole
reason `--depth-backend moge3 --refiner sam2` exists.

The first: a majority or median vote over an odd-length window provably cannot
remove perfect frame-to-frame alternation, because the window centred on a pixel
always holds one more copy of that pixel's own class, so the vote re-elects the
flicker. That is why `temporal.py` runs a second short-run suppression pass, and
why `test_a_majority_vote_alone_cannot_remove_it` exists.

The second: a monocular model run frame by frame re-decides two **global**
quantities per frame — the field of view and the metric scale — and each of them
moves every pixel of the frame by the same factor at once. A windowed per-pixel
median cannot reach that, because the neighbours it votes against are wrong
identically.

So both new backends work by moving a decision from per frame to per clip:

| | per frame, flickers | per clip, cannot flicker |
| --- | --- | --- |
| `moge3` | FOV and metric scale re-inferred | one probed FOV, one levelled scale |
| `sam2` | a label per pixel per frame | one label per masklet, voted over the clip |

`sam2` is worth being precise about, because it is not a segmenter here. The
closed-set trunk says *what*, per frame, and flickers; SAM 2 says *which pixels
are the same surface as before*, across the clip, and does not. Masklets are
seeded from the trunk's own connected components, propagated by SAM 2's video
memory, and then each masklet takes the majority of the trunk's votes pooled over
every frame it appears in. Within a masklet, alternation is not suppressed — it
is unrepresentable. What is left over lives at masklet boundaries and in
whatever no masklet covers, which is what `temporal.py` is still good at, so
both stages run and this one runs first.

**The dangerous half of the depth fix.** `temporal.lock_depth_scale` removes the
high-frequency part of the scale drift and leaves the clip's absolute level
alone, and that distinction is the entire game: renormalising per clip looks
identical in every shape, dtype and range check, and it is the one mistake that
makes a whole batch scrap (`PROXY_DUV_SPEC.md` section 2). The guarantee is that
the mean log correction over a clip is exactly zero, so the geometric mean depth
— where the encoder's log codes sit — comes out as it went in. It is reported as
`scale_mean_log_correction` so a reader can check the claim instead of trusting
it, and `test_the_scale_lock_leaves_the_clips_metric_level_alone` pins it.

Both of `moge3`'s locks are **per call**. `--chunk-frames 32` over a 124-frame
window re-probes the camera and re-levels the scale four times and leaves a step
in depth at each boundary; `frames_in_call` in the report is what explains a
seam. For the clip route, pass `--chunk-frames` >= the window or leave it unset.

## One more thing worth knowing before changing the code

**Output paths.** ABot names every episode `video.mp4` and distinguishes them by
the directory above it, so the parent directory is part of both the scene's
`sample_id` and the condition_root's path. Keying on the stem alone collapses the
whole corpus onto one output.

## Tests

```bash
pytest
```

No GPU and no corpus required — the fixtures synthesise their own footage. The
important ones import the real `cwm_h3_inference.duv` loader and make it read our
output, so the contract is checked against its actual consumer rather than
against this README, and the packaging ones assert that `../RUNBOOK.md`,
`../DATA_F.md` and `../Makefile` still describe the layout the code writes.

The `synthetic` depth and semantic backends exist to run the full pipeline
without a GPU. They are placeholders whose output is structurally
indistinguishable from real output, so every run that uses them says so in its
own `extraction_report.json`; they must never be used for real data.
