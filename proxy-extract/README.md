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
RUNBOOK section 5 has the full comparison.

**Semantic: closed-set trunk, SAM 3 for the gaps.** Six of the twelve CWM classes
are "stuff" — unbounded regions with no instances — which is where concept
detectors are weakest and ADE20K models are strongest. SAM 3 is prompted only for
`animal` and `prop`, which no closed-set label set expresses, and for extra
`infrastructure` detail. SAM 2 cannot do this job at all: it is class-agnostic
and produces masks without labels.

Mappings are written by source-class **name**, not index, and resolved against
each checkpoint's own `id2label`. A checkpoint with a permuted label order would
otherwise silently relabel the whole dataset.

## Two things worth knowing before changing the code

**Flicker.** A majority or median vote over an odd-length window provably cannot
remove perfect frame-to-frame alternation: the window centred on a pixel always
holds one more copy of that pixel's own class, so the vote re-elects the flicker.
That is why `temporal.py` runs a second short-run suppression pass, and why
`test_a_majority_vote_alone_cannot_remove_it` exists.

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
