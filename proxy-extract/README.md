# proxy-extract

Extracts the Depth + Semantic-ID proxy conditions for the v2v dataset, writing a
`condition_root` directory that `code-world-model`'s `prepare` step consumes
unmodified. No changes to the inference repo are needed.

> First time here? Read [`../RUNBOOK.md`](../RUNBOOK.md) instead — it is the
> step-by-step version, in Chinese, with a Docker path. This file is the design
> rationale: why each piece is the way it is.

## What the output has to look like

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
not a geometric distortion - a 2x error costs 10.3% of full range. So one
well-estimated scale per clip is enough, and chasing per-frame absolute metric
accuracy is not worth it.

## Install

```bash
pip install -e .                 # contract, QC, encoding, preview - no GPU needed
pip install -e ".[depth]"        # MapAnything
pip install -e ".[semantic]"     # transformers segmentation
pip install -e ".[sam3]"         # SAM 3 concept refinement
```

Model backends are imported lazily, so the contract, taxonomy, QC and encoding
layers work on a laptop.

**The stages cannot share one environment.** SAM 3 needs Python >= 3.12 and
torch >= 2.7; `code-world-model` pins Python 3.10 and torch 2.9.1. Run extraction
separately and hand over the written `condition_root`.

## Use

```bash
# 1. Screen high/low pairs for geometric drift
proxy-extract qc --dataset handpick29_high_low --report qc_report.json

# 2. Extract. Outputs land at <out>/<track>/<clip>/, so high and low do not collide.
proxy-extract extract \
    --video handpick29_high_low/low/26_trevor_seg_0004.mp4 \
    --out conditions --depth-backend mapanything --semantic-backend ade20k

# 3. Look at what came out
proxy-extract preview --condition-root conditions/low/26_trevor_seg_0004 --out duv.mp4

# 4. Re-read it with every check the consumer applies
proxy-extract validate --condition-root conditions/low/26_trevor_seg_0004 --expect-frames 124
```

Then point a `code-world-model` config's `condition_root` at that directory.

## Stages

1. **Decode** to 1344x768, an exact 4x multiple of the 336x192 condition grid.
2. **Depth** - MapAnything over the whole clip at once.
3. **Calibrate** - scale from the GT camera baseline where available, otherwise
   the backend's own metric estimate. Recorded in the report either way.
4. **Semantic** - a closed-set ADE20K or Cityscapes model projected onto the 12
   classes, optionally refined by SAM 3 for the classes those sets cannot express.
5. **Reduce** to 336x192. Depth by block median, labels by block majority vote -
   averaging depth across a discontinuity invents surfaces, and nearest-neighbour
   sampling of labels keeps or drops thin structures at random.
6. **Stabilise** temporally, flow-compensated.
7. **Encode and validate** by reading the result back.

## Backend choices

**Depth: MapAnything.** It predicts metric scale natively and can ingest known
calibration. VGGT is up-to-scale only. VGGT-Omega scores better but its weights
are gated behind an automated approval that often refuses, and Meta flagged
possible benchmark contamination in the released 1B checkpoint on 2026-08-18.

**Semantic: closed-set trunk, SAM 3 for the gaps.** Six of the twelve classes are
"stuff" - unbounded regions with no instances - which is where concept detectors
are weakest and ADE20K models are strongest. SAM 3 is prompted only for `animal`
and `prop`, which no closed-set label set expresses, and for extra
`infrastructure` detail. SAM 2 cannot do this job at all: it is class-agnostic and
produces masks without labels.

Mappings are written by source-class **name**, not index, and resolved against
each checkpoint's own `id2label`. A checkpoint with a permuted label order would
otherwise silently relabel the whole dataset.

## Quality control

Two commands, and `camera-qc` is the one to use where GT tracks exist.

**`camera-qc` (preferred).** Tracks sparse features and measures how far they
fall from the epipolar lines the known poses predict — a Sampson distance in
pixels. Because the poses are independently trustworthy (median 0.43 px across
the 29 high renders), a large residual is evidence about the *render*, not about
the measurement. Tiers are `keep <= 1 px`, `review <= 3 px`, `drop > 3 px`.

On handpick29 the two tracks separate sharply, which is the whole argument for
extracting from the high render:

| track | keep | review | drop |
| --- | --- | --- | --- |
| high | 28 | 1 | 0 |
| low | 10 | 11 | 8 |

**`qc` (fallback, no cameras needed).** The low-poly track is an AI restyle, not
a re-render of the same 3D scene, so it is free to reinvent geometry. Art styles
differ too much to compare pixels, so this compares each clip's **own optical
flow field**: flow comes from camera motion and scene depth, so a restyle that
kept the geometry must reproduce it.

The grade is `EPE / motion`, with cosine reported alongside but not used to
decide. Flat, low-texture scenes produce near-random flow directions and tank
cosine even when alignment is fine - `26_trevor_seg_0004` scores cos 0.803 but
EPE/motion 0.209 and is visibly well aligned.

| Tier | EPE/motion | handpick29 |
| --- | --- | --- |
| keep | <= 0.22 | 12 |
| review | <= 0.45 | 9 |
| drop | > 0.45 | 8 |

Urban scenes hold up; natural ones (trees, rocks) drift most.

## Two things worth knowing before changing the code

**Flicker.** A majority or median vote over an odd-length window provably cannot
remove perfect frame-to-frame alternation: the window centred on a pixel always
holds one more copy of that pixel's own class, so the vote re-elects the flicker.
That is why `temporal.py` runs a second short-run suppression pass, and why
`test_a_majority_vote_alone_cannot_remove_it` exists.

**Output paths.** The high and low renders of a pair share a file name, so the
track directory is part of the output path. Keying on the stem alone silently
overwrote half the dataset.

## Tests

```bash
pytest
```

131 tests, no GPU required. The important ones import the real
`cwm_h3_inference.duv` loader and make it read our output, so the contract is
checked against its actual consumer rather than against this README. The
`synthetic` depth and semantic backends exist to run the full pipeline on a real
clip without a GPU; they are placeholders and must never be used for real data.
