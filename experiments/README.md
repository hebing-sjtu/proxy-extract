# Feasibility experiments

Everything here was run on the Mac workstation against `../handpick29_high_low`,
using real model weights on the laptop GPU — no synthetic stand-ins. Each script
caches its measurements next to itself as JSON, so re-running only redraws the
figure unless you delete the cache.

**`figures/` and `cond_*/` are not in version control**, and neither is the
corpus: they are frames of commercial game footage. What survives in the repo is
the measurement — the `*.json` caches below and the findings written up at the
end of this file. Redraw a figure by running its script against the corpus.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=$PWD/.mplcache
PYTHONPATH=proxy-extract/src:experiments .venv/bin/python experiments/fig_camera_qc.py
```

| script | question | cache | figure |
| --- | --- | --- | --- |
| `fig_camera_qc.py` | Do the delivered COLMAP tracks describe these videos, and which render? | `camera_qc.json` | `figures/camera_qc_overview.png`, `figures/camera_qc_detail.png` |
| `fig_appearance.py` | What does the low-poly render actually keep and lose? | — | `figures/appearance_gap.png` |
| `fig_scale.py` | How many metres is one world unit? | `scale.json` | `figures/scale_calibration.png` |
| `fig_human_scale.py` | Same question, measured without a depth model | `human_scale.json` | `figures/scale_crosscheck.png` |
| `fig_feasibility.py` | Does the low render support the same extraction as the high one? | `feasibility.json` | `figures/feasibility.png` |
| `fig_coarse6.py` | Same question under the 6-class taxonomy, plus: is hero/npc separable? | `coarse6.json`, `coarse6_labels.npz` | `figures/coarse6.png` |
| `fig_hero.py` | Did the tracker pick the protagonist on the one crowded clip? | reads `cond_c6/` | `figures/hero_split.png` |
| `fig_condition.py` | What does a finished clip actually look like? | reads `cond_c6/` | `figures/condition_output.png` |

`cond_real/{high,low}/26_trevor_seg_0004/` holds a full 124-frame `condition_root`
produced end to end by the real pipeline, with `figures/duv_readback.png` and
`figures/duv_{high,low}.mp4` rendered back out of it. `cond_c6/high/` holds the
same thing under the 6-class taxonomy, previewed as
`figures/duv_c6_*.mp4`.

`fig_condition.py` is the odd one out: it asks no question and proves nothing.
It exists so the runbook can show a newcomer what a correct run looks like.

## What they found

- **The camera tracks are trustworthy.** Median Sampson residual of 0.43 px
  across all 29 high renders. They are also *not metric* — COLMAP fixes geometry
  only up to a similarity, and the world unit lands somewhere between 0.8 and
  2.0 metres depending on the clip.
- **Depth survives the low-poly render**, differing from the high-render
  prediction by ~10% in the median.
- **Semantics largely do not.** Mean IoU between the two renders is ~35%, and
  `vegetation` collapses from 81% (as `road`/`sky` score) to 15%.
- **Coarsening to 6 classes fixes most of that but not vegetation**: background
  84%, road 77%, npc 53%, vehicle 43%, vegetation 21%.
- **`hero` vs `npc` is untested by this sample.** Five of six clips never show a
  second person. In the one that does, the protagonist's silhouette wanders
  3.1 px against 14.3 px for everyone else, which is the signal a tracker would
  use — but that is a single clip.

## Caveats worth keeping attached to these numbers

- Depth Anything V2 Small is a stand-in for the production depth backend. It is
  single-view, so it says nothing about temporal consistency, and its metric
  head saturates near 80 m.
- The high render's own prediction is used as the reference throughout. It is
  not ground truth; it is the answer the pipeline would get with good pixels.
- Two independent estimates of metres-per-unit disagree, so the absolute factor
  is not pinned down. Only the conclusion that it is clip-dependent is safe.
