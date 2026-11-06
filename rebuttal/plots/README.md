# rebuttal/plots — figure layout

Figures are grouped by what they show. Each `_best` variant uses the behaviorally-selected
best head for the head-based models (CNN-/ViT-Transformer, Molmo, Qwen); the unsuffixed /
`_mean` variant uses the predefined mean-over-heads rule. LSTMs have a single attention map,
so `mean == best` for them. The **best-head** figures are currently the featured version;
the averaged-layer (mean) counterparts live in `archive/`.

## wholebrain/ — whole-brain bar figures (one number per model)
Featured whole-brain figures use the **best-head** rule (`_best`); the averaged-layer (mean)
counterparts have been moved to `archive/` (see below).
- `nsd_identical_manuscript_plot_attn_only_no_control_for_clip_best.png`
  Manuscript Fig m6c Panel A reproduced on NSD: attention-**only** R² (no CLIP control).
  The existence check / like-for-like replacement of the old TRIBE panel. Not the headline.
- `nsd_wholebrain_clip_controlled_best.png`
  The controlled companion: unique attention variance = R²(attn+CLIP) − R²(CLIP-only), with
  the Shuffled floor drawn in and paired-t-vs-floor stars. This is the honest whole-brain claim.
- `nsd_primary_manuscript_style_best.png`
  Near-duplicate of the attn-only figure produced by `plot_story_panels.py` (kept for now;
  slated to be retired in favor of the `nsd_identical_manuscript_plot_*` version).
- `*_sigtests.csv` — the paired-t tables behind each bar figure.

## regional/ — where in cortex the signal lives
Featured regional figures use the **best-head** rule (`_best`); the averaged-layer (mean)
counterparts have been moved to `archive/` (see below).
- `nsd_roi_regional_best_holm_corrected_across_model.png` — unique attn-over-CLIP by cortical
  stream (early / ventral / lateral / parietal), floor-corrected vs Shuffled. Stars: Holm WITHIN
  ROI×task **across models** (a model's stars can shift with the head rule because its Holm
  neighbours change — even when its own map does not). Answers "is this model special WITHIN this
  region relative to the others?". Written by `plot_story_panels.py`.
- `nsd_roi_regional_best_holm_corrected_within_model.png` — same bars, but stars Holm-corrected
  WITHIN model×task **across the four streams**. A model's correction then depends only on its own
  p-values, so fixed-map models (both LSTMs: mean==best) get identical stars under either head
  rule. Answers "does THIS model beat its own floor across regions?". Written by
  `plot_regional_rule_invariant.py`.

## archive/ — superseded averaged-layer (mean) figures
Kept for reference; not the featured version. NOTE: re-running any of the plot scripts with
`--head-rule mean` regenerates these into their original subdir (`regional/` or `wholebrain/`),
so they must be re-archived after any such run.

Regional:
- `nsd_roi_regional_holm_corrected_{across,within}_model.png` — the mean/averaged-layer ROI figures.
- `nsd_brainmap_CNN_Transformer_{social,describe}_mean.png` — m6c Panel B surface maps
  (Where / What / Shared variance partition) for CNN-Transformer.

Whole-brain:
- `nsd_identical_manuscript_plot_attn_only_no_control_for_clip_mean.png` (+ `_sigtests.csv`)
- `nsd_wholebrain_clip_controlled_mean.png` (+ `_sigtests.csv`)
- `nsd_primary_manuscript_style.png`

## cross_dataset/ — TRIBE (simulated) vs NSD (real), and the eye-tracking overlap
- `tribe_vs_nsd_consistency{,_best}.png` — same metric on both datasets + cross-dataset ρ.
- `nsd_exploratory_overlap.png` — the 106-image COCO∩NSD overlap (the only set with human gaze).

## head_selection/ — Task 1 / Task 3c behavioral figures
Both scripts stay at the plots/ root because they compute `REBUTTAL = dirname(HERE)`.
- `fig_head_selection_corrected.png` — produced by `plot_fig_head_selection_corrected.py`.
  CV per-task head selection (Task 1) + corrected human between-task reference (Task 2), raw metric.
- `fig_center_corrected.png` — raw vs center-bias-controlled alignment for all six agents plus
  the center prior, each normalized by the ceiling under its own metric, all Transformers/VLMs on
  cross-validated heads. The figure behind "under center-bias control Molmo becomes the
  best-aligned agent".
- `fig_layer_mean_vs_best_head_controlling_for_center_bias.png` — the AC's averaging remedy
  (all layers × heads) for the two trained Transformer decoders, against their CV best head
  and the matched LSTM.

  Both written by `plot_fig_center_corrected.py` from `task3_baselines/run_headrule_center.py`.
  **These supersede the center-controlled numbers in `writeups/task3_baselines.md`**, which used
  the paper's fixed heads for the trained Transformers — see `writeups/task3c_headrule_center.md`.
- `fig_per_layer.png` — the same comparison decomposed one bar per decoder layer (each layer's
  8 heads averaged). Written by `plot_fig_per_layer.py`. Shows that layers 1–2 sit at the center
  prior and layers 3–4 carry the all-layer mean; CNN-Transformer L4 beats CNN-LSTM in Describe.
- `fig_metric_{CC,NSS,AUC,sAUC,KL,SIM,IG}.png` + `fig_metric_overview.png` — the seven-metric
  saliency suite on CV heads, Describe and Social, each normalized by that metric's own
  split-half ceiling, with paired-permutation stars on the two decoder contrasts. Written by
  `plot_fig_metric_suite.py` from `task4_metric_suite/run_metric_suite_cv.py`.
  **Supersedes `writeups/task4_metric_suite.md`** — see `writeups/task4b_metric_suite_cv.md`.

## Which script writes what
| script (in task8_nsd/, except where noted) | output subdir |
|---|---|
| `plot_manuscript_identical.py`        | wholebrain/ |
| `plot_wholebrain_clip_controlled.py`  | wholebrain/ |
| `plot_story_panels.py`                | wholebrain/, regional/, cross_dataset/ |
| `plot_regional_rule_invariant.py`     | regional/ (rule-invariant stars) |
| `plot_nsd_brainmap.py`                | regional/ |
| `plots/plot_fig_head_selection_corrected.py` | head_selection/ |
| `plots/plot_fig_center_corrected.py`  | head_selection/ |
| `plots/plot_fig_per_layer.py`         | head_selection/ |
| `plots/plot_fig_metric_suite.py`      | head_selection/ |
