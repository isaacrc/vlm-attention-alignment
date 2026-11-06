# Task 8 Option A — NSD encoding analysis

**Question.** Which model's attention best predicts brain responses in real fMRI?

Replaces the TRIBE simulated-response analysis that reviewers called "the weakest part of
the paper" with n=1000 real images, 8 real subjects, and a genuine cross-subject noise
ceiling.

## Design

| | TRIBE (current paper) | NSD (this analysis) |
|---|---|---|
| Responses | TRIBE v2 *simulated* | real fMRI, `betas_fithrf_GLMdenoise_RR` |
| Images | 200 | 1000 (shared-1000) |
| Subjects | — | 8, each with 3 repetitions |
| Noise ceiling | none | cross-subject, max r = 0.701 |
| Space | fsaverage5, 20484 vertices | same |

**Primary.** Per subject, banded ridge on shared-1000: attention band (196) + CLIP
ViT-L/14 band (768). 5-fold CV. Vertices restricted to mean ncsnr > 0.2 (5,054 of 20,484).
Bands/solver/CV mirror `tribe/scripts/3b_banded_encoding_model.py` so the two analyses are
directly comparable.

**Head selection.** Transformer attention uses the MEAN over all 8 layer-3 heads — a
predefined rule, addressing rebuttal Task 1. `--head-rule best` reproduces the biased
selection for comparison.

**Exploratory.** The 106 stimuli shared between our 200 and NSD (98 with usable data) are
the only images with both our fixations and real fMRI. Models trained on shared-1000 —
disjoint from the overlap, so no leakage — are applied frozen to them.

## Verified facts (not assumed)

- fsaverage5 is exactly the first 10,242 vertices of fsaverage (max diff 1.03e-07), so
  betas can be sliced by byte range.
- Trial→beta-frame mapping validated against off-by-one controls: correct mapping gives
  within-image repeat r = 0.671 vs between-image 0.479; both wrong offsets give 0.000.
- `imgBrick` in nsd_stimuli.hdf5 is contiguous; stimuli pulled by range request are
  byte-identical to h5py reads.
- Shared-1000 is 100% COCO train2017, which is why none of our val2017 stimuli are in it.
- Subjects 3/4/6/8 stopped at 32/30/32/30 sessions; their shared sets are 930/907/930/907.
  907 images are common to all 8.

See `FINDINGS_overlap.md` and `FINDINGS_scoping.md` for the full derivations.

## Pipeline

```bash
python build_trial_index.py       # which trials to fetch
python download_betas.py          # 1.5 GB, ~10 min, ranged reads
python download_stimuli.py        # 1098 images, 343 MB, 33 s
python prepare_responses.py       # session z-score, rep-average, noise ceiling
sbatch slurm_extract_trained.sh   # 4 trained models x 2 tasks
sbatch slurm_qwen_nsd.sh          # Qwen3.5-9B x 2 tasks
sbatch slurm_molmo_nsd.sh         # Molmo 7B-D x 2 tasks
sbatch slurm_clip.sh              # CLIP ViT-L/14 "what" band
sbatch slurm_fit_encoding.sh      # banded ridge, one job per subject
python summarize_encoding.py      # model comparison + paired contrasts
python build_rois.py              # NSD streams atlas -> fsaverage5 masks
python summarize_by_roi.py        # the same fits, split by cortical stream
python plot_nsd_encoding.py       # figure
sbatch slurm_fit_overlap.sh       # exploratory overlap analysis
```

## Read the ROI split, not the pooled number

Pooling over high-SNR vertices mixes early visual cortex with ventral/lateral cortex. The
top decile by noise ceiling is 34.6% early visual against 12.5% of all selected vertices --
a 2.8x enrichment. Early visual responses are driven by local contrast, so a Sobel edge map
is a strong predictor there almost by construction, and the pooled ranking partly measures
"which map most resembles a contrast map."

`summarize_by_roi.py` regroups the *existing* per-vertex results by NSD's `streams` atlas
(no refitting). The two regimes separate cleanly:

- **early**: EdgeEnergy wins outright. No model comes close. Expected.
- **ventral**: CNN-LSTM (+0.0058) beats EdgeEnergy (+0.0047); Luminance falls to ~0; the
  LSTM > Transformer ordering holds in both tasks.

That contrast is the defensible claim: model attention adds nothing over low-level contrast
in early visual cortex, but CNN-LSTM's attention does add unique variance where semantic
content is represented. Reporting the pooled number alone would invite exactly the
"rediscovering gaze priors" objection that dMXa and hmYo raise.

Always report EdgeEnergy alongside the models. A model that does not beat it in a given
stream has explained nothing there that contrast does not.

## Three metric traps found here (relevant to the other rebuttal tasks)

Each of these produced a plausible-looking number that measured the wrong thing. All three
were caught by an explicit control, not by inspection.

1. **Constant-map baselines are degenerate in an encoding model.** A Center-Gaussian or
   mean-attention map is the right control for per-image fixation alignment (Task 3), but
   an encoding model explains variance ACROSS images, so a map identical for every image
   contributes nothing by construction and returns ~0 regardless of merit. The controls
   that matter here must vary per image: Luminance, EdgeEnergy, Shuffled.

2. **Differencing two independently tuned models has a positive bias.**
   `R2(banded) - R2(clip-only)` is nonzero even when the attention band is uninformative,
   because each fit picks its own regularisation. In ventral cortex that bias (+0.0030) is
   as large as the model effects. It cancels in model-vs-model contrasts but not in
   "is this above zero" claims. Center and Shuffled bracket it.

3. **A dominant nuisance band saturates per-image scores.** In the overlap analysis the
   banded per-image prediction is ~95% CLIP: a *scrambled* attention map scored 0.294
   against CNN-LSTM's 0.314. Correlating that against fixation alignment returns null
   almost regardless of the truth. Fixed by scoring an attention-only model and a per-image
   `r_gain` instead.

The general rule: any measure built from a difference, or from a model with a dominant
nuisance term, needs a control that pins its floor before the numbers mean anything.
**Task 4's metric suite has the same exposure** -- ceiling-normalising each metric is not
sufficient if the underlying quantity is saturated by a shared component.

## Scale of the effects -- state this plainly

Attention-unique R2 is ~0.006 in ventral cortex against a noise ceiling of ~0.25, and the
attention band receives under 1% of the banded-ridge kernel weight against CLIP's 99%.
The supported claim is narrow: *given* a strong semantic baseline, model attention adds a
small, consistent increment that differs by architecture. It is not "attention maps predict
visual cortex."

## Notes

- Compute nodes have no internet: set `HF_HOME=/scratch/gpfs/GRAZIANO/CANS/.hf_cache` and
  `HF_HUB_OFFLINE=1` for CLIP (ViT-L/14 is cached there, not in `GRAZIANO/hf_cache`).
- `cans-env` torch needs `module load anaconda3/2022.5` plus
  `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` — the compute nodes' `/lib64/libstdc++.so.6`
  predates GLIBCXX_3.4.30.
- The encoding fit needs more memory than a login node allows (OOM at exit 137); run it
  through SLURM.
- The Molmo NSD variant reduces attention to `(28 layers, 28 heads, 196)` in-loop, ~307 KB
  per image instead of ~55 MB, avoiding ~120 GB of raw activations.
