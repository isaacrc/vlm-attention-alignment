"""
rerun_appendix.py
-----------------
Re-run the behavioral appendix analyses that involve Qwen3.5 on the thinking-off maps
and heads (rerun_behavioral.py must have run first). The rebuttal scripts are imported
unchanged; only their selected heads, the Qwen activation directory, and their output
directories are swapped, so every non-Qwen row reproduces the rebuttal values.

  1. Winner's-curse inflation      task1_head_selection/analyze_head_selection.py
  2. Center-controlled alignment   task3_baselines/run_headrule_center.py   (Table 2)
  3. Seven-metric saliency suite   task4_metric_suite/run_metric_suite_cv.py + plots/plot_fig_metric_suite.py

Output: neurips_final/final_figures/qwen35_nothink/results/ (+ appendix/ for figures)
"""
import os, sys, json

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
REB  = os.path.join(ROOT, 'rebuttal')
OUT  = os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink')
RES  = os.path.join(OUT, 'results')
FIG  = os.path.join(OUT, 'appendix'); os.makedirs(FIG, exist_ok=True)
for sub in ('shared', 'task1_head_selection', 'task3_baselines', 'task4_metric_suite'):
    sys.path.insert(0, os.path.join(REB, sub))
import vlm_maps as V

PROBE = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output'
V._QCFG = {t: dict(act=os.path.join(PROBE, d, 'qwen35_nothink', 'activations'),
                   mask=os.path.join(PROBE, d, 'qwen35_nothink', 'image_mask.pt'))
           for t, d in (('social', 'social_cap'), ('describe', 'coco'))}
SELECTED = json.load(open(os.path.join(RES, 'selected_heads.json')))

# ── 1. winner's curse ─────────────────────────────────────────────────────────
import analyze_head_selection as AH
AH.RES, AH.FIG = RES, FIG
AH.main()

# ── 2. center-controlled alignment ────────────────────────────────────────────
import run_headrule_center as HC
HC.SELECTED, HC.RES = SELECTED, RES
HC.main()

# ── 3. metric suite + overview figure ─────────────────────────────────────────
import run_metric_suite_cv as MS
MS.SELECTED, MS.RES = SELECTED, RES
MS.main()

src_path = os.path.join(REB, 'plots', 'plot_fig_metric_suite.py')
src = open(src_path).read()
for old, new in [("RES = os.path.join(REBUTTAL, 'task4_metric_suite', 'results')", f"RES = {RES!r}"),
                 ("OUT = os.path.join(HERE, 'head_selection')", f"OUT = {FIG!r}")]:
    assert old in src, old
    src = src.replace(old, new)
exec(compile(src, src_path, 'exec'), {'__file__': src_path, '__name__': '__main__'})
print('DONE rerun_appendix', flush=True)
