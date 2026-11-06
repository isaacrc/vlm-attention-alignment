"""
metric_overview_notitle.py
--------------------------
Re-render the appendix metric-suite overview (plot_fig_metric_suite.py on the thinking-off
results) without the explanatory suptitle, which the paper caption already covers.
Writes final_figures/qwen35_nothink/appendix/fig_metric_overview_notitle.png; the
per-metric figures the source script also draws go to the same directory.
KL and IG are left out of the overview. KL: its split-half human value is not a ceiling (KL rewards
diffuse maps), so every model sat above 100% and the bar misread as beating humans.
"""
import os, re

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
# MOLMO_BILINEAR=1: plot the molmo_bilinear run
NT = os.path.join(ROOT, 'neurips_final', 'final_figures',
                  os.environ.get('MOLMO_BILINEAR_DIR', 'molmo_bilinear') if os.environ.get('MOLMO_BILINEAR') == '1' else 'qwen35_nothink')
RES, FIG = os.path.join(NT, 'results'), os.path.join(NT, 'appendix')
src_path = os.path.join(ROOT, 'rebuttal', 'plots', 'plot_fig_metric_suite.py')
src = open(src_path).read()
for old, new in [("RES = os.path.join(REBUTTAL, 'task4_metric_suite', 'results')", f"RES = {RES!r}"),
                 ("OUT = os.path.join(HERE, 'head_selection')", f"OUT = {FIG!r}"),
                 ("'fig_metric_overview.png'", "'fig_metric_overview_notitle.png'"),
                 ("fig.tight_layout(rect=[0, 0, 1, 0.90])\nf = os.path.join(OUT, 'fig_metric_overview",
                  "fig.tight_layout()\nf = os.path.join(OUT, 'fig_metric_overview")]:
    assert old in src, old
    src = src.replace(old, new)
src, n = re.subn(r"fig\.suptitle\('All seven saliency metrics.*?y=0\.99\)\n", '', src, flags=re.S)
assert n == 1
head, sep, tail = src.partition('# ── overview')
assert sep
tail = tail.replace('np.arange(len(METRICS))', 'np.arange(len(OV_METRICS))', 1)
tail = tail.replace('for metric in METRICS:', 'for metric in OV_METRICS:', 1)
tail = tail.replace('set_xticklabels(METRICS', 'set_xticklabels(OV_METRICS', 1)
assert tail.count('OV_METRICS') == 3
# without the negative IG bars the legend would sit on the bars: put it below the panels
old_leg = "axes[0].legend(fontsize=FS_LEG, ncol=3, loc='lower left', frameon=True,\n               framealpha=0.92, edgecolor='none')"
assert old_leg in tail
tail = tail.replace(old_leg, "fig.legend(*axes[0].get_legend_handles_labels(), fontsize=FS_LEG, ncol=6, "
                             "loc='upper center', bbox_to_anchor=(0.5, 0.0), frameon=False)")
# whole-image metrics (CC, SIM) first, then the fixation-based ones
src = head + "OV_METRICS = ['CC', 'SIM', 'NSS', 'AUC', 'sAUC']\n" + sep + tail
exec(compile(src, src_path, 'exec'), {'__file__': src_path, '__name__': '__main__'})
