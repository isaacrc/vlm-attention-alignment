"""
metric_suite_slide.py
---------------------
The appendix metric-suite overview (paper Fig 6, KL and IG dropped) as one 16:9 slide
(13.33 x 7.5 in), styled like final_figures/slides/fig3_describe_social_slide.png:
no panel letters, labelled human-upperbound line, legend below the panels.
Reuses data loading, colors and normalize() from rebuttal/plots/plot_fig_metric_suite.py.

Output: final_figures/qwen35_nothink/slides/fig6_metric_suite_slide.png
"""
import os
import numpy as np

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
NT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink')
OUT = os.path.join(NT, 'slides')
os.makedirs(OUT, exist_ok=True)

src_path = os.path.join(ROOT, 'rebuttal', 'plots', 'plot_fig_metric_suite.py')
src = open(src_path).read()
old = "RES = os.path.join(REBUTTAL, 'task4_metric_suite', 'results')"
assert old in src
src = src.replace(old, f"RES = {os.path.join(NT, 'results')!r}")
ns = {'__file__': src_path, '__name__': 'metric_src'}
exec(src[:src.index('def panel(')], ns)   # data, colors, normalize
plt, summ, normalize = ns['plt'], ns['summ'], ns['normalize']
COLORS, TICK, ORDER = ns['COLORS'], ns['TICK'], ns['ORDER']
METRICS = [m for m in ns['METRICS'] if m not in ('KL', 'IG')]

FS_TITLE, FS_TICK, FS_LABEL, FS_LEG = 22, 15, 16, 14
fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), sharey=True)
W = 0.8 / len(ORDER)
xm = np.arange(len(METRICS))
for ax, task in zip(axes, ['describe', 'social']):
    for k, m in enumerate(ORDER):
        vals, errs = [], []
        for metric in METRICS:
            sub = summ[(summ.metric == metric) & (summ.task == task)].set_index('model')
            p, e = normalize(metric, float(sub.loc[m, 'value']), float(sub.loc[m, 'sem']),
                             float(sub['ceiling'].iloc[0]))
            vals.append(p); errs.append(e)
        off = (k - (len(ORDER) - 1) / 2) * W
        ax.bar(xm + off, vals, W, color=COLORS[m], label=TICK.get(m, m),
               edgecolor='black', linewidth=0.4, zorder=3)
        ax.errorbar(xm + off, vals, yerr=errs, fmt='none', color='black',
                    capsize=1.5, linewidth=0.8, zorder=4)
    ax.axhline(100.0, color='black', ls='--', lw=1.8, zorder=2)
    ax.text(len(METRICS) - 0.55, 102, 'Human Upperbound', ha='right', va='bottom', fontsize=FS_LEG)
    ax.axhline(0, color='black', lw=0.8, zorder=2)
    ax.set_xticks(xm)
    ax.set_xticklabels(METRICS, fontsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(0, 115)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(ns['TASK_NAME'][task], fontsize=FS_TITLE, fontweight='bold', pad=10)
axes[0].set_ylabel('% of Human\nUpperbound', fontsize=FS_LABEL)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, fontsize=FS_LEG, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.07, 1, 1])
f = os.path.join(OUT, 'fig6_metric_suite_slide.png')
fig.savefig(f, dpi=300)
print('Saved:', f)
