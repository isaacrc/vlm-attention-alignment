"""
plot_blur_social_panels.py
--------------------------
Alignment against Gaussian blur for one task (argv[1]: social, the default, or describe), one panel per model (2 x 3), bars for
sigma = 0, 8, ..., 64 px on the 256x256 map; styled like Figure 1A/B
(rebuttal/plots/plot_fig_head_selection_corrected.py): % of the mean pairwise human
noise ceiling (Fisher-z ratio), SEM across participants. Bar shade darkens with blur.
Data: final_figures/molmo_bilinear_pairwise/blur_sweep/blur_sweep_perimage.parquet
(sigma 40 and 56 from add_blur_sigmas.py). Writes fig_blur_{task}_panels.png and
blur_{task}_panels.csv next to it.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.transforms import ScaledTranslation

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST

FF = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise')
BS = os.path.join(FF, 'blur_sweep')
long = pd.read_parquet(os.path.join(BS, 'blur_sweep_perimage.parquet'))
TASK = sys.argv[1] if len(sys.argv) > 1 else 'social'   # social | describe
SIGMAS = list(range(0, 65, 8))
CEIL_Z = json.load(open(os.path.join(FF, 'results', 'ceilings.json')))[TASK]
assert set(SIGMAS) <= set(long.sigma.unique()), 'run add_blur_sigmas.py first'

COLORS = FST.ALL_COLORS
MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
NAME = {'CNN-Transformer': 'CNN-Transformer', 'ViT-Transformer': 'ViT-Transformer',
        'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen 9B-3.5'}
FS_TITLE, FS_TICK, FS_LABEL, FS_LEGEND = 40, 30, 34, 26
FS_LETTER = 46
TITLE_PAD = 14
BAR_W = 0.72
YMAX = 115


def shade(c, k, n):
    """Model colour from light (no blur) to full (most blur)."""
    c = np.array(to_rgb(c)); w = 0.35 + 0.65 * k / (n - 1)
    return tuple(1 - w * (1 - c))


subj = long[long.task == TASK].groupby(['model', 'sigma', 'subject_id'])['z'].mean()
fig, axes = plt.subplots(2, 3, figsize=(30, 17), sharey=True)
rows = []
for ax, m, letter in zip(axes.ravel(), MODELS_6, 'ABCDEF'):
    zs = [subj.loc[(m, s)] for s in SIGMAS]
    means = np.array([z.mean() for z in zs]) / CEIL_Z * 100
    sems = np.array([z.std(ddof=1) / np.sqrt(len(z)) for z in zs]) / CEIL_Z * 100
    rows += [dict(model=m, sigma=s, pct=p, sem=e) for s, p, e in zip(SIGMAS, means, sems)]
    x = np.arange(len(SIGMAS))
    ax.bar(x, means, BAR_W, color=[shade(COLORS[m], k, len(SIGMAS)) for k in range(len(SIGMAS))],
           edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=4, linewidth=1.6, zorder=4)
    ax.axhline(100.0, color='black', linestyle='--', linewidth=2.5, zorder=2)
    ax.text(0.99, 100.0 / YMAX + 0.015, 'Human Upperbound', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=FS_LEGEND)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SIGMAS], fontsize=FS_TICK)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(0, YMAX)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(NAME.get(m, m), fontsize=FS_TITLE, fontweight='bold', pad=TITLE_PAD)
    off = ScaledTranslation(0, TITLE_PAD / 72, fig.dpi_scale_trans)
    ax.text(-0.02, 1.0, letter, transform=ax.transAxes + off,
            fontsize=FS_LETTER, fontweight='bold', va='baseline', ha='left')
for ax in axes[:, 0]:
    ax.set_ylabel('% of Human\nUpperbound', fontsize=FS_LABEL, labelpad=12)
for ax in axes[1]:
    ax.set_xlabel(r'Gaussian blur $\sigma$ (pixels)', fontsize=FS_LABEL, labelpad=10)
fig.suptitle(f'{TASK.capitalize()} Task', fontsize=FS_TITLE * 1.25, fontweight='bold', y=0.995)
fig.tight_layout(h_pad=3, w_pad=2, rect=[0, 0, 1, 0.95])
f = os.path.join(BS, f'fig_blur_{TASK}_panels.png')
fig.savefig(f, dpi=300, bbox_inches='tight')
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BS, f'blur_{TASK}_panels.csv'), index=False)
print(out.pivot(index='model', columns='sigma', values='pct').round(1).loc[MODELS_6].to_string())
print('wrote', f)
