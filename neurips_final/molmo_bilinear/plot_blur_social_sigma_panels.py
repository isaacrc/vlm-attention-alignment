"""
plot_blur_social_sigma_panels.py
--------------------------------
Alignment for one task (argv[1]: social, the default, or describe) at four fixed Gaussian blurs (sigma = 4, 16, 32, 64 px on the
256x256 map), one panel per blur (2 x 2), each drawn like Figure 1A/B
(rebuttal/plots/plot_fig_head_selection_corrected.py): % of the mean pairwise human noise
ceiling, Fisher-z ratio, SEM across participants. The per-model view of the same sweep is
plot_blur_social_panels.py.
Data: final_figures/molmo_bilinear_pairwise/blur_sweep/blur_sweep_perimage.parquet
Writes fig_blur_{task}_sigma_panels.png and blur_{task}_sigma_panels.csv next to it.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST

FF = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise')
BS = os.path.join(FF, 'blur_sweep')
long = pd.read_parquet(os.path.join(BS, 'blur_sweep_perimage.parquet'))
TASK = sys.argv[1] if len(sys.argv) > 1 else 'social'   # social | describe
# --r: y-axis in Spearman r with both human ceilings (mean pairwise and leave-one-out),
# as in plot_blur_sweep_r.py, instead of % of the pairwise ceiling
RAW = '--r' in sys.argv
LOO_R = json.load(open(os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink', 'blur_sweep',
                                    'ceilings_pairwise_vs_loo.json')))[TASK]['loo_r']
CEIL_Z = json.load(open(os.path.join(FF, 'results', 'ceilings.json')))[TASK]
SIGMAS = [4, 16, 32, 64]

COLORS = FST.ALL_COLORS
MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
TICK = {'CNN-Transformer': 'CNN-Trans.', 'ViT-Transformer': 'ViT-Trans.',
        'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen 9B-3.5'}
FS_TITLE, FS_TICK, FS_LABEL, FS_LEGEND = 48, 36, 40, 32
FS_LETTER = FS_TITLE
TITLE_PAD = 14
BAR_W = 0.55
subj = long[long.task == TASK].groupby(['sigma', 'model', 'subject_id'])['z'].mean()
# room above the tallest bar (blurred Molmo passes the pairwise ceiling in Describe); the
# ceiling label then moves to the left, above the LSTM bars, so no bar covers it
top = max(subj.loc[(s, m)].mean() for s in SIGMAS for m in MODELS_6) / CEIL_Z * 100
YMAX = max(115, 10 * np.ceil((top + 8) / 10))
YMAX_R = 0.42
LAB_X, LAB_HA = (0.01, 'left') if top > 100 else (0.99, 'right')
fig, axes = plt.subplots(2, 2, figsize=(24, 22), sharey=True)
rows = []
for ax, s, letter in zip(axes.ravel(), SIGMAS, 'ABCD'):
    zs = [subj.loc[(s, m)] for m in MODELS_6]
    mz = np.array([z.mean() for z in zs]); sz = np.array([z.std(ddof=1) / np.sqrt(len(z)) for z in zs])
    if RAW:   # back-transformed mean r; SEM carried through tanh by the delta rule
        means = np.tanh(mz); sems = sz * (1 - means ** 2)
    else:
        means, sems = mz / CEIL_Z * 100, sz / CEIL_Z * 100
    rows += [dict(sigma=s, model=m, **{'r' if RAW else 'pct': p}, sem=e) for m, p, e in zip(MODELS_6, means, sems)]
    x = np.arange(len(MODELS_6))
    ax.bar(x, means, BAR_W, color=[COLORS[m] for m in MODELS_6], edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=6, linewidth=2.0, zorder=4)
    if RAW:
        for y, ls, c, lab in ((np.tanh(CEIL_Z), '--', 'black', 'Human ceiling (pairwise)'),
                              (LOO_R, ':', 'dimgray', 'Human ceiling (leave-one-out)')):
            ax.axhline(y, color=c, linestyle=ls, linewidth=3.0, zorder=2)
            ax.text(0.01, y / YMAX_R + 0.012, lab, transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=FS_LEGEND * 0.85, color=c)
    else:
        ax.axhline(100.0, color='black', linestyle='--', linewidth=2.5, zorder=2)
        ax.text(LAB_X, 100.0 / YMAX + 0.015, 'Human Upperbound', transform=ax.transAxes,
                ha=LAB_HA, va='bottom', fontsize=FS_LEGEND)
    ax.set_xticks(x)
    ax.set_xticklabels([TICK.get(m, m) for m in MODELS_6], fontsize=FS_TICK, rotation=45, ha='right')
    ax.margins(x=0.06)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4] if RAW else [0, 50, 100])
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(0, YMAX_R if RAW else YMAX)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(rf'$\sigma$ = {s} px', fontsize=FS_TITLE, fontweight='bold', pad=TITLE_PAD)
    off = ScaledTranslation(0, TITLE_PAD / 72, fig.dpi_scale_trans)
    ax.text(-0.02, 1.0, letter, transform=ax.transAxes + off,
            fontsize=FS_LETTER, fontweight='bold', va='baseline', ha='left')
for ax in axes[:, 0]:
    ax.set_ylabel(r'Spearman $r$' if RAW else '% of Human\nUpperbound', fontsize=FS_LABEL, labelpad=12)
fig.suptitle(f'{TASK.capitalize()} Task', fontsize=FS_TITLE * 1.15, fontweight='bold', y=0.995)
fig.tight_layout(h_pad=3, rect=[0, 0, 1, 0.955])
f = os.path.join(BS, f'fig_blur_{TASK}_sigma_panels' + ('_r' if RAW else '') + '.png')
fig.savefig(f, dpi=300, bbox_inches='tight')
pd.DataFrame(rows).to_csv(os.path.join(BS, f'blur_{TASK}_sigma_panels' + ('_r' if RAW else '') + '.csv'), index=False)
print('wrote', f)
