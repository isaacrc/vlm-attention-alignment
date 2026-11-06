"""
slides_partial_center.py -- 16:9 slide (13.333 x 7.5 in, 300 dpi): alignment after the center prior is
partialled out, Describe and Social side by side, in the Appendix B order (trained models | center prior | VLMs).
Solid bars: partial Spearman r (model map vs fixation map, controlling for the leave-one-image-out center
prior) as % of the human upperbound under the same partial metric. Pale bars behind: the raw r as % of the
raw upperbound. Percent = r / upperbound r, as in Table 2 (make_table_baselines.py); SEM across participants
scaled the same way. Source: molmo_bilinear_pairwise/results/headrule_center_summary.csv and
headrule_center_ceilings.json (trained models and VLMs at their cross-validated heads, all maps unblurred).
Writes final_figures/manuscript_2026-10-23/slides/partial_center_slide.png
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST

Q = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides', 'partial_center_slide.png')
hr = pd.read_csv(os.path.join(Q, 'headrule_center_summary.csv'))
CEIL = json.load(open(os.path.join(Q, 'headrule_center_ceilings.json')))

# (name in csv, head rule, tick label, colour)
ITEMS = [('CNN-LSTM', 'na', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM']),
         ('CNN-Transformer', 'cv_best', 'CNN-Trans.', FST.ALL_COLORS['CNN-Transformer']),
         ('ViT-LSTM', 'na', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM']),
         ('ViT-Transformer', 'cv_best', 'ViT-Trans.', FST.ALL_COLORS['ViT-Transformer']),
         ('CENTER_PRIOR', 'na', 'Center prior', '#8c6d31'),
         ('Molmo', 'cv_best', 'Molmo 7B-D', FST.ALL_COLORS['Molmo']),
         ('Qwen3.5', 'cv_best', 'Qwen 9B-3.5', FST.ALL_COLORS['Qwen3.5'])]
GROUP_ENDS = [4, 5]   # trained | prior | VLMs
TASKS = (('describe', 'Describe Task'), ('social', 'Social Task'))

fig = plt.figure(figsize=(13.333, 7.5), dpi=300, facecolor='white')
fig.text(0.5, 0.955, 'Alignment beyond the center prior', fontsize=30, fontweight='bold', ha='center', va='center')
axes = [fig.add_axes(r) for r in ([0.115, 0.27, 0.415, 0.56], [0.565, 0.27, 0.415, 0.56])]
x = np.arange(len(ITEMS), dtype=float)
for b in GROUP_ENDS:
    x[b:] += 0.5
for k, (ax, (task, title)) in enumerate(zip(axes, TASKS)):
    for xi, (name, rule, _, col) in zip(x, ITEMS):
        row = hr[(hr.model == name) & (hr.rule == rule) & (hr.task == task)].iloc[0]
        raw, raw_se = 100 * row.r_raw / CEIL[task]['raw_r'], 100 * row.sem_raw / CEIL[task]['raw_r']
        par, par_se = 100 * row.r_partial / CEIL[task]['partial_r'], 100 * row.sem_partial / CEIL[task]['partial_r']
        ax.bar(xi, raw, 0.62, color=col, alpha=0.22, edgecolor=col, linewidth=1.0, zorder=2)
        ax.bar(xi, par, 0.62, color=col, edgecolor='black', linewidth=0.6, zorder=3)
        ax.errorbar(xi, par, yerr=par_se, fmt='none', color='black', capsize=4, linewidth=1.5, zorder=4)
        if par > 12:   # value inside the solid bar, clear of the pale raw bar behind it
            ax.text(xi, par - par_se - 2, f'{par:.0f}', ha='center', va='top', fontsize=14, fontweight='bold', color='white', zorder=5)
        else:
            ax.text(xi, max(par, 0) + 2, f'{par:.0f}', ha='center', va='bottom', fontsize=14, fontweight='bold', zorder=5)
    ax.axhline(100, color='black', linestyle='--', linewidth=2.0, zorder=1)
    ax.axhline(0, color='black', linewidth=1.0, zorder=1)
    for b in GROUP_ENDS:
        ax.axvline((x[b - 1] + x[b]) / 2, color='gray', linewidth=1.0, linestyle=':', zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([it[2] for it in ITEMS], rotation=45, ha='right', fontsize=17)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(-8, 122); ax.set_yticks([0, 50, 100]); ax.tick_params(axis='y', labelsize=17, labelleft=(k == 0))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=24, fontweight='bold', pad=10)
axes[0].set_ylabel('% of Human\nUpperbound', fontsize=20, labelpad=8)
axes[1].text(0.99, 101.5, 'Human Upperbound', transform=axes[1].get_yaxis_transform(), ha='right', va='bottom', fontsize=15)
fig.legend(handles=[Patch(facecolor='#777777', edgecolor='black', label='Center prior partialled out'),
                    Patch(facecolor='#777777', alpha=0.22, edgecolor='#777777', label='Raw (for reference)')],
           loc='lower center', ncol=2, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(OUT, dpi=300, facecolor='white'); plt.close(fig)
print('wrote', OUT)
