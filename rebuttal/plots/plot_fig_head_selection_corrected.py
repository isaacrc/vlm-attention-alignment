"""
plot_fig_head_selection_corrected.py
-------------------------------------
Task 1 rebuttal figure. Reuses the manuscript's Fig 3 panel style (colors, bar
layout, fonts) from
architecture_manipulation/manuscript_plots/plot_fig3_qwen35_lg_m6cstyle_m6colors.py,
substituting cross-validated per-task head selection for the paper's fixed heads
(CNN-Transformer L3H5, ViT-Transformer L3H8 used for both tasks; Molmo read from
the wrong layer). Selected heads and per-task/per-subject alignment come from
rebuttal/task1_head_selection/build_cvhonest_table.py's output:
  data/alignment_summary_cv_honest.csv, results/selected_heads.json

CNN-LSTM and ViT-LSTM are unaffected (single attention map, nothing to select)
and use the same values as the original figure.

Panel C carries the Task 2 correction, which is independent of the head fix. The
paper's human between-task reference (0.237) correlated one Social participant
against one Describe participant, while model values are map-to-map with no
noise. The like-for-like human quantity is the group-mean Social map against the
group-mean Describe map: r = 0.658. No reliability or group-size correction is
applied, by decision: the reviewers asked for the human reference to be computed
the way the model reference is, not for it to be corrected for noise. Source:
task2_between_task/results/human_group_between_task.json.

Output: rebuttal/plots/fig_head_selection_corrected.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
import figstyle as FST

import matplotlib.gridspec
from matplotlib.transforms import ScaledTranslation

HERE = os.path.dirname(os.path.abspath(__file__))
REBUTTAL = os.path.dirname(HERE)
DATA = os.path.join(REBUTTAL, 'data')
T1 = os.path.join(REBUTTAL, 'task1_head_selection', 'results')

T2 = os.path.join(REBUTTAL, 'task2_between_task', 'results')

summ = pd.read_csv(os.path.join(DATA, 'alignment_summary_cv_honest.csv'))
ceil = json.load(open(os.path.join(DATA, 'ceilings.json')))
selected = json.load(open(os.path.join(T1, 'selected_heads.json')))
CEIL_SOC, CEIL_DES = ceil['social'], ceil['describe']

bt_model = pd.read_csv(os.path.join(T2, 'model_between_task.csv'))
bt_human = json.load(open(os.path.join(T2, 'human_group_between_task.json')))

COLORS = FST.ALL_COLORS
MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']


def make_tick(m):
    return {'CNN-Transformer': 'CNN-Trans.', 'ViT-Transformer': 'ViT-Trans.',
            'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen 9B-3.5'}.get(m, m)


def draw_between_panel(ax, label):
    """Panel C — between-task similarity, all agents on a map-to-map footing."""
    order = MODELS_6 + ['Human']
    vals, sems = [], []
    for m in order:
        if m == 'Human':
            vals.append(bt_human['between_r']); sems.append(bt_human['sem'])
        else:
            row = bt_model[bt_model.model == m].iloc[0]
            vals.append(float(row['between_r'])); sems.append(float(row['sem']))
    x = np.arange(len(order))
    ax.bar(x, vals, BAR_W, color=[COLORS[m] for m in order],
           edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, vals, yerr=sems, fmt='none', color='black',
                capsize=6, linewidth=2.0, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([make_tick(m) for m in order], fontsize=FS_TICK, rotation=45, ha='right')
    ax.margins(x=0.06)
    ax.tick_params(axis='x', pad=4)
    ax.set_yticks([0.0, 0.3, 0.6, 0.9])
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title('Between-task', fontsize=FS_TITLE, fontweight='bold', pad=TITLE_PAD)
    # Offset the letter by exactly the title's pad and share the title's baseline, so
    # the two sit on one line. A fixed axes-fraction y cannot do this: the same fraction
    # is a different number of points once the canvas height changes.
    _off = ScaledTranslation(0, TITLE_PAD / 72, fig.dpi_scale_trans)
    ax.text(-0.062, 1.0, label, transform=ax.transAxes + _off,
            fontsize=FS_LETTER, fontweight='bold', va='baseline', ha='center')
    ax.set_ylabel('Between-task\nsimilarity (r)', fontsize=FS_LABEL, labelpad=12)


def get(model, task):
    row = summ[(summ.model == model) & (summ.task == task)].iloc[0]
    return float(row['mean_r']), float(row['sem_r'])


bar_data = {m: {'social': get(m, 'social'), 'describe': get(m, 'describe')} for m in MODELS_6}

FS_TITLE, FS_TICK, FS_LABEL, FS_LEGEND = 56, 40, 42, 36
FS_LETTER = FS_TITLE
TITLE_PAD = 14
BAR_W = 0.55


def draw_panel(ax, task, ceil_z, label, show_ylabel=False):
    scale = 100.0 / ceil_z
    yticks = [0, 50, 100]
    x = np.arange(len(MODELS_6))
    means_r = np.array([bar_data[m][task][0] for m in MODELS_6])
    sems_r = np.array([bar_data[m][task][1] for m in MODELS_6])
    means_pct = 100 * np.arctanh(means_r) / ceil_z
    # delta-rule propagation of the r-space SEM through arctanh and the ceiling scale
    sems_pct = sems_r / (1 - means_r ** 2) * scale
    cols = [COLORS[m] for m in MODELS_6]

    ax.bar(x, means_pct, BAR_W, color=cols, edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, means_pct, yerr=sems_pct, fmt='none', color='black',
                capsize=6, linewidth=2.0, zorder=4)
    line = ax.axhline(100.0, color='black', linestyle='--', linewidth=2.5, zorder=2)

    tick_labels = [make_tick(m) for m in MODELS_6]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=FS_TICK, rotation=45, ha='right')
    ax.margins(x=0.06)
    ax.tick_params(axis='x', pad=4)
    ax.set_yticks(yticks)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    if task == 'social':
        ax.tick_params(axis='y', labelleft=False)
    ax.set_ylim(bottom=0, top=yticks[-1] * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    title_map = {'describe': 'Describe Task', 'social': 'Social Task'}
    ax.set_title(title_map[task], fontsize=FS_TITLE, fontweight='bold', pad=TITLE_PAD)
    # Offset the letter by exactly the title's pad and share the title's baseline, so
    # the two sit on one line. A fixed axes-fraction y cannot do this: the same fraction
    # is a different number of points once the canvas height changes.
    _off = ScaledTranslation(0, TITLE_PAD / 72, fig.dpi_scale_trans)
    ax.text(-0.13 if show_ylabel else -0.018, 1.0, label, transform=ax.transAxes + _off,
            fontsize=FS_LETTER, fontweight='bold', va='baseline', ha='left')

    if show_ylabel:
        ax.set_ylabel('% of Human\nUpperbound', fontsize=FS_LABEL, labelpad=12)
    return line


def add_ceil_legend(ax, line):
    """Name the ceiling line in place. A boxed legend anchored below the line covered
    whichever bars sat under it; text above the line cannot overlap any bar."""
    ceil_frac = 100.0 / ax.get_ylim()[1]
    ax.text(0.99, ceil_frac + 0.015, 'Human Upperbound', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=FS_LEGEND)


panel_configs = [('describe', CEIL_DES, 'A', True), ('social', CEIL_SOC, 'B', False)]

# Matches the manuscript Fig 3 layout: two alignment panels, a spacer, then the
# between-task panel, which is on a different y-scale and needs its own axis.
fig = plt.figure(figsize=(34, 9))
gs = matplotlib.gridspec.GridSpec(
    1, 4, width_ratios=[6, 6, 1.15, 7],
    left=0.15, right=0.99, top=0.86, bottom=0.40, wspace=0.12,
    figure=fig
)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])]
lines = [draw_panel(ax, task, ceil_z, label, show_ylabel=show_y)
         for ax, (task, ceil_z, label, show_y) in zip(axes[:2], panel_configs)]

y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1], 122)
axes[0].set_ylim(0, y_max)
axes[1].set_ylim(0, y_max)
add_ceil_legend(axes[1], lines[1])

draw_between_panel(axes[2], 'C')


out = os.path.join(HERE, 'head_selection', 'fig_head_selection_corrected.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved: {out}')

print(f"\nBetween-task: human r = {bt_human['between_r']:.4f} "
      f"(SEM {bt_human['sem']:.4f}), group-size ceiling {bt_human['group_size_ceiling']:.4f}, "
      f"noise-corrected {bt_human['between_r_noise_corrected']:.4f}")

print('\nSelected heads used:')
for k in ['CNN-Transformer_social', 'ViT-Transformer_social', 'Molmo_social', 'Qwen3.5_social',
          'CNN-Transformer_describe', 'ViT-Transformer_describe', 'Molmo_describe', 'Qwen3.5_describe']:
    d = selected[k]
    print(f"  {k:28s} L{d['layer']}H{d['head']}  cv_r={d['cv_honest_r']:.4f}  folds={d['fold_agreement']}")
