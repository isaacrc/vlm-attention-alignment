"""
plot_neglect_comparison.py
--------------------------
Generates two figures from the hemispatial neglect ablation results:

Figure 1 — Cross-architecture comparison (all-layers ablation):
  2-panel (Social | Describe), all 4 conditions on each panel.
  Saves: plots/neglect/neglect_all_layers_comparison.png

Figure 2 — Per-layer breakdown for Transformer conditions:
  2×2 grid (CNN-Transformer | ViT-Transformer) × (Social | Describe).
  Each panel: 4 layer lines (layer0-3) + all-layers dashed.
  Saves: plots/neglect/neglect_per_layer_transformers.png

Usage:
  python plot_neglect_comparison.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

ARCH_DIR = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
OUT_DIR  = os.path.join(ARCH_DIR, 'plots', 'neglect')
os.makedirs(OUT_DIR, exist_ok=True)

# X-axis: 14 ablation levels (fraction of 14 left columns removed)
X_LABELS = [f'{int(round(i/14*100))}%' for i in range(1, 15)]
X        = np.arange(1, 15)   # indices 1-14 in the (200,15) array

# ── style ──────────────────────────────────────────────────────────────────────
BASE_FS = 14
# m6c-matched font sizes (applied explicitly per-call below for plot_all_conditions)
FS_SUPTITLE = 23
FS_TITLE    = 20
FS_TICK     = 18
FS_LABEL    = 19
FS_LEGEND   = 16
mpl.rcParams.update({
    'font.size': BASE_FS, 'axes.labelsize': BASE_FS,
    'xtick.labelsize': BASE_FS - 2, 'ytick.labelsize': BASE_FS - 2,
    'legend.fontsize': BASE_FS - 2, 'axes.titlesize': BASE_FS + 2,
})
sns.set_style('white')

# Project 2 color palette
COLORS = {
    'CNN_LSTM':         '#5D6D7E',
    'CNN_Transformer':  '#E74C3C',
    'ViT_LSTM':         '#27AE60',
    'ViT_Transformer':  '#8E44AD',
}
LABELS = {
    'CNN_LSTM':         'CNN-LSTM',
    'CNN_Transformer':  'CNN-Transformer',
    'ViT_LSTM':         'ViT-LSTM',
    'ViT_Transformer':  'ViT-Transformer',
}

LAYER_COLORS = {
    'layer0':     '#1A73E8',
    'layer1':     '#E67E22',
    'layer2':     '#27AE60',
    'layer3':     '#C0392B',
    'all_layers': '#2C3E50',
}
LAYER_LABELS = {
    'layer0': 'Layer 0', 'layer1': 'Layer 1',
    'layer2': 'Layer 2', 'layer3': 'Layer 3',
    'all_layers': 'All layers',
}


# ── data loading ───────────────────────────────────────────────────────────────

def load(condition, task, layer_str):
    path = os.path.join(ARCH_DIR, condition, 'neglect',
                        f'{task}_neglect_{layer_str}.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)   # (200, 15)


def stats(arr):
    """Return (means, sems) over X indices 1-14."""
    col = arr[:, 1:]                        # (200, 14)
    means = np.nanmean(col, axis=0)         # (14,)
    sems  = np.nanstd(col, axis=0) / np.sqrt(np.sum(~np.isnan(col), axis=0))
    return means, sems


# ── Figure 1: all-conditions all-layers ───────────────────────────────────────

def plot_all_conditions():
    conditions = ['CNN_LSTM', 'CNN_Transformer', 'ViT_LSTM', 'ViT_Transformer']
    tasks      = ['social', 'describe']
    task_titles = {'social': 'Social task', 'describe': 'Describe task'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, dpi=300)

    for ax, task in zip(axes, tasks):
        for cond in conditions:
            arr = load(cond, task, 'all_layers')
            if arr is None:
                print(f'  Missing: {cond}/{task}/all_layers — skipping')
                continue
            means, sems = stats(arr)
            ax.errorbar(X, means, yerr=sems, fmt='o-',
                        color=COLORS[cond], label=LABELS[cond],
                        linewidth=2.5, markersize=6, capsize=4)

        ax.set_title(task_titles[task], fontsize=FS_TITLE, fontweight='bold')
        ax.set_xlabel('Ablation (% left columns removed)', fontsize=FS_LABEL)
        ax.set_xticks(X[::2])
        ax.set_xticklabels(X_LABELS[::2], rotation=30, fontsize=FS_TICK)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        sns.despine(ax=ax)

    axes[0].set_ylabel('Cosine similarity to unablated caption', fontsize=FS_LABEL)
    axes[1].legend(loc='lower left', frameon=True, fontsize=FS_LEGEND)

    plt.suptitle('Hemispatial Neglect',
                 fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'neglect_all_layers_comparison.png')
    out2 = os.path.normpath(os.path.join(OUT_DIR, '../../../manuscript/figures/neglect_all_layers_comparison.png'))
    plt.savefig(out, bbox_inches='tight')
    plt.savefig(out2, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
    print(f'Saved: {out2}')


# ── Figure 2: per-layer breakdown for Transformer conditions ──────────────────

def plot_per_layer():
    transformer_conds = ['CNN_Transformer', 'ViT_Transformer']
    tasks             = ['social', 'describe']
    layer_keys        = ['layer0', 'layer1', 'layer2', 'layer3', 'all_layers']
    task_titles       = {'social': 'Social task', 'describe': 'Describe task'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True, dpi=300)

    for row, cond in enumerate(transformer_conds):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            for lk in layer_keys:
                arr = load(cond, task, lk)
                if arr is None:
                    print(f'  Missing: {cond}/{task}/{lk} — skipping')
                    continue
                means, sems = stats(arr)
                ls = '--' if lk == 'all_layers' else '-'
                lw = 3.0  if lk == 'all_layers' else 2.0
                ax.errorbar(X, means, yerr=sems, fmt='o' + ls,
                            color=LAYER_COLORS[lk], label=LAYER_LABELS[lk],
                            linewidth=lw, markersize=5, capsize=3)

            title = f'{LABELS[cond]} — {task_titles[task]}'
            ax.set_title(title)
            ax.set_xlabel('Ablation (% left columns removed)')
            ax.set_xticks(X[::2])
            ax.set_xticklabels(X_LABELS[::2], rotation=30)
            sns.despine(ax=ax)
            if col == 0:
                ax.set_ylabel('Cosine similarity to unablated caption')
            if row == 0 and col == 1:
                ax.legend(loc='lower left', frameon=True)

    plt.suptitle('Hemispatial Neglect — Per-Layer Ablation (Transformer Decoders)',
                 fontsize=BASE_FS + 2, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'neglect_per_layer_transformers.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Also plot LSTM conditions side-by-side for reference ──────────────────────

def plot_lstm_conditions():
    lstm_conds  = ['CNN_LSTM', 'ViT_LSTM']
    tasks       = ['social', 'describe']
    task_titles = {'social': 'Social task', 'describe': 'Describe task'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, dpi=300)

    for ax, task in zip(axes, tasks):
        for cond in lstm_conds:
            arr = load(cond, task, 'all_layers')
            if arr is None:
                print(f'  Missing: {cond}/{task}/all_layers — skipping')
                continue
            means, sems = stats(arr)
            ax.errorbar(X, means, yerr=sems, fmt='o-',
                        color=COLORS[cond], label=LABELS[cond],
                        linewidth=2.5, markersize=6, capsize=4)

        ax.set_title(task_titles[task])
        ax.set_xlabel('Ablation (% left columns removed)')
        ax.set_xticks(X[::2])
        ax.set_xticklabels(X_LABELS[::2], rotation=30)
        sns.despine(ax=ax)

    axes[0].set_ylabel('Cosine similarity to unablated caption')
    axes[1].legend(loc='lower left', frameon=True)

    plt.suptitle('Hemispatial Neglect — LSTM Architectures',
                 fontsize=BASE_FS + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'neglect_lstm_comparison.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    print('Generating Figure 1: all-conditions all-layers comparison...')
    plot_all_conditions()
    print('Generating Figure 2: per-layer Transformer breakdown...')
    plot_per_layer()
    print('Generating Figure 3: LSTM conditions...')
    plot_lstm_conditions()
    print('Done.')
