"""
5_variance_partition.py
-----------------------
Compute and plot variance partition from existing R² files.

Unique to attn  = R²(banded) - R²(CLIP-only)
Unique to CLIP  = R²(banded) - R²(attn-only)
Shared          = R²(attn-only) + R²(CLIP-only) - R²(banded)  [clipped to 0]
Total           = R²(banded)

Plots: stacked bar per model × task showing the partition over top-5% banded vertices.
Output: figures/sig_tests/variance_partition.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE    = Path('/scratch/gpfs/GRAZIANO/CANS')
OUT_DIR = BASE / 'tribe/outputs'
FIG_DIR = OUT_DIR / 'figures/sig_tests'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
TASKS  = ['describe', 'social']

COLORS = {
    'CNN-LSTM':        '#5D6D7E',
    'ViT-LSTM':        '#27AE60',
    'CNN-Transformer': '#E74C3C',
    'ViT-Transformer': '#8E44AD',
    'Molmo':           '#F39C12',
    'Qwen3.5':         '#1ABC9C',
}

r2_clip_only = np.load(OUT_DIR / 'r2_clip_only.npy')   # (20484,) shared

# ── collect partition values ───────────────────────────────────────────────────
results = {}
for model in MODELS:
    for task in TASKS:
        p_attn   = OUT_DIR / f'r2_{model}_{task}.npy'
        p_banded = OUT_DIR / f'r2_banded_{model}_{task}.npy'
        if not p_attn.exists() or not p_banded.exists():
            print(f'Missing: {model}/{task}')
            continue

        r2_attn   = np.load(p_attn)
        r2_banded = np.load(p_banded)

        # Use top-5% of banded R² vertices as ROI
        thr = np.percentile(r2_banded, 95)
        roi = r2_banded >= thr

        def mean_roi(arr):
            return float(arr[roi].mean())

        unique_attn = np.clip(r2_banded - r2_clip_only, 0, None)
        unique_clip = np.clip(r2_banded - r2_attn,      0, None)
        shared      = np.clip(r2_attn + r2_clip_only - r2_banded, 0, None)
        total       = r2_banded

        results[(model, task)] = {
            'unique_attn': mean_roi(unique_attn),
            'unique_clip': mean_roi(unique_clip),
            'shared':      mean_roi(shared),
            'total':       mean_roi(total),
        }
        print(f'{model}/{task}: total={mean_roi(total):.4f}  '
              f'unique_attn={mean_roi(unique_attn):.4f}  '
              f'unique_clip={mean_roi(unique_clip):.4f}  '
              f'shared={mean_roi(shared):.4f}')

# ── stacked bar chart ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

PART_COLORS = {
    'unique_attn': '#E74C3C',
    'shared':      '#95A5A6',
    'unique_clip': '#3498DB',
}
PART_LABELS = {
    'unique_attn': 'Unique to Attention',
    'shared':      'Shared (Attn ∩ CLIP)',
    'unique_clip': 'Unique to CLIP',
}

for ti, task in enumerate(TASKS):
    ax = axes[ti]
    x  = np.arange(len(MODELS))
    bw = 0.55

    bottoms = np.zeros(len(MODELS))
    for part in ['unique_attn', 'shared', 'unique_clip']:
        vals = [results.get((m, task), {}).get(part, 0) for m in MODELS]
        bars = ax.bar(x, vals, bw, bottom=bottoms,
                      color=PART_COLORS[part],
                      label=PART_LABELS[part] if ti == 0 else '_nolegend_',
                      edgecolor='white', linewidth=0.6)
        # annotate non-trivial stacks
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.005:
                ax.text(xi, b + v / 2, f'{v:.3f}',
                        ha='center', va='center', fontsize=7.5,
                        color='white', fontweight='bold')
        bottoms += np.array(vals)

    # model-color dots along x-axis for identity
    for xi, m in enumerate(MODELS):
        ax.plot(xi, -0.004, 's', color=COLORS[m], markersize=9, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9, rotation=20, ha='right')
    ax.set_title(f'{task.capitalize()} task', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean R² (top-5% banded vertices)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

fig.legend(loc='upper center', ncol=3, fontsize=11, frameon=False,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle('Variance partition: Attention vs. CLIP contributions to brain encoding',
             fontsize=13, fontweight='bold', y=1.06)
fig.tight_layout()
fig.savefig(FIG_DIR / 'variance_partition.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('\nSaved: variance_partition.png')
