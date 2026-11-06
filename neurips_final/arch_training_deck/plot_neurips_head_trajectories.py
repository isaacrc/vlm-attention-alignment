"""
Per-head alignment trajectories across training for the two Transformer decoders,
highlighting the cross-validated heads used in the NeurIPS camera-ready
(Describe: CNN L3H5, ViT L3H8; Social: CNN L4H4, ViT L4H8).

Inputs:
  L3: CANS/architecture_manipulation/{cond}/head_attention_analysis_L3.csv (corrected 2026-04-15)
  L4: headL4_rerun/out/{cond}_head_attention_analysis_L4.csv (corrected-pipeline rerun;
      the CANS L4 CSVs from 2026-04-10 predate the Spearman fix and are inflated/wrong)

Output: fig_neurips_head_trajectories.png (same directory)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans']

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
PHASES = ['phase1', 'phase2', 'social_frozen', 'social_unfrozen']
PHASE_LBL = {'phase1': 'Describe\npretrain', 'phase2': 'Describe\nfinetune',
             'social_frozen': 'Social\nenc. frozen', 'social_unfrozen': 'Social\nenc. unfrozen'}
# checkpoints the paper analyzes (rebuttal/shared/alignment_core.py NPZ_FILE)
PAPER_CKPT = {'CNN_Transformer': {'phase2': 14, 'social_unfrozen': 3},
              'ViT_Transformer': {'phase2': 11, 'social_unfrozen': 1}}
MODELS = [('CNN_Transformer', 'CNN-Transformer', (3, 5), (4, 4)),
          ('ViT_Transformer', 'ViT-Transformer', (3, 8), (4, 8))]
C_DESC, C_SOC = '#D55E00', '#0072B2'


def load(cond):
    l3 = pd.read_csv(os.path.join(ARCH, cond, 'head_attention_analysis_L3.csv')).assign(layer=3)
    l4 = pd.read_csv(os.path.join(HERE, 'headL4_rerun', 'out',
                                  f'{cond}_head_attention_analysis_L4.csv')).assign(layer=4)
    return pd.concat([l3, l4], ignore_index=True)


fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharey=True)
for ax, (cond, name, hd, hs) in zip(axes, MODELS):
    df = load(cond)
    x0, bounds, mids = 0, [], []
    for ph in PHASES:
        p = df[df.phase == ph]
        eps = sorted(p.epoch.unique())
        xmap = {e: x0 + i for i, e in enumerate(eps)}
        for (L, H), g in p.groupby(['layer', 'head']):
            g = g.sort_values('epoch')
            xs = g.epoch.map(xmap)
            if (L, H) == hd:
                kw = dict(color=C_DESC, lw=2.4, zorder=4, marker='o', ms=4)
            elif (L, H) == hs:
                kw = dict(color=C_SOC, lw=2.4, zorder=4, marker='o', ms=4)
            else:
                kw = dict(color='#B8B8B8', lw=0.8, zorder=1, alpha=0.8)
            ax.plot(xs, g.mean_r, **kw)
        ck = PAPER_CKPT[cond].get(ph)
        if ck is not None:
            target = hd if ph == 'phase2' else hs
            row = p[(p.layer == target[0]) & (p['head'] == target[1]) & (p.epoch == ck)]
            ax.scatter(xmap[ck], row.mean_r, s=220, marker='*', facecolor='white',
                       edgecolor='black', lw=1.2, zorder=5)
        mids.append(x0 + (len(eps) - 1) / 2)
        x0 += len(eps)
        bounds.append(x0 - 0.5)
    for b in bounds[:-1]:
        ax.axvline(b, color='#666666', ls=':', lw=0.9)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_xticks(mids)
    ax.set_xticklabels([PHASE_LBL[p] for p in PHASES], fontsize=10)
    ax.set_title(name, fontsize=13, fontweight='bold', loc='left')
    ax.set_ylabel('Spatial Spearman r\n(vs. task-matched human gaze)', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    ax.plot([], [], color=C_DESC, lw=2.4, label=f'L{hd[0]}H{hd[1]} (paper Describe head)')
    ax.plot([], [], color=C_SOC, lw=2.4, label=f'L{hs[0]}H{hs[1]} (paper Social head)')
    ax.plot([], [], color='#B8B8B8', lw=0.8, label='other 14 heads, layers 3-4')
    ax.scatter([], [], s=120, marker='*', facecolor='white', edgecolor='black',
               label='checkpoint analyzed in paper')
    ax.legend(fontsize=8.5, loc='lower left', ncol=2, frameon=False)
fig.tight_layout()
out = os.path.join(HERE, 'fig_neurips_head_trajectories.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
