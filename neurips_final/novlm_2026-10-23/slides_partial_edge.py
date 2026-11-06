"""
slides_partial_edge.py -- partial_face_slide.png with edge energy partialled out instead of the face prior.
Solid bars: partial Spearman controlling for the edge-energy map, as % of the human upperbound under the
same partial metric; pale bars behind: raw alignment on the same images. All 200 images (the face prior
only on the 160 with an annotated face), computed by partial_edge_compute.py.
Mean of the Social and Describe percentages; SEM across participants, task SEMs combined as
sqrt(s_s^2 + s_d^2) / 2. Edge energy itself is left out. Order and colours as the Appendix B figure:
trained models | image statistics | priors | VLMs.
Writes final_figures/manuscript_2026-10-23/slides/partial_face_slide.png
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides', 'partial_edge_slide.png')
D = pd.read_csv(os.path.join(HERE, 'results', 'partial_edge_200.csv'))
CEIL = json.load(open(os.path.join(HERE, 'results', 'partial_edge_200_ceilings.json')))

bsrc_path = os.path.join(HERE, 'fig_baselines_rsa.py')   # ITEMS: labels, colours, hatches
bsrc = open(bsrc_path).read()
bns = {'__file__': bsrc_path, '__name__': 'baselines_src'}
exec(compile(bsrc[:bsrc.index('fig = plt.figure(')], bsrc_path, 'exec'), bns)
BY_NAME = {it[0]: it for it in bns['ITEMS']}
GROUPS = [['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer'],
          ['Itti-Koch', 'GBVS', 'Spectral residual', 'Luminance'],
          ['Center prior', 'Person prior', 'Face prior'], ['Molmo', 'Qwen3.5']]
items = [BY_NAME[m] for g in GROUPS for m in g]
ends = np.cumsum([len(g) for g in GROUPS])[:-1]


def pct(name, metric):
    """Mean over tasks of % upperbound, and its SEM, for metric 'raw' or 'partial'."""
    p, s = [], []
    for t in ('social', 'describe'):
        z = D[(D.model == name) & (D.task == t)][f'z_{metric}']
        cz = CEIL[t][f'{metric}_z']
        p.append(100 * z.mean() / cz); s.append(100 * z.std(ddof=1) / np.sqrt(len(z)) / cz)
    return np.mean(p), np.sqrt(s[0] ** 2 + s[1] ** 2) / 2


fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
ax = fig.add_axes([0.10, 0.30, 0.87, 0.59])
x = np.arange(len(items), dtype=float)
for b in ends:
    x[b:] += 0.6
means, sems = [], []
for xi, it in zip(x, items):
    name, _, _, c, h = it
    r, _ = pct(name, 'raw'); m, se = pct(name, 'partial')
    means.append(m); sems.append(se)
    ax.bar(xi, r, 0.7, color=c, alpha=0.22, edgecolor=c, linewidth=1.0, zorder=2)   # raw, for reference
    ax.bar(xi, m, 0.7, color=c, hatch=h, edgecolor='black', linewidth=0.6, zorder=3)
    if m > 12:   # value inside the solid bar, clear of the pale raw bar behind it
        light = np.dot(matplotlib.colors.to_rgb(c), [0.299, 0.587, 0.114]) > 0.62   # dark text on pale bars
        ax.text(xi, m - se - 2, f'{m:.0f}', ha='center', va='top', fontsize=13, fontweight='bold', zorder=5,
                color='black' if light else 'white', bbox=dict(boxstyle='round,pad=0.12', facecolor=c, edgecolor='none') if h else None)
    else:
        ax.text(xi, max(m, 0) + se + 2, f'{m:.0f}', ha='center', va='bottom', fontsize=13, fontweight='bold', zorder=5)
ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=4, linewidth=1.4, zorder=4)
ax.axhline(100, color='black', linestyle='--', linewidth=2.0, zorder=2)
ax.axhline(0, color='black', linewidth=1.0, zorder=2)
for b in ends:
    ax.axvline((x[b - 1] + x[b]) / 2, color='gray', linewidth=1.0, linestyle=':', zorder=1)
ax.text(0.99, 102, 'Human Upperbound', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=16)
ax.set_xticks(x); ax.set_xticklabels([it[2] for it in items], rotation=45, ha='right', fontsize=18)
ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
ax.set_ylim(-20 if min(means) < 0 else 0, 120); ax.set_yticks([0, 50, 100]); ax.tick_params(axis='y', labelsize=18)
ax.set_ylabel('% of Human\nUpperbound', fontsize=20, labelpad=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0); ax.set_axisbelow(True)
ax.set_title('Alignment beyond edge energy', fontsize=28, fontweight='bold', pad=12)
fig.legend(handles=[Patch(facecolor='#777777', edgecolor='black', label='Edge energy partialled out'),
                    Patch(facecolor='#777777', alpha=0.22, edgecolor='#777777', label='Raw (for reference)')],
           loc='lower center', ncol=2, fontsize=16, frameon=False, bbox_to_anchor=(0.40, 0.0))
fig.text(0.985, 0.022, 'Face prior: 160 images\nwith an annotated face', ha='right', va='bottom', fontsize=12, color='#555555')
fig.savefig(OUT, dpi=300, facecolor='white'); plt.close(fig)
print(' '.join(f'{it[2]}={m:.1f}' for it, m in zip(items, means)))
print('wrote', OUT)
