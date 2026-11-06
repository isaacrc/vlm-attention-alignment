"""
slides_partial_baselines.py -- baselines_alignment_slide.png (slides_appendix.py) with the center prior
partialled out: alignment as % of the human upperbound under the partial metric, mean of the Social and
Describe percentages, every map unblurred. The center prior itself is left out (it is what is removed).
Order and colours as the Appendix B figure: trained models | image statistics | person, face priors | VLMs.

Metric: partial Spearman r between each map and each participant's fixation map, controlling for the
leave-one-image-out center prior (run_center_controlled.partial_rows), Fisher z averaged over images per
participant; percent = mean z / partial-upperbound z (headrule_center_ceilings.json), as Figure 1 and
baselines_alignment_slide.png; SEM across participants, task SEMs combined as sqrt(s_s^2 + s_d^2) / 2.
  Trained models and VLMs   molmo_bilinear_pairwise/results/headrule_center_summary.csv (cross-validated heads)
  Baselines                 computed here from the unblurred maps scored in Appendix B (score_baselines.py),
                            cached in results/partial_baselines_unblurred.csv
Pale bars behind: the raw scores of baselines_alignment_slide.png (same maps, not partialled), to show the drop.
Writes final_figures/manuscript_2026-10-23/slides/partial_baselines_slide.png
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'task3_baselines'))
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
Q = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results')
CACHE = os.path.join(HERE, 'results', 'partial_baselines_unblurred.csv')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides', 'partial_baselines_slide.png')
CEIL = json.load(open(os.path.join(Q, 'headrule_center_ceilings.json')))
BASELINES = ['Itti-Koch', 'GBVS', 'Spectral residual', 'Edge energy', 'Luminance', 'Person prior', 'Face prior']

# per-participant partial z for the baselines (cached; the human maps take a while to load)
if not os.path.exists(CACHE):
    import alignment_core as A, tierB_common as T, run_center_controlled as CC
    bs = open(os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'fig1_baselines', 'score_baselines.py')).read()
    bns = {'__name__': 'baselines'}
    exec(compile(bs[:bs.index('def main():')], 'score_baselines.py', 'exec'), bns)   # cell_maps
    imgs = A.target_images()
    prior_rk = CC.loio_prior_ranked(imgs)
    maps = {'Itti-Koch': T.load_maps('itti_koch'), 'GBVS': T.load_maps('gbvs'),
            'Spectral residual': T.load_maps('spectral_residual'),
            'Person prior': T.load_maps('coco_person'), 'Face prior': T.load_maps('coco_face')}
    maps['Edge energy'], maps['Luminance'] = bns['cell_maps'](imgs)
    rows = []
    for task in ('social', 'describe'):
        for name in BASELINES:
            df = CC.partial_rows(maps[name], task, imgs, prior_rk, name)   # flat maps (no face) are dropped inside
            zs = df.groupby('subject_id').z.mean()
            rows += [dict(model=name, task=task, subject_id=s, z=v, n_images=df.image_id.nunique()) for s, v in zs.items()]
            print(f'{task:8s} {name:18s} r = {np.tanh(zs.mean()):.3f}  ({df.image_id.nunique()} images)', flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pd.DataFrame(rows).to_csv(CACHE, index=False)
base = pd.read_csv(CACHE)


def pct(name, task):
    """(% of the partial upperbound, SEM) for one predictor and task."""
    cz = CEIL[task]['partial_z']
    if name in BASELINES:
        z = base[(base.model == name) & (base.task == task)].z
        return 100 * z.mean() / cz, 100 * z.std(ddof=1) / np.sqrt(len(z)) / cz
    hr = pd.read_csv(os.path.join(Q, 'headrule_center_summary.csv'))
    rule = 'na' if name in ('CNN-LSTM', 'ViT-LSTM') else 'cv_best'
    row = hr[(hr.model == name) & (hr.rule == rule) & (hr.task == task)].iloc[0]
    z = np.arctanh(row.r_partial)
    return 100 * z / cz, 100 * (np.arctanh(row.r_partial + row.sem_partial) - z) / cz


# drawing: slides_appendix.py's alignment_slide, with the percentages swapped for the partial ones
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
bsrc_path = os.path.join(HERE, 'fig_baselines_rsa.py')
bsrc = open(bsrc_path).read()
bns = {'__file__': bsrc_path, '__name__': 'baselines_src'}
exec(compile(bsrc[:bsrc.index('fig = plt.figure(')], bsrc_path, 'exec'), bns)
BY_NAME = {it[0]: it for it in bns['ITEMS']}
GROUPS = [['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer'],
          ['Itti-Koch', 'GBVS', 'Spectral residual', 'Edge energy', 'Luminance'],
          ['Person prior', 'Face prior'], ['Molmo', 'Qwen3.5']]
items = [BY_NAME[m] for g in GROUPS for m in g]
ends = np.cumsum([len(g) for g in GROUPS])[:-1]

fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
ax = fig.add_axes([0.10, 0.30, 0.87, 0.59])
x = np.arange(len(items), dtype=float)
for b in ends:
    x[b:] += 0.6
d, RAW_CEIL = bns['d'], bns['CEIL']


def raw_pct(name):
    """Raw (not partialled) % of upperbound, mean of the two tasks: the bars of baselines_alignment_slide.png."""
    return np.mean([d[(d.task == t) & (d.model == name)].mean_z.mean() / RAW_CEIL[t] * 100 for t in ('social', 'describe')])


means, sems, raws = [], [], []
for it in items:
    (ps, ss), (pd_, sd) = pct(it[0], 'social'), pct(it[0], 'describe')
    means.append((ps + pd_) / 2); sems.append(np.sqrt(ss ** 2 + sd ** 2) / 2); raws.append(raw_pct(it[0]))
for xi, m, se, r, (_, _, _, c, h) in zip(x, means, sems, raws, items):
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
ax.set_title('Alignment beyond the center prior', fontsize=28, fontweight='bold', pad=12)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor='#777777', edgecolor='black', label='Center prior partialled out'),
                    Patch(facecolor='#777777', alpha=0.22, edgecolor='#777777', label='Raw (for reference)')],
           loc='lower center', ncol=2, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(OUT, dpi=300, facecolor='white'); plt.close(fig)
print(' '.join(f'{it[2]}={m:.1f}' for it, m in zip(items, means)))
print('wrote', OUT)
