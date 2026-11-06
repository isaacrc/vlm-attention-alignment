"""
blur_sweep.py
-------------
Smoothness control applied symmetrically to all six models, following Langlois et al.
(2021, arXiv:2107.07013, Sec. 4 and App. D.2/I): blur every model map with a Gaussian of
sigma in SIGMAS (pixels at 256x256), and report alignment at the best sigma per model x
task. The best sigma is chosen two ways:
  in-sample : max over sigma on all 200 images (Langlois' main analysis)
  CV        : 5-fold over images; sigma picked on 4/5 of images, scored on the held-out 1/5
              (Langlois App. I uses split-half for the same purpose)
Maps are the paper's per-task cross-validated heads (selected_heads.json); sigma = 0
reproduces the paper's alignment table. Human maps are untouched.

Outputs (final_figures/qwen35_nothink/blur_sweep/):
  blur_sweep_perimage.parquet   per subject x image z for every model/task/sigma
  blur_sweep_curve.csv          mean r and % ceiling per sigma
  blur_sweep_summary.csv        sigma 0 / in-sample best / CV best, per model x task
  blur_sweep_cv_subject_z.csv   per-subject mean z at the CV-chosen sigma (Figure 1 variant)
  blur_sweep_contrasts.csv      paired subject-level contrasts at the CV-best sigma
  fig_blur_sweep.png
"""
import os, sys, json, itertools
import numpy as np
import pandas as pd
import cv2
from scipy import stats

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
REB = os.path.join(ROOT, 'rebuttal')
# MOLMO_BILINEAR=1: Molmo bilinear maps, heads and leave-one-out ceilings from the
# molmo_bilinear rerun (neurips_final/molmo_bilinear/rerun_molmo_bilinear.py)
BILIN = os.environ.get('MOLMO_BILINEAR') == '1'
# MOLMO_BILINEAR_DIR=molmo_bilinear_pairwise: the same maps, scored against the pairwise ceiling
LOO = os.environ.get('MOLMO_BILINEAR_DIR', 'molmo_bilinear') == 'molmo_bilinear'
NT = os.path.join(ROOT, 'neurips_final', 'final_figures', os.environ.get('MOLMO_BILINEAR_DIR', 'molmo_bilinear') if BILIN else 'qwen35_nothink')
RES = os.path.join(NT, 'results')
OUT = os.path.join(NT, 'blur_sweep'); os.makedirs(OUT, exist_ok=True)
for sub in ('shared', 'task1_head_selection'):
    sys.path.insert(0, os.path.join(REB, sub))
import alignment_core as A
import vlm_maps as V

PROBE = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output'
V._QCFG = {t: dict(act=os.path.join(PROBE, d, 'qwen35_nothink', 'activations'),
                   mask=os.path.join(PROBE, d, 'qwen35_nothink', 'image_mask.pt'))
           for t, d in (('social', 'social_cap'), ('describe', 'coco'))}
if BILIN:
    sys.path.insert(0, os.path.join(ROOT, 'neurips_final', 'molmo_bilinear'))
    import molmo_bilinear_maps as MB
    MB.patch(V)
sel = json.load(open(os.path.join(RES, 'selected_heads.json')))
ceil = json.load(open(os.path.join(RES, 'ceilings.json')))
imgs = A.target_images()
MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
TASKS = ['social', 'describe']
SIGMAS = [0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def maps(m, task):
    if m.endswith('LSTM'):
        return A.trained_maps(m, task, imgs)
    L, H = sel[f'{m}_{task}']['layer'], sel[f'{m}_{task}']['head']
    if m.endswith('Transformer'):
        return A.trained_maps(m, task, imgs, layer=L, head=H)
    return V.vlm_maps(m, task, layer=L, head=H)


pf = os.path.join(OUT, 'blur_sweep_perimage.parquet')
if os.path.exists(pf):
    long = pd.read_parquet(pf)
else:
    frames = []
    for task in TASKS:
        for m in MODELS:
            base = maps(m, task)
            for s in SIGMAS:
                mp = base if s == 0 else {k: cv2.GaussianBlur(np.asarray(v, np.float32), (0, 0), s)
                                          for k, v in base.items()}
                df = A.per_image_z(mp, task, imgs, m); df['sigma'] = s
                frames.append(df)
            print(f'{task} {m}: swept', flush=True)
    long = pd.concat(frames, ignore_index=True)
    long.to_parquet(pf)


def summarize(df):
    sm = df.groupby('subject_id')['z'].mean()
    return np.tanh(sm.mean()), sm


curve, summ, subj = [], [], {}
rng = np.random.default_rng(0)
for task in TASKS:
    img_ids = sorted(long[long.task == task].image_id.unique())
    folds = np.array_split(rng.permutation(len(img_ids)), 5)
    for m in MODELS:
        d = long[(long.task == task) & (long.model == m)]
        pim = d.groupby(['sigma', 'image_id'])['z'].mean().unstack()          # sigma x image
        for s in SIGMAS:
            r, _ = summarize(d[d.sigma == s])
            curve.append(dict(task=task, model=m, sigma=s, mean_r=r, pct=100 * np.arctanh(r) / ceil[task]))
        best_in = int(pim.mean(axis=1).idxmax())
        parts, picks = [], []
        for f in folds:
            te = [img_ids[i] for i in f]; tr = [i for i in img_ids if i not in set(te)]
            s = int(pim[tr].mean(axis=1).idxmax()); picks.append(s)
            parts.append(d[(d.sigma == s) & d.image_id.isin(te)])
        r_cv, sm_cv = summarize(pd.concat(parts))
        subj[(task, m)] = sm_cv
        r0, _ = summarize(d[d.sigma == 0]); rb, _ = summarize(d[d.sigma == best_in])
        pc = lambda r: 100 * np.arctanh(r) / ceil[task]
        summ.append(dict(task=task, model=m, r_sigma0=r0, pct_sigma0=pc(r0), best_sigma_insample=best_in,
                         r_best=rb, pct_best=pc(rb), cv_sigmas='/'.join(map(str, picks)),
                         r_cv=r_cv, pct_cv=pc(r_cv)))
pd.DataFrame([dict(task=t, model=m, subject_id=sid, mean_z=z) for (t, m), sm in subj.items()
              for sid, z in sm.items()]).to_csv(os.path.join(OUT, 'blur_sweep_cv_subject_z.csv'), index=False)
curve = pd.DataFrame(curve); curve.to_csv(os.path.join(OUT, 'blur_sweep_curve.csv'), index=False)
summ = pd.DataFrame(summ); summ.to_csv(os.path.join(OUT, 'blur_sweep_summary.csv'), index=False)

con = []
for task in TASKS:
    for a, b in itertools.combinations(MODELS, 2):
        x, y = subj[(task, a)], subj[(task, b)]
        dd = (x - y).dropna()
        t = stats.ttest_rel(x[dd.index], y[dd.index])
        con.append(dict(task=task, a=a, b=b, diff_z=dd.mean(), dz=dd.mean() / dd.std(ddof=1), p=t.pvalue))
con = pd.DataFrame(con)
def holm(p):
    order = np.argsort(p); m = len(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m); out[order] = adj
    return out


for task in TASKS:
    k = con.task == task
    con.loc[k, 'p_holm'] = holm(con.loc[k, 'p'].values)
con.to_csv(os.path.join(OUT, 'blur_sweep_contrasts.csv'), index=False)

pd.set_option('display.width', 200)
print(summ.round(3).to_string(index=False))
print(con.round(4).to_string(index=False))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(REB, 'shared'))
import figstyle as FST   # the paper's model colours
LBL = {'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen3.5 9B'}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, task in zip(axes, ['describe', 'social']):
    for m in MODELS:
        c = curve[(curve.task == task) & (curve.model == m)]
        ax.plot(c.sigma, c.pct, marker='o', ms=4, label=LBL.get(m, m), color=FST.ALL_COLORS[m])
    ax.axhline(100, ls='--', c='k', lw=1); ax.set_title(f'{task.capitalize()} task')
    ax.set_xlabel('Gaussian blur $\\sigma$ (px, 256 px image)'); ax.set_xscale('symlog', linthresh=2); ax.set_xlim(-0.3, 70)
    ax.set_xticks(SIGMAS); ax.set_xticklabels(SIGMAS, fontsize=8)
axes[0].set_ylabel('% of human noise ceiling' + (' (leave-one-out)' if BILIN and LOO else '')); axes[1].legend(fontsize=9, frameon=False, loc='lower left', ncol=2)
plt.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_blur_sweep.png'), dpi=200)
print('wrote', OUT)
