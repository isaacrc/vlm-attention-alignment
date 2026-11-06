"""
figs_bestblur.py
----------------
Variants of Figures 1 and 2 with every model at its own best Gaussian blur (sigma chosen
by 5-fold CV over images, blur_sweep.py) and, in Figure 1, the leave-one-subject-out
noise ceiling (loo_ceiling.py) instead of the pairwise one.

Figure 1
  A-B : per-subject mean z at the CV-chosen sigma (blur_sweep_cv_subject_z.csv), as % of
        the LOO ceiling; SEM across participants as in the paper.
  C   : between-task similarity on one head per model (fig1_between_besthead.py rule), with
        BOTH task maps blurred at the sigma chosen for that head's task. Human reference
        unchanged (group-mean maps are already smooth).
Figure 2
  The paper's example grid with each model x task map blurred at its CV sigma; headers
  give the head and sigma. Human maps unchanged.

Outputs: final_figures/qwen35_nothink/bestblur/{fig1_bestblur.png, fig2_bestblur.png, *.csv, *.json}
"""
import os, sys, json
from collections import Counter
import numpy as np
import pandas as pd
import cv2
from scipy.stats import spearmanr

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
REB = os.path.join(ROOT, 'rebuttal')
NT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink')
RES, BS = os.path.join(NT, 'results'), os.path.join(NT, 'blur_sweep')
OUT = os.path.join(NT, 'bestblur'); os.makedirs(OUT, exist_ok=True)
for sub in ('shared', 'task1_head_selection'):
    sys.path.insert(0, os.path.join(REB, sub))
import alignment_core as A
import vlm_maps as V

PROBE = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output'
V._QCFG = {t: dict(act=os.path.join(PROBE, d, 'qwen35_nothink', 'activations'),
                   mask=os.path.join(PROBE, d, 'qwen35_nothink', 'image_mask.pt'))
           for t, d in (('social', 'social_cap'), ('describe', 'coco'))}
sel = json.load(open(os.path.join(RES, 'selected_heads.json')))
imgs = A.target_images()
MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']

# sigma per model x task: modal CV pick across the 5 folds
summ = pd.read_csv(os.path.join(BS, 'blur_sweep_summary.csv'))
SIG = {(r.task, r.model): Counter(int(s) for s in r.cv_sigmas.split('/')).most_common(1)[0][0]
       for r in summ.itertuples()}

# ── Figure 1 data ─────────────────────────────────────────────────────────────
loo = json.load(open(os.path.join(BS, 'ceilings_pairwise_vs_loo.json')))
json.dump({t: loo[t]['loo_z'] for t in ('social', 'describe')}, open(os.path.join(OUT, 'ceilings_loo.json'), 'w'))
sz = pd.read_csv(os.path.join(BS, 'blur_sweep_cv_subject_z.csv'))
rows = []
for (m, t), grp in sz.groupby(['model', 'task']):
    z = grp.mean_z.values; mz, se = z.mean(), z.std(ddof=1) / np.sqrt(len(z))
    rows.append(dict(model=m, task=t, mean_r=np.tanh(mz), sem_r=np.tanh(mz + se) - np.tanh(mz), n_sub=len(z),
                     pct_ceiling=100 * mz / loo[t]['loo_z'], sigma=SIG[(t, m)]))
fsum = pd.DataFrame(rows); fsum.to_csv(os.path.join(OUT, 'alignment_summary_bestblur_loo.csv'), index=False)

head_rule = pd.read_csv(os.path.join(RES, 'model_between_task_fig1.csv')).set_index('model')
bt = []
for m in MODELS:
    if m.endswith('LSTM'):
        head, s = None, SIG[('social', m)]          # 16 px in both tasks for both LSTMs
        get = lambda t: A.trained_maps(m, t, imgs)
    else:
        ht = head_rule.loc[m, 'head_task']; head = (sel[f'{m}_{ht}']['layer'], sel[f'{m}_{ht}']['head'])
        s = SIG[(ht, m)]
        get = (lambda t: A.trained_maps(m, t, imgs, layer=head[0], head=head[1])) if m.endswith('Transformer') \
            else (lambda t: V.vlm_maps(m, t, layer=head[0], head=head[1]))
    mp = {t: {k: cv2.GaussianBlur(np.asarray(v, np.float32), (0, 0), s) for k, v in get(t).items()}
          for t in ('social', 'describe')}
    rs = np.array([spearmanr(mp['social'][im].ravel(), mp['describe'][im].ravel())[0] for im in imgs
                   if im in mp['social'] and im in mp['describe']])
    bt.append(dict(model=m, between_r=rs.mean(), sem=rs.std(ddof=1) / np.sqrt(len(rs)), n_img=len(rs),
                   task_diff_index=1 - rs.mean(), head='' if head is None else f'L{head[0]}H{head[1]}', sigma=s))
    print(bt[-1], flush=True)
pd.DataFrame(bt).to_csv(os.path.join(OUT, 'model_between_task_bestblur.csv'), index=False)

src_path = os.path.join(REB, 'plots', 'plot_fig_head_selection_corrected.py')
src = open(src_path).read()
for old, new in [("os.path.join(DATA, 'alignment_summary_cv_honest.csv')", repr(os.path.join(OUT, 'alignment_summary_bestblur_loo.csv'))),
                 ("os.path.join(DATA, 'ceilings.json')", repr(os.path.join(OUT, 'ceilings_loo.json'))),
                 ("os.path.join(T1, 'selected_heads.json')", repr(os.path.join(RES, 'selected_heads.json'))),
                 ("os.path.join(T2, 'model_between_task.csv')", repr(os.path.join(OUT, 'model_between_task_bestblur.csv'))),
                 ("os.path.join(HERE, 'head_selection', 'fig_head_selection_corrected.png')",
                  repr(os.path.join(OUT, 'fig1_bestblur.png')))]:
    assert old in src, old
    src = src.replace(old, new)
exec(src, {'__file__': src_path, '__name__': 'fig1_bestblur'})

# ── Figure 2 ──────────────────────────────────────────────────────────────────
SRC = os.path.join(ROOT, 'archive', 'ICLR', 'scripts', 'regen_fig5_entropy_cvheads.py')
fsrc = open(SRC).read()
g = {'__file__': SRC, '__name__': 'fig2_bestblur'}
exec(compile(fsrc[:fsrc.index('# ── 1. entropy')], SRC, 'exec'), g)
g['OUT_FIG'] = os.path.join(OUT, 'fig2_bestblur.png')
g['HEADS']['Qwen3.5'] = {t: (sel[f'Qwen3.5_{t}']['layer'], sel[f'Qwen3.5_{t}']['head']) for t in ('social', 'describe')}
P = g['PROBING_DIR']
g['QCFG'] = {'social': (os.path.join(P, 'output', 'social_cap', 'qwen35_nothink'),),
             'describe': (os.path.join(P, 'output', 'coco', 'qwen35_nothink'),)}
torch = g['torch']


def molmo_unblurred(task, only=None):
    """28x28 token grid (crop-overlap margins dropped), bilinear to 256; as final_fig5_entropy.py."""
    L, H = g['HEADS']['Molmo'][task]
    act_dir = os.path.join(g['MOLMO_DIR'], 'activations', task)
    grid36 = g['_molmo_grid'](act_dir)[0]
    vidx = np.where(grid36 > 0); pos = grid36[vidx]
    rws, cls = np.any(grid36 > 0, axis=1), np.any(grid36 > 0, axis=0)
    out = {}
    for im in only:
        a = torch.load(os.path.join(act_dir, f'activations_img_{im}.pt'), map_location='cpu')[L - 1, H - 1].float().numpy()
        vol = np.zeros((a.shape[0],) + grid36.shape, np.float32); vol[:, vidx[0], vidx[1]] = a[:, pos]
        out[im] = cv2.resize(vol.sum(0)[rws][:, cls], (g['SIZE'], g['SIZE']), interpolation=cv2.INTER_LINEAR)
    return out


def blurred(fn, model_of):
    def wrap(*a, **k):
        task = a[1] if len(a) > 1 else a[0]
        s = SIG[(task, model_of(*a))]
        return {im: cv2.GaussianBlur(np.asarray(v, np.float32), (0, 0), s) for im, v in fn(*a, **k).items()}
    return wrap


g['lstm_maps'] = blurred(g['lstm_maps'], lambda m, t: m)
g['trans_maps'] = blurred(g['trans_maps'], lambda m, t: m)
g['molmo_maps'] = blurred(molmo_unblurred, lambda t, **k: 'Molmo')
g['qwen_maps'] = blurred(g['qwen_maps'], lambda t, **k: 'Qwen3.5')
g['SIG'] = SIG
fig_src = fsrc[fsrc.index('# ── 2. Fig 5 grid'):]
for old, new in [("    if col in ('CNN-LSTM', 'ViT-LSTM'): return col.replace('-', '\\n')",
                  "    if col in ('CNN-LSTM', 'ViT-LSTM'): return col.replace('-', '\\n') + f'\\nσ = {SIG[(task, col)]}'"),
                 ("    return f'{name}\\n(L{L}H{H})'", "    return f'{name}\\n(L{L}H{H})\\nσ = {SIG[(task, col)]}'"),
                 # four-line headers: push the Describe block further below the divider
                 ("b.y0 - 0.035", "b.y0 - 0.06")]:
    assert old in fig_src, old
    fig_src = fig_src.replace(old, new)
exec(compile(fig_src, SRC, 'exec'), g)
print(fsum.round(3).to_string(index=False))
