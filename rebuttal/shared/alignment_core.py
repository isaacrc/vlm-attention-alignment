"""
alignment_core.py — shared data layer for rebuttal Tasks 1-5.
================================================================
One primitive underlies every task: the per-subject x per-image Spearman-z
between a *chosen* model attention map and human fixations. Build it once here;
Tasks 1/3/4/5 consume the tidy long tables it produces.

Adapts the fast pre-ranked Spearman from analysis/head_attention_analysis.py and
the map conventions from analysis/compute_correlations.py.

All paths point at the CANS working tree (the real data); outputs go under
rebuttal/. See rebuttal/EXECUTION_PLAN_tasks_1-5.md for the full data map.
"""
import os, glob, re
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata as _rankdata

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = '/scratch/gpfs/GRAZIANO/CANS'
ARCH_DIR   = os.path.join(BASE, 'architecture_manipulation')
GROUP_DIR  = os.path.join(BASE, 'group_analysis')
PROC_DIR   = os.path.join(GROUP_DIR, 'processed_correlations')
RNN_DIR    = os.path.join(BASE, 'rnn_model')
TARGET_CSV = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
HM_DIR = {
    'social':   os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social'),
    'describe': os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive'),
}
OUT_SIZE = 256

# Canonical subject exclusions (rebuttal). Describe: drop 3 subjects with corrupted
# caption/audio data -> n=22 (see project_describe_n22 memory / EXECUTION_PLAN D3).
EXCLUDE = {'social': set(), 'describe': {'sub032', 'sub050', 'sub057'}}

# ── Model registry ─────────────────────────────────────────────────────────────
# encoder/decoder factor labels + the NPZ that holds each trained model's maps.
# Phase->task: phase2 = describe, social_unfrozen = social (paper's best epochs).
TRAINED = {
    'CNN-LSTM':        dict(encoder='CNN', decoder='LSTM', cond='CNN_LSTM'),
    'ViT-LSTM':        dict(encoder='ViT', decoder='LSTM', cond='ViT_LSTM'),
    'CNN-Transformer': dict(encoder='CNN', decoder='Transformer', cond='CNN_Transformer',
                            best_head={'social': (3, 5), 'describe': (3, 5)}),  # (layer,head) 1-idx
    'ViT-Transformer': dict(encoder='ViT', decoder='Transformer', cond='ViT_Transformer',
                            best_head={'social': (3, 8), 'describe': (3, 8)}),
}
# best-epoch NPZ filenames per condition/phase (from fig3 + between_task scripts)
NPZ_FILE = {
    'CNN_Transformer': {'describe': 'phase2_beam5_E014.npz', 'social': 'social_unfrozen_beam5_E003.npz'},
    'ViT_Transformer': {'describe': 'phase2_beam5_E011.npz', 'social': 'social_unfrozen_beam5_E001.npz'},
    'ViT_LSTM':        {'describe': 'phase2_beam5_E003.npz', 'social': 'social_unfrozen_beam5_E001.npz'},
    # CNN_LSTM (Project-1 naming): social in cans_beam5.npz; describe handled specially below.
    'CNN_LSTM':        {'social': 'cans_beam5.npz'},
}
VLM = {
    'Molmo':   dict(encoder='VLM', decoder='VLM'),
    'Qwen3.5': dict(encoder='VLM', decoder='VLM'),
}

# ── target images ──────────────────────────────────────────────────────────────
def target_images():
    return pd.read_csv(TARGET_CSV)['image_name'].tolist()

# ── helpers ────────────────────────────────────────────────────────────────────
def _norm(a):
    mn, mx = float(a.min()), float(a.max())
    return (a - mn) / (mx - mn + 1e-10)

def alpha_to_map(vec196):
    """(196,) -> 14x14 -> BILINEAR 256x256 -> minmax [0,1]. Matches paper post-proc."""
    arr = np.asarray(vec196, dtype=np.float32).reshape(14, 14)
    r = np.array(Image.fromarray(arr).resize((OUT_SIZE, OUT_SIZE), resample=Image.BILINEAR),
                 dtype=np.float32)
    return _norm(r)

def _rank(mat, axis=-1):
    return _rankdata(mat, method='average', axis=axis).astype(np.float32)

# ── human loading + pre-ranking (fast Spearman) ────────────────────────────────
_HUMAN_CACHE = {}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_human(task, imgs):
    """Returns (hum_ranked (S,N,65536) float32 pre-ranked, hum_valid (S,N) bool,
    subject_ids list). Keys may be '000..jpg' or bare id. Disk-cached to
    rebuttal/data/human_ranked_{task}.npz so parallel task processes load fast."""
    key = (task, tuple(imgs))
    if key in _HUMAN_CACHE:
        return _HUMAN_CACHE[key]
    # disk cache (valid only when the image list matches)
    cache_f = os.path.join(_CACHE_DIR, f'human_ranked_{task}.npz')
    if os.path.exists(cache_f):
        z = np.load(cache_f, allow_pickle=True)
        if list(z['imgs']) == list(imgs):
            out = (z['ranked'], z['valid'], list(z['subs']))
            _HUMAN_CACHE[key] = out
            return out
    hm_dir = HM_DIR[task]
    files = sorted(f for f in os.listdir(hm_dir)
                   if f.endswith('_heatmaps.npy')
                   and f.replace('_heatmaps.npy', '') not in EXCLUDE[task])
    S, N, D = len(files), len(imgs), OUT_SIZE * OUT_SIZE
    ranked = np.zeros((S, N, D), dtype=np.float32)
    valid  = np.zeros((S, N), dtype=bool)
    subs   = []
    for si, fn in enumerate(files):
        subs.append(fn.replace('_heatmaps.npy', ''))
        sdat = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
        raw, ok = [], []
        for im in imgs:
            hm = sdat.get(im, sdat.get(im.replace('.jpg', ''), sdat.get(im + '.jpg')))
            if hm is None:
                raw.append(np.zeros(D, np.float32)); ok.append(False)
            else:
                h = np.nan_to_num(hm, nan=0.0).astype(np.float32)
                h = np.array(Image.fromarray(h).resize((OUT_SIZE, OUT_SIZE), resample=Image.NEAREST),
                             dtype=np.float32).flatten()
                raw.append(h); ok.append(True)
        valid[si] = ok
        vi = np.where(ok)[0]
        if len(vi):
            ranked[si, vi] = _rank(np.stack([raw[i] for i in vi]), axis=1)
    out = (ranked, valid, subs)
    _HUMAN_CACHE[key] = out
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        np.savez(cache_f, ranked=ranked, valid=valid,
                 subs=np.array(subs), imgs=np.array(list(imgs)))
    except Exception as e:
        print(f'  (human cache write failed: {e})')
    return out

# ── per-subject x per-image z for a set of model maps ──────────────────────────
def per_image_z(model_maps, task, imgs, model_name=None):
    """
    model_maps: {img_name: (256,256) or (65536,) raw map}
    Returns long DataFrame [model, task, subject_id, image_id, r, z].
    Spearman = Pearson on ranks; z = arctanh(r).
    """
    ranked, valid, subs = load_human(task, imgs)
    # rank each model map once
    order = [im for im in imgs if im in model_maps]
    idx   = {im: i for i, im in enumerate(imgs)}
    mv = np.stack([np.asarray(model_maps[im], np.float32).reshape(-1) for im in order])
    keep = mv.std(axis=1) > 1e-12
    order = [order[i] for i in range(len(order)) if keep[i]]
    mv = mv[keep]
    mv_r = _rank(mv, axis=1)
    mv_r = mv_r - mv_r.mean(axis=1, keepdims=True)
    mv_n = mv_r / (np.linalg.norm(mv_r, axis=1, keepdims=True) + 1e-10)
    rows = []
    for si, sid in enumerate(subs):
        for k, im in enumerate(order):
            j = idx[im]
            if not valid[si, j]:
                continue
            hv = ranked[si, j]
            hv = hv - hv.mean()
            nrm = np.linalg.norm(hv)
            if nrm < 1e-10:
                continue
            r = float(np.dot(mv_n[k], hv / nrm))
            r = min(max(r, -0.999999), 0.999999)
            rows.append(dict(model=model_name, task=task, subject_id=sid,
                             image_id=im, r=r, z=float(np.arctanh(r))))
    return pd.DataFrame(rows)

def subject_means(df):
    """Fisher-z avg over images per subject, back-transform. -> [model,task,subject_id,mean_r,mean_z]."""
    g = df.groupby(['model', 'task', 'subject_id'])['z'].mean().reset_index()
    g['mean_r'] = np.tanh(g['z']); g = g.rename(columns={'z': 'mean_z'})
    return g

def agg(df):
    """model/task summary: mean_r (Fisher), SEM across subjects, n."""
    sm = subject_means(df)
    out = []
    for (m, t), grp in sm.groupby(['model', 'task']):
        z = grp['mean_z'].values
        out.append(dict(model=m, task=t, mean_r=float(np.tanh(z.mean())),
                        sem_r=float(np.tanh(z.std(ddof=1)/np.sqrt(len(z)) + z.mean()) - np.tanh(z.mean()))
                        if len(z) > 1 else 0.0, n_sub=len(z)))
    return pd.DataFrame(out)

# ── ceilings ───────────────────────────────────────────────────────────────────
_CEIL_CACHE = {}

def ceiling(task, imgs=None, as_r=False):
    """Human-human noise ceiling over the CANONICAL (exclusion-respecting) subject
    set. Per-image mean of pairwise Fisher-z Spearman, averaged over images.
    Returns mean-z by default (paper convention: social 0.284, describe(22) 0.254)."""
    if task in _CEIL_CACHE:
        z = _CEIL_CACHE[task]
        return float(np.tanh(z)) if as_r else z
    if imgs is None:
        imgs = target_images()
    ranked, valid, subs = load_human(task, imgs)   # already excludes EXCLUDE[task]
    per_img = []
    for j in range(len(imgs)):
        vi = np.where(valid[:, j])[0]
        if len(vi) < 2:
            continue
        V = ranked[vi, j]
        Vc = V - V.mean(axis=1, keepdims=True)
        Vn = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-10)
        C = Vn @ Vn.T
        iu = np.triu_indices(len(vi), 1)
        z = np.arctanh(np.clip(C[iu], -0.999999, 0.999999))
        per_img.append(z.mean())
    zc = float(np.mean(per_img))
    _CEIL_CACHE[task] = zc
    return float(np.tanh(zc)) if as_r else zc

# ── trained-model attention maps ───────────────────────────────────────────────
def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}

def trained_maps(model_name, task, imgs, layer=None, head=None):
    """
    Return {img: (256,256)} for a trained model on a task.
    LSTM: uses 'alphas'. Transformer: uses head_layer_alphas[layer-1, head-1] if
    (layer,head) given, else best_head, else 'alphas'.
    """
    info = TRAINED[model_name]; cond = info['cond']
    if model_name == 'CNN-LSTM' and task == 'describe':
        # Project-1 CNN-LSTM describe maps (alphas already 14x14).
        npz = os.path.join(GROUP_DIR, 'processed_heatmaps_rnn', 'h_c_alpha_n200_describe_final.npz')
        alphas = _load_npz(npz)['alphas']
        return {im: alpha_to_map(alphas[k])
                for im in imgs
                for k in [im if im in alphas else im.replace('.jpg', '')]
                if k in alphas}
    npz = os.path.join(ARCH_DIR, cond, 'ckpt_cache', NPZ_FILE[cond][task])
    d = _load_npz(npz)
    if info['decoder'] == 'LSTM':
        alphas = d['alphas']
        return {im: alpha_to_map(alphas[k])
                for im in imgs
                for k in [im if im in alphas else im.replace('.jpg', '')]
                if k in alphas}
    # Transformer
    if layer is None or head is None:
        layer, head = info['best_head'][task]
    hla = d['head_layer_alphas']
    out = {}
    for im in imgs:
        k = im if im in hla else im.replace('.jpg', '')
        if k not in hla:
            continue
        arr = hla[k]  # (n_layers, n_heads, 196)
        li = min(layer - 1, arr.shape[0] - 1); hi = min(head - 1, arr.shape[1] - 1)
        out[im] = alpha_to_map(arr[li, hi])
    return out

def transformer_head_grid(model_name, task, imgs):
    """Yield (layer, head, {img:(256,256)}) for ALL layers x heads. For Task 1/4."""
    info = TRAINED[model_name]; cond = info['cond']
    npz = os.path.join(ARCH_DIR, cond, 'ckpt_cache', NPZ_FILE[cond][task])
    d = _load_npz(npz); hla = d['head_layer_alphas']
    a0 = next(iter(hla.values())); nL, nH = a0.shape[0], a0.shape[1]
    for li in range(nL):
        for hi in range(nH):
            maps = {}
            for im in imgs:
                k = im if im in hla else im.replace('.jpg', '')
                if k in hla:
                    maps[im] = alpha_to_map(hla[k][li, hi])
            yield li + 1, hi + 1, maps


if __name__ == '__main__':
    # ── sanity: reproduce paper headline numbers ──
    imgs = target_images()
    print('Ceilings: social=%.4f describe=%.4f' % (ceiling('social'), ceiling('describe')))
    for m in ['CNN-LSTM', 'ViT-LSTM']:
        mm = trained_maps(m, 'social', imgs)
        df = per_image_z(mm, 'social', imgs, m)
        a = agg(df).iloc[0]
        print(f'{m:16s} social mean_r={a.mean_r:.4f} (n_sub={a.n_sub})  '
              f'[paper: CNN-LSTM .238 / ViT-LSTM .223]')
    for m in ['CNN-Transformer', 'ViT-Transformer']:
        mm = trained_maps(m, 'social', imgs)
        df = per_image_z(mm, 'social', imgs, m)
        a = agg(df).iloc[0]
        print(f'{m:16s} social mean_r={a.mean_r:.4f} (best head)  '
              f'[paper: CNN-Trans .166 / ViT-Trans .112]')
