"""
3b_banded_encoding_model.py
---------------------------
Banded ridge encoding model: attention + CLIP as separate bands.

For each model × task (12 total):
  Band 1 — X_attn: (200, 196)  model attention map
  Band 2 — X_clip: (200, 768)  CLIP ViT-L/14 visual embedding (shared)

Uses himalaya MultipleKernelRidgeCV with separate alpha per band.
Evaluates with Pearson r per vertex; saves R² = r².

Outputs per model × task:
  tribe/outputs/r2_banded_{model}_{task}.npy    shape (200, 20484)
  tribe/outputs/deltas_banded_{model}_{task}.npy shape (200, 2)  band weights

Run on CPU.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE    = Path('/scratch/gpfs/GRAZIANO/CANS')
OUT_DIR = BASE / 'tribe/outputs'

try:
    from himalaya.kernel_ridge import MultipleKernelRidgeCV, Kernelizer, ColumnKernelizer
    from himalaya.scoring import correlation_score
    HAS_HIMALAYA = True
except ImportError:
    print('WARNING: himalaya not found — banded ridge requires himalaya. Exiting.')
    raise SystemExit(1)

MODELS = [
    'CNN-LSTM', 'ViT-LSTM',
    'CNN-Transformer', 'ViT-Transformer',
    'Molmo', 'Qwen3.5',
]
TASKS = ['describe', 'social']

Y     = np.load(OUT_DIR / 'tribe_Y.npy')        # (200, 20484)
X_clip = np.load(OUT_DIR / 'X_clip.npy')         # (200, 768)
print(f'Loaded Y: {Y.shape}  X_clip: {X_clip.shape}')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model in MODELS:
    for task in TASKS:
        out_r2  = OUT_DIR / f'r2_banded_{model}_{task}.npy'
        out_del = OUT_DIR / f'deltas_banded_{model}_{task}.npy'
        if out_r2.exists():
            print(f'Skipping (exists): {out_r2.name}')
            continue

        x_path = OUT_DIR / f'X_attn_{model}_{task}.npy'
        if not x_path.exists():
            print(f'Missing X_attn: {x_path.name} — skipping')
            continue

        X_attn = np.load(x_path)   # (200, 196)
        print(f'\n{model} / {task}  X_attn={X_attn.shape}  X_clip={X_clip.shape}  Y={Y.shape}')

        # Band split indices for ColumnKernelizer
        n_attn = X_attn.shape[1]   # 196
        n_clip = X_clip.shape[1]   # 768
        X_all  = np.concatenate([X_attn, X_clip], axis=1)  # (200, 964)

        r_folds  = []
        del_folds = []

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_all)):
            X_tr, X_te = X_all[train_idx], X_all[test_idx]
            Y_tr, Y_te = Y[train_idx],     Y[test_idx]

            # Scale each band independently before kernelizing
            sc_attn = StandardScaler()
            sc_clip = StandardScaler()
            X_tr_attn = sc_attn.fit_transform(X_tr[:, :n_attn])
            X_te_attn = sc_attn.transform(X_te[:, :n_attn])
            X_tr_clip = sc_clip.fit_transform(X_tr[:, n_attn:])
            X_te_clip = sc_clip.transform(X_te[:, n_attn:])

            X_tr_scaled = np.concatenate([X_tr_attn, X_tr_clip], axis=1)
            X_te_scaled = np.concatenate([X_te_attn, X_te_clip], axis=1)

            ck = ColumnKernelizer(
                [
                    ('attn', Kernelizer(kernel='linear'), slice(0, n_attn)),
                    ('clip', Kernelizer(kernel='linear'), slice(n_attn, n_attn + n_clip)),
                ]
            )

            mkr = MultipleKernelRidgeCV(
                kernels='precomputed',
                solver='random_search',
                solver_params=dict(n_iter=20, alphas=np.logspace(1, 10, 10)),
                random_state=42,
                cv=KFold(n_splits=4),
            )

            Ks_tr = ck.fit_transform(X_tr_scaled)
            mkr.fit(Ks_tr, Y_tr)

            Ks_te = ck.transform(X_te_scaled)
            Y_pred = mkr.predict(Ks_te)       # (n_test, 20484)

            r = correlation_score(Y_te, Y_pred)   # (20484,)
            r = np.nan_to_num(r, nan=0.0)
            r_folds.append(r)

            # deltas: (2,) log band weights averaged across vertices
            deltas_mean = mkr.deltas_.mean(axis=0).cpu().numpy() if hasattr(mkr.deltas_, 'cpu') else mkr.deltas_.mean(axis=0)
            del_folds.append(deltas_mean)

            print(f'  fold {fold_i+1}/5  mean_r={r.mean():.4f}  max_r={r.max():.4f}  deltas={np.exp(deltas_mean).round(3)}')

        r_mean = np.stack(r_folds).mean(axis=0)
        r2     = r_mean ** 2
        deltas = np.stack(del_folds).mean(axis=0)   # (2,) mean log-deltas

        np.save(str(out_r2),  r2)
        np.save(str(out_del), deltas)

        thresh = np.percentile(r2, 95)
        print(f'Saved {out_r2.name}  top-5% R²={r2[r2>=thresh].mean():.4f}  '
              f'mean R²={r2.mean():.4f}  band weights attn/clip={np.exp(deltas).round(3)}')

print('\nBanded encoding models complete.')
