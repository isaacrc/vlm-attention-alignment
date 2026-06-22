"""
3_encoding_model.py
-------------------
For each model × task (12 total), fit a kernel ridge encoding model:
  X: (200, 196) attention map features
  Y: (200, 20484) TRIBE-predicted fMRI

Uses himalaya KernelRidgeCV with K=5 fold CV across images.
Evaluates with Pearson r per vertex; saves R² = r².

Output: tribe/outputs/r2_{model}_{task}.npy  shape (20484,)

Run on CPU.
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

BASE    = Path('/scratch/gpfs/GRAZIANO/CANS')
OUT_DIR = BASE / 'tribe/outputs'

try:
    from himalaya.kernel_ridge import KernelRidgeCV, Kernelizer
    from himalaya.scoring import correlation_score
    from sklearn.pipeline import make_pipeline
    HAS_HIMALAYA = True
except ImportError:
    print('WARNING: himalaya not found, falling back to sklearn RidgeCV')
    from sklearn.linear_model import RidgeCV
    HAS_HIMALAYA = False

MODELS = [
    'CNN-LSTM', 'ViT-LSTM',
    'CNN-Transformer', 'ViT-Transformer',
    'Molmo', 'Qwen3.5',
]
TASKS = ['describe', 'social']

Y = np.load(OUT_DIR / 'tribe_Y.npy')   # (200, 20484)
print(f'Loaded Y: {Y.shape}')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model in MODELS:
    for task in TASKS:
        out_path = OUT_DIR / f'r2_{model}_{task}.npy'
        if out_path.exists():
            print(f'Skipping (exists): {out_path.name}')
            continue

        x_path = OUT_DIR / f'X_attn_{model}_{task}.npy'
        if not x_path.exists():
            print(f'Missing X: {x_path.name} — skipping')
            continue

        X = np.load(x_path)   # (200, 196)
        print(f'\n{model} / {task}  X={X.shape}  Y={Y.shape}')

        r_folds = []
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            if HAS_HIMALAYA:
                pipeline = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    Kernelizer(kernel='linear'),
                    KernelRidgeCV(
                        alphas=np.logspace(1, 10, 10),
                        cv=KFold(n_splits=4),
                    ),
                )
                pipeline.fit(X_train, Y_train)
                Y_pred = pipeline.predict(X_test)          # (40, 20484)
                r = correlation_score(Y_test, Y_pred)      # (20484,)
            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train)
                X_te_s = scaler.transform(X_test)
                ridge  = RidgeCV(alphas=np.logspace(1, 10, 10), cv=4)
                ridge.fit(X_tr_s, Y_train)
                Y_pred = ridge.predict(X_te_s)
                # compute per-vertex pearson r manually
                r = np.array([
                    np.corrcoef(Y_test[:, v], Y_pred[:, v])[0, 1]
                    for v in range(Y_test.shape[1])
                ])

            r = np.nan_to_num(r, nan=0.0)
            r_folds.append(r)
            print(f'  fold {fold_i+1}/5 done  mean_r={r.mean():.4f}  max_r={r.max():.4f}')

        r_mean = np.stack(r_folds).mean(axis=0)   # (20484,)
        r2     = r_mean ** 2
        np.save(str(out_path), r2)

        thresh = np.percentile(r2, 95)
        print(f'Saved {out_path.name}  top-5% R²={r2[r2>=thresh].mean():.4f}  '
              f'mean R²={r2.mean():.4f}')

print('\nEncoding models complete.')
