"""
build_cvhonest_table.py — alignment table under HONEST, PER-TASK head selection.

The default table (data/alignment_long_best_head.parquet) uses the head the paper
designates for each Transformer model (CNN-Transformer L3H5, ViT-Transformer L3H8)
for BOTH tasks. The full head sweep shows that designation is the true maximum in
the Describe task but not in the Social task, where it understates both models:

    CNN-Transformer social: L4H4 r=.203 vs designated L3H5 r=.166
    ViT-Transformer social: L4H8 r=.157 vs designated L3H8 r=.112

This script rebuilds the long table selecting each Transformer/VLM head by the
cross-validated rule (5-fold over images; the selection is re-made on each training
split, so the reported value is not the in-sample maximum). Winner's-curse inflation
is < .0014 r in every cell, so CV-honest and global-best coincide here; the point of
the table is that the SELECTION is defensible, not that the number moves.

LSTM models have a single attention map and are unchanged.

Outputs: data/alignment_long_cv_honest.parquet, data/alignment_summary_cv_honest.csv,
         results/selected_heads.json
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import alignment_core as A
import vlm_maps as V

RES = os.path.join(HERE, 'results')
DATA = os.path.join(HERE, '..', 'data')
RNG = np.random.default_rng(0)

FACTOR = {'CNN-LSTM': ('CNN', 'LSTM'), 'ViT-LSTM': ('ViT', 'LSTM'),
          'CNN-Transformer': ('CNN', 'Transformer'),
          'ViT-Transformer': ('ViT', 'Transformer'),
          'Molmo': ('VLM', 'VLM'), 'Qwen3.5': ('VLM', 'VLM')}
SEL_MODELS = ['CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']


def cv_selected_head(pim, model, task, k=5):
    """5-fold CV over images. Returns (layer, head) chosen on the majority of
    training folds, plus the CV-honest mean z (selection made out-of-sample)."""
    sub = pim[(pim['model'] == model) & (pim['task'] == task)]
    M = sub.pivot_table(index=['layer', 'head'], columns='image_id', values='mean_z')
    cols = list(M.columns); n = len(cols)
    idx = RNG.permutation(n); folds = np.array_split(idx, k)
    picks, ev = [], []
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(k) if j != f])
        tr_c = [cols[i] for i in tr]; te_c = [cols[i] for i in te]
        best = M[tr_c].mean(axis=1).idxmax()
        picks.append(best)
        ev.append(M.loc[best, te_c].mean())
    # modal pick across folds
    lay, hd = max(set(picks), key=picks.count)
    return int(lay), int(hd), float(np.mean(ev)), picks


def main():
    imgs = A.target_images()
    ceil = json.load(open(os.path.join(DATA, 'ceilings.json')))
    pim = pd.read_parquet(os.path.join(RES, 'head_table_perimage.parquet'))

    selected, frames = {}, []
    for task in ['social', 'describe']:
        # LSTMs: single map, nothing to select
        for m in ['CNN-LSTM', 'ViT-LSTM']:
            frames.append(A.per_image_z(A.trained_maps(m, task, imgs), task, imgs, m))
            print(f'  {task:9s} {m:16s} (single map)', flush=True)
        for m in SEL_MODELS:
            L, H, cvz, picks = cv_selected_head(pim, m, task)
            stable = sum(1 for p in picks if tuple(p) == (L, H))
            selected[f'{m}_{task}'] = dict(layer=L, head=H, cv_honest_z=cvz,
                                           cv_honest_r=float(np.tanh(cvz)),
                                           fold_agreement=f'{stable}/5')
            maps = (V.vlm_maps(m, task, L, H) if m in ('Molmo', 'Qwen3.5')
                    else A.trained_maps(m, task, imgs, layer=L, head=H))
            frames.append(A.per_image_z(maps, task, imgs, m))
            print(f'  {task:9s} {m:16s} -> L{L}H{H} '
                  f'(cv r={np.tanh(cvz):.4f}, folds agree {stable}/5)', flush=True)

    json.dump(selected, open(os.path.join(RES, 'selected_heads.json'), 'w'), indent=2)

    long = pd.concat(frames, ignore_index=True)
    long['encoder'] = long['model'].map(lambda m: FACTOR[m][0])
    long['decoder'] = long['model'].map(lambda m: FACTOR[m][1])
    long.to_parquet(os.path.join(DATA, 'alignment_long_cv_honest.parquet'))

    summ = A.agg(long)
    summ['pct_ceiling'] = summ.apply(
        lambda r: 100 * np.arctanh(r['mean_r']) / ceil[r['task']], axis=1)
    summ.to_csv(os.path.join(DATA, 'alignment_summary_cv_honest.csv'), index=False)
    for t in ['social', 'describe']:
        print(f'\n=== {t.upper()} (CV-honest heads) ===')
        print(summ[summ.task == t].sort_values('mean_r', ascending=False)
              [['model', 'mean_r', 'pct_ceiling']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
