"""
build_alignment_table.py — canonical tidy alignment tables for Tasks 1/3/4/5.
Produces per-subject x per-image Spearman-z for all 6 models under the DEFAULT
(best-head / single-map) selection, both tasks, on the canonical subject sets
(social n=27, describe n=22). Also caches ranked humans + recomputed ceilings.

Outputs (rebuttal/data/):
  human_ranked_{task}.npz        (written lazily by alignment_core.load_human)
  ceilings.json                  {task: mean_z}
  alignment_long_best_head.parquet   [model,encoder,decoder,task,subject_id,image_id,r,z]
  alignment_subject_means_best_head.csv
  alignment_summary_best_head.csv    [model,task,mean_r,sem_r,pct_ceiling,n_sub]
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import alignment_core as A
import vlm_maps as V

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA, exist_ok=True)

FACTOR = {  # encoder/decoder labels for the mixed model
    'CNN-LSTM':        ('CNN', 'LSTM'),
    'ViT-LSTM':        ('ViT', 'LSTM'),
    'CNN-Transformer': ('CNN', 'Transformer'),
    'ViT-Transformer': ('ViT', 'Transformer'),
    'Molmo':           ('VLM', 'VLM'),
    'Qwen3.5':         ('VLM', 'VLM'),
}


def main():
    imgs = A.target_images()
    # ceilings (also warms the human disk cache)
    ceil = {t: A.ceiling(t, imgs) for t in ['social', 'describe']}
    json.dump(ceil, open(os.path.join(DATA, 'ceilings.json'), 'w'), indent=2)
    print('Ceilings (mean-z):', {k: round(v, 4) for k, v in ceil.items()}, flush=True)

    frames = []
    for task in ['social', 'describe']:
        for m in ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer']:
            maps = A.trained_maps(m, task, imgs)
            df = A.per_image_z(maps, task, imgs, m)
            frames.append(df)
            print(f'  {task:9s} {m:16s} n_rows={len(df)}', flush=True)
        for m in ['Molmo', 'Qwen3.5']:
            maps = V.vlm_maps(m, task)
            df = A.per_image_z(maps, task, imgs, m)
            frames.append(df)
            print(f'  {task:9s} {m:16s} n_rows={len(df)} (VLM best head)', flush=True)

    long = pd.concat(frames, ignore_index=True)
    long['encoder'] = long['model'].map(lambda m: FACTOR[m][0])
    long['decoder'] = long['model'].map(lambda m: FACTOR[m][1])
    long.to_parquet(os.path.join(DATA, 'alignment_long_best_head.parquet'))

    sm = A.subject_means(long)
    sm.to_csv(os.path.join(DATA, 'alignment_subject_means_best_head.csv'), index=False)

    rows = []
    for (m, t), g in sm.groupby(['model', 'task']):
        z = g['mean_z'].values
        mr = float(np.tanh(z.mean()))
        sem = float(z.std(ddof=1) / np.sqrt(len(z))) if len(z) > 1 else 0.0
        rows.append(dict(model=m, task=t, mean_r=mr, sem_z=sem,
                         pct_ceiling=100 * z.mean() / ceil[t], n_sub=len(z)))
    summ = pd.DataFrame(rows).sort_values(['task', 'mean_r'], ascending=[True, False])
    summ.to_csv(os.path.join(DATA, 'alignment_summary_best_head.csv'), index=False)
    print('\n=== Summary (best head / single map) ===')
    print(summ.to_string(index=False))


if __name__ == '__main__':
    main()
