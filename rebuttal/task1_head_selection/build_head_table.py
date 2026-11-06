"""
Task 1 (step 1) — build per-(model,layer,head,image) alignment table.
Transformers (CNN/ViT) via head_layer_alphas; Qwen via .pt reconstruction; Molmo
merged from the sweep parquet. Subject-averaged Fisher-z per image (for CV selection
+ winner's-curse) plus per-subject table for transformers/Qwen (participant split).

Outputs (task1_head_selection/results/):
  head_table_perimage.parquet    [model,task,layer,head,image_id,mean_z]
  head_table_persubject.parquet  [model,task,layer,head,subject_id,mean_z]  (trained+Qwen)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
import alignment_core as A
import vlm_maps as V

RES = os.path.join(os.path.dirname(__file__), 'results'); os.makedirs(RES, exist_ok=True)
QWEN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
N_QHEAD = 16


def subjmean_and_persub(df):
    """df from per_image_z -> (per-image subject-mean z, per-subject mean z)."""
    pim = df.groupby('image_id')['z'].mean().reset_index().rename(columns={'z': 'mean_z'})
    psb = df.groupby('subject_id')['z'].mean().reset_index().rename(columns={'z': 'mean_z'})
    return pim, psb


def main():
    imgs = A.target_images()
    perimg, persub = [], []

    for task in ['social', 'describe']:
        # ── Transformers ──
        for m in ['CNN-Transformer', 'ViT-Transformer']:
            for L, H, maps in A.transformer_head_grid(m, task, imgs):
                df = A.per_image_z(maps, task, imgs, m)
                pim, psb = subjmean_and_persub(df)
                pim['model'], pim['task'], pim['layer'], pim['head'] = m, task, L, H
                psb['model'], psb['task'], psb['layer'], psb['head'] = m, task, L, H
                perimg.append(pim); persub.append(psb)
            print(f'  {task} {m}: done (4x8 heads)', flush=True)
        # ── Qwen ──
        for L in QWEN_LAYERS:
            for H in range(1, N_QHEAD + 1):
                maps = V.qwen_maps(task, L, H)
                df = A.per_image_z(maps, task, imgs, 'Qwen3.5')
                pim, psb = subjmean_and_persub(df)
                pim['model'], pim['task'], pim['layer'], pim['head'] = 'Qwen3.5', task, L, H
                psb['model'], psb['task'], psb['layer'], psb['head'] = 'Qwen3.5', task, L, H
                perimg.append(pim); persub.append(psb)
            print(f'  {task} Qwen3.5 layer {L}: done', flush=True)

    # ── Molmo from sweep parquet (per-image subject-mean already) ──
    for task in ['social', 'describe']:
        f = os.path.join(RES, f'molmo_allL_perimage_{task}.parquet')
        if os.path.exists(f):
            md = pd.read_parquet(f)[['task', 'layer', 'head', 'image_id', 'mean_z']].copy()
            md['model'] = 'Molmo'
            perimg.append(md)
            print(f'  merged Molmo {task} sweep', flush=True)
        else:
            print(f'  WARNING: Molmo sweep parquet missing for {task}', flush=True)

    pd.concat(perimg, ignore_index=True).to_parquet(os.path.join(RES, 'head_table_perimage.parquet'))
    pd.concat(persub, ignore_index=True).to_parquet(os.path.join(RES, 'head_table_persubject.parquet'))
    print('Saved head tables.', flush=True)


if __name__ == '__main__':
    main()
