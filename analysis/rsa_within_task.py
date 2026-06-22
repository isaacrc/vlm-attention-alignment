#!/usr/bin/env python
# coding: utf-8

"""
Human–Human RSA Analysis

For each task:
1. Compute subject-level image–image RSA matrices
2. Pairwise subject RSA-of-RSA correlations
3. ISC: subject RSA correlated with leave-one-out mean RSA

Outputs:
- hum_hum_pairwise_{task}.csv
- hum_hum_isc_{task}.csv
"""

import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy.stats import spearmanr
from itertools import combinations

from utils import load_human_subject_data

# --------------------
# Paths
# --------------------
SOC_ALPHA_HUM_DIR = './processed_heatmaps_by_subject_social'
DESC_ALPHA_HUM_DIR = './processed_heatmaps_by_subject_descriptive'
OUT_DIR = './processed_correlations'
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------
# Main loop
# --------------------
for task in ["social", "describe"]:

    print(f"\n===== TASK: {task} =====")

    TARG_DIR = SOC_ALPHA_HUM_DIR if task == "social" else DESC_ALPHA_HUM_DIR
    data = load_human_subject_data(TARG_DIR)

    subject_ids = sorted(data.keys())

    # Collect all image IDs
    image_ids = sorted({
        img.replace(".jpg", "")
        for s_dat in data.values()
        for img in s_dat.keys()
    })
    n_images = len(image_ids)

    print(f"Subjects: {len(subject_ids)} | Images: {n_images}")

    # --------------------
    # 1. Subject RSAs
    # --------------------
    subject_corr_mats = []

    for sub_id in tqdm(subject_ids, desc="Computing subject RSAs"):

        s_dat = data[sub_id]
        vecs = []

        for img_id in image_ids:
            fname = f"{img_id}.jpg"

            if fname in s_dat:
                vec = np.nan_to_num(s_dat[fname].flatten())
                vec = Image.fromarray(vec).resize((256, 256), Image.NEAREST)
                vec = np.array(vec).flatten()
                vecs.append(vec if np.std(vec) > 0 else np.full_like(vec, np.nan))
            else:
                ref = next(iter(s_dat.values())).flatten()
                ref = Image.fromarray(np.nan_to_num(ref)).resize((256, 256), Image.NEAREST)
                ref = np.array(ref).flatten()
                vecs.append(np.full_like(ref, np.nan))

        vecs = np.stack(vecs)

        rsa = np.full((n_images, n_images), np.nan)
        for i in range(n_images):
            for j in range(i, n_images):
                valid = ~np.isnan(vecs[i]) & ~np.isnan(vecs[j])
                if np.sum(valid) > 1:
                    r = np.corrcoef(vecs[i][valid], vecs[j][valid])[0, 1]
                    rsa[i, j] = rsa[j, i] = r

        subject_corr_mats.append(rsa)

    subject_corr_mats = np.stack(subject_corr_mats)
    subject_corr_mats_z = np.arctanh(
        np.clip(subject_corr_mats, -0.999999, 0.999999)
    )

    n_subjects = subject_corr_mats_z.shape[0]
    triu_idx = np.triu_indices(n_images, k=1)

    # --------------------
    # 2. Pairwise hum–hum
    # --------------------
    pairwise_results = []

    for i, j in combinations(range(n_subjects), 2):

        vec_i = subject_corr_mats_z[i][triu_idx]
        vec_j = subject_corr_mats_z[j][triu_idx]
        valid = ~np.isnan(vec_i) & ~np.isnan(vec_j)

        rho, pval = spearmanr(vec_i[valid], vec_j[valid])

        pairwise_results.append({
            "subject_i": subject_ids[i],
            "subject_j": subject_ids[j],
            "spearman_rho": rho,
            "p_value": pval,
            "n_pairs": np.sum(valid),
            "task": task
        })

    df_pairwise = pd.DataFrame(pairwise_results)
    df_pairwise.to_csv(
        f"{OUT_DIR}/hum_hum_pairwise_{task}.csv", index=False
    )

    # --------------------
    # 3. ISC (leave-one-out)
    # --------------------
    isc_results = []

    for s in range(n_subjects):

        subj_rsa = subject_corr_mats_z[s]

        others = np.delete(subject_corr_mats_z, s, axis=0)
        mean_others = np.nanmean(others, axis=0)

        vec_sub = subj_rsa[triu_idx]
        vec_mean = mean_others[triu_idx]
        valid = ~np.isnan(vec_sub) & ~np.isnan(vec_mean)

        rho, pval = spearmanr(vec_sub[valid], vec_mean[valid])

        isc_results.append({
            "left_out_subject": subject_ids[s],
            "spearman_rho": rho,
            "p_value": pval,
            "n_pairs": np.sum(valid),
            "task": task
        })

    df_isc = pd.DataFrame(isc_results)
    df_isc.to_csv(
        f"{OUT_DIR}/hum_hum_isc_{task}.csv", index=False
    )

    print(f"Saved hum–hum results for {task}")
    gc.collect()
