#!/usr/bin/env python
# coding: utf-8

# # Welcome! This code always wanted to do great things, and it does just that ! 
'''
- Iterate for each task
- load models and pearson r RSA  (average RSAs across layers if necessary)
- load subject RSAs and pearson r RSA for each sub
-  correlate pearson Rs lower triangle from models to humans
- save for each subject
- also correlate all subs pearson RsA to one another, and output
- lastly get the SEM for models and humans for plotting. This requires boostrapping over models lol

you need not change anything, just run
OUTPUT:
second_order_rsa_between_tasks.csv

'''

# In[28]:


import os

from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import gc
import cv2
from scipy.stats import spearmanr
import random




# In[4]:


TOP_DIR = os.getcwd()
IMG_DIR = 'imgs'
CAP_DIR = 'human_captions'
MOD_CAP_DIR = './captions_model'
SOC_ALPHA_HUM_DIR = './processed_heatmaps_by_subject_social'
DESC_ALPHA_HUM_DIR = './processed_heatmaps_by_subject_descriptive'
SOC_ALPHA_MOD_DIR = './processed_heatmaps_rnn'
DESC_ALPHA_MOD_DIR = './processed_heatmaps_rnn'
ALPHA_MOD_DIR_molmo = './processed_heatmaps_molmo'
SOC_CAP_HUM_DIR = os.path.join(CAP_DIR, 'CLEANED_trans_social')
DESC_CAP_HUM_DIR = os.path.join(CAP_DIR, 'CLEANED_trans_desc')
SOC_ALPHA_MOD_DIR_molmo = os.path.join(TOP_DIR, 'processed_heatmaps_molmo')
DESC_ALPHA_MOD_DIR_molmo = os.path.join(TOP_DIR, 'processed_heatmaps_molmo')
SOC_ALPHA_MOD_DIR_qwen = os.path.join(TOP_DIR, 'processed_heatmaps_qwen')
DESC_ALPHA_MOD_DIR_qwen = os.path.join(TOP_DIR, 'processed_heatmaps_qwen')
PLOT_DIR = os.path.join(SOC_ALPHA_MOD_DIR_molmo, 'ppt_plots')



# --- Directory Paths ---
# IMPORTANT: Adjust these paths to match your actual file structure.
COND_A_DIR = 'processed_heatmaps_by_subject_social'
COND_B_DIR = 'processed_heatmaps_by_subject_descriptive'
OUT_DIR = 'processed_correlations'
AGENT_H_MAP_DIR = 'processed_heatmaps_rnn'
MOLMO_H_MAP_DIR = 'processed_heatmaps_molmo'
QWEN_H_MAP_DIR = 'processed_heatmaps_qwen'


# --- Functions ---
from utils import load_human_subject_data, normalize_attention

big_dic = {"social": {}, "describe": {}}

for task in ['social', 'describe']:
    #####################################
    # ## RNN
    #####################################
    ## set the path to models alphas ##
    AGENT_B_HEATMAPS_FILE = f'{AGENT_H_MAP_DIR}/h_c_alpha_n200_{task}_final.npz'
    print(f"Loading Agent B (gated) heatmaps from {AGENT_B_HEATMAPS_FILE}")

    # --- Load Agent A (VLM) Heatmaps ---
    try:
        agent_b_data = np.load(f'{AGENT_B_HEATMAPS_FILE}', allow_pickle=True)
        gated_h_state = agent_b_data['h'].item()
        gated_c_state = agent_b_data['c'].item()
        gated_emb = agent_b_data['emb'].item()
        gated_heatmaps_l = agent_b_data['alphas'].item()
        gated_heatmaps = {}
        for k in sorted(gated_heatmaps_l.keys()):
            ## RESIZE
            ## Blur via Bilinear
            ## Normalize between 0 and 1
            gated_heatmaps[k[:-4]] = normalize_attention(np.array(Image.fromarray(gated_heatmaps_l[k]).resize((256, 256), resample=Image.BILINEAR)))


        #print(f"Loaded Agent B (gated) heatmaps from {AGENT_B_HEATMAPS_FILE}")
        #print(f"Number of images in Agent A data: {len(agent_b_heatmaps)}")
    except FileNotFoundError:
        print(f"Error: Agent B heatmaps file '{AGENT_B_HEATMAPS_FILE}' not found.")
        # Consider handling this more gracefully, e.g., setting agent_a_heatmaps to empty dict or exiting.
        # For now, we'll exit as it's critical data.
    except Exception as e:
        print(f"Error loading Agent B heatmaps: {e}")

    # Collect flattened attention maps for all images
    vecs = [gated_heatmaps[img].flatten() for img in gated_heatmaps]
    vecs = np.stack(vecs)  # shape: [num_images, num_pixels]

    # --- 1. Compute correlation matrix across images ---
    corr_matrix_gated = np.corrcoef(vecs)  # shape: [num_images, num_images]
    big_dic[task]["Gated"] = corr_matrix_gated

    #####################################
    # MOLMO
    #####################################
    AGENT_A_HEATMAPS_FILE = f'{MOLMO_H_MAP_DIR}/attention_maps_{task}.npz' # Path to your LLM data
    # --- Load Agent A (VLM) Heatmaps ---
    try:
        molmo_heatmaps_l = np.load(AGENT_A_HEATMAPS_FILE, allow_pickle=True)
        print(f"Loaded Agent A (LLM) heatmaps from {AGENT_A_HEATMAPS_FILE}")
        print(f"Number of images in molmo data: {len(molmo_heatmaps_l)}")
        molmo_heatmaps = {}
        # must normalize and smooth
        for image_id in tqdm(sorted(molmo_heatmaps_l.keys()), desc="Processing images"):
            layer_dic = {}
            layers = list(molmo_heatmaps_l[image_id].item().keys())[:-1]  # skip last if needed
            for idx, layer in enumerate(layers):
                # optionally smooth and normalize
                img = molmo_heatmaps_l[image_id].item()[layer]
                ## RESIZE
                attn_img = Image.fromarray(img).resize((256, 256), resample=Image.NEAREST)
                ## BLUR
                sigma = 10
                attn_img = cv2.GaussianBlur(np.array(attn_img).astype(np.float32), ksize=(0,0), sigmaX=sigma, sigmaY=sigma)
                # normalize 
                attn_img = normalize_attention(np.array(attn_img))
                layer_dic[f'layer_{idx}'] = attn_img
            molmo_heatmaps[image_id] = layer_dic
        del molmo_heatmaps_l
        gc.collect()

    except FileNotFoundError:
        print(f"Error: Agent A heatmaps file '{AGENT_A_HEATMAPS_FILE}' not found.")
        # Consider handling this more gracefully, e.g., setting agent_a_heatmaps to empty dict or exiting.
        # For now, we'll exit as it's critical data.
    except Exception as e:
        print(f"Error loading Agent A heatmaps: {e}")
    
    #### GET CORRELATION MATRICES ACROSS LAYERS ### 
    # List of images
    image_ids = sorted(molmo_heatmaps.keys())
    n_images = len(image_ids)
    # List of images
    image_ids = sorted(molmo_heatmaps.keys())
    n_images = len(image_ids)
    layers = list(next(iter(molmo_heatmaps.values())).keys())

    # --- Step 1: compute per-layer correlation matrices ---
    corr_mats_z = []

    for layer in layers:
        # Stack image vectors for this layer
        layer_vecs = np.stack([
            molmo_heatmaps[img_id][layer].flatten()
            for img_id in image_ids
        ])  # shape: [images, pixels]

        # Pearson correlation across images
        corr_r = np.corrcoef(layer_vecs)

        # Fisher z-transform (clip for numerical stability)
        corr_z = np.arctanh(np.clip(corr_r, -0.999999, 0.999999))
        corr_mats_z.append(corr_z)

    # --- Step 2: average correlation matrices in z-space ---
    mean_corr_z = np.mean(corr_mats_z, axis=0)

    # --- Step 3: convert back to r for interpretability ---
    corr_matrix_molmo = np.tanh(mean_corr_z)  # final RSA matrix
    big_dic[task]["Molmo"] = corr_matrix_molmo


    #####################################
    # QWEN 
    #####################################
    AGENT_C_HEATMAPS_FILE = f'{QWEN_H_MAP_DIR}/attention_maps_{task}.npz' # Path to your LLM data
    # --- Load Agent A (VLM) Heatmaps ---
    try:
        qwen_heatmaps_l = np.load(AGENT_C_HEATMAPS_FILE, allow_pickle=True)
        print(f"Loaded Qwen heatmaps from {AGENT_C_HEATMAPS_FILE}")
        print(f"Number of images in Qwen data: {len(qwen_heatmaps_l)}")
        qwen_heatmaps = {}
        # must normalize and smooth
        for image_id in tqdm(sorted(qwen_heatmaps_l.keys()), desc="Processing images"):
            layer_dic = {}
            for layer in qwen_heatmaps_l[image_id].item().keys():
                ## smooth
                #sigma = 7.2
                #img = gaussian_filter(qwen_heatmaps[image_id].item()[layer], sigma=sigma)
                img = qwen_heatmaps_l[image_id].item()[layer]
                attn_img = Image.fromarray(img).resize((256, 256), resample=Image.BILINEAR)
                # normalize
                attn_img = normalize_attention(np.array(attn_img))
                layer_dic[layer] = attn_img
            ## set cleaned heatmaps dict
            qwen_heatmaps[image_id] = layer_dic
        del qwen_heatmaps_l
        gc.collect()

    except FileNotFoundError:
        print(f"Error: Qwen heatmaps file '{AGENT_C_HEATMAPS_FILE}' not found.")
        # Consider handling this more gracefully, e.g., setting agent_a_heatmaps to empty dict or exiting.
        # For now, we'll exit as it's critical data.
    except Exception as e:
        print(f"Error loading Qwen heatmaps: {e}")

    #### AVERAGE CORRELATION MATRICES ACROSS LAYERS ### 
    # List of images
    image_ids = sorted(qwen_heatmaps.keys())
    n_images = len(image_ids)
    layers = list(next(iter(qwen_heatmaps.values())).keys())

    # --- Step 1: compute per-layer correlation matrices ---
    corr_mats_z = []

    for layer in layers:
        # Stack image vectors for this layer
        layer_vecs = np.stack([
            qwen_heatmaps[img_id][layer].flatten()
            for img_id in image_ids
        ])  # shape: [images, pixels]

        # Pearson correlation across images
        corr_r = np.corrcoef(layer_vecs)

        # Fisher z-transform (clip for numerical stability)
        corr_z = np.arctanh(np.clip(corr_r, -0.999999, 0.999999))
        corr_mats_z.append(corr_z)

    # --- Step 2: average correlation matrices in z-space ---
    mean_corr_z = np.mean(corr_mats_z, axis=0)

    # --- Step 3: convert back to r for interpretability ---
    corr_matrix_qwen = np.tanh(mean_corr_z)  # final RSA matrix
    big_dic[task]["Qwen"] = corr_matrix_qwen


    #####################################
    # --- Load Human Subject Heatmaps ---
    #####################################
    if task == "social":
        TARG_DIR = SOC_ALPHA_HUM_DIR 
    else:
        TARG_DIR = DESC_ALPHA_HUM_DIR 

    data = load_human_subject_data(TARG_DIR)

    # --- Define full set of 200 images ---
    image_ids = sorted({img_id.replace(".jpg", "") for s_dat in data.values() for img_id in s_dat.keys()})
    n_images = len(image_ids)

    subject_corr_mats = []

    for sub_id, s_dat in tqdm(data.items(), desc="Processing subjects"):

        # Stack image vectors for this subject, fill missing images with NaNs
        vecs = []
        for img_id in image_ids:
            fname = f"{img_id}.jpg"
            if fname in s_dat:
                vec = np.nan_to_num(s_dat[fname].flatten())
                vec = Image.fromarray(np.nan_to_num(vec, nan=0.0)).resize((256, 256), resample=Image.NEAREST)
                vec = np.array(vec).flatten()
                if np.std(vec) > 0:
                    vecs.append(vec)
                else:
                    vecs.append(np.full_like(vec, np.nan))  # keep shape
            else:
                # Image missing → fill with NaNs of the correct vector length
                # Assuming at least one image exists for shape reference
                any_vec = next(iter(s_dat.values())).flatten()
                vec = Image.fromarray(np.nan_to_num(any_vec, nan=0.0)).resize((256, 256), resample=Image.NEAREST)
                vec = np.array(vec).flatten()
                vecs.append(np.full_like(vec, np.nan))
            
        vecs = np.stack(vecs)  # shape: [n_images, n_pixels]

        # Subject-level image–image correlation matrix
        # np.corrcoef cannot handle NaNs directly, so we compute pairwise correlations manually
        sub_corr = np.full((n_images, n_images), np.nan)
        for i in range(n_images):
            for j in range(i, n_images):
                vec_i = vecs[i]
                vec_j = vecs[j]

                # Only compute if both vectors have at least one non-NaN
                valid_mask = ~np.isnan(vec_i) & ~np.isnan(vec_j)
                if np.sum(valid_mask) > 1:
                    r = np.corrcoef(vec_i[valid_mask], vec_j[valid_mask])[0, 1]
                    sub_corr[i, j] = r
                    sub_corr[j, i] = r

        subject_corr_mats.append(sub_corr)

    # Stack all subject correlation matrices
    subject_corr_mats = np.stack(subject_corr_mats)  # [n_subjects, n_images, n_images]

    # --- Fisher z transform & average ignoring NaNs ---
    subject_corr_mats_z = np.arctanh(np.clip(subject_corr_mats, -0.999999, 0.999999))
    mean_corr_matrix_human = np.tanh(np.nanmean(subject_corr_mats_z, axis=0))  # shape: [n_images, n_images]
    
    big_dic[task]["human"] = np.tanh(subject_corr_mats_z)

    
#####################################
# # SECOND ORDER CORS ### 
# ## SUBJECT LEVEL: model correlated to each human
#####################################
n_images = next(iter(big_dic['social'].values())).shape[0]
tril_idx = np.tril_indices(n_images, k=-1)

rows = []

for agent in ["Molmo", "Qwen", "Gated"]:
    social_rdm = big_dic["social"][agent]
    describe_rdm = big_dic["describe"][agent]

    vec_soc = social_rdm[tril_idx]
    vec_des = describe_rdm[tril_idx]

    valid = ~np.isnan(vec_soc) & ~np.isnan(vec_des)
    rho, pval = spearmanr(vec_soc[valid], vec_des[valid])

    rows.append({
        "agent": agent,
        "level": "model",
        "subject": None,
        "spearman_rho": rho,
        "p_value": pval,
        "n_pairs": np.sum(valid)
    })


## subjects 

n_subj_soc = big_dic["social"]["human"].shape[0]
n_subj_des = big_dic["describe"]["human"].shape[0]

# Precompute task-wise means
mean_social = np.nanmean(big_dic["social"]["human"], axis=0)
mean_describe = np.nanmean(big_dic["describe"]["human"], axis=0)

for sub_idx in range(n_subj_soc):

    # Social subject vs Describe group mean
    subj_soc = big_dic["social"]["human"][sub_idx]
    vec_sub = subj_soc[tril_idx]
    vec_ref = mean_describe[tril_idx]

    valid = ~np.isnan(vec_sub) & ~np.isnan(vec_ref)
    rho, pval = spearmanr(vec_sub[valid], vec_ref[valid])

    rows.append({
        "agent": "Human",
        "level": "subject",
        "subject": sub_idx,
        "spearman_rho": rho,
        "p_value": pval,
        "n_pairs": np.sum(valid)
    })

for sub_idx in range(n_subj_des):

    # Describe subject vs Social group mean
    subj_des = big_dic["describe"]["human"][sub_idx]
    vec_sub = subj_des[tril_idx]
    vec_ref = mean_social[tril_idx]

    valid = ~np.isnan(vec_sub) & ~np.isnan(vec_ref)
    rho, pval = spearmanr(vec_sub[valid], vec_ref[valid])

    rows.append({
        "agent": "Human",
        "level": "subject",
        "subject": sub_idx,
        "spearman_rho": rho,
        "p_value": pval,
        "n_pairs": np.sum(valid)
    })

df = pd.DataFrame(rows)
#df.to_csv(f"{OUT_DIR}/rsa_between_tasks.csv", index=False)
print(f"Saved RSA between tasks results to {OUT_DIR}/rsa_between_tasks.csv")



# -----------------------------
# Compute SEMs and summary table
# -----------------------------
summary_rows = []

# 1) Subjects: SEM across held-out subjects
for agent in ["Human"]:
    for task_pair in [("social", "describe")]:
        # Extract only subject-level rows
        subj_rows = df[(df["agent"] == agent) & (df["level"] == "subject")]
        spearman_values = subj_rows["spearman_rho"].values
        mean_r = np.mean(spearman_values)
        sem_r = np.std(spearman_values, ddof=1) / np.sqrt(len(spearman_values))

        summary_rows.append({
            "agent": agent,
            "mean_r": mean_r,
            "sem_r": sem_r
        })

# 2) Models: bootstrap across image pairs to get SEM
n_boot = 1000  # number of bootstrap samples
for agent in ["Molmo", "Qwen", "Gated"]:
    # Extract upper-triangle values from both tasks
    vec_soc = big_dic["social"][agent][tril_idx]
    vec_des = big_dic["describe"][agent][tril_idx]
    valid = ~np.isnan(vec_soc) & ~np.isnan(vec_des)
    vec_soc = vec_soc[valid]
    vec_des = vec_des[valid]

    # Compute bootstrap samples
    boot_rhos = []
    n_pairs = len(vec_soc)
    for _ in range(n_boot):
        idxs = np.random.choice(n_pairs, n_pairs, replace=True)
        boot_rho, _ = spearmanr(vec_soc[idxs], vec_des[idxs])
        boot_rhos.append(boot_rho)
    boot_rhos = np.array(boot_rhos)

    summary_rows.append({
        "agent": agent,
        "mean_r": np.mean(boot_rhos),
        "sem_r": np.std(boot_rhos, ddof=1)
    })

# Save summary table
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f"{OUT_DIR}/rsa_second_order_summary_between_tasks.csv", index=False)
print(f"Saved RSA second-order summary (mean + SEM) to {OUT_DIR}/rsa_second_order_summary_between_tasks.csv")


