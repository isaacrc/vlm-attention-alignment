#!/usr/bin/env python
# coding: utf-8

# # Welcome! This code reprocuces Table 1 muahaha

# - more specifically (and seriously) (kind of) We calculat correlations between 3 model attentions and human eyetracking heatmaps for two tasks
# - but wait there's more ! we also correlate captions between the two tasks. Fun!
# - shout out within_task_hum_hum_corr_analysis_spearman.py which calculates the eye track correlation ceiling between humans


import os
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.misc import imread, imresize
from imageio import imread
from skimage.transform import resize
import torch
import json
from PIL import Image
import skimage.transform
import numpy as np
from scipy.spatial.distance import squareform
import random
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
from tqdm import tqdm
from itertools import combinations
import cv2
from scipy.stats import spearmanr
import numpy as np




# In[2]:


TOP_DIR = os.getcwd()
IMG_DIR = 'imgs'
CAP_DIR = 'captions_model'
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

# --- rsync ---
'''
rsync -rvp isaacrc@della.princeton.edu:/scratch/gpfs/GRAZIANO/CANS/molmo/molmo/activations/social/gener* .
rsync -rvp isaacrc@della.princeton.edu:/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output/social_cap/resp* .

'''

from utils import load_human_subject_data, normalize_attention, average_layers_all_images


### CHANGE ME ### 
SAVE_NAME = "all_models_all_tasks_correlations_final"


# # Load Model Data
all_results = []   # will store all model–task correlations


for task in ['social', 'describe']:
    
    # MOLMO 
    AGENT_A_HEATMAPS_FILE = f'{MOLMO_H_MAP_DIR}/attention_maps_{task}.npz' # Path to your LLM data
    # --- Load Agent A (VLM) Heatmaps ---
    try:
        molmo_heatmaps_l = np.load(AGENT_A_HEATMAPS_FILE, allow_pickle=True)
        print(f"Loaded molmo heatmaps from {AGENT_A_HEATMAPS_FILE}")
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
        print(f"Error: molmo heatmaps file '{AGENT_A_HEATMAPS_FILE}' not found.")
        # Consider handling this more gracefully, e.g., setting agent_a_heatmaps to empty dict or exiting.
        # For now, we'll exit as it's critical data.
    except Exception as e:
        print(f"Error loading molmo heatmaps: {e}")




    # ## RNN
    ## set the path to models alphas ##
    AGENT_B_HEATMAPS_FILE = f'{AGENT_H_MAP_DIR}/h_c_alpha_n200_{task}_final.npz' # Path to your VLM data
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


        print(f"Loaded Gated heatmaps from {AGENT_B_HEATMAPS_FILE}")
        #print(f"Number of images in Agent A data: {len(agent_b_heatmaps)}")
    except FileNotFoundError:
        print(f"Error: Agent B heatmaps file '{AGENT_B_HEATMAPS_FILE}' not found.")
        # Consider handling this more gracefully, e.g., setting agent_a_heatmaps to empty dict or exiting.
        # For now, we'll exit as it's critical data.
    except Exception as e:
        print(f"Error loading Agent B heatmaps: {e}")




    # QWEN
    AGENT_C_HEATMAPS_FILE = f'{QWEN_H_MAP_DIR}/attention_maps_{task}.npz' # Path to your LLM data
    # --- Load Agent C (VLM) Heatmaps ---
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






    # # Calculate correlations: Agent --> Hum Soc, Hum Desc 

    # - this code takes a while but you can bet its pretty fun! 
    # - the goal is to find the mean correlation, across subs, for each image
    # - the correlation is between model and sub 1, 2, 3, etc.
    # - we perform this operation for both social and non-social datasets, for each image_id

    # In[14]:


    model_attn_maps = {
        "Gated": gated_heatmaps,
        "Molmo": average_layers_all_images(molmo_heatmaps),
        "Qwen": average_layers_all_images(qwen_heatmaps)
    }


    for model in model_attn_maps.keys():
        out_name = f'{model}_{task}_correlated_to_hum_{task}' # molmo-soc-hum-soc_cor_ztrans_mean_results
        agent_a_heatmaps = model_attn_maps[model]
        print(f'out_name: {out_name}')

        if task == 'social':
            d_set = COND_A_DIR
        else:
            d_set = COND_B_DIR

        

        analysis_results = []
        subject_results = []   

        print("\nCalculating correlations per image across subjects and conditions...")

        # --- Loop through each image from Agent A's heatmaps ---
        for image_id in tqdm(sorted(agent_a_heatmaps.keys()), desc="Processing images"):
            agent_a_array_flattened = np.nan_to_num(agent_a_heatmaps[image_id].flatten())
            image_id_jpg = f"{image_id}.jpg"
            #print(image_id_jpg)

            image_result = {'image_id': image_id_jpg}

            # --- Loop over target set
            data = load_human_subject_data(d_set)

            # track raw r per subject 
            for sub_id in data.keys():

                s_dat = data[sub_id]

                if image_id_jpg not in s_dat:
                    continue
                
                # resize human attention heatmap
                hum_alpha_resize = Image.fromarray(np.nan_to_num(s_dat[image_id_jpg], nan=0.0)).resize((256, 256), resample=Image.NEAREST)
                human_array_flat = np.nan_to_num(np.array(hum_alpha_resize).flatten())

                # skip mismatched spatial resolutions
                if agent_a_array_flattened.size != human_array_flat.size:
                    continue

                # valid data?
                if np.std(agent_a_array_flattened) > 0 and np.std(human_array_flat) > 0:

                    # --- SPEARMAN correlation ---
                    r_val, _ = spearmanr(agent_a_array_flattened, human_array_flat)
                else:
                    r_val = np.nan

                # store per-image r's for Fisher-z mean
                if not np.isnan(r_val):
                    r_val = np.clip(r_val, -0.999999, 0.999999)
                    z_val = np.arctanh(r_val)
                    #current_image_corrs_z.append(np.arctanh(r_val))

                # store per-subject individual correlations ---
                subject_results.append({
                    "subject_id": sub_id,
                    "image_id": image_id_jpg,
                    "spearman_r": r_val,
                    "spearman_z": z_val
                })
        df_subject = pd.DataFrame(subject_results)
        df_subject["model"] = model
        df_subject["task"] = task

        all_results.append(df_subject)
        df_subject.to_csv(f'{OUT_DIR}/{out_name}_subject.csv')
df_big = pd.concat(all_results, ignore_index=True)
df_big.to_csv(f"{OUT_DIR}/{SAVE_NAME}.csv", index=False)

