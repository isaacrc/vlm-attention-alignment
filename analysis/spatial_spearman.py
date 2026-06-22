import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from scipy.stats import spearmanr


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Replace with your actual directory paths
TOP_DIR = os.getcwd()
IMG_DIR = 'imgs'
CAP_DIR = 'captions_model'
ALPHA_DIR = '../eye_analysis/analysis'
SOC_ALPHA_HUM_DIR = './processed_heatmaps_by_subject_social'
DESC_ALPHA_HUM_DIR = './processed_heatmaps_by_subject_descriptive'
OUT_DIR = 'processed_correlations'

COND_A_DIR = DESC_ALPHA_HUM_DIR # SOC_ALPHA_HUM_DIR #

def load_single_subject_heatmaps(directory, subject_id):
    """Loads heatmap data for a single subject from a given directory."""
    filename = f"{subject_id}_heatmaps.npy"
    filepath = os.path.join(directory, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found for subject '{subject_id}' at {filepath}")

    try:
        subject_heatmaps = np.load(filepath, allow_pickle=True).item()
        return subject_heatmaps
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}: {e}")

def load_human_subject_data(directory):
    """Load all subject IDs from heatmap files in the given directory."""
    data = {}
    for file in os.listdir(directory):
        if file.endswith('_heatmaps.npy'):
            subject_id = file.split('_')[0]
            try:
                data[subject_id] = load_single_subject_heatmaps(directory, subject_id)
            except Exception as e:
                logger.warning(f"Skipping {file}: {e}")
    return data

def main():
    logger.info("Loading all image IDs...")
    hum_agent_heatmaps = load_single_subject_heatmaps(SOC_ALPHA_HUM_DIR, 'sub005')
    image_ids = sorted(hum_agent_heatmaps.keys())

    logger.info("\nLoading subject lists...")
    data_a = load_human_subject_data(COND_A_DIR)
    sub_list_a = list(data_a.keys())


    analysis_results = []

    logger.info("\nCalculating ISC (subject ↔ group-mean) per image...")
    for image_id in tqdm(image_ids, desc="Processing images"):

        # Preload all subject heatmaps for this image
        subj_maps = {}
        for sub_id in sub_list_a:
            if image_id in data_a[sub_id]:
                subj_maps[sub_id] = np.nan_to_num(
                    data_a[sub_id][image_id].flatten()
                )

        if len(subj_maps) < 2:
            continue  # need at least 2 subjects

        for sub_id, sub_vec in subj_maps.items():

            if np.std(sub_vec) == 0:
                continue

            # Leave-one-out mean of other subjects
            other_vecs = [
                v for sid, v in subj_maps.items() if sid != sub_id
            ]

            mean_other = np.nanmean(other_vecs, axis=0)

            if mean_other.shape != sub_vec.shape:
                continue
            if np.std(mean_other) == 0:
                continue

            # Spearman correlation
            r_val, _ = spearmanr(sub_vec, mean_other)

            if np.isnan(r_val):
                continue

            task = "social" if COND_A_DIR == SOC_ALPHA_HUM_DIR else "describe"

            analysis_results.append({
                "image_id": image_id,
                "left_out_subject": sub_id,
                "isc_r": r_val,
                "isc_z": np.arctanh(np.clip(r_val, -0.999999, 0.999999)),
                "task": task
            })
    df_analysis = pd.DataFrame(analysis_results)
    df_analysis.to_csv(os.path.join(OUT_DIR, f"within_task_hum_hum_corr_analysis_spearman_{task}.csv"), index=False)




if __name__ == "__main__":
    main()
