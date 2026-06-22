import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from glob import glob
import sys
from itertools import combinations

# -----------------------------
# Directories
# -----------------------------
OUT_DIR = './processed_correlations'
os.makedirs(OUT_DIR, exist_ok=True)

task = 'social'  # ****** CHANGE ME ******

OUT_NAME = f'HUM-HUM_cosine_{task[:3]}_{task[:3]}_200_images_captions'
CAP_DIR = './captions_humans/CLEANED_trans_desc' if task == 'describe' else './captions_humans/CLEANED_trans_social'

# -----------------------------
# Load sentence embedding model
# -----------------------------
embedder = SentenceTransformer("all-mpnet-base-v2")

# -----------------------------
# Utility functions
# -----------------------------
def read_csv_remove_commas(csv_path: str) -> pd.DataFrame:
    """Reads a CSV, removes commas inside captions, keeps the header, returns DataFrame."""
    cleaned_lines = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = next(f).strip().split(',')
        for line in f:
            line = line.strip()
            if not line:
                continue
            index, caption = line.split(',', 1)
            caption = caption.replace(',', '')
            cleaned_lines.append((index, caption))
    df = pd.DataFrame(cleaned_lines, columns=header)
    return df

# -----------------------------
# Collect human subjects
# -----------------------------
human_caption_files = sorted(glob(os.path.join(CAP_DIR, "*_trans.csv")))
subjects = [os.path.basename(f).split("_")[0] for f in human_caption_files]
print("Subjects:", subjects)

# -----------------------------
# Pre-encode captions for all subjects
# -----------------------------
subject_caps = {}  # dict: subject_id -> {image_name: caption}
subject_embs = {}  # dict: subject_id -> {image_name: embedding}

for sub_id in subjects:
    csv_path = os.path.join(CAP_DIR, f"{sub_id}_trans.csv")
    
    if task == 'describe':
        df = read_csv_remove_commas(csv_path)
    else:
        try:
            df = pd.read_csv(csv_path, dtype={"index": str})
        except Exception:
            df = pd.read_csv(csv_path, sep="\t", dtype={"index": str})
        # Special fixes for certain subjects
        if sub_id in ["sub-014", "sub-022", "sub-023"]:
            df = pd.read_csv(csv_path, sep="\t", dtype={"index": str})
    
    if "index" not in df.columns:
        df.rename(columns={df.columns[0]: "index"}, inplace=True)
    
    df = df[df["caption"] != "NOT FOUND"].copy()
    df["image_name"] = df["index"]
    
    captions = dict(zip(df["image_name"], df["caption"]))
    subject_caps[sub_id] = captions
    
    # Pre-encode all captions for this subject
    images = list(captions.keys())
    embeddings = embedder.encode(list(captions.values()), convert_to_numpy=True)
    subject_embs[sub_id] = dict(zip(images, embeddings))

# -----------------------------
# Compute human-human cosine similarity per image
# -----------------------------
results = []

for sub_i, sub_j in tqdm(combinations(subjects, 2), desc="Computing similarities"):

    caps_i = subject_caps[sub_i]
    caps_j = subject_caps[sub_j]
    emb_i = subject_embs[sub_i]
    emb_j = subject_embs[sub_j]

    # Find common images
    common_images = sorted(set(caps_i.keys()) & set(caps_j.keys()))
    if not common_images:
        print(f"No common images between {sub_i} and {sub_j}")
        continue

    for img in common_images:
        # Vectorized cosine similarity
        vec_i = emb_i[img]
        vec_j = emb_j[img]
        sim = np.dot(vec_i, vec_j) / (norm(vec_i) * norm(vec_j))
        
        results.append({
            "subject_i": sub_i,
            "subject_j": sub_j,
            "image_id": img,
            "similarity": sim
        })

# -----------------------------
# Save results
# -----------------------------
df_results = pd.DataFrame(results)
output_csv = os.path.join(OUT_DIR, f"{OUT_NAME}_by_image.csv")
df_results.to_csv(output_csv, index=False)
print(f"Done! Results saved to {output_csv}")
