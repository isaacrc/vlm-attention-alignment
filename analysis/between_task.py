"""
compute_between_task.py
-----------------------
Compute between-task attention alignment for the 3 new architecture conditions
(CNN_Transformer, ViT_LSTM, ViT_Transformer).  CNN_LSTM data already exists in
Project 1's processed_correlations/ and is read from there.

Between-task measures how similar each model's own attention is between its
Social task checkpoint and its Describe task checkpoint.  Higher = less flexible.

For each new condition:
  - Describe attention: last phase2 NPZ
  - Social attention:   last social_unfrozen NPZ
  - Spatial between-task: per-image Spearman r → Mean, Median, SEM across 200 images
  - RSA between-task:     Spearman r between lower triangles of 200×200 RDMs
                          SEM via image jackknife (N=200 leave-one-out replicates)

Outputs:
  architecture_manipulation/manuscript_plots/between_task_spatial.csv
    columns: Model, Mean, Median, SEM
    (matches format of processed_correlations/SUMMARY_between_tasks_spearman_correlations.csv)

  architecture_manipulation/manuscript_plots/between_task_rsa.csv
    columns: agent, mean_r, sem_r
    (matches format of processed_correlations/rsa_second_order_summary_between_tasks.csv)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, pearsonr

ARCH_DIR  = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
GROUP_DIR = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
PROC_DIR  = os.path.join(GROUP_DIR, 'processed_correlations')
RNN_DIR   = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
OUT_DIR   = os.path.join(ARCH_DIR, 'manuscript_plots')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_CSV = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')

# Last-epoch NPZ for each condition
LAST_EPOCHS = {
    'CNN_Transformer': {'phase2': 'phase2_beam5_E014.npz',
                        'social_unfrozen': 'social_unfrozen_beam5_E003.npz'},
    'ViT_LSTM':        {'phase2': 'phase2_beam5_E003.npz',
                        'social_unfrozen': 'social_unfrozen_beam5_E001.npz'},
    'ViT_Transformer': {'phase2': 'phase2_beam5_E011.npz',
                        'social_unfrozen': 'social_unfrozen_beam5_E001.npz'},
}

# Label mapping for output CSVs (to match Project 1 naming convention)
LABEL_MAP = {
    'CNN_Transformer': 'CNN_Transformer',
    'ViT_LSTM':        'ViT_LSTM',
    'ViT_Transformer': 'ViT_Transformer',
}


def load_alphas(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data['alphas'].item()


def normalize_map(a):
    mn, mx = float(a.min()), float(a.max())
    return (a - mn) / (mx - mn + 1e-10)


def alpha_to_heatmap(alpha_vec, size=256):
    """Reshape (196,) → 14×14, upsample to size×size, normalize."""
    arr = alpha_vec.reshape(14, 14).astype(np.float32)
    resized = np.array(Image.fromarray(arr).resize((size, size),
                                                    resample=Image.BILINEAR))
    return normalize_map(resized).flatten()


def load_target_names(csv_path):
    df = pd.read_csv(csv_path)
    return list(df['image_name'])


def compute_spatial_between_task(soc_alphas, des_alphas, target_imgs):
    """
    Per-image Spearman r between social and describe attention maps.
    Returns array of r values for all common target images.
    """
    rs = []
    for img_name in target_imgs:
        # Handle both 'img_id.jpg' and 'img_id' keys
        soc_key = img_name if img_name in soc_alphas else img_name.replace('.jpg', '')
        des_key = img_name if img_name in des_alphas else img_name.replace('.jpg', '')
        if soc_key not in soc_alphas or des_key not in des_alphas:
            continue
        soc_vec = alpha_to_heatmap(soc_alphas[soc_key])
        des_vec = alpha_to_heatmap(des_alphas[des_key])
        r, _ = spearmanr(soc_vec, des_vec)
        rs.append(r)
    return np.array(rs)


def build_rdm(alphas, target_imgs):
    """Build 200×200 RDM (pairwise Pearson r) for all target images."""
    heatmaps = []
    valid_imgs = []
    for img_name in target_imgs:
        key = img_name if img_name in alphas else img_name.replace('.jpg', '')
        if key not in alphas:
            continue
        heatmaps.append(alpha_to_heatmap(alphas[key]))
        valid_imgs.append(img_name)
    n = len(heatmaps)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = pearsonr(heatmaps[i], heatmaps[j])
            rdm[i, j] = r
            rdm[j, i] = r
    return rdm, valid_imgs


def tril_indices(n):
    return np.tril_indices(n, k=-1)


def compute_rsa_between_task(soc_alphas, des_alphas, target_imgs):
    """
    Spearman r between social RDM lower triangle and describe RDM lower triangle.
    Also computes jackknife SEM.
    """
    soc_rdm, soc_imgs = build_rdm(soc_alphas, target_imgs)
    des_rdm, des_imgs = build_rdm(des_alphas, target_imgs)

    # Use intersection of valid images
    common = [img for img in soc_imgs if img in des_imgs]
    soc_idx = {img: i for i, img in enumerate(soc_imgs)}
    des_idx = {img: i for i, img in enumerate(des_imgs)}
    c_s = [soc_idx[img] for img in common]
    c_d = [des_idx[img] for img in common]

    soc_sub = soc_rdm[np.ix_(c_s, c_s)]
    des_sub = des_rdm[np.ix_(c_d, c_d)]

    n = len(common)
    idx = tril_indices(n)
    r_full, _ = spearmanr(soc_sub[idx], des_sub[idx])

    # Jackknife SEM: leave one image out at a time
    jack_rs = []
    for leave_out in range(n):
        keep = [i for i in range(n) if i != leave_out]
        s_sub = soc_sub[np.ix_(keep, keep)]
        d_sub = des_sub[np.ix_(keep, keep)]
        idx_k = tril_indices(len(keep))
        r_j, _ = spearmanr(s_sub[idx_k], d_sub[idx_k])
        jack_rs.append(r_j)

    jack_rs = np.array(jack_rs)
    jack_mean = jack_rs.mean()
    sem = np.sqrt((n - 1) / n * np.sum((jack_rs - jack_mean) ** 2))

    return float(r_full), float(sem)


# ── Load target images ────────────────────────────────────────────────────────
print('Loading target image list...')
target_imgs = load_target_names(TARGET_CSV)
print(f'  {len(target_imgs)} target images')

# ── Load CNN_LSTM between-task data from Project 1 ───────────────────────────
print('\nLoading CNN_LSTM between-task data from Project 1...')
p1_spat = pd.read_csv(os.path.join(PROC_DIR,
    'SUMMARY_between_tasks_spearman_correlations.csv'))
p1_rsa  = pd.read_csv(os.path.join(PROC_DIR,
    'rsa_second_order_summary_between_tasks.csv'))

cnn_lstm_spat_row = p1_spat[p1_spat['Model'].str.lower() == 'gated'].iloc[0]
cnn_lstm_rsa_row  = p1_rsa[p1_rsa['agent'].str.lower() == 'gated'].iloc[0]

spat_rows = [{
    'Model': 'CNN_LSTM',
    'Mean':   float(cnn_lstm_spat_row['Mean']),
    'Median': float(cnn_lstm_spat_row['Median']),
    'SEM':    float(cnn_lstm_spat_row['SEM']),
}]
rsa_rows = [{
    'agent':  'CNN_LSTM',
    'mean_r': float(cnn_lstm_rsa_row['mean_r']),
    'sem_r':  float(cnn_lstm_rsa_row['sem_r']),
}]

# Also pull human floor from Project 1
human_spat_row = p1_spat[p1_spat['Model'].str.lower() == 'human'].iloc[0]
human_rsa_row  = p1_rsa[p1_rsa['agent'].str.lower() == 'human'].iloc[0]

# ── Compute between-task for new conditions ───────────────────────────────────
for cond, epochs in LAST_EPOCHS.items():
    print(f'\n=== {cond} ===')
    cond_dir  = os.path.join(ARCH_DIR, cond)
    cache_dir = os.path.join(cond_dir, 'ckpt_cache')

    des_path = os.path.join(cache_dir, epochs['phase2'])
    soc_path = os.path.join(cache_dir, epochs['social_unfrozen'])

    print(f'  Describe NPZ: {os.path.basename(des_path)}')
    print(f'  Social NPZ:   {os.path.basename(soc_path)}')

    soc_alphas = load_alphas(soc_path)
    des_alphas = load_alphas(des_path)
    print(f'  Loaded {len(soc_alphas)} social, {len(des_alphas)} describe maps')

    # Spatial between-task
    print('  Computing spatial between-task...')
    rs = compute_spatial_between_task(soc_alphas, des_alphas, target_imgs)
    print(f'    {len(rs)} images, mean r = {rs.mean():.4f}, SEM = {rs.std()/np.sqrt(len(rs)):.4f}')
    spat_rows.append({
        'Model':  LABEL_MAP[cond],
        'Mean':   float(rs.mean()),
        'Median': float(np.median(rs)),
        'SEM':    float(rs.std() / np.sqrt(len(rs))),
    })

    # RSA between-task
    print('  Computing RSA between-task (with jackknife — may take a few minutes)...')
    mean_r, sem_r = compute_rsa_between_task(soc_alphas, des_alphas, target_imgs)
    print(f'    mean_r = {mean_r:.4f}, sem_r = {sem_r:.4f}')
    rsa_rows.append({
        'agent':  LABEL_MAP[cond],
        'mean_r': mean_r,
        'sem_r':  sem_r,
    })

# ── Save outputs ──────────────────────────────────────────────────────────────
spat_df = pd.DataFrame(spat_rows)
rsa_df  = pd.DataFrame(rsa_rows)

# Append human floor rows (for plotting reference)
spat_df = pd.concat([spat_df, pd.DataFrame([{
    'Model': 'Human',
    'Mean':   float(human_spat_row['Mean']),
    'Median': float(human_spat_row.get('Median', np.nan)),
    'SEM':    float(human_spat_row['SEM']),
}])], ignore_index=True)
rsa_df = pd.concat([rsa_df, pd.DataFrame([{
    'agent':  'Human',
    'mean_r': float(human_rsa_row['mean_r']),
    'sem_r':  float(human_rsa_row['sem_r']),
}])], ignore_index=True)

spat_out = os.path.join(OUT_DIR, 'between_task_spatial.csv')
rsa_out  = os.path.join(OUT_DIR, 'between_task_rsa.csv')
spat_df.to_csv(spat_out, index=False)
rsa_df.to_csv(rsa_out,  index=False)
print(f'\nSaved: {spat_out}')
print(f'Saved: {rsa_out}')
print('\nSpatial between-task:')
print(spat_df.to_string(index=False))
print('\nRSA between-task:')
print(rsa_df.to_string(index=False))
