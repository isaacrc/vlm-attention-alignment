"""
head_attention_rsa.py
----------------------
Compute RSA (representational similarity analysis) for each attention head in a
given decoder layer across all training phases, and plot one line per head.

RSA metric: Spearman rho between the model head's image-pairwise correlation
structure and each human subject's heatmap correlation structure, averaged
(in Fisher-z space) across subjects.

Usage:
  python head_attention_rsa.py --condition CNN_Transformer --layer 3
  python head_attention_rsa.py --condition ViT_Transformer --layer 3

Reads:
  {COND_DIR}/ckpt_cache/{phase}_beam5_E{epoch:03d}_head_vecs_L{layer}.npz
    -> head_vecs : (nhead, n_common, 65536)
    -> common_idx: (n_common,)

Writes:
  {COND_DIR}/plots/head_attention_rsa_{condition}_L{layer}.png
  {COND_DIR}/head_attention_rsa_L{layer}.csv
  plots/head_attention/head_attention_rsa_{condition}_L{layer}.png
"""

import os, sys, re, glob, argparse, shutil
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARCH_DIR      = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
TOP_PLOTS_DIR = os.path.join(ARCH_DIR, 'plots', 'head_attention')
RNN_DIR       = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
GROUP_DIR     = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
TARGET_CSV    = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
HUM_HM_DIR_SOC  = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social')
HUM_HM_DIR_DESC = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive')
DESCRIBE_PHASES = {'phase1', 'phase2'}

ALL_PHASES = ['phase1', 'phase2', 'social_unfrozen', 'social_frozen']
PHASE_LABELS = {
    'phase1':          'Describe Pretrain',
    'phase2':          'Describe Finetune',
    'social_unfrozen': 'Social – Unfrozen',
    'social_frozen':   'Social – Frozen',
}

HEAD_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
]


def epoch_from_path(path):
    m = re.search(r'_E(\d+)_head_vecs', os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_human_heatmaps_flat(target_imgs, hm_dir):
    """
    Load and resize all human heatmaps once.
    Returns:
        hum_matrix : (n_subjects, n_imgs, 256*256) float32, NaN rows where missing
        valid_imgs : list of img_names in same order as axis 1
    """
    sub_dats = []
    for fn in sorted(os.listdir(hm_dir)):
        if fn.endswith('_heatmaps.npy'):
            sdat = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
            sub_dats.append(sdat)

    if not sub_dats:
        return np.zeros((0, 0, 256 * 256)), []

    all_keys = set()
    for sd in sub_dats:
        all_keys.update(sd.keys())
    valid_imgs = [im for im in target_imgs if im in all_keys]
    n_imgs = len(valid_imgs)
    n_subs = len(sub_dats)
    hum_matrix = np.full((n_subs, n_imgs, 256 * 256), np.nan, dtype=np.float32)

    for si, sdat in enumerate(sub_dats):
        for ii, img_name in enumerate(valid_imgs):
            if img_name not in sdat:
                continue
            hum = np.nan_to_num(sdat[img_name], nan=0.0).astype(np.float32)
            resized = np.array(
                Image.fromarray(hum).resize((256, 256), resample=Image.NEAREST),
                dtype=np.float32)
            hum_matrix[si, ii] = resized.flatten()

    return hum_matrix, valid_imgs


def build_human_rdms(hum_matrix, common_idx):
    """
    Precompute Fisher-z RDM upper-triangle vector for each subject over the
    given set of image indices.

    Returns:
        human_rdms : list of (valid_local_idx, sub_z_vec) per subject, or None
            where a subject has too few valid images.
    """
    hm = hum_matrix[:, common_idx, :]   # (n_subs, n_common, 65536)
    n_common = len(common_idx)
    human_rdms = []
    for si in range(hm.shape[0]):
        hv = hm[si]                       # (n_common, 65536)
        valid_rows = ~np.any(np.isnan(hv), axis=1)
        valid_idx = np.where(valid_rows)[0]
        if len(valid_idx) < 2:
            human_rdms.append(None)
            continue
        hv_v = hv[valid_idx]              # (n_valid, 65536)
        sub_corr = np.corrcoef(hv_v)      # (n_valid, n_valid)
        sub_z = np.arctanh(np.clip(sub_corr, -0.999999, 0.999999))
        human_rdms.append((valid_idx, sub_z))
    return human_rdms


def compute_rsa_per_head(model_vecs, human_rdms):
    """
    RSA between a single head's attention map structure and human heatmap structure.

    model_vecs  : (n_common, 65536) — one head's attention maps across images
    human_rdms  : precomputed list from build_human_rdms()

    Returns mean RSA Spearman rho across subjects (Fisher-z averaged, then tanh).
    """
    n_common = model_vecs.shape[0]
    model_corr = np.corrcoef(model_vecs)   # (n_common, n_common)
    model_z    = np.arctanh(np.clip(model_corr, -0.999999, 0.999999))

    zs = []
    for entry in human_rdms:
        if entry is None:
            continue
        valid_idx, sub_z = entry

        # Restrict model to the same valid images as this subject
        mod_sub = model_z[np.ix_(valid_idx, valid_idx)]
        n_v = len(valid_idx)
        triu = np.triu_indices(n_v, k=1)
        vec_mod = mod_sub[triu]
        vec_sub = sub_z[triu]

        valid = ~np.isnan(vec_mod) & ~np.isnan(vec_sub)
        if valid.sum() < 2:
            continue
        rho, _ = spearmanr(vec_mod[valid], vec_sub[valid])
        if not np.isnan(rho):
            zs.append(np.arctanh(np.clip(float(rho), -0.999999, 0.999999)))

    if not zs:
        return np.nan
    return float(np.tanh(np.mean(zs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True,
                        choices=['CNN_Transformer', 'ViT_Transformer'])
    parser.add_argument('--layer', type=int, default=3,
                        help='1-indexed decoder layer (default: 3)')
    args = parser.parse_args()

    cond_dir  = os.path.join(ARCH_DIR, args.condition)
    cache_dir = os.path.join(cond_dir, 'ckpt_cache')
    plots_dir = os.path.join(cond_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
    print(f'Target images: {len(target_imgs)}')

    # Heatmaps loaded lazily per phase-type to reduce peak memory usage
    _heatmap_cache = {}

    def get_heatmap_matrix(phase_type):
        if phase_type not in _heatmap_cache:
            other = 'soc' if phase_type == 'desc' else 'desc'
            if other in _heatmap_cache:
                del _heatmap_cache[other]
            hm_dir = HUM_HM_DIR_DESC if phase_type == 'desc' else HUM_HM_DIR_SOC
            lbl    = 'describe' if phase_type == 'desc' else 'social'
            print(f'Loading {lbl} heatmaps...')
            mat, imgs = load_human_heatmaps_flat(target_imgs, hm_dir)
            print(f'  {mat.shape[0]} subjects, {len(imgs)} images')
            _heatmap_cache[phase_type] = mat
        return _heatmap_cache[phase_type]

    records = []

    # Cache human RDMs keyed by (phase_type, common_idx_tuple) to avoid recomputing
    # across epochs that share the same image set.
    _human_rdm_cache = {}

    for phase in ALL_PHASES:
        phase_type = 'desc' if phase in DESCRIBE_PHASES else 'soc'
        hum_matrix = get_heatmap_matrix(phase_type)

        # Look for pre-built head_vecs cache files
        pattern = os.path.join(cache_dir,
                               f'{phase}_beam5_E[0-9][0-9][0-9]_head_vecs_L{args.layer}.npz')
        hvec_paths = sorted(glob.glob(pattern), key=epoch_from_path)
        if not hvec_paths:
            print(f'  No head_vecs cache for phase: {phase}  '
                  f'(run head_attention_analysis.py first)')
            continue
        print(f'\nPhase: {phase}  ({len(hvec_paths)} epochs)')

        for hvec_path in tqdm(hvec_paths, desc=f'  {phase}'):
            epoch = epoch_from_path(hvec_path)
            hvec_data  = np.load(hvec_path, allow_pickle=True)
            head_vecs  = hvec_data['head_vecs']      # (nhead, n_common, 65536)
            common_idx = hvec_data['common_idx'].tolist()
            nhead      = head_vecs.shape[0]

            # Get (or compute) human RDMs for this common_idx set
            phase_type = 'desc' if phase in DESCRIBE_PHASES else 'soc'
            cache_key  = (phase_type, tuple(common_idx))
            if cache_key not in _human_rdm_cache:
                _human_rdm_cache[cache_key] = build_human_rdms(hum_matrix, common_idx)
            human_rdms = _human_rdm_cache[cache_key]

            for head_idx in range(nhead):
                model_vecs = head_vecs[head_idx]     # (n_common, 65536)
                mean_rsa   = compute_rsa_per_head(model_vecs, human_rdms)
                records.append({
                    'phase': phase,
                    'epoch': epoch,
                    'head':  head_idx + 1,
                    'rsa':   mean_rsa,
                })

    if not records:
        print('No data found. Exiting.')
        return

    df = pd.DataFrame(records)
    csv_out = os.path.join(cond_dir, f'head_attention_rsa_L{args.layer}.csv')
    df.to_csv(csv_out, index=False)
    print(f'\nSaved CSV: {csv_out}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    nhead = int(df['head'].max())
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f'{args.condition}  –  RSA (Spearman \u03c1) per Attention Head  (Layer {args.layer})',
        fontsize=12, fontweight='bold')

    x_cursor   = 0
    boundaries = []
    phase_mids = []
    phase_lbls = []

    for phase in ALL_PHASES:
        ph_df = df[df['phase'] == phase]
        if ph_df.empty:
            continue
        epochs     = sorted(ph_df['epoch'].unique())
        n_epochs   = len(epochs)
        epoch_to_x = {ep: x_cursor + i for i, ep in enumerate(epochs)}

        for head_idx in range(1, nhead + 1):
            hd    = ph_df[ph_df['head'] == head_idx].sort_values('epoch')
            xs    = [epoch_to_x[ep] for ep in hd['epoch'].values]
            ys    = hd['rsa'].values
            color = HEAD_COLORS[(head_idx - 1) % len(HEAD_COLORS)]
            label = f'Head {head_idx}' if phase == ALL_PHASES[0] else None
            ax.plot(xs, ys, color=color, label=label,
                    linewidth=1.4, marker='o', ms=3)

        if x_cursor > 0:
            boundaries.append(x_cursor - 0.5)
        phase_mids.append(x_cursor + (n_epochs - 1) / 2)
        phase_lbls.append(PHASE_LABELS.get(phase, phase))
        x_cursor += n_epochs

    for b in boundaries:
        ax.axvline(b, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

    ax.set_xticks(phase_mids)
    ax.set_xticklabels(phase_lbls, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_ylabel('Mean RSA (Spearman \u03c1)')
    ax.grid(alpha=0.25, axis='y')
    ax.legend(fontsize=8, loc='lower right', ncol=2)

    out = os.path.join(plots_dir,
                       f'head_attention_rsa_{args.condition}_L{args.layer}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot: {out}')

    os.makedirs(TOP_PLOTS_DIR, exist_ok=True)
    shutil.copy2(out, os.path.join(TOP_PLOTS_DIR, os.path.basename(out)))
    print('Done.')


if __name__ == '__main__':
    main()
