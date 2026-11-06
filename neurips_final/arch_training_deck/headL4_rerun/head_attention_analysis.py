"""
head_attention_analysis.py
---------------------------
Compute spatial Spearman r for each attention head in a given layer
across all training phases, and plot one line per head.

Usage:
  python head_attention_analysis.py --condition CNN_Transformer --layer 3
  python head_attention_analysis.py --condition ViT_Transformer --layer 3

Reads:
  {COND_DIR}/ckpt_cache/{phase}_beam5_E{epoch:03d}.npz
    -> head_layer_alphas: dict {img_name: (num_layers, nhead, 196)}

Writes:
  {COND_DIR}/plots/head_attention_analysis_{condition}_L{layer}.png
"""

import os, sys, re, glob, argparse, shutil
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata as scipy_rankdata
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARCH_DIR   = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
TOP_PLOTS_DIR = os.path.join(ARCH_DIR, 'plots', 'head_attention')
RNN_DIR    = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
GROUP_DIR  = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
TARGET_CSV = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
HUM_HM_DIR_SOC  = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social')
HUM_HM_DIR_DESC = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive')
DESCRIBE_PHASES = {'phase1', 'phase2'}

ALL_PHASES = ['phase1', 'phase2', 'social_unfrozen', 'social_frozen']
PHASE_COLORS = {
    'phase1':          '#95a5a6',
    'phase2':          '#f39c12',
    'social_unfrozen': '#e74c3c',
    'social_frozen':   '#3498db',
}
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
    m = re.search(r'_E(\d+)\.npz$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        v = data[k]
        out[k] = v.item() if v.ndim == 0 else v
    return out


def normalize_map(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-10)


def load_human_heatmaps_flat(target_imgs, hm_dir):
    """
    Load, resize, and pre-rank all human heatmaps once.
    Pre-ranking eliminates repeated ranking inside the per-epoch/per-head loop.
    Returns:
        hum_ranked: (n_subjects, n_imgs, 256*256) float32 — ranked rows (avg ties);
                    zero-filled for missing images
        hum_valid:  (n_subjects, n_imgs) bool — True where image exists for subject
        valid_imgs: list of img_names in same order as axis 1
    """
    sub_dats = []
    for fn in sorted(os.listdir(hm_dir)):
        if fn.endswith('_heatmaps.npy'):
            sdat = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
            sub_dats.append(sdat)

    if not sub_dats:
        return np.zeros((0, 0, 256*256), dtype=np.float32), np.zeros((0, 0), dtype=bool), []

    all_keys = set()
    for sd in sub_dats:
        all_keys.update(sd.keys())
    valid_imgs = [im for im in target_imgs if im in all_keys]
    n_imgs = len(valid_imgs)
    n_subs = len(sub_dats)

    hum_ranked = np.zeros((n_subs, n_imgs, 256*256), dtype=np.float32)
    hum_valid  = np.zeros((n_subs, n_imgs), dtype=bool)

    for si, sdat in enumerate(sub_dats):
        rows_raw  = []
        row_valid = []
        for img_name in valid_imgs:
            if img_name not in sdat:
                rows_raw.append(np.zeros(256*256, dtype=np.float32))
                row_valid.append(False)
            else:
                hum = np.nan_to_num(sdat[img_name], nan=0.0).astype(np.float32)
                flat = np.array(
                    Image.fromarray(hum).resize((256, 256), resample=Image.NEAREST),
                    dtype=np.float32).flatten()
                rows_raw.append(flat)
                row_valid.append(True)

        hum_valid[si] = row_valid
        valid_i = np.where(row_valid)[0]
        if len(valid_i):
            mat = np.stack([rows_raw[i] for i in valid_i])   # (n_valid, 65536)
            hum_ranked[si, valid_i] = scipy_rankdata(
                mat, method='average', axis=1).astype(np.float32)
        del rows_raw

    return hum_ranked, hum_valid, valid_imgs


def rank_matrix(mat):
    """Rank each row of mat (n, d) → (n, d) float32, averaging tied values."""
    return scipy_rankdata(mat, method='average', axis=1).astype(np.float32)


def compute_spatial_mean_r_fast(model_vecs, hum_ranked, hum_valid, valid_img_indices):
    """
    Vectorized spatial Spearman r using pre-ranked human heatmaps.
    model_vecs:       (n_imgs, 256*256) float32 — raw attention maps
    hum_ranked:       (n_subjects, n_imgs, 256*256) float32 — pre-ranked (avg ties)
    hum_valid:        (n_subjects, n_imgs) bool
    valid_img_indices: which image indices (into hum_ranked axis 1) to use
    Returns (mean_r, sem_r) via per-subject Fisher-z averaging.
    """
    mv   = model_vecs[valid_img_indices]              # (n_valid, d)
    mv_r = rank_matrix(mv)                            # rank model vecs once
    mv_z = mv_r - mv_r.mean(axis=1, keepdims=True)
    mv_n = mv_z / (np.linalg.norm(mv_z, axis=1, keepdims=True) + 1e-10)

    all_z = []
    for si in range(hum_ranked.shape[0]):
        v = hum_valid[si, valid_img_indices]          # (n_valid,) bool
        if v.sum() < 2:
            continue
        hv   = hum_ranked[si][valid_img_indices][v]   # (n_present, d) — already ranked
        mv_  = mv_n[v]

        hv_z = hv - hv.mean(axis=1, keepdims=True)
        hv_n = hv_z / (np.linalg.norm(hv_z, axis=1, keepdims=True) + 1e-10)

        rs = np.einsum('ij,ij->i', mv_, hv_n)
        rs = np.clip(rs, -0.999999, 0.999999)
        zs = np.arctanh(rs)
        zs = zs[~np.isnan(zs)]
        if len(zs):
            all_z.append(zs.mean())

    if not all_z:
        return np.nan, np.nan
    arr    = np.array(all_z)
    mean_r = float(np.tanh(arr.mean()))
    sem_r  = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean_r, sem_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True,
                        choices=['CNN_Transformer', 'ViT_Transformer'])
    parser.add_argument('--layer', type=int, default=3,
                        help='1-indexed decoder layer (default: 3)')
    args = parser.parse_args()

    layer_idx = args.layer - 1  # 0-indexed

    cond_dir  = os.path.join(ARCH_DIR, args.condition)
    cache_dir = os.path.join(cond_dir, 'ckpt_cache')
    plots_dir = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/arch_training_deck/headL4_rerun/out'
    os.makedirs(plots_dir, exist_ok=True)

    target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
    print(f'Target images: {len(target_imgs)}')

    # Heatmaps loaded lazily per phase-type to reduce peak memory usage
    _heatmap_cache = {}

    def get_heatmaps(phase_type):
        """Load and pre-rank heatmaps on demand; keep only one type in memory."""
        if phase_type not in _heatmap_cache:
            other = 'soc' if phase_type == 'desc' else 'desc'
            if other in _heatmap_cache:
                del _heatmap_cache[other]
            hm_dir = HUM_HM_DIR_DESC if phase_type == 'desc' else HUM_HM_DIR_SOC
            lbl    = 'describe' if phase_type == 'desc' else 'social'
            print(f'Loading {lbl} heatmaps (+ pre-ranking)...')
            ranked, valid, imgs = load_human_heatmaps_flat(target_imgs, hm_dir)
            print(f'  {ranked.shape[0]} subjects, {len(imgs)} images')
            _heatmap_cache[phase_type] = (ranked, valid, imgs,
                                          {im: i for i, im in enumerate(imgs)})
        return _heatmap_cache[phase_type]

    # Pre-build model heatmap cache: {phase_E###: (nhead, n_common, 256*256)}
    # Stored on disk as {cache_dir}/{phase}_beam5_E{epoch:03d}_head_vecs_L{layer}.npz
    # to avoid rebuilding from raw attention NPZs on every run.
    records = []

    for phase in ALL_PHASES:
        # Select the correct human heatmaps for this phase (loaded lazily)
        phase_type = 'desc' if phase in DESCRIBE_PHASES else 'soc'
        hum_ranked, hum_valid, valid_imgs, img_to_idx = get_heatmaps(phase_type)

        # Only match raw attention NPZs — exclude any _head_vecs_* cache files
        pattern   = os.path.join(cache_dir, f'{phase}_beam5_E*.npz')
        all_paths = glob.glob(pattern)
        npz_paths = sorted(
            [p for p in all_paths if '_head_vecs_' not in os.path.basename(p)],
            key=epoch_from_path)
        if not npz_paths:
            print(f'  No NPZ files for phase: {phase}')
            continue
        print(f'\nPhase: {phase}  ({len(npz_paths)} epochs)')

        for npz_path in tqdm(npz_paths, desc=f'  {phase}'):
            epoch = epoch_from_path(npz_path)

            # Check if head-vector cache exists
            hvec_cache = npz_path.replace(
                f'_beam5_E{epoch:03d}.npz',
                f'_beam5_E{epoch:03d}_head_vecs_L{args.layer}.npz')

            if os.path.exists(hvec_cache):
                hvec_data   = np.load(hvec_cache, allow_pickle=True)
                head_vecs   = hvec_data['head_vecs']    # (nhead, n_common, 65536)
                common_idx  = hvec_data['common_idx'].tolist()
                nhead       = head_vecs.shape[0]
            else:
                data = load_npz(npz_path)
                hla  = data.get('head_layer_alphas', {})
                if not hla:
                    print(f'  WARNING: no head_layer_alphas in {os.path.basename(npz_path)}, skipping')
                    continue

                first_arr  = next(iter(hla.values()))
                nhead      = first_arr.shape[1]
                common_imgs = [im for im in valid_imgs if im in hla]
                if not common_imgs:
                    continue
                common_idx = [img_to_idx[im] for im in common_imgs]

                head_vecs = np.zeros((nhead, len(common_imgs), 256*256), dtype=np.float32)
                for hi in range(nhead):
                    for ii, img_name in enumerate(common_imgs):
                        arr = hla[img_name]
                        li  = min(layer_idx, arr.shape[0] - 1)
                        attn = arr[li, hi].reshape(14, 14).astype(np.float32)
                        resized = np.array(
                            Image.fromarray(attn).resize((256, 256), resample=Image.BILINEAR),
                            dtype=np.float32)
                        head_vecs[hi, ii] = resized.flatten()

                # Save to disk
                np.savez_compressed(hvec_cache,
                                    head_vecs=head_vecs,
                                    common_idx=np.array(common_idx, dtype=np.int32))

            for head_idx in range(nhead):
                model_mat = head_vecs[head_idx]          # (n_common, 65536)
                mean_r, sem_r = compute_spatial_mean_r_fast(
                    model_mat, hum_ranked, hum_valid, common_idx)
                records.append({
                    'phase':       phase,
                    'epoch':       epoch,
                    'head':        head_idx + 1,
                    'mean_r':      mean_r,
                    'spatial_SEM': sem_r,
                })

    if not records:
        print('No data found. Exiting.')
        return

    df = pd.DataFrame(records)
    csv_out = os.path.join('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/arch_training_deck/headL4_rerun/out', args.condition + '_' + f'head_attention_analysis_L{args.layer}.csv')
    df.to_csv(csv_out, index=False)
    print(f'\nSaved CSV: {csv_out}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    nhead = df['head'].max()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f'{args.condition}  –  Spatial Spearman r per Attention Head  (Layer {args.layer})',
        fontsize=12, fontweight='bold')

    x_cursor   = 0
    boundaries = []
    phase_mids = []
    phase_lbls = []

    for phase in ALL_PHASES:
        ph_df = df[df['phase'] == phase]
        if ph_df.empty:
            continue
        epochs = sorted(ph_df['epoch'].unique())
        n_epochs = len(epochs)
        epoch_to_x = {ep: x_cursor + i for i, ep in enumerate(epochs)}

        for head_idx in range(1, nhead + 1):
            hd = ph_df[ph_df['head'] == head_idx].sort_values('epoch')
            xs = [epoch_to_x[ep] for ep in hd['epoch'].values]
            ys = hd['mean_r'].values
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
    ax.set_ylabel('Mean Spatial Spearman r')
    ax.grid(alpha=0.25, axis='y')
    ax.legend(fontsize=8, loc='lower right', ncol=2)

    out = os.path.join(plots_dir, f'head_attention_analysis_{args.condition}_L{args.layer}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot: {out}')

    # Also copy to top-level plots/head_attention/
    os.makedirs(TOP_PLOTS_DIR, exist_ok=True)
    pass  # shutil.copy2(out, os.path.join(TOP_PLOTS_DIR, os.path.basename(out)))
    print('Done.')


if __name__ == '__main__':
    main()
