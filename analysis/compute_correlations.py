"""
compute_correlations.py
-----------------------
Compute spatial Spearman r, RSA, and caption cosine similarity for all
cached NPZ files from a given condition × phase.

Usage:
  python compute_correlations.py --condition ViT_LSTM --phase social_unfrozen
  python compute_correlations.py --condition CNN_Transformer --phase social_frozen

Reads:
  {COND_DIR}/ckpt_cache/{phase}_beam5_E{epoch:03d}.npz

Writes:
  {COND_DIR}/correlations/spatial_corr_{phase}.csv
  {COND_DIR}/correlations/rsa_corr_{phase}.csv
  {COND_DIR}/correlations/caption_cosine_{phase}.csv

CSV columns:
  spatial_corr  : condition, phase, epoch, subject_id, mean_r, fisher_z
  rsa_corr      : condition, phase, epoch, subject_id, spearman_rho
  caption_cosine: condition, phase, epoch, subject_id, mean_cosine_sim
"""
import os, sys, re, json, argparse, glob
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm

ARCH_DIR    = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
RNN_DIR     = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
GROUP_DIR   = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
TARGET_CSV  = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
HUM_HM_DIR_SOC   = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social')
HUM_HM_DIR_DESC  = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive')
HUM_CAP_DIR_SOC  = os.path.join(GROUP_DIR, 'captions_humans', 'CLEANED_trans_social')
HUM_CAP_DIR_DESC = os.path.join(GROUP_DIR, 'captions_humans', 'CLEANED_trans_desc')

# Phases that compare against describe-task human data rather than social
DESCRIBE_PHASES = {'phase1', 'phase2'}


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


def load_human_heatmaps(hm_dir):
    hdata = {}
    for fn in os.listdir(hm_dir):
        if fn.endswith('_heatmaps.npy'):
            sub_id = fn.replace('_heatmaps.npy', '')
            hdata[sub_id] = np.load(os.path.join(hm_dir, fn),
                                    allow_pickle=True).item()
    return hdata


def normalize_map(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-10)


def build_model_heatmaps(npz_alphas, target_imgs):
    heatmaps = {}
    for img_name in target_imgs:
        if img_name not in npz_alphas:
            continue
        arr = npz_alphas[img_name].reshape(14, 14).astype(np.float32)
        resized = np.array(Image.fromarray(arr).resize((256, 256),
                                                       resample=Image.BILINEAR))
        heatmaps[img_name] = normalize_map(resized)
    return heatmaps


def compute_spatial(heatmaps, hdata, target_imgs):
    rows = []
    img_keys = [im for im in target_imgs if im in heatmaps]
    for sub_id, sdat in tqdm(hdata.items(), desc='  spatial'):
        z_vals = []
        for img_name in img_keys:
            fname = img_name  # keys in hdata are e.g. '000000017883.jpg'
            if fname not in sdat:
                continue
            model_flat = heatmaps[img_name].flatten()
            hum = np.nan_to_num(sdat[fname], nan=0.0)
            hum_resized = np.array(
                Image.fromarray(hum).resize((256, 256), resample=Image.NEAREST))
            hum_flat = np.nan_to_num(hum_resized.flatten())
            if np.std(model_flat) == 0 or np.std(hum_flat) == 0:
                continue
            r, _ = spearmanr(model_flat, hum_flat)
            if not np.isnan(r):
                z_vals.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
        if z_vals:
            fisher_z = np.mean(z_vals)
            rows.append({'subject_id': sub_id,
                         'mean_r':    float(np.tanh(fisher_z)),
                         'fisher_z':  float(fisher_z)})
    return rows


def compute_rsa(heatmaps, hdata, target_imgs):
    img_keys = [im for im in target_imgs if im in heatmaps]
    n = len(img_keys)
    if n < 2:
        return []

    vecs_model = np.stack([heatmaps[im].flatten() for im in img_keys])
    model_corr = np.corrcoef(vecs_model)
    triu        = np.triu_indices(n, k=1)
    vec_model   = model_corr[triu]

    rows = []
    for sub_id, sdat in tqdm(hdata.items(), desc='  RSA'):
        vecs = []
        for img_name in img_keys:
            fname = img_name
            if fname in sdat:
                v = np.nan_to_num(sdat[fname], nan=0.0)
                v = np.array(Image.fromarray(v).resize(
                    (256, 256), resample=Image.NEAREST)).flatten()
                vecs.append(v if np.std(v) > 0 else np.full_like(v, np.nan))
            else:
                any_v = next(iter(sdat.values()))
                sz = np.array(Image.fromarray(np.nan_to_num(any_v, nan=0.0)).resize(
                    (256, 256), resample=Image.NEAREST)).flatten()
                vecs.append(np.full(sz.shape, np.nan))
        vecs = np.stack(vecs)

        sub_corr = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i, n):
                mask = ~np.isnan(vecs[i]) & ~np.isnan(vecs[j])
                if mask.sum() > 1:
                    r = np.corrcoef(vecs[i][mask], vecs[j][mask])[0, 1]
                    sub_corr[i, j] = sub_corr[j, i] = r
        sub_z   = np.arctanh(np.clip(sub_corr, -0.999999, 0.999999))
        vec_sub = sub_z[triu]
        valid   = ~np.isnan(vec_sub) & ~np.isnan(vec_model)
        if valid.sum() < 2:
            continue
        rho, _ = spearmanr(vec_sub[valid], vec_model[valid])
        rows.append({'subject_id': sub_id, 'spearman_rho': float(rho)})
    return rows


def compute_caption_cosine(npz_captions, target_imgs, hum_cap_dir=None):
    """Compute cosine similarity between model captions and human captions."""
    if hum_cap_dir is None:
        hum_cap_dir = HUM_CAP_DIR_SOC
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print('  sentence_transformers not available, skipping caption cosine')
        return []

    embedder = SentenceTransformer('all-mpnet-base-v2')

    # Load human captions
    is_desc = (os.path.realpath(hum_cap_dir) == os.path.realpath(HUM_CAP_DIR_DESC))
    hum_caps = {}  # {sub_id: {img_name: caption}}
    for fn in sorted(os.listdir(hum_cap_dir)):
        if not fn.endswith('_trans.csv'):
            continue
        sub_id = fn.split('_')[0]
        fpath  = os.path.join(hum_cap_dir, fn)
        if is_desc:
            # Describe CSVs can have commas embedded in captions; manual parse
            rows = []
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                header = next(f).strip().lstrip('\ufeff').split(',')
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    idx, cap = line.split(',', 1)
                    cap = cap.replace(',', '').strip().strip('"')
                    rows.append((idx.strip(), cap))
            if len(header) < 2:
                continue
            df = pd.DataFrame(rows, columns=['index', 'caption'])
        else:
            df = pd.read_csv(fpath, sep=None, engine='python', dtype=str,
                             encoding_errors='replace')
            df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
            if 'index' not in df.columns:
                df.rename(columns={df.columns[0]: 'index'}, inplace=True)
            if 'caption' not in df.columns:
                continue
        df = df[df['caption'].astype(str).str.strip() != 'NOT FOUND'].copy()
        hum_caps[sub_id] = dict(zip(df['index'].astype(str), df['caption']))

    img_keys = [im for im in target_imgs if im in npz_captions]
    model_texts = [npz_captions[im] for im in img_keys]
    model_embs  = embedder.encode(model_texts, show_progress_bar=False,
                                  normalize_embeddings=True)  # (N, D)

    rows = []
    for sub_id, scaps in tqdm(hum_caps.items(), desc='  caption cosine'):
        sims = []
        for i, img_name in enumerate(img_keys):
            key = img_name  # '000000017883.jpg' or without ext
            cap = scaps.get(key) or scaps.get(img_name[:-4])
            if not cap or str(cap).strip() == '':
                continue
            hum_emb = embedder.encode([str(cap)], normalize_embeddings=True)[0]
            sim = float(np.dot(model_embs[i], hum_emb))
            sims.append(sim)
        if sims:
            rows.append({'subject_id': sub_id,
                         'mean_cosine_sim': float(np.mean(sims))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True,
                        choices=['CNN_Transformer', 'ViT_LSTM', 'ViT_Transformer'])
    parser.add_argument('--phase', required=True,
                        choices=['phase1', 'phase2', 'social_unfrozen', 'social_frozen'])
    parser.add_argument('--layer', type=int, default=None,
                        help='1-indexed layer to use for spatial/RSA (default: last layer via alphas key)')
    args = parser.parse_args()

    cond_dir  = os.path.join(ARCH_DIR, args.condition)
    cache_dir = os.path.join(cond_dir, 'ckpt_cache')
    corr_dir  = os.path.join(cond_dir, 'correlations')
    os.makedirs(corr_dir, exist_ok=True)

    target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
    print(f'Target images: {len(target_imgs)}')

    # Use describe heatmaps for phase1/phase2, social heatmaps for social phases
    hm_dir = HUM_HM_DIR_DESC if args.phase in DESCRIBE_PHASES else HUM_HM_DIR_SOC
    print(f'Loading human heatmaps from: {os.path.basename(hm_dir)}')
    hdata = load_human_heatmaps(hm_dir)
    print(f'  {len(hdata)} subjects')

    pattern = os.path.join(cache_dir, f'{args.phase}_beam5_E[0-9][0-9][0-9].npz')
    npz_paths = sorted(glob.glob(pattern), key=epoch_from_path)
    print(f'Found {len(npz_paths)} cached NPZ files for {args.condition}/{args.phase}')

    sp_rows, rsa_rows, cos_rows = [], [], []

    for npz_path in npz_paths:
        epoch = epoch_from_path(npz_path)
        print(f'\n── Epoch {epoch:03d}: {os.path.basename(npz_path)} ──')
        data     = load_npz(npz_path)
        captions = data.get('caption', {})

        # Choose which attention layer to use for spatial/RSA
        if args.layer is not None:
            layer_idx = args.layer - 1  # convert 1-indexed to 0-indexed
            layer_alphas_dict = data.get('layer_alphas', {})
            alphas = {}
            for img_name, la in layer_alphas_dict.items():
                # la shape: (num_layers, 196)
                if layer_idx < la.shape[0]:
                    alphas[img_name] = la[layer_idx]
                else:
                    print(f'  WARNING: layer {args.layer} out of range for {img_name}, using last')
                    alphas[img_name] = la[-1]
        else:
            alphas = data.get('alphas', {})

        heatmaps = build_model_heatmaps(alphas, target_imgs)
        print(f'  Model heatmaps: {len(heatmaps)} images')

        print('  Spatial correlations...')
        for row in compute_spatial(heatmaps, hdata, target_imgs):
            row.update({'condition': args.condition, 'phase': args.phase,
                        'epoch': epoch})
            sp_rows.append(row)

        print('  RSA...')
        for row in compute_rsa(heatmaps, hdata, target_imgs):
            row.update({'condition': args.condition, 'phase': args.phase,
                        'epoch': epoch})
            rsa_rows.append(row)

        print('  Caption cosine...')
        hum_cap_dir = HUM_CAP_DIR_DESC if args.phase in DESCRIBE_PHASES else HUM_CAP_DIR_SOC
        for row in compute_caption_cosine(captions, target_imgs, hum_cap_dir=hum_cap_dir):
            row.update({'condition': args.condition, 'phase': args.phase,
                        'epoch': epoch})
            cos_rows.append(row)

    # Save CSVs
    pd.DataFrame(sp_rows).to_csv(
        os.path.join(corr_dir, f'spatial_corr_{args.phase}.csv'), index=False)
    pd.DataFrame(rsa_rows).to_csv(
        os.path.join(corr_dir, f'rsa_corr_{args.phase}.csv'), index=False)
    pd.DataFrame(cos_rows).to_csv(
        os.path.join(corr_dir, f'caption_cosine_{args.phase}.csv'), index=False)

    print(f'\nSaved CSVs to {corr_dir}')

    if sp_rows:
        by_epoch = pd.DataFrame(sp_rows).groupby('epoch')['mean_r'].mean()
        print('\nSpatial r by epoch:')
        print(by_epoch.to_string())
    if rsa_rows:
        by_epoch = pd.DataFrame(rsa_rows).groupby('epoch')['spearman_rho'].mean()
        print('\nRSA rho by epoch:')
        print(by_epoch.to_string())


if __name__ == '__main__':
    main()
