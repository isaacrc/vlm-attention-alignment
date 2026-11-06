"""
extract_molmo_alllayers.py  (Task 1, D4)
=========================================
Full 28-layer x 28-head Molmo sweep. Adapts analysis/vlm_head_molmo.py (which only
did layer 9) to every layer, and saves per-(layer,head,image) SUBJECT-AVERAGED
Fisher-z so downstream Task 1 can do:
  - best-head / mean-head / median-head selection rules
  - CV-honest selection over images
  - winner's-curse bootstrap over images

Outputs (rebuttal/task1_head_selection/results/):
  molmo_allL_perimage_{task}.parquet   cols: task,layer,head,image_id,mean_z,mean_r,n_sub
  molmo_allL_summary_{task}.csv         cols: task,layer,head,mean_r,sem_r   (subject-level SEM)
Run via SLURM (CPU, ~1-2h). See run_molmo_sweep.slurm.
"""
import os, sys
import numpy as np
import pandas as pd
import torch, cv2
from PIL import Image
from scipy.stats import rankdata as _rankdata
from tqdm import tqdm

BASE       = '/scratch/gpfs/GRAZIANO/CANS'
GROUP_DIR  = os.path.join(BASE, 'group_analysis')
MOLMO_DIR  = os.path.join(BASE, 'molmo', 'molmo')
RNN_DIR    = os.path.join(BASE, 'rnn_model')
TARGET_CSV = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
HM_DIR = {'social':   os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social'),
          'describe': os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive')}
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS, N_HEADS, OUT_SIZE = 28, 28, 256
NUM_CROPS_X = NUM_CROPS_Y = 3

target_imgs = [x.replace('.jpg', '') for x in pd.read_csv(TARGET_CSV)['image_name'].tolist()]


def fast_rank(mat):
    """Vectorized ordinal ranks (K,D)->(K,D) float32. Model maps are bilinear-
    upsampled with few exact ties, so ordinal ~ average-tie for Spearman."""
    return mat.argsort(axis=1).argsort(axis=1).astype(np.float32)


def setup_grid(act_dir):
    idx = torch.load(os.path.join(act_dir, 'image_input_idx.pt'), map_location='cpu')[0].numpy()
    img_idx_all = (idx[1:].reshape(NUM_CROPS_X, NUM_CROPS_Y, 12, 12)
                   .transpose(0, 2, 1, 3).reshape(NUM_CROPS_X * 12, NUM_CROPS_Y * 12))
    valid_idx = np.where(img_idx_all > 0)
    H, W = NUM_CROPS_X * 336, NUM_CROPS_Y * 336
    resized = cv2.resize(img_idx_all, (W, H), interpolation=cv2.INTER_NEAREST)
    return img_idx_all, valid_idx, resized, (H, W)


def all_layer_head_maps(act_path, img_idx_all, valid_idx, resized, shape):
    """Return (N_LAYERS, N_HEADS, OUT_SIZE*OUT_SIZE) float32 for one image."""
    attn = torch.load(act_path, map_location='cpu').numpy()  # (28,28,n_tok,1028)
    n_tok = attn.shape[2]
    H, W = shape
    Hg, Wg = img_idx_all.shape
    pos = img_idx_all[valid_idx]                      # 1D indices into the 1028 vector
    row_valid = np.any(resized > 0, axis=1); col_valid = np.any(resized > 0, axis=0)
    out = np.zeros((N_LAYERS, N_HEADS, OUT_SIZE * OUT_SIZE), dtype=np.float32)
    for L in range(N_LAYERS):
        for h in range(N_HEADS):
            vol = np.zeros((n_tok, Hg, Wg), dtype=np.float32)
            vol[:, valid_idx[0], valid_idx[1]] = attn[L, h, :, pos].T  # (n_tok, n_valid)
            grid = vol.sum(axis=0)
            big = cv2.resize(grid, (W, H), interpolation=cv2.INTER_NEAREST)
            crop = big[row_valid][:, col_valid]
            out[L, h] = cv2.resize(crop, (OUT_SIZE, OUT_SIZE),
                                   interpolation=cv2.INTER_AREA).flatten()
    return out


def load_prerank_human(task, imgs):
    hm_dir = HM_DIR[task]
    files = sorted(f for f in os.listdir(hm_dir) if f.endswith('_heatmaps.npy'))
    S, N, D = len(files), len(imgs), OUT_SIZE * OUT_SIZE
    ranked = np.zeros((S, N, D), np.float32); valid = np.zeros((S, N), bool)
    for si, fn in enumerate(files):
        sd = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
        raw, ok = [], []
        for im in imgs:
            hm = sd.get(im + '.jpg', sd.get(im, None))
            if hm is None:
                raw.append(np.zeros(D, np.float32)); ok.append(False)
            else:
                h = np.nan_to_num(hm, nan=0.0).astype(np.float32)
                h = np.array(Image.fromarray(h).resize((OUT_SIZE, OUT_SIZE), resample=Image.NEAREST),
                             np.float32).flatten()
                raw.append(h); ok.append(True)
        valid[si] = ok; vi = np.where(ok)[0]
        if len(vi):
            ranked[si, vi] = _rankdata(np.stack([raw[i] for i in vi]),
                                       method='average', axis=1).astype(np.float32)
    # pre-normalize human ranks for fast cosine
    hn = ranked - ranked.mean(axis=2, keepdims=True)
    hn = hn / (np.linalg.norm(hn, axis=2, keepdims=True) + 1e-10)
    return hn, valid


def main():
    for task in ['social', 'describe']:
        done = os.path.join(OUT_DIR, f'molmo_allL_perimage_{task}.parquet')
        if os.path.exists(done):
            print(f'=== Molmo {task}: already done, skipping ===', flush=True)
            continue
        print(f'=== Molmo {task} ===', flush=True)
        act_dir = os.path.join(MOLMO_DIR, 'activations', task)
        grid = setup_grid(act_dir)
        avail = {f.replace('activations_img_', '').replace('.pt', '')
                 for f in os.listdir(act_dir)
                 if f.startswith('activations_img_') and f.endswith('.pt')}
        imgs = [im for im in target_imgs if im in avail]
        print(f'  images={len(imgs)}', flush=True)
        hn, valid = load_prerank_human(task, imgs)   # (S,N,D) normed, (S,N)
        S = hn.shape[0]
        # accumulate per (layer,head): per-subject z-sum for SEM; per-image mean_z
        perimg_rows = []
        # subject-level accumulation: sum of per-image z and counts, per (L,h,subject)
        subj_z_sum = np.zeros((N_LAYERS, N_HEADS, S), np.float64)
        subj_z_cnt = np.zeros((N_LAYERS, N_HEADS, S), np.int64)
        for ni, im in enumerate(tqdm(imgs, desc=f'  {task}')):
            maps = all_layer_head_maps(os.path.join(act_dir, f'activations_img_{im}.pt'), *grid)
            # rank+normalize all 784 model maps for this image (fast argsort ranks)
            mr = fast_rank(maps.reshape(N_LAYERS * N_HEADS, -1))
            mr = mr - mr.mean(axis=1, keepdims=True)
            mr = mr / (np.linalg.norm(mr, axis=1, keepdims=True) + 1e-10)  # (784, D)
            subj_mask = valid[:, ni]                       # (S,)
            if subj_mask.sum() == 0:
                continue
            Hsub = hn[subj_mask, ni]                        # (Ssub, D)
            R = mr @ Hsub.T                                  # (784, Ssub)
            R = np.clip(R, -0.999999, 0.999999)
            Z = np.arctanh(R)                                # (784, Ssub)
            zmean = Z.mean(axis=1).reshape(N_LAYERS, N_HEADS)
            for L in range(N_LAYERS):
                for h in range(N_HEADS):
                    perimg_rows.append((task, L + 1, h + 1, im,
                                        float(zmean[L, h]), float(np.tanh(zmean[L, h])),
                                        int(subj_mask.sum())))
            si_idx = np.where(subj_mask)[0]
            Zfull = Z.reshape(N_LAYERS, N_HEADS, -1)
            subj_z_sum[:, :, si_idx] += Zfull
            subj_z_cnt[:, :, si_idx] += 1
        # summary with subject-level SEM
        with np.errstate(invalid='ignore'):
            subj_mean = subj_z_sum / np.maximum(subj_z_cnt, 1)   # (L,h,S) mean z per subject
        sum_rows = []
        for L in range(N_LAYERS):
            for h in range(N_HEADS):
                zc = subj_mean[L, h][subj_z_cnt[L, h] > 0]
                if len(zc) == 0:
                    continue
                mr_ = float(np.tanh(zc.mean()))
                sem = float(zc.std(ddof=1) / np.sqrt(len(zc))) if len(zc) > 1 else 0.0
                sum_rows.append(dict(task=task, layer=L + 1, head=h + 1,
                                     mean_r=mr_, sem_r=sem, n_sub=len(zc)))
        pd.DataFrame(perimg_rows, columns=['task', 'layer', 'head', 'image_id',
                                           'mean_z', 'mean_r', 'n_sub']
                     ).to_parquet(os.path.join(OUT_DIR, f'molmo_allL_perimage_{task}.parquet'))
        sdf = pd.DataFrame(sum_rows)
        sdf.to_csv(os.path.join(OUT_DIR, f'molmo_allL_summary_{task}.csv'), index=False)
        best = sdf.loc[sdf['mean_r'].idxmax()]
        print(f"  best: L{int(best['layer'])}H{int(best['head'])} "
              f"r={float(best['mean_r']):.4f}", flush=True)


if __name__ == '__main__':
    main()
