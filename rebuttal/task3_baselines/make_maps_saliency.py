"""
Tier-B, part 2 -- classic bottom-up saliency.

pysaliency's Itti-Koch and GBVS are thin wrappers around the authors' MATLAB code,
and neither MATLAB nor Octave is installed on Della, so these are reimplemented
directly from the papers:

  itti_koch          Itti, Koch & Niebur (1998), PAMI 20(11)
  gbvs               Harel, Koch & Perona (2006), NIPS -- graph-based activation
                     and normalization via the principal eigenvector of a Markov
                     chain, run over the same feature maps
  spectral_residual  Hou & Zhang (2007), CVPR

These are our own implementations, not the reference code, and the writeup says so.
They are here to answer "does it beat classic bottom-up saliency?"; DeepGaze IIE
(make_maps_deepgaze.py) is the strong modern reference.

Outputs: data/tierB_maps/{name}.npz
"""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
import tierB_common as T
import alignment_core as A

BASE = 512          # level-0 side; level L has side BASE >> L
CS_C = [2, 3, 4]    # center levels
CS_D = [3, 4]       # center-surround offsets
OUT_LEVEL = 4       # conspicuity maps live at level 4 (32x32)
THETAS = [0, 45, 90, 135]


# ── Itti-Koch ─────────────────────────────────────────────────────────────────
def pyramid(img, n=9):
    p = [img.astype(np.float32)]
    for _ in range(n - 1):
        p.append(cv2.pyrDown(p[-1]))
    return p


def cs_diff(pyr, c, s):
    """Across-scale subtraction |pyr(c) - pyr(s)|, interpolated to level c."""
    a = pyr[c]
    b = cv2.resize(pyr[s], (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.abs(a - b)


def norm_op(m):
    """Itti's N(.): scale to [0,1], then multiply by (global_max - mean_other_max)^2.
    Local maxima are found on a coarse grid, as in the original."""
    m = m - m.min()
    mx = m.max()
    if mx < 1e-9:
        return m
    m = m / mx
    # local maxima above threshold, sampled over a 10x10 block grid
    h, w = m.shape
    bh, bw = max(h // 10, 1), max(w // 10, 1)
    maxima = []
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            blk = m[i:i + bh, j:j + bw]
            if blk.size:
                maxima.append(blk.max())
    maxima = np.array(maxima)
    gmax = m.max()
    others = maxima[maxima < gmax - 1e-6]
    mbar = others.mean() if len(others) else 0.0
    return m * (gmax - mbar) ** 2


def across_scale_add(maps, level=OUT_LEVEL):
    tgt = (BASE >> level, BASE >> level)
    out = np.zeros(tgt, np.float32)
    for m in maps:
        out += cv2.resize(norm_op(m), (tgt[1], tgt[0]), interpolation=cv2.INTER_LINEAR)
    return out


def itti_features(rgb):
    """Returns (intensity, color, orientation) conspicuity maps at OUT_LEVEL,
    plus the raw feature-map list that GBVS reuses."""
    im = cv2.resize(rgb.astype(np.float32), (BASE, BASE), interpolation=cv2.INTER_AREA)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    I = (r + g + b) / 3.0

    # broadly-tuned color channels, decorrelated from intensity (Itti eq. 1-4)
    Imax = I.max()
    mask = I > 0.1 * Imax
    denom = np.where(mask, I, np.inf)
    R = np.clip(r - (g + b) / 2, 0, None) / denom
    G = np.clip(g - (r + b) / 2, 0, None) / denom
    B = np.clip(b - (r + g) / 2, 0, None) / denom
    Y = np.clip((r + g) / 2 - np.abs(r - g) / 2 - b, 0, None) / denom

    pI = pyramid(I)
    pR, pG, pB, pY = pyramid(R), pyramid(G), pyramid(B), pyramid(Y)
    pO = {}
    for th in THETAS:
        k = cv2.getGaborKernel((11, 11), 2.0, np.deg2rad(th), 6.0, 0.5, 0, cv2.CV_32F)
        pO[th] = [cv2.filter2D(x, cv2.CV_32F, k) for x in pI]

    feats_I, feats_C, feats_O = [], [], []
    for c in CS_C:
        for d in CS_D:
            s = c + d
            feats_I.append(cs_diff(pI, c, s))
            # double-opponent color (Itti eq. 6-7)
            rg_c = pR[c] - pG[c]
            rg_s = cv2.resize(pG[s] - pR[s], rg_c.shape[::-1], interpolation=cv2.INTER_LINEAR)
            feats_C.append(np.abs(rg_c - rg_s))
            by_c = pB[c] - pY[c]
            by_s = cv2.resize(pY[s] - pB[s], by_c.shape[::-1], interpolation=cv2.INTER_LINEAR)
            feats_C.append(np.abs(by_c - by_s))
            for th in THETAS:
                feats_O.append(cs_diff(pO[th], c, s))

    cI = across_scale_add(feats_I)
    cC = across_scale_add(feats_C)
    cO = across_scale_add(feats_O)
    return cI, cC, cO, (feats_I, feats_C, feats_O)


def itti_koch(rgb):
    cI, cC, cO, _ = itti_features(rgb)
    S = (norm_op(cI) + norm_op(cC) + norm_op(cO)) / 3.0
    return T.normalize(cv2.resize(S, (T.SIZE, T.SIZE), interpolation=cv2.INTER_CUBIC))


# ── GBVS ──────────────────────────────────────────────────────────────────────
def _spatial_falloff(n, sigma_frac=0.15):
    """F(i,j) = exp(-|pos_i - pos_j|^2 / (2 sigma^2)) over an n x n lattice."""
    idx = np.arange(n * n)
    yy, xx = np.divmod(idx, n)
    d2 = (yy[:, None] - yy[None, :]) ** 2 + (xx[:, None] - xx[None, :]) ** 2
    sigma = sigma_frac * n
    return np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)


def _markov_equilibrium(W):
    """Principal eigenvector of the column-normalized Markov matrix, by power
    iteration. W[i,j] = weight of edge j -> i."""
    W = W / (W.sum(axis=0, keepdims=True) + 1e-12)
    v = np.ones(W.shape[0], np.float32) / W.shape[0]
    for _ in range(60):
        v_new = W @ v
        s = v_new.sum()
        if s < 1e-12:
            break
        v_new /= s
        if np.abs(v_new - v).sum() < 1e-9:
            v = v_new
            break
        v = v_new
    return v


def gbvs(rgb, n=32, F=None):
    """Graph-based activation + normalization over the Itti feature maps."""
    _, _, _, (fI, fC, fO) = itti_features(rgb)
    if F is None:
        F = _spatial_falloff(n)

    def activate(M):
        M = cv2.resize(M, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
        M = M - M.min()
        M = M / (M.max() + 1e-12) + 1e-3          # keep the log ratio finite
        f = M.ravel()
        # dissimilarity d(i,j) = |log(f_i / f_j)|, weighted by spatial proximity
        L = np.log(f)
        W = np.abs(L[:, None] - L[None, :]) * F
        return _markov_equilibrium(W).reshape(n, n)

    def normalize_graph(A_):
        a = A_.ravel()
        # Mass must flow toward high-activation nodes, so the weight of edge j->i
        # carries the activation of the DESTINATION i: w(j->i) = A(i) * F(i,j).
        # Putting it on the source (a[None, :]) makes it cancel exactly under the
        # column normalization in _markov_equilibrium, leaving a map that depends
        # only on F -- identical for every image.
        W = a[:, None] * F
        return _markov_equilibrium(W).reshape(n, n)

    total = np.zeros((n, n), np.float32)
    for group in (fI, fC, fO):
        acc = np.zeros((n, n), np.float32)
        for M in group:
            acc += normalize_graph(activate(M))
        total += acc / max(len(group), 1)
    return T.normalize(cv2.resize(total, (T.SIZE, T.SIZE), interpolation=cv2.INTER_CUBIC))


# ── Spectral residual ─────────────────────────────────────────────────────────
def spectral_residual(rgb, work=64):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, (work, work), interpolation=cv2.INTER_AREA)
    F = np.fft.fft2(g)
    logA = np.log(np.abs(F) + 1e-9)
    phase = np.angle(F)
    avg = cv2.blur(logA, (3, 3))
    sr = logA - avg
    S = np.abs(np.fft.ifft2(np.exp(sr + 1j * phase))) ** 2
    S = cv2.GaussianBlur(S, (0, 0), 2.5)
    return T.normalize(cv2.resize(S, (T.SIZE, T.SIZE), interpolation=cv2.INTER_CUBIC))


def main():
    imgs = A.target_images()
    F = _spatial_falloff(32)
    out = {'itti_koch': {}, 'gbvs': {}, 'spectral_residual': {}}
    for i, name in enumerate(imgs):
        rgb = T.load_rgb(name)
        out['itti_koch'][name] = itti_koch(rgb)
        out['gbvs'][name] = gbvs(rgb, F=F)
        out['spectral_residual'][name] = spectral_residual(rgb)
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(imgs)}', flush=True)
    for k, v in out.items():
        arr = np.stack(list(v.values()))
        print(f'{k}: mean={arr.mean():.4f} std={arr.std():.4f} '
              f'finite={np.isfinite(arr).all()}', flush=True)
        T.save_maps(k, v)


if __name__ == '__main__':
    main()
