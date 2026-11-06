"""
saliency_metrics.py — standard saliency-evaluation metrics (Bylinskii et al. 2019),
self-contained (numpy/scipy only). All take model map P and human map/fixations G,
both 2D float arrays same shape. Higher = better except KL (lower better).
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def _norm01(x):
    x = x - x.min(); m = x.max()
    return x / (m + 1e-12)


def _prob(x):
    x = np.clip(x, 0, None); s = x.sum()
    return x / (s + 1e-12)


def cc(P, G):
    """Pearson correlation coefficient."""
    p = (P - P.mean()); g = (G - G.mean())
    return float((p * g).sum() / (np.sqrt((p**2).sum() * (g**2).sum()) + 1e-12))


def sim(P, G):
    """Histogram intersection of the two prob maps."""
    p, g = _prob(P), _prob(G)
    return float(np.minimum(p, g).sum())


def kl(P, G):
    """KL(G || P) — how well P approximates the human distribution G."""
    p, g = _prob(P), _prob(G)
    return float((g * np.log(g / (p + 1e-12) + 1e-12)).sum())


def nss(P, fix_binary):
    """Normalized Scanpath Saliency at fixated locations (binary map)."""
    z = (P - P.mean()) / (P.std() + 1e-12)
    f = fix_binary > 0
    return float(z[f].mean()) if f.sum() else np.nan


def auc_judd(P, fix_binary):
    """AUC-Judd."""
    P = _norm01(P).ravel(); f = (fix_binary.ravel() > 0)
    n_fix = f.sum()
    if n_fix == 0:
        return np.nan
    thresholds = np.sort(P[f])[::-1]
    tp = np.zeros(len(thresholds) + 2); fp = np.zeros(len(thresholds) + 2)
    tp[0] = fp[0] = 0; tp[-1] = fp[-1] = 1
    n_pix = len(P)
    for i, th in enumerate(thresholds):
        above = (P >= th).sum()
        tp[i + 1] = (i + 1) / n_fix
        fp[i + 1] = (above - (i + 1)) / (n_pix - n_fix)
    return float(np.trapz(tp, fp))


def auc_shuffled(P, fix_binary, other_fix, n_splits=10):
    """Shuffled AUC (sAUC): negatives drawn from other images' fixations."""
    P = _norm01(P).ravel(); f = (fix_binary.ravel() > 0); o = (other_fix.ravel() > 0)
    Sf = P[f]; So = P[o]
    if len(Sf) == 0 or len(So) == 0:
        return np.nan
    aucs = []
    rng = np.random.default_rng(0)
    for _ in range(n_splits):
        neg = rng.choice(So, size=len(Sf), replace=len(So) < len(Sf))
        alls = np.concatenate([Sf, neg])
        th = np.sort(alls)[::-1]
        tp = [0]; fp = [0]
        for t in th:
            tp.append((Sf >= t).mean()); fp.append((neg >= t).mean())
        tp.append(1); fp.append(1)
        aucs.append(np.trapz(tp, fp))
    return float(np.mean(aucs))


def info_gain(P, fix_binary, baseline):
    """Information gain of P over a baseline (center-bias) at fixations, in bits."""
    p = _prob(P).ravel(); b = _prob(baseline).ravel(); f = (fix_binary.ravel() > 0)
    if f.sum() == 0:
        return np.nan
    return float((np.log2(p[f] + 1e-12) - np.log2(b[f] + 1e-12)).mean())


def entropy_norm(P):
    """Normalized Shannon entropy of the map as a distribution (1=uniform,0=peak)."""
    p = _prob(P).ravel(); p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(_prob(P).ravel())))


def fixation_binary(human_map, top_frac=0.05):
    """Binarize a continuous human heatmap to pseudo-fixation points (top fraction)."""
    thr = np.quantile(human_map, 1 - top_frac)
    return (human_map >= thr).astype(np.float32)
