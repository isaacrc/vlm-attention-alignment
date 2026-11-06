"""Primary analysis: which model's attention best predicts NSD brain responses?

Per subject, per model x task, fit a banded ridge on shared-1000 with two bands:
    attention band  (196)  - the model's spatial attention map, the variable of interest
    CLIP band       (768)  - ViT-L/14 embedding, a fixed "what" control

Bands, solver, and CV settings mirror tribe/scripts/3b_banded_encoding_model.py so the NSD
result is directly comparable to the existing TRIBE analysis. Differences: n=1000 instead
of 200, real fMRI instead of simulated, and 8 real subjects with a cross-subject ceiling.

Head selection (rebuttal Task 1): for Transformer models the attention band is the MEAN
over all 8 layer-3 heads, a predefined rule, not the best head. Pass --head-rule best to
reproduce the biased selection for comparison.

--head-rule cvbehav (ICLR revision, 2026-09-23) uses the behaviourally cross-validated
head for every model that has heads to choose from, per task. This is what `best` was
meant to be: `best` was pinned to layer 3 because only layer 3 had been extracted on the
NSD stimuli, but Task 1's sweep puts the Social-task maxima in layer 4, and Molmo's
Describe head at L8H7 rather than the in-sample argmax L10H27. Heads are still chosen on
fixation data and scored on fMRI, so nothing is selected on the data it is evaluated
against. Requires attn/{model}_{task}_alllayers.npz (extract_nsd_trained.py --all-layers)
and attn/Molmo_{task}_cvbehav.npz (build_molmo_cvbehav.py).

Also fits an attention-only and a CLIP-only model per condition so unique variance can be
partitioned without differencing two banded fits.

Outputs (encoding/):
  r2_{model}_{task}_subj{NN}.npz   per-vertex R2 for banded / attn-only / clip-only
  weights_{model}_{task}_subj{NN}.npz  refit-on-all weights, for the overlap analysis
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV, KernelRidgeCV
from himalaya.scoring import correlation_score, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
OUT = HERE / "encoding"
OUT.mkdir(exist_ok=True)

MODELS = ["CNN_LSTM", "ViT_LSTM", "CNN_Transformer", "ViT_Transformer", "Molmo", "Qwen35",
          # Baselines. An encoding model explains variance ACROSS images, so the
          # controls that matter here must vary per image (see
          # make_lowlevel_baselines.py). Luminance and EdgeEnergy are the real
          # competing low-level hypotheses; Shuffled is the noise floor; Center is
          # kept only as a pipeline sanity check and must come out at ~0, because a
          # map identical for every image cannot explain across-image variance.
          "Luminance", "EdgeEnergy", "Shuffled", "Center"]
TASKS = ["describe", "social"]
ALPHAS = np.logspace(1, 10, 10)

# Behaviourally cross-validated heads, 1-indexed (layer, head), from
# task1_head_selection/results/selected_heads.json. Qwen35's npz holds only the layer of
# its selected head, so only its head index is used; Molmo is served by a prebuilt cvbehav
# file. Qwen35 (camera-ready, 2026-10-20): thinking-off re-extraction, heads from
# neurips_final/final_figures/qwen35_nothink/results/selected_heads.json; npz built by
# neurips_final/qwen35_nothink/nsd_prepare.py. The thinking-on value was L15H12 both tasks.
CVBEHAV_HEADS = {
    "CNN_Transformer": {"social": (4, 4), "describe": (3, 5)},
    "ViT_Transformer": {"social": (4, 8), "describe": (3, 8)},
    "Qwen35":          {"social": (15, 12), "describe": (7, 9)},
}


def _align(ids, feats, order):
    """Identical indexing to the tail of load_attention, factored out."""
    lut = {int(i): k for k, i in enumerate(ids)}
    rows = np.array([lut.get(int(i), -1) for i in order])
    X = np.zeros((len(order), feats.shape[1]), dtype=np.float32)
    ok = rows >= 0
    X[ok] = feats[rows[ok]]
    return X, ok


def _load_cvbehav(model, task, order):
    """Behaviourally cross-validated head for the head-bearing models; `alphas` otherwise."""
    if model in ("CNN_Transformer", "ViT_Transformer"):
        path = HERE / "attn" / f"{model}_{task}_alllayers.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run extract_nsd_trained.py --all-layers first")
        d = np.load(path, allow_pickle=True)
        L, H = CVBEHAV_HEADS[model][task]
        feats = d["heads_all"][:, L - 1, H - 1, :]
    elif model == "Molmo":
        path = HERE / "attn" / f"Molmo_{task}_cvbehav.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run build_molmo_cvbehav.py first")
        d = np.load(path, allow_pickle=True)
        feats = d["alphas"]
    else:
        path = HERE / "attn" / f"{model}_{task}.npz"
        if not path.exists():
            return None
        d = np.load(path, allow_pickle=True)
        if model == "Qwen35" and "heads" in d:
            feats = d["heads"][:, CVBEHAV_HEADS[model][task][1] - 1, :]
        else:
            feats = d["alphas"]      # LSTMs and the baselines have a single map
    return _align(d["nsdId"], np.asarray(feats, dtype=np.float32), order)


def load_attention(model, task, order, head_rule="mean"):
    """Attention features aligned to `order` (the shared-1000 nsdId order)."""
    if head_rule == "cvbehav":
        return _load_cvbehav(model, task, order)
    path = HERE / "attn" / f"{model}_{task}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    ids = d["nsdId"]
    if head_rule == "best" and "heads" in d:
        # The paper's best heads, 0-indexed. Retained only for comparison against the
        # predefined mean-over-heads rule; these were selected on fixation alignment and
        # are the selection bias rebuttal Task 1 is about.
        best = {"CNN_Transformer": 4, "ViT_Transformer": 7,
                "Qwen35": 11,
                "Molmo": {"social": 23, "describe": 26}.get(task)}.get(model)
        feats = d["heads"][:, best, :] if best is not None else d["alphas"]
    elif head_rule == "mean" and "heads" in d:
        feats = d["heads"].mean(axis=1)
    else:
        feats = d["alphas"]
    lut = {int(i): k for k, i in enumerate(ids)}
    missing = [i for i in order if int(i) not in lut]
    if missing:
        print(f"  WARNING {model}/{task}: {len(missing)} images missing attention")
    rows = np.array([lut.get(int(i), -1) for i in order])
    X = np.zeros((len(order), feats.shape[1]), dtype=np.float32)
    ok = rows >= 0
    X[ok] = feats[rows[ok]]
    return X, ok


def _aggregate(folds, metric):
    """Combine per-fold scores into a per-vertex metric.

    metric='corr2' (default/final analysis): mean Pearson r across folds, then signed square
    (r**2 * sign(r)) -- a cross-validated squared correlation.
    metric='r2' (robustness supplement): mean of himalaya's r2_score (coefficient of
    determination) across folds, returned as-is. Variance partitioning is defined on true R2,
    so this checks whether the unique-variance ordering survives the metric choice.
    """
    s = np.mean(folds, axis=0)
    return s if metric == "r2" else s ** 2 * np.sign(s)


def _score(Y_te, pred, metric):
    fn = r2_score if metric == "r2" else correlation_score
    return np.nan_to_num(np.asarray(fn(Y_te, pred)), nan=0.0)


def fit_banded(X_attn, X_clip, Y, n_folds=5, seed=42, metric="corr2"):
    """Banded ridge with per-band alphas. Returns per-vertex metric and band weights."""
    n_a = X_attn.shape[1]
    X = np.concatenate([X_attn, X_clip], axis=1)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r_folds, d_folds = [], []
    for tr, te in kf.split(X):
        sa, sc = StandardScaler(), StandardScaler()
        Xtr = np.concatenate([sa.fit_transform(X[tr, :n_a]), sc.fit_transform(X[tr, n_a:])], axis=1)
        Xte = np.concatenate([sa.transform(X[te, :n_a]), sc.transform(X[te, n_a:])], axis=1)
        ck = ColumnKernelizer([
            ("attn", Kernelizer(kernel="linear"), slice(0, n_a)),
            ("clip", Kernelizer(kernel="linear"), slice(n_a, X.shape[1])),
        ])
        mkr = MultipleKernelRidgeCV(
            kernels="precomputed", solver="random_search",
            solver_params=dict(n_iter=20, alphas=ALPHAS),
            random_state=seed, cv=KFold(n_splits=4))
        mkr.fit(ck.fit_transform(Xtr), Y[tr])
        r_folds.append(_score(Y[te], mkr.predict(ck.transform(Xte)), metric))
        dm = mkr.deltas_.mean(axis=1)
        d_folds.append(np.asarray(dm.cpu() if hasattr(dm, "cpu") else dm))
    return _aggregate(r_folds, metric), np.mean(d_folds, axis=0)


def fit_single(X, Y, n_folds=5, seed=42, metric="corr2"):
    """Single-band ridge, for variance partitioning."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r_folds = []
    for tr, te in kf.split(X):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        kr = KernelRidgeCV(kernel="linear", alphas=ALPHAS, cv=KFold(n_splits=4))
        kr.fit(Xtr, Y[tr])
        r_folds.append(_score(Y[te], kr.predict(Xte), metric))
    return _aggregate(r_folds, metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--head-rule", default="mean", choices=["mean", "best", "cvbehav"])
    ap.add_argument("--metric", default="corr2", choices=["corr2", "r2"],
                    help="corr2 = final analysis (signed squared Pearson r); "
                         "r2 = robustness SUPPLEMENT (himalaya coefficient of determination), "
                         "written to encoding_r2score_supplement/ so it never mixes with final.")
    ap.add_argument("--ncsnr-thresh", type=float, default=0.2,
                    help="keep vertices whose mean ncsnr exceeds this")
    args = ap.parse_args()

    # The supplement metric writes to its own directory; the final `encoding/` set is untouched.
    out_dir = HERE / ("encoding_r2score_supplement" if args.metric == "r2" else "encoding")
    out_dir.mkdir(exist_ok=True)

    subj = args.subject
    resp = HERE / "responses"
    Y_full = np.load(resp / f"Y_shared_subj{subj:02d}.npy")
    order = np.load(resp / f"shared_nsdIds_subj{subj:02d}.npy")
    nc = np.load(resp / "noise_ceiling.npz")
    # `valid` drops the few vertices that are non-finite in any subject, so all subjects
    # are fit on an identical vertex set (see prepare_responses.py).
    keep = np.where((nc["ncsnr"] > args.ncsnr_thresh) & nc["valid"])[0]
    Y = Y_full[:, keep]
    assert np.isfinite(Y).all(), f"subj{subj}: non-finite values survived the vertex mask"
    print(f"subj{subj:02d}: Y {Y_full.shape} -> {Y.shape} "
          f"({len(keep)} vertices with ncsnr>{args.ncsnr_thresh}, finite in all subjects)")

    X_clip = np.load(HERE / "features" / "X_clip_nsd.npy")
    stim = pd.read_csv(HERE / "stimuli_index.csv")
    clip_lut = {int(i): k for k, i in enumerate(stim["nsdId"])}
    X_clip = X_clip[[clip_lut[int(i)] for i in order]]

    r2_clip = None
    for model in args.models:
        for task in args.tasks:
            out = out_dir / f"r2_{model}_{task}_subj{subj:02d}_{args.head_rule}.npz"
            if out.exists():
                print(f"  exists: {out.name}")
                continue
            loaded = load_attention(model, task, order, args.head_rule)
            if loaded is None:
                print(f"  missing attention: {model}/{task}")
                continue
            X_attn, ok = loaded
            if ok.sum() < len(order) * 0.9:
                print(f"  too few images for {model}/{task} ({ok.sum()}), skipping")
                continue

            t0 = time.time()
            r2_banded, deltas = fit_banded(X_attn, X_clip, Y, metric=args.metric)
            r2_attn = fit_single(X_attn, Y, metric=args.metric)
            if r2_clip is None:
                r2_clip = fit_single(X_clip, Y, metric=args.metric)
            np.savez(out, r2_banded=r2_banded, r2_attn=r2_attn, r2_clip=r2_clip,
                     deltas=deltas, vertices=keep, ceiling=nc["ceiling"][keep])
            print(f"  {model:16s} {task:8s} banded R2 mean={r2_banded.mean():.4f} "
                  f"top5%={np.percentile(r2_banded, 95):.4f} | attn-only={r2_attn.mean():.4f} "
                  f"| clip-only={r2_clip.mean():.4f} | band w={np.exp(deltas).round(2)} "
                  f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
