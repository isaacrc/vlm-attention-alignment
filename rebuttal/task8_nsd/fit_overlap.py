"""Exploratory analysis: do the 106 overlap images link fixation alignment to neural
predictivity?

These are the only images with BOTH our eye-tracking fixations and real NSD fMRI. Each was
seen by exactly one subject (98 have usable data), so this is a small, single-subject-per-
image test and is reported as exploratory, not as a primary result.

Design (avoids leakage): the encoding model is trained on that subject's shared-1000
images, which are disjoint from the overlap set, then applied frozen to their overlap
images. Prediction accuracy is scored per image.

Outputs: results/overlap_predictions.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from himalaya.kernel_ridge import (ColumnKernelizer, Kernelizer,
                                   KernelRidgeCV, MultipleKernelRidgeCV)
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
ALPHAS = np.logspace(1, 10, 10)

import sys
sys.path.insert(0, str(HERE))
from fit_encoding import MODELS, TASKS, load_attention  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="mean")
    ap.add_argument("--ncsnr-thresh", type=float, default=0.2)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    resp = HERE / "responses"
    nc = np.load(resp / "noise_ceiling.npz")
    keep = np.where((nc["ncsnr"] > args.ncsnr_thresh) & nc["valid"])[0]

    X_clip_all = np.load(HERE / "features" / "X_clip_nsd.npy")
    stim = pd.read_csv(HERE / "stimuli_index.csv")
    clip_lut = {int(i): k for k, i in enumerate(stim["nsdId"])}

    rows = []
    for subj in args.subjects:
        tr_ids = np.load(resp / f"shared_nsdIds_subj{subj:02d}.npy")
        te_path = resp / f"overlap_nsdIds_subj{subj:02d}.npy"
        if not te_path.exists():
            continue
        te_ids = np.load(te_path)
        Y_tr = np.load(resp / f"Y_shared_subj{subj:02d}.npy")[:, keep]
        Y_te = np.load(resp / f"Y_overlap_subj{subj:02d}.npy")[:, keep]
        Ctr = X_clip_all[[clip_lut[int(i)] for i in tr_ids]]
        Cte = X_clip_all[[clip_lut[int(i)] for i in te_ids]]
        print(f"subj{subj:02d}: train {Y_tr.shape} -> test {Y_te.shape}")
        clip_only_pred = None   # shared across models within a subject

        for model in args.models:
            for task in TASKS:
                a_tr = load_attention(model, task, tr_ids, args.head_rule)
                a_te = load_attention(model, task, te_ids, args.head_rule)
                if a_tr is None or a_te is None:
                    continue
                Atr, ok_tr = a_tr
                Ate, ok_te = a_te
                if not ok_tr.all() or not ok_te.all():
                    print(f"  {model}/{task}: incomplete attention, skipping")
                    continue

                n_a = Atr.shape[1]
                sa, sc = StandardScaler(), StandardScaler()
                Xtr = np.concatenate([sa.fit_transform(Atr), sc.fit_transform(Ctr)], axis=1)
                Xte = np.concatenate([sa.transform(Ate), sc.transform(Cte)], axis=1)
                ck = ColumnKernelizer([
                    ("attn", Kernelizer(kernel="linear"), slice(0, n_a)),
                    ("clip", Kernelizer(kernel="linear"), slice(n_a, Xtr.shape[1])),
                ])
                mkr = MultipleKernelRidgeCV(
                    kernels="precomputed", solver="random_search",
                    solver_params=dict(n_iter=20, alphas=ALPHAS),
                    random_state=42, cv=4)
                mkr.fit(ck.fit_transform(Xtr), Y_tr)
                pred = np.asarray(mkr.predict(ck.transform(Xte)))

                # The banded prediction is ~95% CLIP: a scrambled attention map scores
                # almost as well as a real one, so r_pred alone cannot detect a link
                # between fixation alignment and neural predictivity. Score an
                # ATTENTION-ONLY model too -- that is the quantity that actually varies
                # with the attention map. CLIP-only is fit once per subject as the
                # reference both share.
                kr_a = KernelRidgeCV(kernel="linear", alphas=ALPHAS, cv=4)
                kr_a.fit(sa.transform(Atr), Y_tr)
                pred_a = np.asarray(kr_a.predict(sa.transform(Ate)))
                if clip_only_pred is None:
                    kr_c = KernelRidgeCV(kernel="linear", alphas=ALPHAS, cv=4)
                    kr_c.fit(sc.transform(Ctr), Y_tr)
                    clip_only_pred = np.asarray(kr_c.predict(sc.transform(Cte)))

                def _r(p_vec, o_vec):
                    a_ = p_vec - p_vec.mean()
                    b_ = o_vec - o_vec.mean()
                    d_ = np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum())
                    return float((a_ * b_).sum() / d_) if d_ > 0 else np.nan

                # per-image correlation across vertices between predicted and observed
                for k, nsd_id in enumerate(te_ids):
                    r_band = _r(pred[k], Y_te[k])
                    r_clip = _r(clip_only_pred[k], Y_te[k])
                    rows.append({
                        "subject": subj, "nsdId": int(nsd_id),
                        "cocoId": int(stim.loc[stim["nsdId"] == nsd_id, "cocoId"].iloc[0]),
                        "model": model, "task": task,
                        "r_pred": r_band,
                        "r_attn_only": _r(pred_a[k], Y_te[k]),
                        "r_clip_only": r_clip,
                        "r_gain": r_band - r_clip,   # attention's per-image increment
                    })
                recent = rows[-len(te_ids):]
                print(f"  {model:16s} {task:8s} banded r={np.mean([r['r_pred'] for r in recent]):.4f} "
                      f"attn-only r={np.nanmean([r['r_attn_only'] for r in recent]):.4f} "
                      f"gain={np.nanmean([r['r_gain'] for r in recent]):+.4f}")

    if rows:
        df = pd.DataFrame(rows)
        tag = "_".join(str(s) for s in args.subjects) if len(args.subjects) < 8 else "all"
        df.to_csv(OUT / f"overlap_predictions_{args.head_rule}_subj{tag}.csv", index=False)
        print("\nmean per-image prediction r by model x task:")
        print(df.groupby(["task", "model"])["r_pred"].agg(["mean", "sem", "size"]).to_string())


if __name__ == "__main__":
    main()
