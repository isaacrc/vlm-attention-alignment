"""Exploratory: does fixation alignment track neural predictivity, image by image?

The 106 overlap stimuli (98 with usable data) are the only images with BOTH our eye-tracking
fixations and real NSD fMRI, so they are the only place this question can be asked directly.

Two quantities per image:
  r_pred  - how well that subject's frozen shared-1000 encoding model predicts the NSD
            response to that image (from fit_overlap.py). Shared-1000 and the overlap set
            are disjoint, so there is no leakage.
  fix_z   - how well the same model's attention map matches HUMAN FIXATIONS on that image,
            Fisher-z Spearman averaged over our eye-tracking participants
            (group_analysis/processed_correlations/all_models_all_tasks_correlations_final.csv).

If the two covary, fixation alignment and neural predictivity are measuring related things.
If they do not, they are dissociated -- which is the claim the paper currently makes from
simulated responses, and which this tests on real fMRI.

Caveats, to be stated wherever this is reported: each overlap image was seen by exactly one
NSD subject, n is ~98 image-model pairs per model, and the NSD task (continuous recognition
under central fixation) differs from both of our eye-tracking tasks. Exploratory only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
RES = HERE / "results"
FIX = Path("/scratch/gpfs/GRAZIANO/CANS/group_analysis/processed_correlations/"
           "all_models_all_tasks_correlations_final.csv")

# model naming differs between the two pipelines
FIX_NAME = {"CNN_LSTM": "Gated", "ViT_LSTM": "ViT_LSTM",
            "CNN_Transformer": "CNN_Transformer", "ViT_Transformer": "ViT_Transformer",
            "Molmo": "Molmo", "Qwen35": "Qwen"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="mean")
    args = ap.parse_args()

    files = sorted(RES.glob(f"overlap_predictions_{args.head_rule}_subj*.csv"))
    if not files:
        print("no overlap results yet")
        return
    pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pred = pred.dropna(subset=["r_pred"])
    print(f"{len(files)} subject files | {len(pred)} image x model x task predictions "
          f"| {pred.nsdId.nunique()} unique images")

    grp = (pred.groupby(["task", "model"])["r_pred"]
           .agg(["mean", "sem", "size"]).reset_index()
           .sort_values(["task", "mean"], ascending=[True, False]))
    grp.to_csv(RES / f"overlap_group_{args.head_rule}.csv", index=False)
    print("\nneural prediction accuracy on the overlap images (per-image r):")
    print(grp.to_string(index=False))

    if not FIX.exists():
        print(f"\nfixation alignment file not found: {FIX}")
        return

    fix = pd.read_csv(FIX)
    fix["cocoId"] = fix["image_id"].str.replace(".jpg", "", regex=False).astype(int)
    # average over eye-tracking participants -> one alignment value per image x model x task
    fix_img = (fix.groupby(["cocoId", "model", "task"])["spearman_z"]
               .mean().reset_index().rename(columns={"spearman_z": "fix_z"}))

    rows = []
    for (task, model), g in pred.groupby(["task", "model"]):
        fname = FIX_NAME.get(model)
        if fname is None:
            continue
        f = fix_img[(fix_img.model == fname) & (fix_img.task == task)]
        m = g.merge(f, on="cocoId", how="inner")
        if len(m) < 10:
            continue
        rec = {"task": task, "model": model, "n_images": len(m),
               "mean_r_pred": m["r_pred"].mean(),
               "mean_attn_only": m["r_attn_only"].mean(),
               "mean_gain": m["r_gain"].mean(),
               "mean_fix_z": m["fix_z"].mean()}
        # r_pred is ~95% CLIP and cannot detect a link; attn_only and gain can.
        for col, tag in (("r_pred", "banded"), ("r_attn_only", "attn"), ("r_gain", "gain")):
            if col in m:
                rr, pp = stats.spearmanr(m[col], m["fix_z"])
                rec[f"rho_{tag}"] = rr
                rec[f"p_{tag}"] = pp
        rows.append(rec)

    if not rows:
        print("\nno model matched between the two pipelines -- check FIX_NAME mapping")
        print(f"fixation file models: {sorted(fix['model'].unique())}")
        return

    link = pd.DataFrame(rows).sort_values(["task", "model"])
    link.to_csv(RES / f"overlap_fixation_link_{args.head_rule}.csv", index=False)
    print("\nimage-wise link between fixation alignment and neural predictivity:")
    print(link.to_string(index=False))

    # pooled across models, within task, since per-model n is small
    print("\npooled across models, using attention-only prediction "
          "(z-scored within model x task):")
    for task, g in pred.groupby("task"):
        parts = []
        for model, gg in g.groupby("model"):
            fname = FIX_NAME.get(model)
            if fname is None:
                continue
            f = fix_img[(fix_img.model == fname) & (fix_img.task == task)]
            m = gg.merge(f, on="cocoId", how="inner")
            if len(m) < 10:
                continue
            col = "r_attn_only" if "r_attn_only" in m else "r_pred"
            m = m.assign(zp=stats.zscore(m[col]), zf=stats.zscore(m["fix_z"]))
            parts.append(m[["zp", "zf"]])
        if parts:
            allm = pd.concat(parts)
            r, p = stats.spearmanr(allm["zp"], allm["zf"])
            print(f"  {task}: rho={r:+.3f}  p={p:.4f}  (n={len(allm)} image-model pairs)")


if __name__ == "__main__":
    main()
