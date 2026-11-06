"""Model comparison split by cortical stream.

No refitting: the per-subject encoding results already store per-vertex R2 together with
the vertex indices they were fit on, so the ROI split is a regrouping of existing output.

Why this matters. Pooling over all high-SNR vertices mixes early visual cortex -- where
responses are driven by local contrast, and where a Sobel edge map is a strong predictor
almost by construction -- with ventral/lateral cortex, where higher-order scene content
lives. A model whose attention merely tracks contrast will look good in the pooled number.
Splitting by stream asks the sharper question: does model attention add anything BEYOND
low-level image structure in the areas where semantic content is represented?

Read the EdgeEnergy row as the reference line in every ROI. A model that does not beat it
in a given stream has not explained anything there that contrast does not.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
ENC = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/encoding_norm')
OUT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/results_norm')
OUT.mkdir(exist_ok=True)
ROI_ORDER = ["grp_early", "grp_ventral", "grp_lateral", "grp_parietal", "nsdgeneral"]
ROI_LABEL = {"grp_early": "early", "grp_ventral": "ventral", "grp_lateral": "lateral",
             "grp_parietal": "parietal", "nsdgeneral": "nsdgeneral"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="mean")
    ap.add_argument("--min-vertices", type=int, default=25)
    args = ap.parse_args()

    masks = np.load(HERE / "rois" / "roi_masks_fs5.npz")
    rows = []
    for f in sorted(ENC.glob(f"r2_*_{args.head_rule}.npz")):
        stem = f.stem[len("r2_"):-len(f"_{args.head_rule}")]
        parts = stem.rsplit("_", 2)
        # Task-independent fits (r2_ShufCLIP_subjNN_*, fit_clip_floor.py) and banded
        # CLIP-floor fits (r2_bandedShufCLIP_*) are not encoding models of a task; skip them.
        if len(parts) != 3 or parts[1] not in ("social", "describe") or stem.startswith("bandedShufCLIP"):
            continue
        model, task, subj = parts
        d = np.load(f)
        verts = d["vertices"]                       # indices into the 20484-vertex space
        unique = d["r2_banded"] - d["r2_clip"]
        for roi in ROI_ORDER:
            sel = masks[roi][verts]
            if sel.sum() < args.min_vertices:
                continue
            rows.append({
                "roi": ROI_LABEL[roi], "model": model, "task": task,
                "subject": int(subj.replace("subj", "")),
                "n_vertices": int(sel.sum()),
                "attn_unique": float(unique[sel].mean()),
                "r2_attn": float(d["r2_attn"][sel].mean()),
                "r2_clip": float(d["r2_clip"][sel].mean()),
                "r2_banded": float(d["r2_banded"][sel].mean()),
                "ceiling": float(d["ceiling"][sel].mean()),
            })

    if not rows:
        print("no encoding results yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"encoding_by_roi_per_subject_{args.head_rule}.csv", index=False)

    grp = (df.groupby(["roi", "task", "model"])
             .agg(n=("subject", "size"),
                  attn_unique=("attn_unique", "mean"),
                  sem_=("attn_unique", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
                  r2_attn=("r2_attn", "mean"),
                  ceiling=("ceiling", "mean"))
             .reset_index())
    grp.to_csv(OUT / f"encoding_by_roi_{args.head_rule}.csv", index=False)

    for roi in [ROI_LABEL[r] for r in ROI_ORDER]:
        sub = grp[grp.roi == roi]
        if sub.empty:
            continue
        print(f"\n=== {roi} ===")
        for task in ("describe", "social"):
            s = sub[sub.task == task].sort_values("attn_unique", ascending=False)
            if s.empty:
                continue
            print(f"  [{task}]")
            for _, r in s.iterrows():
                star = "  <- low-level baseline" if r.model in ("EdgeEnergy", "Luminance") else ""
                sem = f"+-{r['sem_']:.4f}" if np.isfinite(r["sem_"]) else "       "
                print(f"    {r.model:<16} {r.attn_unique:+.4f} {sem}  (n={int(r.n)}){star}")


if __name__ == "__main__":
    main()
