"""Task 8 Option A, step 1: overlap between the 200 stimuli and the NSD image set.

Inputs
  data/target_images.csv                     - 200 stimulus filenames (COCO 2017 val)
  rebuttal/data/nsd_stim_info_merged.csv     - NSD stimulus table (73k images, cocoId/cocoSplit/shared1000)

Outputs
  rebuttal/task8_nsd/nsd_overlap.csv         - per-stimulus overlap table
  rebuttal/task8_nsd/nsd_overlap_summary.json
"""

import json
from pathlib import Path

import pandas as pd

REPO = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment")
OUT = REPO / "rebuttal" / "task8_nsd"
OUT.mkdir(parents=True, exist_ok=True)

targets = pd.read_csv(REPO / "data" / "target_images.csv")
targets["cocoId"] = targets["image_name"].str.replace(".jpg", "", regex=False).astype(int)

nsd = pd.read_csv(REPO / "rebuttal" / "data" / "nsd_stim_info_merged.csv", index_col=0)

subj_cols = [f"subject{s}" for s in range(1, 9)]
merged = targets.merge(
    nsd[["cocoId", "cocoSplit", "nsdId", "shared1000", "flagged", "BOLD5000"] + subj_cols],
    on="cocoId",
    how="left",
)
merged["in_nsd"] = merged["nsdId"].notna()
merged["n_subjects"] = merged[subj_cols].sum(axis=1)

merged.to_csv(OUT / "nsd_overlap.csv", index=False)

n_total = len(merged)
n_in = int(merged["in_nsd"].sum())
in_nsd = merged[merged["in_nsd"]]
n_shared = int(in_nsd["shared1000"].sum()) if n_in else 0

summary = {
    "n_stimuli": n_total,
    "n_in_nsd": n_in,
    "pct_in_nsd": round(100 * n_in / n_total, 1),
    "n_in_nsd_shared1000": n_shared,
    "n_in_nsd_subject_specific": n_in - n_shared,
    "n_flagged": int(in_nsd["flagged"].sum()) if n_in else 0,
    "subjects_per_overlapping_image": (
        in_nsd["n_subjects"].value_counts().sort_index().to_dict() if n_in else {}
    ),
    "n_seen_by_all_8": int((in_nsd["n_subjects"] == 8).sum()) if n_in else 0,
    "nsd_total_images": len(nsd),
    "nsd_val2017_images": int((nsd["cocoSplit"] == "val2017").sum()),
}
summary["subjects_per_overlapping_image"] = {
    int(k): int(v) for k, v in summary["subjects_per_overlapping_image"].items()
}

with open(OUT / "nsd_overlap_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
if n_in:
    print("\nOverlapping images:")
    print(
        in_nsd[["image_name", "nsdId", "shared1000", "n_subjects"]]
        .sort_values("nsdId")
        .to_string(index=False)
    )
