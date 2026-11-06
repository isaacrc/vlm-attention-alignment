"""Establish the NSD design facts needed to scope Option A."""

from pathlib import Path

import pandas as pd

REPO = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment")
nsd = pd.read_csv(REPO / "rebuttal" / "data" / "nsd_stim_info_merged.csv", index_col=0)
SESSIONS = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}

print("=== global ===")
print(f"total NSD images: {len(nsd)}")
print(nsd["cocoSplit"].value_counts().to_string())
print(f"shared1000: {int(nsd['shared1000'].sum())}")
print("shared1000 by coco split:")
print(nsd[nsd["shared1000"]]["cocoSplit"].value_counts().to_string())

print("\n=== how many subjects saw each image ===")
subj_cols = [f"subject{s}" for s in range(1, 9)]
n_sub = nsd[subj_cols].sum(axis=1)
print(pd.crosstab(n_sub, nsd["shared1000"]).to_string())

print("\n=== per subject: unique vs shared, and val2017 content ===")
for s in range(1, 9):
    seen = nsd[nsd[f"subject{s}"] == 1]
    shared = int(seen["shared1000"].sum())
    val = int((seen["cocoSplit"] == "val2017").sum())
    cap = SESSIONS[s] * 750
    cols = seen[[f"subject{s}_rep{k}" for k in range(3)]].to_numpy()
    scanned = ((cols != 0) & (cols <= cap)).sum(axis=1)
    sh = seen["shared1000"].to_numpy()
    print(
        f"subj{s}: {len(seen)} assigned = {shared} shared + {len(seen)-shared} unique | "
        f"val2017 {val} | >=1 rep: {(scanned>0).sum()} "
        f"(shared {(scanned[sh]>0).sum()}/1000, unique {(scanned[~sh]>0).sum()}) | "
        f"3 reps: {(scanned==3).sum()}"
    )

print("\n=== our 200 stimuli ===")
targets = pd.read_csv(REPO / "data" / "target_images.csv")
targets["cocoId"] = targets["image_name"].str.replace(".jpg", "", regex=False).astype(int)
inn = nsd[nsd["cocoId"].isin(targets["cocoId"])]
print(f"in NSD: {len(inn)} | in shared1000: {int(inn['shared1000'].sum())}")
print(f"all our stimuli are cocoSplit: {sorted(targets['cocoId'].isin(nsd['cocoId']).unique())}")
print(inn["cocoSplit"].value_counts().to_string())
