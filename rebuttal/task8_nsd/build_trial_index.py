"""Build the trial index: for each (subject, image) we need betas for, list the trials.

Covers two image sets:
  shared1000 - 1000 images x 8 subjects, the primary analysis set
  overlap106 - our 200 stimuli that appear in NSD, each seen by exactly 1 subject

Filters planned-but-never-scanned trials against real session counts (subjects 3/4/6/8
stopped early), so every trial listed here has data behind it.

Output: nsd_trial_index.csv  (subject, nsdId, cocoId, image_set, rep, trial, session, frame)
"""

from pathlib import Path

import pandas as pd

REPO = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment")
OUT = Path(__file__).parent
SESSIONS = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}
TPS = 750

nsd = pd.read_csv(REPO / "rebuttal" / "data" / "nsd_stim_info_merged.csv", index_col=0)
targets = pd.read_csv(REPO / "data" / "target_images.csv")
target_ids = set(targets["image_name"].str.replace(".jpg", "", regex=False).astype(int))

rows = []
for _, r in nsd.iterrows():
    coco = int(r["cocoId"])
    if r["shared1000"]:
        image_set = "shared1000"
    elif coco in target_ids:
        image_set = "overlap106"
    else:
        continue
    for s in range(1, 9):
        if r[f"subject{s}"] != 1:
            continue
        cap = SESSIONS[s] * TPS
        for k in range(3):
            trial = int(r[f"subject{s}_rep{k}"])
            if trial == 0 or trial > cap:
                continue
            ses, frame = divmod(trial - 1, TPS)
            rows.append({
                "subject": s, "nsdId": int(r["nsdId"]), "cocoId": coco,
                "image_set": image_set, "rep": k, "trial": trial,
                "session": ses + 1, "frame": frame,
            })

idx = pd.DataFrame(rows).sort_values(["subject", "session", "frame"]).reset_index(drop=True)
idx.to_csv(OUT / "nsd_trial_index.csv", index=False)

print(f"total trials to fetch: {len(idx)}")
print(idx.groupby("image_set").agg(trials=("trial", "size"), images=("nsdId", "nunique")).to_string())
print("\nby subject:")
print(
    idx.pivot_table(index="subject", columns="image_set", values="trial", aggfunc="size", fill_value=0).to_string()
)
mb = len(idx) * 2 * 10242 * 4 / 1e6
print(f"\nranged reads: {len(idx) * 2} (both hemis) -> {mb:.0f} MB")
