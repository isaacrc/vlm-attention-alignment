"""nsd_stim_info_merged encodes the FULL PLANNED design (30000 trials/subject = 40
sessions x 750). Subjects 3, 4, 6, 8 stopped early, so trials with an id beyond
750 * n_sessions_completed were never actually scanned. Recount the usable reps for
the 106 overlapping images under the real session counts.

Session counts are the published NSD values; verified against max trial id below.
"""

from pathlib import Path

import pandas as pd

REPO = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment")
OUT = REPO / "rebuttal" / "task8_nsd"

SESSIONS = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}
TRIALS_PER_SESSION = 750

nsd = pd.read_csv(REPO / "rebuttal" / "data" / "nsd_stim_info_merged.csv", index_col=0)
sub = nsd.set_index("cocoId")
ov = pd.read_csv(OUT / "nsd_overlap.csv")
ov = ov[ov["in_nsd"]]

rows = []
for coco in ov["cocoId"]:
    r = sub.loc[coco]
    for s in range(1, 9):
        if r[f"subject{s}"] != 1:
            continue
        trials = [r[f"subject{s}_rep{k}"] for k in range(3)]
        planned = [t for t in trials if t != 0]
        usable = [t for t in planned if t <= SESSIONS[s] * TRIALS_PER_SESSION]
        rows.append({"cocoId": coco, "nsdId": int(r["nsdId"]), "subject": s,
                     "n_reps_planned": len(planned), "n_reps_scanned": len(usable)})

det = pd.DataFrame(rows)
det.to_csv(OUT / "nsd_overlap_usable_reps.csv", index=False)

print("reps actually scanned, per overlapping image:")
print(det["n_reps_scanned"].value_counts().sort_index().to_string())
print(f"\nimages with >=1 scanned rep: {(det['n_reps_scanned'] > 0).sum()} / {len(det)}")
print(f"images with 3 scanned reps:  {(det['n_reps_scanned'] == 3).sum()} / {len(det)}")
print("\nby subject (assigned -> >=1 scanned rep):")
print(
    det.groupby("subject")
    .agg(assigned=("cocoId", "size"), with_data=("n_reps_scanned", lambda x: (x > 0).sum()))
    .to_string()
)

# Same recount for the whole NSD design, as context for the full-scale option.
print("\n--- full NSD, images with >=1 scanned rep per subject ---")
for s in range(1, 9):
    seen = nsd[nsd[f"subject{s}"] == 1]
    cap = SESSIONS[s] * TRIALS_PER_SESSION
    cols = seen[[f"subject{s}_rep{k}" for k in range(3)]].to_numpy()
    any_scanned = (((cols != 0) & (cols <= cap)).sum(axis=1) > 0).sum()
    all3 = (((cols != 0) & (cols <= cap)).sum(axis=1) == 3).sum()
    print(f"subject{s} ({SESSIONS[s]} sessions): {any_scanned} images, {all3} with 3 reps")
