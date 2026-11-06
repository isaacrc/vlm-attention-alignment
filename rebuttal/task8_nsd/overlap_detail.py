"""Follow-up detail on the 106 overlapping images: which NSD subject saw each one,
and how many repetitions were actually collected (rep trial ids of 0 = not collected)."""

from pathlib import Path

import pandas as pd

REPO = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment")
OUT = REPO / "rebuttal" / "task8_nsd"

nsd = pd.read_csv(REPO / "rebuttal" / "data" / "nsd_stim_info_merged.csv", index_col=0)
ov = pd.read_csv(OUT / "nsd_overlap.csv")
ov = ov[ov["in_nsd"]].copy()

sub = nsd.set_index("cocoId")
rep_cols = [f"subject{s}_rep{r}" for s in range(1, 9) for r in range(3)]

rows = []
for coco in ov["cocoId"]:
    r = sub.loc[coco]
    subj = [s for s in range(1, 9) if r[f"subject{s}"] == 1]
    for s in subj:
        reps = [r[f"subject{s}_rep{k}"] for k in range(3)]
        rows.append(
            {"cocoId": coco, "nsdId": r["nsdId"], "subject": s,
             "n_reps_collected": sum(1 for x in reps if x != 0)}
        )
det = pd.DataFrame(rows)
det.to_csv(OUT / "nsd_overlap_subject_detail.csv", index=False)

print("images per NSD subject (of the 106 overlapping):")
print(det["subject"].value_counts().sort_index().to_string())
print("\nrepetitions actually collected:")
print(det["n_reps_collected"].value_counts().sort_index().to_string())
print(f"\ntotal image x subject observations: {len(det)}")
print(f"with >=1 rep collected: {(det['n_reps_collected'] > 0).sum()}")
print(f"with 3 reps collected: {(det['n_reps_collected'] == 3).sum()}")

# Context: how big is the usable NSD set overall, ignoring our 200.
print("\n--- NSD totals for reference ---")
print(f"shared1000 images: {int(nsd['shared1000'].sum())}")
print(f"  of which val2017: {int(((nsd['shared1000']) & (nsd['cocoSplit'] == 'val2017')).sum())}")
for s in range(1, 9):
    seen = nsd[nsd[f"subject{s}"] == 1]
    collected = sum(
        1 for _, r in seen.iterrows()
        if any(r[f"subject{s}_rep{k}"] != 0 for k in range(3))
    )
    print(f"subject{s}: {len(seen)} assigned, {collected} with >=1 rep collected")
