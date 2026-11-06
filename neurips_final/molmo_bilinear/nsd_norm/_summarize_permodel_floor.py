"""Regional stats with one permutation floor PER MODEL.

Replaces the single CNN-LSTM-derived `Shuffled` floor that summarize_final.py subtracted
from every model. Each model is now referenced to its own permuted maps (Shuf{model}),
which is what the Methods and the Figure 3 caption describe.

Holm family matches the Methods: within model and task, across the four cortical streams.

Output: results/permodel_per_subject_{rule}.csv   per-subject floor-corrected values
        results/permodel_vs_floor_{rule}.csv      one row per model x task x region
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
OUT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/results_norm')
STREAMS = ["early", "ventral", "lateral", "parietal"]
MODELS = ["CNN_LSTM", "ViT_LSTM", "CNN_Transformer", "ViT_Transformer",
          "Molmo", "Qwen35", "EdgeEnergy", "Luminance"]
MIN_N = 3


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="cvbehav")
    a = ap.parse_args()

    per = pd.read_csv(OUT / f"encoding_by_roi_per_subject_{a.head_rule}.csv")
    per = per[per.roi.isin(STREAMS)]

    key = ["subject", "roi", "task"]
    rows = []
    for m in MODELS:
        real = per[per.model == m].set_index(key)["attn_unique"].rename("attn_unique")
        floor = per[per.model == f"Shuf{m}"].set_index(key)["attn_unique"].rename("floor")
        if floor.empty:
            print(f"  no floor fits for {m} -- skipped")
            continue
        j = pd.concat([real, floor], axis=1, join="inner").reset_index()
        j["model"] = m
        j["attn_corrected"] = j["attn_unique"] - j["floor"]
        rows.append(j)
    if not rows:
        print("no per-model floors found")
        return
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(OUT / f"permodel_per_subject_{a.head_rule}.csv", index=False)

    recs = []
    for (m, task), g in d.groupby(["model", "task"]):
        cells, ps = [], []
        for roi in STREAMS:
            gg = g[g.roi == roi]["attn_corrected"].dropna()
            if len(gg) < MIN_N:
                continue
            t, p = stats.ttest_1samp(gg, 0.0)
            sd = gg.std(ddof=1)
            cells.append(dict(model=m, task=task, roi=roi, n=len(gg),
                              raw=g[g.roi == roi]["attn_unique"].mean(),
                              floor=g[g.roi == roi]["floor"].mean(),
                              corrected=gg.mean(), sem=gg.sem(), t=t,
                              cohens_d=gg.mean() / sd if sd > 0 else np.nan))
            ps.append(p)
        for c, ph in zip(cells, holm(ps)):
            c["p_holm"] = ph
            recs.append(c)
    v = pd.DataFrame(recs)
    v.to_csv(OUT / f"permodel_vs_floor_{a.head_rule}.csv", index=False)
    print(f"wrote {OUT}/permodel_vs_floor_{a.head_rule}.csv  ({len(v)} rows)")
    print(v.pivot_table(index=["model"], columns=["roi", "task"],
                        values="corrected").round(4).to_string())


if __name__ == "__main__":
    main()
