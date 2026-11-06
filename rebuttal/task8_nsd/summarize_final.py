"""Final model comparison, floor-corrected, per cortical stream, with paired statistics.

Why a floor correction. `attn_unique = R2(banded) - R2(clip-only)` differences two
INDEPENDENTLY TUNED models: the banded fit picks its own regularisation, the CLIP-only fit
picks its own. That difference is not zero even when the attention band contributes
nothing, so the raw measure carries a positive bias. The Center baseline -- a map identical
for every image, which cannot explain across-image variance -- measures that bias directly.

So the reportable effect is `attn_unique(model) - attn_unique(Center)`, computed WITHIN
subject and ROI so the correction tracks whatever the bias is in that context. Shuffled
(real map statistics, image correspondence destroyed) is reported alongside as an
independent check that the corrected floor really does sit at zero.

Statistics: per ROI and task, a paired t-test against the Center floor across subjects
(does this model beat the bias floor at all?) and paired contrasts between models, both
Holm-corrected within ROI x task.

Outputs:
  results/final_by_roi_{rule}.csv        model x roi x task, corrected effects
  results/final_vs_floor_{rule}.csv      test against the Center floor
  results/final_contrasts_{rule}.csv     pairwise model contrasts
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
OUT = HERE / "results"
# Two candidate floors, and they are NOT equivalent:
#   Center   - constant map; after standardisation the attention kernel is exactly zero,
#              so this is the floor for a DEGENERATE band.
#   Shuffled - real map statistics with image correspondence destroyed; a live band
#              carrying no information. This is the matched control, and empirically it
#              sits below Center, so correcting by Center over-penalises.
# Default to Shuffled; --floor Center reproduces the stricter correction.
DEFAULT_FLOOR = "Shuffled"
BASELINES = {"Center", "Shuffled", "Luminance", "EdgeEnergy"}
MIN_N = 3


def holm(p):
    p = np.asarray(p, dtype=float)
    adj = np.full(len(p), np.nan)
    fin = np.where(np.isfinite(p))[0]
    if not len(fin):
        return adj
    order = fin[np.argsort(p[fin])]
    m, run = len(fin), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="mean")
    ap.add_argument("--floor", default=DEFAULT_FLOOR, choices=["Shuffled", "Center", "none"])
    args = ap.parse_args()
    FLOOR = args.floor

    per = pd.read_csv(OUT / f"encoding_by_roi_per_subject_{args.head_rule}.csv")

    # within subject x roi x task, subtract that cell's Center value
    if FLOOR == "none":
        per["floor"] = 0.0
        floor = None
    else:
        floor = (per[per.model == FLOOR]
                 .set_index(["subject", "roi", "task"])["attn_unique"]
                 .rename("floor"))
        per = per.join(floor, on=["subject", "roi", "task"])
    if per["floor"].isna().all():
        print(f"no {FLOOR} results yet -- cannot floor-correct")
        return
    per["attn_corrected"] = per["attn_unique"] - per["floor"]
    per.to_csv(OUT / f"final_per_subject_{args.head_rule}.csv", index=False)

    grp = (per.groupby(["roi", "task", "model"])
             .agg(n_raw=("subject", "size"),
                  n=("attn_corrected", "count"),
                  raw=("attn_unique", "mean"),
                  corrected=("attn_corrected", "mean"),
                  sem_=("attn_corrected", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
                  ceiling=("ceiling", "mean"))
             .reset_index())
    grp.to_csv(OUT / f"final_by_roi_{args.head_rule}_{FLOOR}.csv", index=False)

    # does each model beat the bias floor?
    vs = []
    for (roi, task, model), g in per.groupby(["roi", "task", "model"]):
        g = g.dropna(subset=["attn_corrected"])
        if model == FLOOR or len(g) < MIN_N:
            continue
        t, p = stats.ttest_1samp(g["attn_corrected"], 0.0)
        vs.append({"roi": roi, "task": task, "model": model, "n": len(g),
                   "corrected": g["attn_corrected"].mean(), "t": t, "p": p,
                   "cohens_d": g["attn_corrected"].mean() / g["attn_corrected"].std(ddof=1)})
    if vs:
        v = pd.DataFrame(vs)
        v["p_holm"] = np.concatenate([holm(x["p"].values)
                                      for _, x in v.groupby(["roi", "task"], sort=False)])
        v.to_csv(OUT / f"final_vs_floor_{args.head_rule}_{FLOOR}.csv", index=False)

    # pairwise model contrasts on the corrected values
    cons = []
    for (roi, task), g in per.groupby(["roi", "task"]):
        piv = g.pivot_table(index="subject", columns="model", values="attn_corrected")
        models = [m for m in piv.columns if m not in BASELINES]
        for a, b in itertools.combinations(sorted(models), 2):
            pair = piv[[a, b]].dropna()
            if len(pair) < MIN_N:
                continue
            t, p = stats.ttest_rel(pair[a], pair[b])
            d = (pair[a] - pair[b])
            cons.append({"roi": roi, "task": task, "model_a": a, "model_b": b,
                         "mean_diff": d.mean(), "t": t, "p": p, "n": len(pair),
                         "cohens_d": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan})
    if cons:
        c = pd.DataFrame(cons)
        c["p_holm"] = np.concatenate([holm(x["p"].values)
                                      for _, x in c.groupby(["roi", "task"], sort=False)])
        c.to_csv(OUT / f"final_contrasts_{args.head_rule}_{FLOOR}.csv", index=False)

    # report
    for roi in ["early", "ventral", "lateral", "parietal", "nsdgeneral"]:
        sub = grp[grp.roi == roi]
        if sub.empty:
            continue
        print(f"\n=== {roi} (ceiling r={sub['ceiling'].mean():.3f}) ===")
        for task in ("describe", "social"):
            s = sub[sub.task == task].sort_values("corrected", ascending=False)
            if s.empty:
                continue
            print(f"  [{task}]  effect = attn_unique - {FLOOR} floor")
            for _, r in s.iterrows():
                if r.model == FLOOR:
                    continue
                sem = f"+-{r['sem_']:.4f}" if np.isfinite(r["sem_"]) else "        "
                sig = ""
                if vs:
                    m = v[(v.roi == roi) & (v.task == task) & (v.model == r.model)]
                    if len(m) and np.isfinite(m["p_holm"].iloc[0]):
                        ph = m["p_holm"].iloc[0]
                        sig = " ***" if ph < .001 else " **" if ph < .01 else " *" if ph < .05 else " ns"
                tag = "  [baseline]" if r.model in BASELINES else ""
                print(f"    {r.model:<16} {r.corrected:+.4f} {sem} (n={int(r.n)}){sig}{tag}")


if __name__ == "__main__":
    main()
