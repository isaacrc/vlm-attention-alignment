"""Aggregate per-subject encoding fits into the model comparison table.

The question this answers: which model's attention best predicts NSD brain responses?

Reported per model x task:
  attn_unique  - R2(banded) - R2(clip-only), the variance the attention map explains
                 beyond the CLIP "what" band. This is the quantity of interest.
  r2_attn      - attention-only R2
  r2_banded    - full model R2
  pct_ceiling  - attn_unique as a fraction of the cross-subject noise ceiling (r^2)

Vertices are restricted to those above the ncsnr threshold used at fit time, and summary
values are means over the top-decile vertices by ceiling (whole-cortex means are dominated
by unresponsive vertices).

Group stats: paired t-test across the 8 subjects for each pairwise model contrast, Holm
corrected, matching the rebuttal Task 5 requirement that no comparative claim is untested.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
ENC = HERE / "encoding"
MIN_SUBJECTS_FOR_TEST = 3
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def _band_share(deltas):
    """Attention band's share of total kernel weight, from log-space deltas.

    deltas are [attn, clip] in log space. Softmax gives the relative weighting, which is
    scale-invariant and therefore the interpretable quantity; 0.5 means the two bands are
    weighted equally.
    """
    d = np.asarray(deltas, dtype=float)
    e = np.exp(d - d.max())
    return e[0] / e.sum()


def holm(pvals):
    """Holm-Bonferroni. NaN inputs (e.g. a contrast with too few subjects) stay NaN
    rather than being sorted to the end and silently emerging as p=0."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full(len(p), np.nan)
    finite = np.where(np.isfinite(p))[0]
    if len(finite) == 0:
        return adj
    order = finite[np.argsort(p[finite])]
    m = len(finite)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="mean")
    ap.add_argument("--top-frac", type=float, default=0.1,
                    help="fraction of vertices (by noise ceiling) used for summaries")
    args = ap.parse_args()

    rows = []
    for f in sorted(ENC.glob(f"r2_*_{args.head_rule}.npz")):
        stem = f.stem[len("r2_"):-len(f"_{args.head_rule}")]
        model, task, subj = stem.rsplit("_", 2)
        d = np.load(f)
        ceiling = d["ceiling"]
        n_top = max(1, int(len(ceiling) * args.top_frac))
        top = np.argsort(ceiling)[-n_top:]
        unique = d["r2_banded"] - d["r2_clip"]
        ceil2 = np.clip(ceiling[top], 1e-6, None) ** 2
        rows.append({
            "model": model, "task": task, "subject": int(subj.replace("subj", "")),
            "r2_banded": d["r2_banded"][top].mean(),
            "r2_attn": d["r2_attn"][top].mean(),
            "r2_clip": d["r2_clip"][top].mean(),
            "attn_unique": unique[top].mean(),
            "pct_ceiling": float(np.mean(unique[top] / ceil2) * 100),
            # Himalaya's deltas are log-space kernel weights whose absolute scale is
            # absorbed by the ridge penalty, so exp() alone is uninterpretable (it can
            # round to 0 for both bands). The meaningful quantity is the relative share.
            "band_w_attn": float(_band_share(d["deltas"])),
            "n_vertices": len(ceiling),
        })

    if not rows:
        print(f"no encoding results found in {ENC} for head-rule={args.head_rule}")
        return

    df = pd.DataFrame(rows).sort_values(["task", "model", "subject"])
    df.to_csv(OUT / f"encoding_per_subject_{args.head_rule}.csv", index=False)

    grp = (df.groupby(["task", "model"])
             .agg(n=("subject", "size"),
                  attn_unique=("attn_unique", "mean"),
                  sem=("attn_unique", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                  r2_attn=("r2_attn", "mean"),
                  r2_banded=("r2_banded", "mean"),
                  pct_ceiling=("pct_ceiling", "mean"))
             .reset_index()
             .sort_values(["task", "attn_unique"], ascending=[True, False]))
    grp.to_csv(OUT / f"encoding_group_{args.head_rule}.csv", index=False)
    print(grp.to_string(index=False))

    # paired contrasts across subjects, per task
    contrasts = []
    for task, sub in df.groupby("task"):
        piv = sub.pivot(index="subject", columns="model", values="attn_unique").dropna()
        for a, b in itertools.combinations(sorted(piv.columns), 2):
            # A paired test needs at least 3 subjects to give a meaningful variance
            # estimate; below that scipy returns NaN and the contrast is not reportable.
            if len(piv) < MIN_SUBJECTS_FOR_TEST:
                continue
            t, p = stats.ttest_rel(piv[a], piv[b])
            diff = (piv[a] - piv[b]).mean()
            sd = (piv[a] - piv[b]).std(ddof=1)
            contrasts.append({"task": task, "model_a": a, "model_b": b,
                              "mean_diff": diff, "t": t, "p": p, "n": len(piv),
                              "cohens_d": diff / sd if sd > 0 else np.nan})
    if contrasts:
        cdf = pd.DataFrame(contrasts)
        cdf["p_holm"] = np.concatenate([
            holm(g["p"].values) for _, g in cdf.groupby("task", sort=False)])
        cdf.to_csv(OUT / f"encoding_contrasts_{args.head_rule}.csv", index=False)
        print(f"\nsignificant contrasts (Holm p<0.05, n>={MIN_SUBJECTS_FOR_TEST} subjects):")
        sig = cdf[cdf["p_holm"].notna() & (cdf["p_holm"] < 0.05)]
        print(sig.to_string(index=False) if len(sig) else "  none")

    with open(OUT / f"encoding_summary_{args.head_rule}.json", "w") as fh:
        json.dump({"top_frac": args.top_frac, "head_rule": args.head_rule,
                   "n_subjects": int(df["subject"].nunique()),
                   "best_by_task": {
                       t: g.sort_values("attn_unique", ascending=False)["model"].iloc[0]
                       for t, g in grp.groupby("task")}}, fh, indent=2)


if __name__ == "__main__":
    main()
