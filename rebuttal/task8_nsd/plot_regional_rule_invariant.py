"""Regional floor-corrected figure with RULE-INVARIANT stars.

The story figure (plot_story_panels.fig_roi) Holm-corrects the floor test WITHIN each
ROI x task family, across all models. That couples every model in a cell: switching the
head rule changes the Transformer/VLM p-values, which reshuffles the Holm ladder and moves
the ADJUSTED p of models whose maps never changed. The LSTMs have a single attention map,
so mean == best for them; their raw floor test (t, p, corrected effect) is byte-identical
between rules, yet their stars flip near p=0.05 purely from the company they keep.

Fix: correct WITHIN each model x task family, across the four disjoint cortical streams
(early, ventral, lateral, parietal). Now a model's correction depends only on its own
p-values, so a fixed-map model gets identical stars under either head rule -- the honest
claim, since its brain predictivity does not depend on how other models' heads were pooled.

This is a supplement to nsd_roi_regional.png, not a replacement: it answers "does THIS
model beat its own floor across regions?" rather than "is this model special within THIS
region relative to the others?" The one-sided star rule (clears floor: corrected>0 AND
p_holm<.05) is unchanged. The raw per-model floor test is unchanged -- only the family the
multiple-comparison correction runs over differs.

Outputs (rebuttal/plots/regional/):
  nsd_roi_regional{_best}_holm_corrected_within_model.png   the figure, per head rule
  ../.. results/final_vs_floor_ruleinvariant_{rule}.csv       the numbers
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
RES = HERE / "results"
REGIONAL = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots/regional")
REGIONAL.mkdir(parents=True, exist_ok=True)

MODELS = ["CNN_LSTM", "ViT_LSTM", "CNN_Transformer", "ViT_Transformer", "Molmo", "Qwen35"]
COLORS = {"CNN_LSTM": "#ff7f0e", "ViT_LSTM": "#2ca02c",
          "CNN_Transformer": "#d62728", "ViT_Transformer": "#9467bd",
          "Molmo": "#8c564b", "Qwen35": "#17becf"}
LABELS = {"CNN_LSTM": "CNN-LSTM", "ViT_LSTM": "ViT-LSTM",
          "CNN_Transformer": "CNN-Transf.", "ViT_Transformer": "ViT-Transf.",
          "Molmo": "Molmo", "Qwen35": "Qwen3.5"}
BASELINES = ["EdgeEnergy", "Luminance"]
TASKS = ["social", "describe"]
STREAMS = ["early", "ventral", "lateral", "parietal"]   # the disjoint Holm family
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


def rule_invariant_floor_test(per):
    """Paired-t of (unique - Shuffled floor) vs 0 per (model, task, stream), then Holm
    WITHIN (model, task) across the four streams. Returns a lookup + tidy DataFrame."""
    rows = []
    sub = per[per.roi.isin(STREAMS)]
    for (model, task), g in sub.groupby(["model", "task"]):
        recs = []
        for roi in STREAMS:
            gg = g[g.roi == roi]["attn_corrected"].dropna()
            if len(gg) < MIN_N:
                continue
            t, p = stats.ttest_1samp(gg, 0.0)
            recs.append({"model": model, "task": task, "roi": roi, "n": len(gg),
                         "corrected": gg.mean(), "t": t, "p": p,
                         "cohens_d": gg.mean() / gg.std(ddof=1)})
        if not recs:
            continue
        d = pd.DataFrame(recs)
        d["p_holm"] = holm(d["p"].values)   # family = this model's four streams
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    lut = {(r.task, r.roi, r.model): (r.corrected, r.p_holm) for r in out.itertuples()}
    return lut, out


def fig_roi_invariant(per, sig_lut, rule, suffix):
    star = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    rois = [r for r in STREAMS if r in set(per.roi)]
    fig, axes = plt.subplots(2, len(rois), figsize=(3.4 * len(rois), 8), sharex=True)
    base_grey = "#9e9e9e"
    plot_models = MODELS + BASELINES
    for row, task in enumerate(TASKS):
        for col, roi in enumerate(rois):
            ax = axes[row, col]
            sub = per[(per.task == task) & (per.roi == roi)]
            means, sems, cols, hats, labs, keys = [], [], [], [], [], []
            for m in plot_models:
                g = sub[sub.model == m]["attn_corrected"].dropna()
                if len(g) < MIN_N:
                    continue
                means.append(g.mean())
                sems.append(g.std(ddof=1) / np.sqrt(len(g)))
                is_b = m in BASELINES
                cols.append(base_grey if is_b else COLORS.get(m, "#666"))
                hats.append("///" if is_b else "")
                labs.append(LABELS.get(m, m))
                keys.append(m)
            xs = np.arange(len(labs))
            bars = ax.bar(xs, means, yerr=sems, capsize=2, width=0.7, color=cols,
                          edgecolor="white", linewidth=0.9, zorder=3)
            for b, h in zip(bars, hats):
                if h:
                    b.set_hatch(h)
            ax.axhline(0, color="#333", lw=0.8, zorder=2)
            ytop = max([mn + se for mn, se in zip(means, sems)] + [0.0])
            off = 0.035 * ytop if ytop > 0 else 1e-5
            for xi, (mn, se, k) in enumerate(zip(means, sems, keys)):
                corr, ph = sig_lut.get((task, roi, k), (np.nan, np.nan))
                mark = star(ph) if (np.isfinite(ph) and corr > 0) else ""
                if mark:
                    ax.text(xs[xi], mn + se + off, mark, ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color="#222")
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, max(hi, ytop + 6 * off))
            ax.set_xticks(xs)
            ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=7.5)
            if row == 0:
                ceil = sub["ceiling"].mean()
                ax.set_title(f"{roi}\n(ceiling r={ceil:.2f})", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{task.capitalize()} task\n"
                              r"attn R² over CLIP, vs Shuffled floor", fontsize=9)
            ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle("Model attention beyond CLIP, by cortical stream — Holm corrected WITHIN model\n"
                 f"(head rule: {rule}; Holm within model×task across the 4 streams, so a "
                 "fixed-map model's stars do not depend on the rule)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.05, "Grey hatched = low-level baselines (Sobel edge energy, luminance). "
             "Bars: mean over 8 subjects, SEM. Stars = significantly clears the Shuffled floor "
             "(paired t vs floor, Holm within MODEL×task across the four streams: *** p<.001, "
             "** p<.01, * p<.05). The LSTMs have a single attention map (mean==best), so their "
             "stars here are identical under either head rule. y-scales differ by row/column.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    out = REGIONAL / f"nsd_roi_regional{suffix}_holm_corrected_within_model.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="both", choices=["mean", "best", "cvbehav", "both"])
    args = ap.parse_args()
    rules = ["mean", "best"] if args.head_rule == "both" else [args.head_rule]

    lstm_check = {}
    for rule in rules:
        suffix = "" if rule == "mean" else f"_{rule}"
        per = pd.read_csv(RES / f"final_per_subject_{rule}.csv")
        sig_lut, tidy = rule_invariant_floor_test(per)
        tidy.to_csv(RES / f"final_vs_floor_ruleinvariant_{rule}.csv", index=False)
        fig_roi_invariant(per, sig_lut, rule, suffix)
        for (task, roi, m), (corr, ph) in sig_lut.items():
            if m in ("CNN_LSTM", "ViT_LSTM"):
                lstm_check[(rule, m, task, roi)] = (round(corr, 6), round(ph, 6))

    # prove invariance: the LSTM rows must match between mean and best
    if {"mean", "best"} <= set(rules):
        print("\nLSTM stars identical across head rules? (must be True for every row)")
        mismatch = 0
        for (rule, m, task, roi), v in lstm_check.items():
            if rule != "mean":
                continue
            other = lstm_check.get(("best", m, task, roi))
            ok = other == v
            mismatch += not ok
            flag = "" if ok else "   <-- MISMATCH"
            print(f"  {m:9s} {task:8s} {roi:9s}  mean={v}  best={other}{flag}")
        print("ALL IDENTICAL" if mismatch == 0 else f"{mismatch} MISMATCHES")


if __name__ == "__main__":
    main()
