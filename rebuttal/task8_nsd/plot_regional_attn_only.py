"""Regional figure on RAW attention-only encoding accuracy — no banded ridge.

Companion to plot_regional_rule_invariant.py, which plots `attn_unique` = R2(banded) -
R2(clip-only), floor-corrected. That quantity answers "what does attention add BEYOND
CLIP?", and because CLIP takes almost all the kernel weight the residual is tiny (~0.004)
and its sign depends on differencing two independently tuned fits.

This figure asks the simpler question: how well does a model's attention map predict
NSD responses ON ITS OWN? The bars are `r2_attn`, the single-band linear KernelRidgeCV on
the 196-dim attention features alone (fit_encoding.fit_single), which fit_encoding already
computes and stores in every r2_*.npz. No refit is needed and nothing here is partialled.

Metric is the same as the rest of the final analysis: mean Pearson r across the 5 outer
folds, then signed square (r^2 * sign(r)) -- a cross-validated squared correlation, not a
true coefficient of determination.

Bars are raw. The Shuffled baseline (real map statistics, image correspondence destroyed)
is plotted alongside as the empirical floor and sits at ~0, as it must: a single-band fit
has no second band to donate variance, so unlike attn_unique the raw measure carries no
positive bias and needs no floor correction. Stars still test against Shuffled within
subject, so the significance family matches the source figure exactly: Holm WITHIN
model x task across the four disjoint streams, which keeps a fixed-map model's stars
independent of how other models' heads were pooled.

Outputs (rebuttal/plots/regional/):
  nsd_roi_regional{_best}_attn_only_raw.png              the figure, per head rule
  results/attn_only_raw_by_roi_{rule}.csv                group means + floor test
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
          "Molmo": "Molmo", "Qwen35": "Qwen3.5", "Shuffled": "Shuffled (floor)"}
BASELINES = ["EdgeEnergy", "Luminance", "Shuffled"]
FLOOR = "Shuffled"
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


def floor_test(per):
    """Paired t of (r2_attn - Shuffled r2_attn) vs 0 per (model, task, stream), Holm
    WITHIN (model, task) across the four streams. Returns a lookup + tidy DataFrame."""
    rows = []
    sub = per[per.roi.isin(STREAMS)]
    for (model, task), g in sub.groupby(["model", "task"]):
        if model == FLOOR:
            continue
        recs = []
        for roi in STREAMS:
            gg = g[g.roi == roi].dropna(subset=["r2_attn_vs_floor"])
            if len(gg) < MIN_N:
                continue
            t, p = stats.ttest_1samp(gg["r2_attn_vs_floor"], 0.0)
            recs.append({"model": model, "task": task, "roi": roi, "n": len(gg),
                         "raw": gg["r2_attn"].mean(),
                         "vs_floor": gg["r2_attn_vs_floor"].mean(), "t": t, "p": p,
                         "cohens_d": gg["r2_attn_vs_floor"].mean()
                                     / gg["r2_attn_vs_floor"].std(ddof=1)})
        if not recs:
            continue
        d = pd.DataFrame(recs)
        d["p_holm"] = holm(d["p"].values)   # family = this model's four streams
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    lut = {(r.task, r.roi, r.model): (r.vs_floor, r.p_holm) for r in out.itertuples()}
    return lut, out


def fig_attn_only(per, sig_lut, rule, suffix):
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
                g = sub[sub.model == m]["r2_attn"].dropna()
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
                vf, ph = sig_lut.get((task, roi, k), (np.nan, np.nan))
                mark = star(ph) if (np.isfinite(ph) and vf > 0) else ""
                if mark:
                    ax.text(xs[xi], mn + se + off, mark, ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color="#222")
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, 0), max(hi, ytop + 6 * off))
            ax.set_xticks(xs)
            ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=7.5)
            if row == 0:
                ceil = sub["ceiling"].mean()
                ax.set_title(f"{roi}\n(ceiling r={ceil:.2f})", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{task.capitalize()} task\n"
                              r"attention-only R² (raw, no CLIP band)", fontsize=9)
            ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle("Raw encoding accuracy of model attention alone, by cortical stream\n"
                 f"(single-band ridge on the 196-dim attention map; head rule: {rule}; "
                 "no CLIP band, no variance partition)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06,
             "Bars: cross-validated signed r² (mean r over 5 folds, then r²·sign r), mean over 8 subjects, SEM.\n"
             "Grey hatched = baselines: Sobel edge energy, luminance, and the Shuffled floor (real map\n"
             "statistics, image correspondence destroyed). Stars = clears the Shuffled floor (paired t within\n"
             "subject, Holm within MODEL×task across the four streams: *** p<.001, ** p<.01, * p<.05).\n"
             "Unlike the banded attn-over-CLIP measure this is not a difference of two tuned fits, so the floor\n"
             "sits at ~0 and bars are plotted uncorrected. y-scales differ by row/column.",
             ha="center", va="top", fontsize=8, color="#555")
    fig.tight_layout()
    out = REGIONAL / f"nsd_roi_regional{suffix}_attn_only_raw.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="best", choices=["mean", "best", "both"])
    args = ap.parse_args()
    rules = ["mean", "best"] if args.head_rule == "both" else [args.head_rule]

    for rule in rules:
        suffix = "" if rule == "mean" else f"_{rule}"
        per = pd.read_csv(RES / f"encoding_by_roi_per_subject_{rule}.csv")
        # within subject x roi x task, the Shuffled cell's own attention-only score
        fl = (per[per.model == FLOOR]
              .set_index(["subject", "roi", "task"])["r2_attn"].rename("floor"))
        per = per.join(fl, on=["subject", "roi", "task"])
        per["r2_attn_vs_floor"] = per["r2_attn"] - per["floor"]

        sig_lut, tidy = floor_test(per)
        tidy.to_csv(RES / f"attn_only_raw_by_roi_{rule}.csv", index=False)
        fig_attn_only(per, sig_lut, rule, suffix)

        print(f"\n=== attention-only raw R², head rule {rule} ===")
        for task in TASKS:
            print(f"\n[{task}]")
            t = (tidy[tidy.task == task]
                 .pivot_table(index="model", columns="roi", values="raw")
                 .reindex(columns=STREAMS))
            print(t.round(4).to_string())


if __name__ == "__main__":
    main()
