"""Figure: which model's attention best predicts NSD brain responses?

Form: grouped bars. The job is magnitude comparison across a categorical dimension
(model) split by task, so bars with SEM across the 8 subjects, one axis, sorted by effect.

Colour: the manuscript's existing model mapping is kept so a reader can match this figure
to Fig 3. That palette is NOT colorblind-safe (see FINDINGS_palette.md: CNN-LSTM vs
ViT-LSTM separate by only dE 0.7 under protanopia), so identity never rests on colour
here -- every bar is direct-labelled on the x axis, and baselines are drawn in neutral grey
with hatching to mark their different role.

Reference lines: the noise ceiling is the honest upper bound; the Shuffled baseline is the
floor. A model is only interesting if it clears the low-level baselines.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"

# Manuscript mapping (plot_fig3_qwen35_lg_m6cstyle_m6colors.py). Kept for cross-figure
# consistency; see module docstring on why colour is never the sole identity channel.
COLORS = {
    "CNN_LSTM": "#ff7f0e", "ViT_LSTM": "#2ca02c",
    "CNN_Transformer": "#d62728", "ViT_Transformer": "#9467bd",
    "Molmo": "#8c564b", "Qwen35": "#17becf",
}
LABELS = {
    "CNN_LSTM": "CNN-LSTM", "ViT_LSTM": "ViT-LSTM",
    "CNN_Transformer": "CNN-Transf.", "ViT_Transformer": "ViT-Transf.",
    "Molmo": "Molmo", "Qwen35": "Qwen3.5",
    "Luminance": "Luminance", "EdgeEnergy": "Edge energy",
    "Shuffled": "Shuffled", "Center": "Center",
}
BASELINES = ["Luminance", "EdgeEnergy", "Shuffled", "Center"]
BASELINE_GREY = "#9e9e9e"
TASK_ORDER = ["describe", "social"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="mean")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per = pd.read_csv(RES / f"encoding_per_subject_{args.head_rule}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    # order models by describe-task effect so the ranking reads left to right
    ref = (per[per.task == "describe"].groupby("model")["attn_unique"].mean()
           .sort_values(ascending=False))
    models = [m for m in ref.index if m not in BASELINES]
    order = models + [b for b in BASELINES if b in ref.index]

    for ax, task in zip(axes, TASK_ORDER):
        sub = per[per.task == task]
        xs, means, sems, colors, hatches = [], [], [], [], []
        for i, m in enumerate(order):
            g = sub[sub.model == m]["attn_unique"]
            if len(g) == 0:
                continue
            xs.append(i)
            means.append(g.mean())
            sems.append(g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0)
            is_base = m in BASELINES
            colors.append(BASELINE_GREY if is_base else COLORS.get(m, "#666666"))
            hatches.append("///" if is_base else "")

        bars = ax.bar(xs, means, yerr=sems, capsize=3, width=0.68,
                      color=colors, edgecolor="white", linewidth=1.2, zorder=3)
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)

        # per-subject points, so the reader sees the spread rather than only the SEM
        for i, m in enumerate(order):
            g = sub[sub.model == m]["attn_unique"]
            if len(g):
                ax.scatter(np.full(len(g), i) + np.random.default_rng(0)
                           .uniform(-0.12, 0.12, len(g)), g,
                           s=9, color="#333333", alpha=0.55, zorder=4, linewidths=0)

        ax.axhline(0, color="#333333", lw=0.8, zorder=2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([LABELS.get(m, m) for m in order], rotation=38,
                           ha="right", fontsize=9)
        ax.set_title(f"{task.capitalize()} task", fontsize=11)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Unique variance from attention\n$R^2$(attention+CLIP) $-$ $R^2$(CLIP)",
                       fontsize=10)
    # Descriptive, not conclusive: the title states what is plotted, so it cannot
    # prejudge a ranking that the data have not yet settled.
    n_subj = per["subject"].nunique()
    fig.suptitle("Unique contribution of model attention to NSD response prediction "
                 f"(n = {n_subj} subjects, shared-1000 images)", fontsize=12, y=1.02)
    fig.text(0.5, -0.14, "Bars: mean across subjects, error bars SEM, dots individual "
             "subjects. Grey hatched = baselines.\nHeads averaged within the analysed "
             "layer (predefined rule, not best-head selection).",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    out = Path(args.out) if args.out else HERE / "figures" / f"nsd_encoding_{args.head_rule}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
