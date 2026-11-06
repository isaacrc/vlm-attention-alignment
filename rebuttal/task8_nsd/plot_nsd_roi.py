"""Figure: unique contribution of model attention to NSD responses, by cortical stream.

Form: small multiples (one panel per stream), grouped bars within panel. The comparison is
magnitude across a categorical dimension repeated in several contexts, which is what small
multiples are for; a single pooled panel would hide the early-vs-higher-order dissociation
that is the actual result.

Shared y axis across panels so stream magnitudes are directly comparable. One axis per
panel, never two.

Colour: the manuscript's model mapping, kept so this figure can be read against Fig 3.
That palette is not colourblind-safe (FINDINGS_palette.md), so identity never depends on
colour alone -- every bar is direct-labelled on the x axis and baselines are grey with
hatching to mark their different role.
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

COLORS = {"CNN_LSTM": "#ff7f0e", "ViT_LSTM": "#2ca02c",
          "CNN_Transformer": "#d62728", "ViT_Transformer": "#9467bd",
          "Molmo": "#8c564b", "Qwen35": "#17becf"}
LABELS = {"CNN_LSTM": "CNN-LSTM", "ViT_LSTM": "ViT-LSTM",
          "CNN_Transformer": "CNN-Transf.", "ViT_Transformer": "ViT-Transf.",
          "Molmo": "Molmo", "Qwen35": "Qwen3.5",
          "EdgeEnergy": "Edge energy", "Luminance": "Luminance",
          "Shuffled": "Shuffled", "Center": "Center"}
MODEL_ORDER = ["CNN_LSTM", "ViT_LSTM", "CNN_Transformer", "ViT_Transformer", "Molmo", "Qwen35"]
BASELINE_ORDER = ["EdgeEnergy", "Luminance", "Shuffled", "Center"]
ROIS = ["early", "ventral", "lateral", "parietal"]
GREY = "#9e9e9e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="describe", choices=["describe", "social"])
    ap.add_argument("--head-rule", default="mean")
    ap.add_argument("--min-n", type=int, default=3)
    args = ap.parse_args()

    per = pd.read_csv(RES / f"encoding_by_roi_per_subject_{args.head_rule}.csv")
    per = per[per.task == args.task]

    rois = [r for r in ROIS if r in set(per.roi)]
    fig, axes = plt.subplots(1, len(rois), figsize=(3.6 * len(rois), 4.8), sharey=False)
    if len(rois) == 1:
        axes = [axes]

    order = [m for m in MODEL_ORDER if m in set(per.model)] + \
            [b for b in BASELINE_ORDER if b in set(per.model)]

    for ax, roi in zip(axes, rois):
        sub = per[per.roi == roi]
        xs, mu, se, col, hat, labs, ns = [], [], [], [], [], [], []
        for i, m in enumerate(order):
            g = sub[sub.model == m]["attn_unique"].dropna()
            if len(g) < args.min_n:
                continue
            xs.append(i)
            mu.append(g.mean())
            se.append(g.std(ddof=1) / np.sqrt(len(g)))
            is_b = m in BASELINE_ORDER
            col.append(GREY if is_b else COLORS.get(m, "#666"))
            hat.append("///" if is_b else "")
            labs.append(LABELS.get(m, m))
            ns.append(len(g))

        bars = ax.bar(xs, mu, yerr=se, capsize=2.5, width=0.7, color=col,
                      edgecolor="white", linewidth=1.1, zorder=3)
        for b, h in zip(bars, hat):
            if h:
                b.set_hatch(h)
        for i, m in enumerate(order):
            g = sub[sub.model == m]["attn_unique"].dropna()
            if len(g) >= args.min_n:
                jit = np.random.default_rng(1).uniform(-0.13, 0.13, len(g))
                ax.scatter(np.full(len(g), i) + jit, g, s=8, color="#2b2b2b",
                           alpha=0.5, zorder=4, linewidths=0)

        ax.axhline(0, color="#333", lw=0.8, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=8)
        # n printed at the base of each bar rather than in the tick label
        for x, n_i in zip(xs, ns):
            ax.annotate(str(n_i), (x, 0), xytext=(0, -3), textcoords="offset points",
                        ha="center", va="top", fontsize=6, color="#777")
        ceil = sub["ceiling"].mean()
        ax.set_title(f"{roi}\n(ceiling r = {ceil:.2f})", fontsize=10)
        ax.grid(axis="y", color="#ececec", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("Unique variance from attention\n"
                       r"$R^2$(attention+CLIP) $-$ $R^2$(CLIP)", fontsize=9.5)
    fig.suptitle(f"Model attention beyond CLIP, by cortical stream — {args.task} task",
                 fontsize=12.5, y=1.03)
    fig.text(0.5, -0.10,
             "Bars: mean across NSD subjects, error bars SEM, dots individual subjects. "
             "Grey hatched = baselines; small figures are n subjects.\n"
             "NOTE: y scales differ per panel — early-visual effects are ~6x larger. "
             "Heads averaged within the analysed layer "
             "(predefined rule, not best-head selection). Between-model differences are "
             "unaffected by the\nridge-tuning bias in this measure, since it is common to "
             "all models within a subject and stream.",
             ha="center", fontsize=7.6, color="#555")
    fig.tight_layout()
    out = HERE / "figures" / f"nsd_roi_{args.task}_{args.head_rule}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
