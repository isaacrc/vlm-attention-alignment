"""Whole-brain CLIP-controlled companion to the manuscript attention-only bar.

The manuscript-style figure (plot_manuscript_identical.py) plots whole-brain attention-ONLY
R^2 -- the existence check, no semantic control. That number rewards any attention map that
tracks brain-predictive structure, including structure a CLIP embedding already captures, so
its model ordering is driven partly by CLIP-shared variance. This figure plots the honest,
controlled quantity instead:

  metric = unique attention variance = R^2(banded: attn+CLIP) - R^2(CLIP-only)
           i.e. what attention adds to a CLIP embedding, per vertex, averaged over the
           reliable vertices (ncsnr>0.2). Same vertices as the attention-only figure.
  floor  = the Shuffled model's own unique variance (real maps, permuted image labels),
           drawn as a dashed reference line; a bar only counts if it clears this.
  bars   = 6 models, two task panels sharing y, no Human bar (shared-1000 has no eye-tracking).

Significance: paired t across the 8 subjects, each model's unique variance vs the Shuffled
floor (Holm-corrected within task). Stars are printed above each bar.

Usage:  python plot_wholebrain_clip_controlled.py --head-rule {mean,best}
Output: rebuttal/plots/nsd_wholebrain_clip_controlled_{rule}.png
        rebuttal/plots/nsd_wholebrain_clip_controlled_{rule}_sigtests.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
ENC = HERE / "encoding"
PLOTS = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots/wholebrain")
PLOTS.mkdir(parents=True, exist_ok=True)

# ---- styling copied verbatim from plot_manuscript_identical.py (== m6c Panel A) ----------
ATTN_MODELS = ["CNN-LSTM", "ViT-LSTM", "CNN-Transformer", "ViT-Transformer", "Molmo", "Qwen3.5"]
TASKS = ["social", "describe"]
COLORS_BAR = {
    "CNN-LSTM": "#ff7f0e", "ViT-LSTM": "#2ca02c",
    "CNN-Transformer": "#d62728", "ViT-Transformer": "#9467bd",
    "Molmo": "#8c564b", "Qwen3.5": "#17becf",
}
TASK_LABELS = {"social": "Social task", "describe": "Describe task"}
DISPLAY_NAMES = {
    "CNN-LSTM": "CNN-LSTM", "ViT-LSTM": "ViT-LSTM",
    "CNN-Transformer": "CNN-Transformer", "ViT-Transformer": "ViT-Transformer",
    "Molmo": "Molmo 7B-D", "Qwen3.5": "Qwen 9B-3.5",
}
FS_TITLE, FS_TICK, FS_LABEL, PANEL_LABEL_FS = 20, 18, 19, 31
BAR_W = 0.6
KEY = {"CNN-LSTM": "CNN_LSTM", "ViT-LSTM": "ViT_LSTM", "CNN-Transformer": "CNN_Transformer",
       "ViT-Transformer": "ViT_Transformer", "Molmo": "Molmo", "Qwen3.5": "Qwen35"}


def per_subject_unique(rule):
    """Per-subject whole-brain unique attention variance (banded - CLIP), incl. Shuffled floor."""
    rows = []
    for disp in ATTN_MODELS + ["__Shuffled__"]:
        mkey = "Shuffled" if disp == "__Shuffled__" else KEY[disp]
        for task in TASKS:
            for subj in range(1, 9):
                f = ENC / f"r2_{mkey}_{task}_subj{subj:02d}_{rule}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                uniq = float((d["r2_banded"] - d["r2_clip"]).mean())
                rows.append({"model": disp, "task": task, "subject": subj, "unique": uniq})
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run, m = 0.0, len(p)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def sig_vs_floor(df):
    """Paired t across subjects: each model's unique variance vs the Shuffled floor, Holm/task."""
    out = []
    for task in TASKS:
        piv = df[df.task == task].pivot_table(index="subject", columns="model", values="unique")
        rows = []
        for a in ATTN_MODELS:
            pair = piv[[a, "__Shuffled__"]].dropna()
            t, p = stats.ttest_rel(pair[a], pair["__Shuffled__"])
            d = pair[a] - pair["__Shuffled__"]
            rows.append({"task": task, "model": a, "unique": pair[a].mean(),
                         "floor": pair["__Shuffled__"].mean(), "diff": d.mean(),
                         "t": t, "p": p,
                         "dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan})
        r = pd.DataFrame(rows)
        r["p_holm"] = holm(r["p"].values)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="mean", choices=["mean", "best", "cvbehav"])
    args = ap.parse_args()
    rule = args.head_rule

    df = per_subject_unique(rule)
    if df.empty:
        print(f"no fits for head-rule={rule} yet")
        return
    sig = sig_vs_floor(df)
    sig.to_csv(PLOTS / f"nsd_wholebrain_clip_controlled_{rule}_sigtests.csv", index=False)

    means = (df[df.model != "__Shuffled__"].groupby(["model", "task"])["unique"]
             .agg(["mean", "sem"]).reset_index())
    floor = (df[df.model == "__Shuffled__"].groupby("task")["unique"].mean().to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    x_pos = np.arange(len(ATTN_MODELS))
    y_top = max((means["mean"] + means["sem"]).max(), max(floor.values()))
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        vals = {m: means[(means.model == m) & (means.task == task)]["mean"].iloc[0]
                for m in ATTN_MODELS}
        sems = {m: means[(means.model == m) & (means.task == task)]["sem"].iloc[0]
                for m in ATTN_MODELS}
        sg = sig[sig.task == task].set_index("model")
        for xi, model in enumerate(ATTN_MODELS):
            ax.bar(xi, vals[model], BAR_W, yerr=sems[model], capsize=3,
                   color=COLORS_BAR[model], edgecolor="white", linewidth=0.5)
            ph = sg.loc[model, "p_holm"]
            clears = (sg.loc[model, "diff"] > 0) and (ph < .05)
            # star answers ONLY "clears the floor" -> one-sided. A bar that is significantly
            # BELOW floor gets "ns" (it does not exceed floor), not a star.
            mark = star(ph) if clears else "ns"
            y_mark = max(vals[model] + sems[model], floor[task]) + y_top * 0.02
            ax.text(xi, y_mark, mark, ha="center", va="bottom", fontsize=12,
                    color="#222" if clears else "#999")
        # Shuffled floor reference line
        ax.axhline(floor[task], color="#333", ls="--", lw=1.3, zorder=4)
        ax.text(len(ATTN_MODELS) - 0.4, floor[task], "  Shuffled floor",
                va="center", ha="left", fontsize=11, color="#333")
        ax.set_ylim(0, y_top * 1.22)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([DISPLAY_NAMES[m] for m in ATTN_MODELS], fontsize=FS_TICK,
                           rotation=30, ha="right")
        ax.set_title(TASK_LABELS[task], fontsize=FS_TITLE, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        if ti == 0:
            ax.set_ylabel("Unique attn R² over CLIP\n(whole brain)", fontsize=FS_LABEL)

    head_txt = "layer-3 head mean" if rule == "mean" else "best head (behavioral)"
    fig.suptitle("Attention Beyond CLIP Predicts Real Brain Activity (NSD)",
                 fontsize=23, fontweight="bold", y=1.01)
    fig.text(0.005, 0.99, "A", fontsize=PANEL_LABEL_FS, fontweight="bold",
             va="top", ha="left", transform=fig.transFigure)
    fig.text(0.5, -0.06,
             f"Unique attention variance = R²(attn+CLIP) − R²(CLIP-only), n=8 NSD subjects, "
             f"shared-1000, reliable vertices (ncsnr>0.2). Attention band = {head_txt}. "
             f"Dashed line = Shuffled floor; stars = significantly ABOVE floor (paired t, Holm/task); "
             f"ns = does not clear floor. Error bars SEM.",
             ha="center", fontsize=10.5, color="#555")
    fig.tight_layout()
    out = PLOTS / f"nsd_wholebrain_clip_controlled_{rule}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)

    print(f"\nWhole-brain unique attn R² over CLIP vs Shuffled floor (head-rule={rule}):")
    for task in TASKS:
        print(f"\n=== {task} (floor={floor[task]:.4f}) ===")
        s = sig[sig.task == task].sort_values("unique", ascending=False)
        for _, r in s.iterrows():
            print(f"    {r['model']:16s} unique={r['unique']:.4f}  "
                  f"vs floor diff={r['diff']:+.4f}  dz={r['dz']:+.2f}  "
                  f"p_holm={r['p_holm']:.3f} {star(r['p_holm'])}")


if __name__ == "__main__":
    main()
