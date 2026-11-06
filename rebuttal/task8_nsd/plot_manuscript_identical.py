"""Reproduce manuscript Fig m6c Panel A *identically* in style, with NSD real-fMRI data.

The original (tribe/scripts/m6c_combined_panel_m4d_top.py, Panel A) plots whole-brain
attention-only mean R^2 as one bar per model, two task panels sharing y, with a specific
palette and font scheme. This copies that styling verbatim and swaps in the NSD numbers:

  metric   = attention-only ridge R^2 (no CLIP control), the same quantity the original
             Panel A plots (r2_{model}_{task}.npy there == r2_attn here)
  vertices = the 5,032 analysed vertices (ncsnr>0.2, finite in all subjects). The original
             averaged over its full simulated surface; on real fMRI the low-SNR vertices are
             noise, so the honest analogue is the reliable-vertex mean. Only the y-label wording
             differs from the manuscript; the visual style does not.
  bars     = 6 models, no Human bar (shared-1000 has no eye-tracking) -- the shared version.

Also computes bar-to-bar significance: paired t across the 8 subjects for every model pair
(Holm-corrected within task) and each model vs the Shuffled floor.

Usage: python plot_manuscript_identical.py --head-rule {mean,best}
Output: rebuttal/plots/nsd_identical_manuscript_plot_attn_only_no_control_for_clip_{rule}.png
        rebuttal/plots/nsd_identical_manuscript_plot_attn_only_no_control_for_clip_{rule}_sigtests.csv
"""

import argparse
import itertools
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

# ---- styling copied verbatim from m6c_combined_panel_m4d_top.py Panel A -----------------
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

# map manuscript display names <-> NSD file model keys
KEY = {"CNN-LSTM": "CNN_LSTM", "ViT-LSTM": "ViT_LSTM", "CNN-Transformer": "CNN_Transformer",
       "ViT-Transformer": "ViT_Transformer", "Molmo": "Molmo", "Qwen3.5": "Qwen35"}


def per_subject_wholebrain(rule):
    """Per-subject whole-brain (reliable-vertex) attention-only R^2, plus the Shuffled floor."""
    rows = []
    for disp in ATTN_MODELS + ["__Shuffled__"]:
        mkey = "Shuffled" if disp == "__Shuffled__" else KEY[disp]
        for task in TASKS:
            for subj in range(1, 9):
                f = ENC / f"r2_{mkey}_{task}_subj{subj:02d}_{rule}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                rows.append({"model": disp, "task": task, "subject": subj,
                             "r2_attn": float(d["r2_attn"].mean())})
    return pd.DataFrame(rows)


def sig_tests(df):
    """Paired t across subjects: every model pair + each model vs Shuffled, Holm within task."""
    def holm(p):
        p = np.asarray(p, float)
        order = np.argsort(p)
        adj = np.empty_like(p)
        run = 0.0
        m = len(p)
        for rank, i in enumerate(order):
            run = max(run, (m - rank) * p[i])
            adj[i] = min(run, 1.0)
        return adj

    out = []
    for task in TASKS:
        sub = df[df.task == task]
        piv = sub.pivot_table(index="subject", columns="model", values="r2_attn")
        # model-vs-model
        rows = []
        for a, b in itertools.combinations(ATTN_MODELS, 2):
            pair = piv[[a, b]].dropna()
            t, p = stats.ttest_rel(pair[a], pair[b])
            d = pair[a] - pair[b]
            rows.append({"task": task, "kind": "model-vs-model", "a": a, "b": b,
                         "mean_a": pair[a].mean(), "mean_b": pair[b].mean(),
                         "diff": d.mean(), "t": t, "p": p,
                         "dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan})
        # model-vs-Shuffled floor
        for a in ATTN_MODELS:
            pair = piv[[a, "__Shuffled__"]].dropna()
            t, p = stats.ttest_rel(pair[a], pair["__Shuffled__"])
            d = pair[a] - pair["__Shuffled__"]
            rows.append({"task": task, "kind": "vs-Shuffled", "a": a, "b": "Shuffled",
                         "mean_a": pair[a].mean(), "mean_b": pair["__Shuffled__"].mean(),
                         "diff": d.mean(), "t": t, "p": p,
                         "dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan})
        r = pd.DataFrame(rows)
        r["p_holm"] = holm(r["p"].values)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="mean", choices=["mean", "best"])
    args = ap.parse_args()
    rule = args.head_rule

    df = per_subject_wholebrain(rule)
    if df.empty:
        print(f"no fits for head-rule={rule} yet")
        return

    means = (df[df.model != "__Shuffled__"].groupby(["model", "task"])["r2_attn"]
             .agg(["mean", "sem"]).reset_index())

    # ---- figure, styled identically to m6c Panel A ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    x_pos = np.arange(len(ATTN_MODELS))
    for ti, task in enumerate(TASKS):
        ax = axes[ti]
        vals = {m: means[(means.model == m) & (means.task == task)]["mean"].iloc[0]
                for m in ATTN_MODELS}
        sems = {m: means[(means.model == m) & (means.task == task)]["sem"].iloc[0]
                for m in ATTN_MODELS}
        y_top = max(vals.values())
        for xi, model in enumerate(ATTN_MODELS):
            ax.bar(xi, vals[model], BAR_W, yerr=sems[model], capsize=3,
                   color=COLORS_BAR[model], edgecolor="white", linewidth=0.5)
        ax.set_ylim(0, y_top * 1.18)
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
            ax.set_ylabel("Mean R² (whole brain)", fontsize=FS_LABEL)

    head_txt = "layer-3 head mean" if rule == "mean" else "best head (behavioral)"
    fig.suptitle(f"Predicting Real Brain Activity (NSD) from Attention Maps",
                 fontsize=23, fontweight="bold", y=1.01)
    fig.text(0.005, 0.99, "A", fontsize=PANEL_LABEL_FS, fontweight="bold",
             va="top", ha="left", transform=fig.transFigure)
    fig.text(0.5, -0.05,
             f"Attention-only R² (no CLIP control), n=8 NSD subjects, shared-1000, "
             f"reliable vertices (ncsnr>0.2). Attention band = {head_txt}. Error bars SEM.",
             ha="center", fontsize=11, color="#555")
    fig.tight_layout()
    out = PLOTS / f"nsd_identical_manuscript_plot_attn_only_no_control_for_clip_{rule}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)

    sig = sig_tests(df)
    sig.to_csv(PLOTS / f"nsd_identical_manuscript_plot_attn_only_no_control_for_clip_{rule}_sigtests.csv",
               index=False)
    print(f"\nBar-to-bar significance (head-rule={rule}), whole-brain attention-only R²:")
    for task in TASKS:
        print(f"\n=== {task} ===")
        s = sig[(sig.task == task)]
        print("  vs Shuffled floor:")
        for _, r in s[s.kind == "vs-Shuffled"].sort_values("mean_a", ascending=False).iterrows():
            ph = r["p_holm"]
            star = "***" if ph < .001 else "**" if ph < .01 else "*" if ph < .05 else "ns"
            print(f"    {r['a']:16s} R2={r['mean_a']:.4f}  vs floor {r['mean_b']:.4f}  "
                  f"p_holm={ph:.3f} {star}")
        print("  model-vs-model (significant pairs, Holm within task):")
        mm = s[(s.kind == "model-vs-model") & (s["p_holm"] < .05)].sort_values("p_holm")
        if mm.empty:
            print("    (none reach p_holm<.05)")
        for _, r in mm.iterrows():
            # order the printed arrow by which mean is higher, not by combination order
            hi, lo = (r["a"], r["b"]) if r["diff"] > 0 else (r["b"], r["a"])
            print(f"    {hi:16s} > {lo:16s} |diff|={abs(r['diff']):.4f}  "
                  f"|dz|={abs(r['dz']):.2f}  p_holm={r['p_holm']:.3f}")


if __name__ == "__main__":
    main()
