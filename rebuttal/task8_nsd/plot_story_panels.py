"""Story panels for the NSD encoding analysis, written for the rebuttal reasoning.

Produces four figures in rebuttal/plots/:
  1. nsd_primary_manuscript_style.png  - shared-1000, whole-brain attention-only mean R2,
     6 models x 2 tasks. Matches manuscript Fig m6c Panel A (which plots attention-only R2),
     minus the Human bar (no eye-tracking exists for shared-1000).
  2. tribe_vs_nsd_consistency.png      - the SAME metric (whole-brain attention-only mean R2)
     on TRIBE (simulated) and NSD (real), side by side, so the model ordering can be compared
     across datasets. This is the cross-dataset replication question.
  3. nsd_roi_regional.png              - the rigorous result: unique variance over CLIP by
     cortical stream, floor-corrected against Shuffled, both tasks. Visualises the ROI table.
  4. nsd_exploratory_overlap.png       - the 106-image overlap: per-image attention-only
     prediction r by model, both tasks. This is the set where human gaze also exists.

All NSD numbers are read straight from the per-vertex encoding npz, averaged across the
analysed (ncsnr>0.2, all-subject-finite) vertices, then across the 8 subjects.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
ENC = HERE / "encoding"
RES = HERE / "results"
PLOTS = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots")
# figures are grouped into subdirs by what they show (see plots/README.md)
WHOLEBRAIN = PLOTS / "wholebrain"
REGIONAL = PLOTS / "regional"
CROSS_DATASET = PLOTS / "cross_dataset"
for _d in (WHOLEBRAIN, REGIONAL, CROSS_DATASET):
    _d.mkdir(parents=True, exist_ok=True)
TRIBE = Path("/scratch/gpfs/GRAZIANO/CANS/tribe/outputs")

MODELS = ["CNN_LSTM", "ViT_LSTM", "CNN_Transformer", "ViT_Transformer", "Molmo", "Qwen35"]
TRIBE_NAME = {"CNN_LSTM": "CNN-LSTM", "ViT_LSTM": "ViT-LSTM",
              "CNN_Transformer": "CNN-Transformer", "ViT_Transformer": "ViT-Transformer",
              "Molmo": "Molmo", "Qwen35": "Qwen3.5"}
COLORS = {"CNN_LSTM": "#ff7f0e", "ViT_LSTM": "#2ca02c",
          "CNN_Transformer": "#d62728", "ViT_Transformer": "#9467bd",
          "Molmo": "#8c564b", "Qwen35": "#17becf"}
LABELS = {"CNN_LSTM": "CNN-LSTM", "ViT_LSTM": "ViT-LSTM",
          "CNN_Transformer": "CNN-Transf.", "ViT_Transformer": "ViT-Transf.",
          "Molmo": "Molmo", "Qwen35": "Qwen3.5"}
TASKS = ["social", "describe"]
HEAD_RULE = "mean"          # overwritten by --head-rule in __main__
SUFFIX = ""                 # "" for mean, "_best" for best-head; appended to filenames


# ----------------------------------------------------------------------------------------
# Load NSD per-subject whole-brain metrics straight from the per-vertex npz
# ----------------------------------------------------------------------------------------
def nsd_wholebrain(rule=HEAD_RULE):
    """Return long df: subject, model, task, r2_attn (attention-only), attn_unique."""
    rows = []
    for model in MODELS:
        for task in TASKS:
            for subj in range(1, 9):
                f = ENC / f"r2_{model}_{task}_subj{subj:02d}_{rule}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                rows.append({
                    "subject": subj, "model": model, "task": task,
                    "r2_attn": float(d["r2_attn"].mean()),
                    "attn_unique": float((d["r2_banded"] - d["r2_clip"]).mean()),
                })
    return pd.DataFrame(rows)


def bar_panel(ax, order, means, sems, colors, hatches, ylabel=None, title=None):
    xs = np.arange(len(order))
    bars = ax.bar(xs, means, yerr=sems, capsize=3, width=0.62,
                  color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    ax.axhline(0, color="#333", lw=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", color="#ececec", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return bars


# ========================================================================================
# FIGURE 1: primary, manuscript style (attention-only whole-brain R2, shared-1000)
# ========================================================================================
def fig_primary(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, task in zip(axes, TASKS):
        sub = df[df.task == task]
        means, sems, cols = [], [], []
        for m in MODELS:
            g = sub[sub.model == m]["r2_attn"]
            means.append(g.mean())
            sems.append(g.std(ddof=1) / np.sqrt(len(g)))
            cols.append(COLORS[m])
        bar_panel(ax, [LABELS[m] for m in MODELS], means, sems, cols,
                  [""] * len(MODELS),
                  ylabel="Mean attention-only R²\n(whole brain, ncsnr>0.2 vertices)"
                  if task == "social" else None,
                  title=f"{task.capitalize()} task")
    fig.suptitle("Predicting REAL brain activity (NSD) from attention maps alone\n"
                 "n = 8 subjects, shared-1000 images", fontsize=13, fontweight="bold",
                 y=1.06)
    fig.text(0.5, -0.13,
             "Attention-only ridge (no CLIP control), matching manuscript Fig m6c Panel A. "
             "Bars: mean over 8 NSD subjects, error bars SEM.\n"
             "Human-gaze bar omitted: shared-1000 has no eye-tracking. This is the pooled "
             "whole-brain number; see nsd_roi_regional.png for the CLIP-controlled, "
             "region-resolved result.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    out = WHOLEBRAIN / f"nsd_primary_manuscript_style{SUFFIX}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# ========================================================================================
# FIGURE 2: TRIBE vs NSD, same metric, cross-dataset consistency
# ========================================================================================
def fig_consistency(df):
    tribe = {}
    for task in TASKS:
        for m in MODELS:
            p = TRIBE / f"r2_{TRIBE_NAME[m]}_{task}.npy"
            tribe[(m, task)] = float(np.load(p).mean()) if p.exists() else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    rho_lines = []
    for col, task in enumerate(TASKS):
        sub = df[df.task == task]
        nsd_means = {m: sub[sub.model == m]["r2_attn"].mean() for m in MODELS}
        nsd_sems = {m: sub[sub.model == m]["r2_attn"].std(ddof=1) / np.sqrt(8) for m in MODELS}
        tri_means = {m: tribe[(m, task)] for m in MODELS}

        # top row: paired bars (TRIBE vs NSD) on independent y-scales
        ax = axes[0, col]
        xs = np.arange(len(MODELS))
        w = 0.38
        # normalise each dataset to its own max so ORDERING is comparable despite scale
        tmax = max(tri_means.values())
        nmax = max(nsd_means.values())
        ax.bar(xs - w / 2, [tri_means[m] / tmax for m in MODELS], w,
               color=[COLORS[m] for m in MODELS], alpha=0.55, edgecolor="white",
               label="TRIBE (simulated)", hatch="///")
        ax.bar(xs + w / 2, [nsd_means[m] / nmax for m in MODELS], w,
               color=[COLORS[m] for m in MODELS], edgecolor="white",
               label="NSD (real)")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{task.capitalize()} task — attention-only R² "
                     "(each dataset scaled to its own max)", fontsize=11)
        ax.set_ylabel("Relative attention-only R²", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if col == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper right")

        # bottom row: scatter TRIBE vs NSD, one point per model
        ax2 = axes[1, col]
        xt = np.array([tri_means[m] for m in MODELS])
        yn = np.array([nsd_means[m] for m in MODELS])
        yn_sem = np.array([nsd_sems[m] for m in MODELS])
        for m, x, y in zip(MODELS, xt, yn):
            ax2.scatter(x, y, s=130, color=COLORS[m], edgecolor="k", linewidth=0.6, zorder=3)
        ax2.errorbar(xt, yn, yerr=yn_sem, fmt="none", ecolor="#888", lw=1, zorder=2)
        for m, x, y in zip(MODELS, xt, yn):
            ax2.annotate(LABELS[m], (x, y), fontsize=8, xytext=(5, 4),
                         textcoords="offset points")
        rho, p = stats.spearmanr(xt, yn)
        rho_lines.append(f"{task}: Spearman rho={rho:+.2f} (p={p:.2f})")
        ax2.set_xlabel("TRIBE attention-only R² (simulated)", fontsize=9)
        ax2.set_ylabel("NSD attention-only R² (real)", fontsize=9)
        ax2.set_title(f"Cross-dataset model ordering — Spearman ρ = {rho:+.2f}", fontsize=11)
        ax2.grid(color="#eee", lw=0.7, zorder=0)
        ax2.set_axisbelow(True)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)

    fig.suptitle("Does the model ordering replicate across simulated (TRIBE) and real (NSD) "
                 "brain data?\nSame metric on both: whole-brain attention-only mean R²",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = CROSS_DATASET / f"tribe_vs_nsd_consistency{SUFFIX}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out, "|", "  ".join(rho_lines))


# ========================================================================================
# FIGURE 3: ROI regional (the rigorous, CLIP-controlled, floor-corrected result)
# ========================================================================================
def fig_roi():
    per = pd.read_csv(RES / f"final_per_subject_{HEAD_RULE}.csv")  # attn_corrected vs Shuffled
    # paired-t of (unique - floor) vs 0, Holm-corrected within ROI x task -> the floor test
    floorsig = pd.read_csv(RES / f"final_vs_floor_{HEAD_RULE}_Shuffled.csv")
    sig_lut = {(r.task, r.roi, r.model): (r.corrected, r.p_holm)
               for r in floorsig.itertuples()}
    star = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    ROIS = ["early", "ventral", "lateral", "parietal"]
    rois = [r for r in ROIS if r in set(per.roi)]
    fig, axes = plt.subplots(2, len(rois), figsize=(3.4 * len(rois), 8), sharex=True)
    base_grey = "#9e9e9e"
    baselines = ["EdgeEnergy", "Luminance"]
    plot_models = MODELS + baselines
    for row, task in enumerate(TASKS):
        for col, roi in enumerate(rois):
            ax = axes[row, col]
            sub = per[(per.task == task) & (per.roi == roi)]
            means, sems, cols, hats, labs, keys = [], [], [], [], [], []
            for m in plot_models:
                g = sub[sub.model == m]["attn_corrected"].dropna()
                if len(g) < 3:
                    continue
                means.append(g.mean())
                sems.append(g.std(ddof=1) / np.sqrt(len(g)))
                is_b = m in baselines
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
            # stars: only when the bar significantly CLEARS the floor (corrected>0 & p_holm<.05)
            ytop = max([mn + se for mn, se in zip(means, sems)] + [0.0])
            off = 0.035 * ytop if ytop > 0 else 1e-5
            for xi, (mn, se, k) in enumerate(zip(means, sems, keys)):
                corr, ph = sig_lut.get((task, roi, k), (np.nan, np.nan))
                mark = star(ph) if (np.isfinite(ph) and corr > 0) else ""
                if mark:
                    ax.text(xs[xi], mn + se + off, mark, ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color="#222")
            lo, hi = ax.get_ylim()  # headroom so stars are not clipped by the panel top
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
    fig.suptitle("Model attention beyond CLIP, by cortical stream (floor-corrected vs "
                 "Shuffled)\nOnly CNN-LSTM clears the floor in ventral/lateral; every model "
                 "clears it in early visual", fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, "Grey hatched = low-level baselines (Sobel edge energy, luminance). "
             "Bars: mean over 8 subjects, SEM. Stars = significantly clears the Shuffled floor "
             "(paired t vs floor, Holm within ROI×task: *** p<.001, ** p<.01, * p<.05). "
             "y-scales differ by row/column — early-visual effects are ~6x larger.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    out = REGIONAL / f"nsd_roi_regional{SUFFIX}_holm_corrected_across_model.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# ========================================================================================
# FIGURE 4: exploratory overlap (per-image attention-only prediction r)
# ========================================================================================
def fig_overlap():
    f = sorted(RES.glob("overlap_predictions_mean_subj*.csv"))
    df = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
    order = ["CNN_LSTM", "ViT_LSTM", "CNN_Transformer", "ViT_Transformer",
             "Molmo", "Qwen35"]
    base = ["Shuffled", "Luminance"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, task in zip(axes, TASKS):
        sub = df[df.task == task]
        means, sems, cols, hats, labs = [], [], [], [], []
        for m in order + base:
            g = sub[sub.model == m]["r_attn_only"].dropna()
            if len(g) == 0:
                continue
            means.append(g.mean())
            sems.append(g.std(ddof=1) / np.sqrt(len(g)))
            is_b = m in base
            cols.append("#9e9e9e" if is_b else COLORS.get(m, "#666"))
            hats.append("///" if is_b else "")
            labs.append(LABELS.get(m, m))
        bar_panel(ax, labs, means, sems, cols, hats,
                  ylabel="Mean per-image prediction r\n(attention-only, held-out overlap "
                  "images)" if task == "social" else None,
                  title=f"{task.capitalize()} task")
    fig.suptitle("Exploratory: attention-only prediction of REAL fMRI on the 106 overlap "
                 "images\n(the only images with both our eye-tracking and NSD; each seen by "
                 "one subject, n≈98)", fontsize=12.5, fontweight="bold", y=1.06)
    fig.text(0.5, -0.13, "Only 3 of these 6 models (CNN-LSTM, Molmo, Qwen3.5) also have "
             "per-image human-fixation alignment, so only they enter the fixation↔neural "
             "link test.\nGrey hatched = baselines. Per-image r, not whole-brain R² — a "
             "different (noisier) metric than the shared-1000 panel.", ha="center",
             fontsize=8, color="#555")
    fig.tight_layout()
    out = CROSS_DATASET / "nsd_exploratory_overlap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="mean", choices=["mean", "best", "cvbehav"])
    args = ap.parse_args()
    HEAD_RULE = args.head_rule
    SUFFIX = "" if HEAD_RULE == "mean" else f"_{HEAD_RULE}"

    df = nsd_wholebrain(HEAD_RULE)
    print(f"[head-rule={HEAD_RULE}] NSD whole-brain attention-only R² (mean over 8 subj):")
    print(df.groupby(["task", "model"])["r2_attn"].mean().round(5).to_string())
    fig_primary(df)
    fig_consistency(df)
    fig_roi()
    # Overlap analysis only exists for the mean rule (fit_overlap.py --head-rule mean);
    # skip it for best-head unless those fits are also produced.
    if HEAD_RULE == "mean":
        fig_overlap()
