"""TRIBE vs NSD on the ATTENTION-ONLY metric, by cortical stream.

The CLIP-controlled (unique-over-CLIP) partition is not reliably estimable on TRIBE: with 200
images and a CLIP band that already explains R^2~0.31 of the simulated responses, the banded
ridge's kernel-weight selection is unstable and drives banded R^2 below CLIP-only for the
high-variance LSTM maps (verified: an NSD-recipe refit reproduces banded=0.168 < clip=0.312).
That instability is the "underpowered variance partition" the reviewers flagged about the TRIBE
analysis, and it is why NSD was needed. So on TRIBE we report the STABLE quantity — attention-only
R^2 — the same metric the manuscript's Panel A and the whole-brain TRIBE-vs-NSD figure use.

TRIBE has no subject dimension, so its bars are point estimates (r2_{model}_{task}.npy averaged
over a vertex set). NSD attention-only comes from the per-subject ROI fits (8-subject mean, SEM).
Both use the same fsaverage5 streams atlas; NSD additionally restricts to ncsnr>0.2 within each
stream (TRIBE has no noise ceiling).

Outputs (rebuttal/plots/):
  regional/tribe_roi_attn_only.png            TRIBE point estimates, per stream, both tasks
  regional/tribe_vs_nsd_roi_attn_only.png     matched TRIBE (point) vs NSD (mean+-SEM) per stream
  wholebrain/tribe_roi_attn_only_values.csv   the numbers
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
TRIBE = Path("/scratch/gpfs/GRAZIANO/CANS/tribe/outputs")
ROIS = HERE / "rois" / "roi_masks_fs5.npz"
RES = HERE / "results"
PLOTS = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots")
(PLOTS / "regional").mkdir(parents=True, exist_ok=True)
(PLOTS / "wholebrain").mkdir(parents=True, exist_ok=True)

# display (TRIBE file) <-> underscore (NSD csv) names, shared palette
TRIBE_NAME = ["CNN-LSTM", "ViT-LSTM", "CNN-Transformer", "ViT-Transformer", "Molmo", "Qwen3.5"]
NSD_KEY = {"CNN-LSTM": "CNN_LSTM", "ViT-LSTM": "ViT_LSTM", "CNN-Transformer": "CNN_Transformer",
           "ViT-Transformer": "ViT_Transformer", "Molmo": "Molmo", "Qwen3.5": "Qwen35"}
COLORS = {"CNN-LSTM": "#ff7f0e", "ViT-LSTM": "#2ca02c", "CNN-Transformer": "#d62728",
          "ViT-Transformer": "#9467bd", "Molmo": "#8c564b", "Qwen3.5": "#17becf"}
TASKS = ["social", "describe"]
ROI_MASK = {"early": "grp_early", "ventral": "grp_ventral",
            "lateral": "grp_lateral", "parietal": "grp_parietal"}
ROIS_ORDER = ["early", "ventral", "lateral", "parietal"]


def tribe_attn_only(masks):
    rows = []
    for m in TRIBE_NAME:
        for task in TASKS:
            r2 = np.load(TRIBE / f"r2_{m}_{task}.npy")
            rows.append({"model": m, "task": task, "roi": "nsdgeneral",
                         "tribe": float(r2[masks["nsdgeneral"]].mean())})
            for roi in ROIS_ORDER:
                rows.append({"model": m, "task": task, "roi": roi,
                             "tribe": float(r2[masks[ROI_MASK[roi]]].mean())})
    return pd.DataFrame(rows)


def nsd_attn_only():
    """8-subject mean +- SEM of attention-only R2 per ROI, from the per-subject ROI fits.

    Head-rule note: TRIBE's attention features (2_feature_extraction.py) are the paper's fixed
    best heads (CNN-T L3H5, ViT-T L3H8, Molmo per-task, Qwen L15H12). NSD's `best` rule uses the
    SAME fixed heads (fit_encoding.py: CNN_T=4, ViT_T=7, Qwen=11, Molmo 23/26), so we read the
    NSD `_best` fits here to keep the two sides head-matched. LSTMs have a single map either way.
    """
    per = pd.read_csv(RES / "encoding_by_roi_per_subject_best.csv")
    inv = {v: k for k, v in NSD_KEY.items()}
    per = per[per.model.isin(NSD_KEY.values())].copy()
    per["model"] = per.model.map(inv)
    g = (per.groupby(["roi", "task", "model"])["r2_attn"]
         .agg(nsd="mean", nsd_sem=lambda x: x.std(ddof=1) / np.sqrt(len(x))).reset_index())
    return g


def fig_tribe_roi(df):
    fig, axes = plt.subplots(2, len(ROIS_ORDER), figsize=(3.4 * len(ROIS_ORDER), 8), sharex=True)
    for r, task in enumerate(TASKS):
        for c, roi in enumerate(ROIS_ORDER):
            ax = axes[r, c]
            sub = df[(df.task == task) & (df.roi == roi)]
            vals = [sub[sub.model == m]["tribe"].iloc[0] for m in TRIBE_NAME]
            ax.bar(np.arange(len(TRIBE_NAME)), vals, width=0.7,
                   color=[COLORS[m] for m in TRIBE_NAME], edgecolor="white", linewidth=0.9, zorder=3)
            ax.axhline(0, color="#333", lw=0.8)
            ax.set_xticks(np.arange(len(TRIBE_NAME)))
            ax.set_xticklabels(TRIBE_NAME, rotation=55, ha="right", fontsize=7.5)
            if r == 0:
                ax.set_title(roi, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{task.capitalize()} task\nattention-only R²", fontsize=9)
            ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle("TRIBE (simulated): attention-only R² by cortical stream\n"
                 "Point estimates — no subject dimension", fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, "r2_{model}_{task} over the streams atlas. y-scales differ by column.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    out = PLOTS / "regional" / "tribe_roi_attn_only.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def fig_matched(df, nsd):
    merged = df[df.roi != "nsdgeneral"].merge(nsd, on=["roi", "task", "model"], how="left")
    fig, axes = plt.subplots(2, len(ROIS_ORDER), figsize=(3.6 * len(ROIS_ORDER), 8), sharex=True)
    x = np.arange(len(TRIBE_NAME))
    w = 0.4
    rho_txt = {}
    for r, task in enumerate(TASKS):
        for c, roi in enumerate(ROIS_ORDER):
            ax = axes[r, c]
            sub = merged[(merged.task == task) & (merged.roi == roi)].set_index("model").loc[TRIBE_NAME]
            nsd_v = sub["nsd"].values
            nsd_e = sub["nsd_sem"].values
            tri_v = sub["tribe"].values
            ax.bar(x - w / 2, nsd_v, w, yerr=nsd_e, capsize=2,
                   color=[COLORS[m] for m in TRIBE_NAME], edgecolor="white", linewidth=0.6,
                   label="NSD (real)")
            ax.bar(x + w / 2, tri_v, w, color=[COLORS[m] for m in TRIBE_NAME], alpha=0.55,
                   hatch="///", edgecolor="white", linewidth=0.6, label="TRIBE (sim)")
            ax.axhline(0, color="#333", lw=0.8)
            rho, p = stats.spearmanr(nsd_v, tri_v)
            rho_txt[(task, roi)] = (rho, p)
            ax.set_xticks(x)
            ax.set_xticklabels(TRIBE_NAME, rotation=55, ha="right", fontsize=7)
            if r == 0:
                ax.set_title(f"{roi}\nρ(NSD,TRIBE)={rho:+.2f}", fontsize=9.5)
            else:
                ax.set_title(f"ρ={rho:+.2f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{task.capitalize()} task\nattention-only R²", fontsize=9)
            ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, -1].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle("Attention-only R² by cortical stream: NSD (real, solid) vs TRIBE (simulated, "
                 "hatched)\nDo the two datasets agree on model ordering within each region?",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, "NSD: 8-subject mean ± SEM, ncsnr>0.2 within stream. TRIBE: point "
             "estimate over the same stream. Both sides use the paper's fixed best heads "
             "(LSTMs single map). ρ = Spearman across the 6 models. y-scales differ.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    out = PLOTS / "regional" / "tribe_vs_nsd_roi_attn_only.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)
    return rho_txt


def main():
    masks = {k: np.load(ROIS)[k].astype(bool) for k in
             ["nsdgeneral"] + list(ROI_MASK.values())}
    df = tribe_attn_only(masks)
    nsd = nsd_attn_only()
    df.to_csv(PLOTS / "wholebrain" / "tribe_roi_attn_only_values.csv", index=False)
    fig_tribe_roi(df)
    rho = fig_matched(df, nsd)

    print("\nTRIBE attention-only R² (point estimates):")
    for task in TASKS:
        print(f"\n=== {task} ===")
        for roi in ["nsdgeneral"] + ROIS_ORDER:
            s = df[(df.task == task) & (df.roi == roi)].sort_values("tribe", ascending=False)
            top = ", ".join(f"{r['model']}={r['tribe']:.4f}" for _, r in s.head(3).iterrows())
            print(f"  {roi:11s}: {top}")
    print("\nTRIBE-vs-NSD Spearman across 6 models (attention-only), per ROI:")
    for (task, roi), (r, p) in rho.items():
        print(f"  {task:9s} {roi:9s}: rho={r:+.2f}  p={p:.3f}")


if __name__ == "__main__":
    main()
