"""ROBUSTNESS SUPPLEMENT -- metric check, NOT a final analysis.

The final whole-brain / ROI story reports unique attention variance = R2(banded) - R2(clip-only)
where R2 = signed squared cross-validated Pearson r ("corr2"). Variance partitioning is really
defined on the coefficient of determination, so this figure re-runs the identical banded-ridge
pipeline scored with himalaya's r2_score ("r2", encoding_r2score_supplement/) and asks the only
question that matters: does the model ordering and the floor-clearance of the unique-over-CLIP
signal survive the metric swap?

Both sides: best-head rule, Shuffled floor, ncsnr>0.2 vertices, 8 subjects.
Reads:  encoding/ (corr2)  and  encoding_r2score_supplement/ (r2)
Writes: plots/supplement/nsd_wholebrain_unique_SUPPLEMENT_corr2_vs_r2score.png
        plots/supplement/nsd_wholebrain_unique_SUPPLEMENT_corr2_vs_r2score.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
DIRS = {"corr2": HERE / "encoding", "r2": HERE / "encoding_r2score_supplement"}
PLOTS = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots/supplement")
PLOTS.mkdir(parents=True, exist_ok=True)
RULE = "best"

ATTN_MODELS = ["CNN-LSTM", "ViT-LSTM", "CNN-Transformer", "ViT-Transformer", "Molmo", "Qwen3.5"]
KEY = {"CNN-LSTM": "CNN_LSTM", "ViT-LSTM": "ViT_LSTM", "CNN-Transformer": "CNN_Transformer",
       "ViT-Transformer": "ViT_Transformer", "Molmo": "Molmo", "Qwen3.5": "Qwen35"}
COLORS = {"CNN-LSTM": "#ff7f0e", "ViT-LSTM": "#2ca02c", "CNN-Transformer": "#d62728",
          "ViT-Transformer": "#9467bd", "Molmo": "#8c564b", "Qwen3.5": "#17becf"}
TASKS = ["social", "describe"]


def per_subject_unique(metric):
    """Per-subject whole-brain unique variance (banded - clip), floor-corrected vs Shuffled."""
    enc = DIRS[metric]
    rows = []
    for disp in ATTN_MODELS + ["__Shuffled__"]:
        mkey = "Shuffled" if disp == "__Shuffled__" else KEY[disp]
        for task in TASKS:
            for subj in range(1, 9):
                f = enc / f"r2_{mkey}_{task}_subj{subj:02d}_{RULE}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                rows.append({"model": disp, "task": task, "subject": subj,
                             "unique": float((d["r2_banded"] - d["r2_clip"]).mean())})
    df = pd.DataFrame(rows)
    # floor-correct each subject/task by its own Shuffled unique variance
    floor = (df[df.model == "__Shuffled__"].set_index(["subject", "task"])["unique"].rename("floor"))
    df = df[df.model != "__Shuffled__"].join(floor, on=["subject", "task"])
    df["corrected"] = df["unique"] - df["floor"]
    return df


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run, m = 0.0, len(p)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def summarize(df):
    """mean/SEM + one-sided floor test (corrected>0 & Holm p<.05), per model x task."""
    g = (df.groupby(["task", "model"])["corrected"]
         .agg(mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x))).reset_index())
    sig = {}
    for task, sub in df.groupby("task"):
        piv = sub.pivot_table(index="subject", columns="model", values="corrected")
        models = [m for m in ATTN_MODELS if m in piv.columns]
        t, p = zip(*[stats.ttest_1samp(piv[m].dropna(), 0.0) for m in models])
        ph = holm(np.array(p))
        for m, corr, pp in zip(models, [piv[m].mean() for m in models], ph):
            sig[(task, m)] = "*" if (corr > 0 and pp < .05) else ""
    return g, sig


def main():
    missing = [m for m, d in DIRS.items() if not any(d.glob(f"r2_*_{RULE}.npz"))]
    if missing:
        print(f"NOT READY -- no {RULE} npz in: {', '.join(DIRS[m].name for m in missing)}")
        print("Run the r2score supplement SLURM job first (slurm_fit_encoding_r2score_supplement.sh).")
        return

    data = {mt: summarize(per_subject_unique(mt)) for mt in DIRS}
    rows = []
    for mt, (g, sig) in data.items():
        for r in g.itertuples():
            rows.append({"metric": mt, "task": r.task, "model": r.model,
                         "corrected": r.mean, "sem": r.sem, "clears_floor": sig[(r.task, r.model)]})
    tidy = pd.DataFrame(rows)
    tidy.to_csv(PLOTS / "nsd_wholebrain_unique_SUPPLEMENT_corr2_vs_r2score.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    x = np.arange(len(ATTN_MODELS))
    w = 0.38
    rho_txt = {}
    for ax, task in zip(axes, TASKS):
        c_g, c_sig = data["corr2"]
        r_g, r_sig = data["r2"]
        cg = c_g[c_g.task == task].set_index("model").reindex(ATTN_MODELS)
        rg = r_g[r_g.task == task].set_index("model").reindex(ATTN_MODELS)
        ax.bar(x - w / 2, cg["mean"], w, yerr=cg["sem"], capsize=2,
               color=[COLORS[m] for m in ATTN_MODELS], edgecolor="white",
               label="corr² (final)")
        ax.bar(x + w / 2, rg["mean"], w, yerr=rg["sem"], capsize=2,
               color=[COLORS[m] for m in ATTN_MODELS], alpha=0.5, hatch="///",
               edgecolor="white", label="r2_score (supplement)")
        ax.axhline(0, color="#333", lw=0.8)
        # stars above whichever metric clears its floor
        for xi, m in enumerate(ATTN_MODELS):
            if c_sig[(task, m)]:
                ax.text(x[xi] - w / 2, cg["mean"][m] + cg["sem"][m] + 1e-4, "*",
                        ha="center", fontweight="bold")
            if r_sig[(task, m)]:
                ax.text(x[xi] + w / 2, rg["mean"][m] + rg["sem"][m] + 1e-4, "*",
                        ha="center", fontweight="bold")
        rho, p = stats.spearmanr(cg["mean"], rg["mean"])
        rho_txt[task] = (rho, p)
        ax.set_title(f"{task.capitalize()} task\nρ(corr², r2_score orderings)={rho:+.2f}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(ATTN_MODELS, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("Unique attn variance over CLIP\n(floor-corrected vs Shuffled)", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("SUPPLEMENT — metric robustness of the unique-over-CLIP signal (best head)\n"
                 "Same banded-ridge pipeline scored two ways: signed squared Pearson r vs true R² "
                 "(himalaya r2_score)", fontsize=12.5, fontweight="bold", y=1.04)
    fig.text(0.5, -0.06, "Solid = corr² (the final metric). Hatched = r2_score (coefficient of "
             "determination). Bars: 8-subject mean ± SEM, whole-brain over ncsnr>0.2 vertices, "
             "floor-corrected by each subject's Shuffled unique variance. * clears the Shuffled "
             "floor (one-sided paired t, Holm within task, p<.05). If the ordering and the "
             "CNN-LSTM floor-clearance hold across both, the metric choice is not driving the result.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    out = PLOTS / "nsd_wholebrain_unique_SUPPLEMENT_corr2_vs_r2score.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)
    print("\nSUPPLEMENT whole-brain unique variance, corr2 vs r2_score (best head):")
    print(tidy.to_string(index=False))
    print("\nSpearman(corr2 ordering, r2_score ordering):")
    for task, (r, p) in rho_txt.items():
        print(f"  {task:9s}: rho={r:+.2f}  p={p:.3f}")


if __name__ == "__main__":
    main()
