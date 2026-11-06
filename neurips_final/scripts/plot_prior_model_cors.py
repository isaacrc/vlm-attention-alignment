"""Model-to-prior map similarity figure (not in the manuscript; for talks).

Alignment scores say how well a model predicts human fixation; two maps can tie on that
while looking nothing alike. This plots the direct per-image Spearman correlation between
each model's attention map and each attentional prior, which is what would license a claim
that a model "recovers" a bias.

Source: rebuttal/task3_baselines/results/model_prior_similarity.csv
        (rebuttal/task3_baselines/model_prior_similarity.py)
Output: ATTN_PRIOR_TO_MODEL_CORS.png / .csv
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, "/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/shared")
import figstyle as FST

SRC = "/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task3_baselines/results/model_prior_similarity.csv"
OUT = "/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/results"
MODELS = ["CNN-LSTM", "ViT-LSTM", "CNN-Transformer", "ViT-Transformer", "Molmo", "Qwen3.5"]
SHORT = {"CNN-LSTM": "CNN-LSTM", "ViT-LSTM": "ViT-LSTM", "CNN-Transformer": "CNN-Transf.",
         "ViT-Transformer": "ViT-Transf.", "Molmo": "Molmo 7B-D", "Qwen3.5": "Qwen 9B-3.5"}
PRIORS = ["Center prior", "Face prior", "Person prior"]
# Three slots from the manuscript palette; validated (all checks pass) for three categories.
PC = {"Center prior": "#c86313", "Face prior": "#1374aa", "Person prior": "#139a75"}
TASKS = [("social", "Social Task"), ("describe", "Describe Task")]

df = pd.read_csv(SRC)
df.to_csv(os.path.join(OUT, "ATTN_PRIOR_TO_MODEL_CORS.csv"), index=False)

fs = FST.font_sizes(0.72)
fig, axes = plt.subplots(1, 2, figsize=(26, 9), sharey=True)
x = np.arange(len(MODELS)); w = 0.26
for ax, (task, title) in zip(axes, TASKS):
    for k, pr in enumerate(PRIORS):
        sub = df[(df.task == task) & (df.prior == pr)].set_index("model")
        vals = [sub.loc[m, "r"] for m in MODELS]
        sems = [sub.loc[m, "sem"] for m in MODELS]
        off = (k - 1) * w
        ax.bar(x + off, vals, w, color=PC[pr], edgecolor=FST.BAR_EDGE,
               linewidth=FST.BAR_EDGE_LW, zorder=3)
        ax.errorbar(x + off, vals, yerr=sems, **FST.errorbar_kwargs(0.72))
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=fs["tick"], rotation=45, ha="right")
    ax.set_title(title, fontsize=fs["title"], fontweight="bold", pad=16)
    FST.style_axes(ax, fs)
axes[0].set_ylabel("Map-to-map Spearman $r$\n(model attention vs. prior)",
                   fontsize=fs["label"], labelpad=14)
fig.legend(handles=[mpatches.Patch(facecolor=PC[p], edgecolor="black", label=p) for p in PRIORS],
           loc="lower center", ncol=3, fontsize=fs["legend"], frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("How much does each model's attention resemble each attentional prior?",
             fontsize=fs["title"], fontweight="bold", y=1.01)
fig.tight_layout(rect=[0, 0.09, 1, 1])
out = os.path.join(OUT, "ATTN_PRIOR_TO_MODEL_CORS.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
print(df.pivot_table(index="model", columns=["prior", "task"], values="r").round(3).to_string())
