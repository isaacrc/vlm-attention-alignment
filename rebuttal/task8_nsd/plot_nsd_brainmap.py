"""Reproduce manuscript Fig m6c Panel B with NSD real-fMRI data.

The original renders, for ONE model (CNN-Transformer, Social), the variance partition of the
banded encoding model on the cortical surface, three bands x four views:
    row 1  Unique attention R^2   ("Where")  = R2(banded) - R2(CLIP-only)
    row 2  Unique CLIP R^2        ("What")   = R2(banded) - R2(attention-only)
    row 3  Shared R^2                        = R2(attn) + R2(CLIP) - R2(banded)

Here the per-vertex R^2 come from the NSD per-subject fits (encoding/r2_*.npz), which store
values only on the analysed vertices (ncsnr>0.2). We scatter each subject's values back into
the full 20484-vertex fsaverage5 space (0 elsewhere), average the partition across the 8
subjects, and plot the group mean exactly like the original (nilearn plot_surf_stat_map,
'hot', clipped at 0). Styling copied from m6c_combined_panel_m4d_top.py Panel B.

Usage:  python plot_nsd_brainmap.py --model CNN_Transformer --task social --head-rule mean
Output: rebuttal/plots/nsd_brainmap_{model}_{task}_{rule}.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from nilearn import plotting, datasets as nidatasets

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
import figstyle as FST

HERE = Path(__file__).parent
ENC = HERE / "encoding"
PLOTS = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots/regional")
PLOTS.mkdir(parents=True, exist_ok=True)

N_TOTAL = 20484
N_LH = 10242
VIEWS = [("left", "lateral"), ("left", "medial"),
         ("right", "lateral"), ("right", "medial")]
ROW_LABELS = ['Unique Attn R²  ("Where")', 'Unique CLIP R²  ("What")', "Shared R²"]
DISPLAY = {"CNN_Transformer": "CNN-Transformer", "CNN_LSTM": "CNN-LSTM",
           "ViT_LSTM": "ViT-LSTM", "ViT_Transformer": "ViT-Transformer",
           "Molmo": "Molmo", "Qwen35": "Qwen3.5"}


def group_partition(model, task, rule):
    """Average the three variance-partition bands across the 8 subjects, full-space."""
    ua, uc, sh = np.zeros(N_TOTAL), np.zeros(N_TOTAL), np.zeros(N_TOTAL)
    n = 0
    for subj in range(1, 9):
        f = ENC / f"r2_{model}_{task}_subj{subj:02d}_{rule}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        v = d["vertices"]
        rb, ra, rc = d["r2_banded"], d["r2_attn"], d["r2_clip"]
        full_ua = np.zeros(N_TOTAL); full_uc = np.zeros(N_TOTAL); full_sh = np.zeros(N_TOTAL)
        full_ua[v] = np.clip(rb - rc, 0, None)
        full_uc[v] = np.clip(rb - ra, 0, None)
        full_sh[v] = np.clip(ra + rc - rb, 0, None)
        ua += full_ua; uc += full_uc; sh += full_sh
        n += 1
    if n == 0:
        raise SystemExit(f"no fits for {model}/{task}/{rule}")
    return ua / n, uc / n, sh / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CNN_Transformer")
    ap.add_argument("--task", default="social", choices=["social", "describe"])
    ap.add_argument("--head-rule", default="mean", choices=["mean", "best", "cvbehav"])
    ap.add_argument("--panel-letter", default=None,
                    help="bare bold panel letter drawn top-left, for the merged "
                         "manuscript figure; omit for standalone/appendix renders")
    ap.add_argument("--vmax-pct", type=float, default=99.0,
                    help="percentile of positive values used as the colour ceiling per row; "
                         "99 lets early visual dominate and hides small higher-order effects")
    args = ap.parse_args()

    ua, uc, sh, n = group_partition(args.model, args.task, args.head_rule)
    rows = [ua, uc, sh]
    print(f"{args.model}/{args.task} ({args.head_rule}), n={n} subjects | "
          f"unique-attn>0 vertices: {(ua>0).sum()}  "
          f"max where={ua.max():.4f} what={uc.max():.4f} shared={sh.max():.4f}")

    fs5 = nidatasets.fetch_surf_fsaverage("fsaverage5")
    sulc = {"left": fs5.get("sulc_left", fs5.get("curv_left")),
            "right": fs5.get("sulc_right", fs5.get("curv_right"))}
    pial = {"left": fs5["pial_left"], "right": fs5["pial_right"]}

    fig, axes = plt.subplots(3, 4, figsize=(14, 10),
                             subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.01,
                        hspace=-0.4, wspace=0.01)

    for ri in range(3):
        pos0 = axes[ri, 0].get_position()
        pos3 = axes[ri, 3].get_position()
        fig.text((pos0.x0 + pos3.x1) / 2, pos0.y1 - 0.01, ROW_LABELS[ri],
                 ha="center", va="top", fontsize=26, fontweight="bold")

    for ri, stat_map in enumerate(rows):
        pos = stat_map[stat_map > 0]
        vmax = max(float(np.percentile(pos, args.vmax_pct)) if len(pos) > 10 else 1e-5, 1e-5)
        for ci, (hemi, view) in enumerate(VIEWS):
            ax = axes[ri, ci]
            data = stat_map[:N_LH] if hemi == "left" else stat_map[N_LH:]
            plotting.plot_surf_stat_map(
                surf_mesh=pial[hemi], stat_map=data, hemi=hemi, view=view,
                bg_map=sulc[hemi], colorbar=(ci == 3), cmap="hot",
                vmin=0, vmax=vmax, threshold=None, axes=ax)
            ax.set_title("")

    if args.panel_letter:
        FST.panel_letter(fig, args.panel_letter)

    tag = "" if args.vmax_pct == 99.0 else f"_p{int(args.vmax_pct)}"
    out = PLOTS / f"nsd_brainmap_{args.model}_{args.task}_{args.head_rule}{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
