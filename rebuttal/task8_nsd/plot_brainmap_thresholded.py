"""Whole-brain map of the ATTENTION-ONLY encoding fit, statistically thresholded.

Each vertex is tested against that model's own permutation floor across the 8 NSD
participants and thresholded at FDR q<.05; vertices that do not survive are left grey.
Unlike the unthresholded variance-partition maps, every coloured vertex here is one that
passed a whole-brain corrected test.

Usage: python plot_brainmap_thresholded.py --model CNN_LSTM
Output: rebuttal/plots/nsd_brainmap_thresh_{model}.png   (rows: Social, Describe)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting, datasets as nidatasets
from scipy import stats

HERE = Path(__file__).parent
PLOTS = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots")
VERTS = 20484


def fdr(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    r = np.empty(n); r[o] = np.minimum(q, 1.0)
    return r


def thresholded(model, task):
    R, F = [], []
    for s in range(1, 9):
        a = np.load(HERE / "encoding" / f"r2_{model}_{task}_subj{s:02d}_cvbehav.npz")
        b = np.load(HERE / "encoding" / f"r2_Shuf{model}_{task}_subj{s:02d}_cvbehav.npz")
        R.append(a["r2_attn"]); F.append(b["r2_attn"])
        verts = a["vertices"]
    C = np.stack(R) - np.stack(F)
    t, p = stats.ttest_1samp(C, 0, axis=0)
    sig = (fdr(p) < .05) & (C.mean(0) > 0)
    full = np.zeros(VERTS, dtype=np.float32)
    full[verts[sig]] = C.mean(0)[sig]
    return full, int(sig.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CNN_LSTM")
    ap.add_argument("--panel-letter", default=None)
    a = ap.parse_args()

    fs5 = nidatasets.fetch_surf_fsaverage("fsaverage5")
    sulc = {"left": fs5.get("sulc_left", fs5.get("curv_left")),
            "right": fs5.get("sulc_right", fs5.get("curv_right"))}
    pial = {"left": fs5["pial_left"], "right": fs5["pial_right"]}
    views = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]

    maps, counts = {}, {}
    for task in ("social", "describe"):
        maps[task], counts[task] = thresholded(a.model, task)
        print(f"{a.model}/{task}: {counts[task]} vertices survive FDR q<.05")
    vmax = max(np.percentile(m[m > 0], 99) if (m > 0).any() else 1e-6 for m in maps.values())

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=0.01, right=0.93, top=1.0, bottom=0.0, hspace=-0.25, wspace=0.01)
    for r, task in enumerate(("social", "describe")):
        half = {"left": maps[task][:VERTS // 2], "right": maps[task][VERTS // 2:]}
        for c, (hemi, view) in enumerate(views):
            ax = axes[r, c]
            plotting.plot_surf_stat_map(
                surf_mesh=pial[hemi], stat_map=half[hemi], hemi=hemi, view=view,
                bg_map=sulc[hemi], cmap="hot", vmin=0, vmax=vmax,
                threshold=1e-9, colorbar=(c == 3), axes=ax)
            ax.set_title("")
        # negative hspace makes the two rows' axes overlap, so fixed figure
        # coordinates are needed here -- deriving y from the axes collides the labels.
        lab = f"{task.capitalize()} Task\n{counts[task]} vertices, FDR $q<.05$"
        fig.text(0.015, 0.72 if r == 0 else 0.28, lab, rotation=90, va="center",
                 ha="center", fontsize=20, fontweight="bold", linespacing=1.5)
    if a.panel_letter:
        import sys
        sys.path.insert(0, str(HERE.parent / "shared"))
        import figstyle as FST
        FST.panel_letter(fig, a.panel_letter)
    out = PLOTS / f"nsd_brainmap_thresh_{a.model}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
