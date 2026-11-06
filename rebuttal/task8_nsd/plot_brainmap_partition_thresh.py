"""Variance partition on the surface, statistically thresholded.

Same three rows as plot_nsd_brainmap.py (unique attention, unique CLIP, shared) but
each vertex must survive a one-sample t across the 8 participants at FDR q<.05.
Grey means it did not.

The three rows are NOT tested against the same reference, and the row labels say so:
unique attention is tested against that model's own permutation floor (a Shuf fit
exists for it), while unique CLIP and shared are tested against zero, because no
banded-with-shuffled-CLIP floor was ever fit. Testing against zero is the weaker bar.

Usage:  python plot_brainmap_partition_thresh.py --model CNN_Transformer --task social
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CNN_Transformer")
    ap.add_argument("--task", default="social")
    ap.add_argument("--panel-letter", default=None)
    a = ap.parse_args()

    real = [np.load(HERE / "encoding" / f"r2_{a.model}_{a.task}_subj{s:02d}_cvbehav.npz") for s in range(1, 9)]
    shuf = [np.load(HERE / "encoding" / f"r2_Shuf{a.model}_{a.task}_subj{s:02d}_cvbehav.npz") for s in range(1, 9)]
    verts = real[0]["vertices"]

    ua = np.stack([d["r2_banded"] - d["r2_clip"] for d in real])
    ua_f = np.stack([d["r2_banded"] - d["r2_clip"] for d in shuf])
    uc = np.stack([d["r2_banded"] - d["r2_attn"] for d in real])
    sh = np.stack([d["r2_attn"] + d["r2_clip"] - d["r2_banded"] for d in real])

    # CLIP floors, when the permuted-CLIP banded fit exists:
    #   unique CLIP  = R2(banded) - R2(attn);  floor = R2(banded,shufCLIP) - R2(attn)
    #                  so the difference reduces to R2(banded) - R2(banded,shufCLIP)
    #   shared       = R2(attn)+R2(CLIP)-R2(banded)
    #                  floor = R2(attn)+R2(shufCLIP)-R2(banded,shufCLIP)
    bsc_p = [HERE / "encoding" / f"r2_bandedShufCLIP_{a.model}_{a.task}_subj{s:02d}_cvbehav.npz"
             for s in range(1, 9)]
    sc_p = [HERE / "encoding" / f"r2_ShufCLIP_subj{s:02d}_cvbehav.npz" for s in range(1, 9)]
    if all(f.exists() for f in bsc_p + sc_p):
        bsc = np.stack([np.load(f)["r2_banded_shufclip"] for f in bsc_p])
        csh = np.stack([np.load(f)["r2_clip_shuf"] for f in sc_p])
        clip_rows = [("Unique CLIP $R^2$ (\"What\")", uc - (bsc - np.stack([d["r2_attn"] for d in real])),
                      uc, "vs permuted-CLIP floor"),
                     ("Shared $R^2$",
                      sh - (np.stack([d["r2_attn"] for d in real]) + csh - bsc),
                      sh, "vs permuted-CLIP floor")]
        print("  using the permuted-CLIP floor for the CLIP and shared rows")
    else:
        clip_rows = [("Unique CLIP $R^2$ (\"What\")", uc, uc, "vs zero"),
                     ("Shared $R^2$", sh, sh, "vs zero")]
        print("  no permuted-CLIP fit found; CLIP and shared tested against zero")
    rows = [("Unique Attn $R^2$ (\"Where\")", ua - ua_f, ua, "vs own permutation floor")] + clip_rows

    fs5 = nidatasets.fetch_surf_fsaverage("fsaverage5")
    sulc = {"left": fs5.get("sulc_left", fs5.get("curv_left")),
            "right": fs5.get("sulc_right", fs5.get("curv_right"))}
    pial = {"left": fs5["pial_left"], "right": fs5["pial_right"]}
    views = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]

    fig, axes = plt.subplots(3, 4, figsize=(15, 11), subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=0.07, right=0.90, top=0.99, bottom=0.01, hspace=-0.22, wspace=0.01)

    for ri, (title, tested, shown, basis) in enumerate(rows):
        t, p = stats.ttest_1samp(tested, 0, axis=0)
        sig = (fdr(p) < .05) & (tested.mean(0) > 0)
        full = np.zeros(VERTS, dtype=np.float32)
        full[verts[sig]] = shown.mean(0)[sig]          # colour by the component's own value
        n = int(sig.sum())
        vmax = np.percentile(full[full > 0], 99) if n else 1.0
        print(f"  {title}: {n} vertices survive ({basis})")
        half = {"left": full[:VERTS // 2], "right": full[VERTS // 2:]}
        for ci, (hemi, view) in enumerate(views):
            ax = axes[ri, ci]
            plotting.plot_surf_stat_map(
                surf_mesh=pial[hemi], stat_map=half[hemi], hemi=hemi, view=view,
                bg_map=sulc[hemi], cmap="hot", vmin=0, vmax=vmax,
                threshold=1e-9, colorbar=(ci == 3 and n > 0), axes=ax)
            ax.set_title("")
        # derive the label position from the row's own axes, as plot_nsd_brainmap.py
        # does; the rows overlap (negative hspace) so fixed offsets land on the brains
        p0 = axes[ri, 0].get_position(); p3 = axes[ri, 3].get_position()
        xc = (p0.x0 + p3.x1) / 2
        fig.text(xc, p0.y1 - 0.005, title, ha="center", va="top",
                 fontsize=20, fontweight="bold")
        note = f"{n} vertices, FDR $q<.05$, {basis}" if n else f"no vertices survive ({basis})"
        fig.text(xc, p0.y1 - 0.038, note, ha="center", va="top", fontsize=13, color="#555")

    if a.panel_letter:
        import sys
        sys.path.insert(0, str(HERE.parent / "shared"))
        import figstyle as FST
        FST.panel_letter(fig, a.panel_letter)
    out = PLOTS / f"nsd_brainmap_partition_thresh_{a.model}_{a.task}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
