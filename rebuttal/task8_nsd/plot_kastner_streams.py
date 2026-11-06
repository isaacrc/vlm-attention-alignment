"""Kastner2015 visual field maps on fsaverage5, coloured by stream.

One hue per stream, stepped light-to-dark along the hierarchy, so a map's colour
says both which stream it belongs to and how far along that stream it sits. Early
visual areas belong to no stream and are rendered neutral, which also keeps only
three chromatic hues on screen at once (the all-pairs CVD case).

Hues match plots/nsd_roi_atlas_grouped.png so the two figures agree.

Usage:  python plot_kastner_streams.py
Output: rebuttal/plots/nsd_kastner_streams.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
from nilearn import plotting, datasets as nidatasets

HERE = Path(__file__).parent
PLOTS = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots")
V = 10242
NAMES = ["Unknown", "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2",
         "PHC1", "PHC2", "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1",
         "IPS2", "IPS3", "IPS4", "IPS5", "SPL1", "FEF"]

# stream -> (anchor hue, its maps in hierarchical order)
STREAMS = {
    "Early visual": ("#E69F00", ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"]),
    "Ventral":      ("#009E73", ["hV4", "VO1", "VO2", "PHC1", "PHC2"]),
    "Lateral":      ("#CC79A7", ["LO1", "LO2", "TO1", "TO2"]),
    "Dorsal":       ("#0072B2", ["V3A", "V3B", "IPS0", "IPS1", "IPS2", "IPS3",
                                 "IPS4", "IPS5", "SPL1", "FEF"]),
}
NEUTRAL = "#E8E6E1"   # unlabelled cortex


def ramp(anchor, n, lo=0.62, hi=-0.34):
    """n steps of one hue, light to dark; lo/hi are tint and shade fractions."""
    base = np.array(to_rgb(anchor))
    out = []
    for f in np.linspace(lo, hi, n):
        c = base + (1 - base) * f if f > 0 else base * (1 + f)
        out.append(tuple(np.clip(c, 0, 1)))
    return out


def main():
    lh = np.asarray(nib.load(str(HERE / "rois" / "lh.Kastner2015.mgz")).get_fdata()).squeeze()[:V]
    rh = np.asarray(nib.load(str(HERE / "rois" / "rh.Kastner2015.mgz")).get_fdata()).squeeze()[:V]
    kast = np.concatenate([lh, rh]).astype(int)

    order, colors, legend = [], [], []
    for stream, (anchor, maps) in STREAMS.items():
        for name, col in zip(maps, ramp(anchor, len(maps))):
            idx = NAMES.index(name)
            n = int((kast == idx).sum())
            if n == 0:
                print(f"  (no vertices for {name}, skipped)")
                continue
            order.append(idx)
            colors.append(col)
            legend.append((stream, name, col, n))

    # relabel to a dense 1..N so one ListedColormap indexes them in stream order
    relab = np.zeros_like(kast)
    for new, old in enumerate(order, start=1):
        relab[kast == old] = new

    fs5 = nidatasets.fetch_surf_fsaverage("fsaverage5")
    sulc = {"left": fs5.get("sulc_left", fs5.get("curv_left")),
            "right": fs5.get("sulc_right", fs5.get("curv_right"))}
    pial = {"left": fs5["pial_left"], "right": fs5["pial_right"]}
    half = {"left": relab[:V], "right": relab[V:]}
    cmap = ListedColormap([to_rgb(NEUTRAL)] + colors)

    views = [("left", "lateral"), ("left", "medial"),
             ("right", "lateral"), ("right", "medial")]
    fig, axes = plt.subplots(1, 4, figsize=(23, 6.3), subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=0.01, right=0.99, top=1.04, bottom=0.30, wspace=0.01)
    for ax, (hemi, view) in zip(axes, views):
        plotting.plot_surf_roi(surf_mesh=pial[hemi], roi_map=half[hemi], hemi=hemi,
                               view=view, bg_map=sulc[hemi], cmap=cmap,
                               vmin=0, vmax=len(order), avg_method="median",
                               axes=ax, colorbar=False)
        ax.set_title(f"{hemi[0].upper()}H {view}", fontsize=21, fontweight="bold", y=0.04)

    # one legend column per stream, so the grouping shows in the key itself
    for i, (stream, (anchor, _)) in enumerate(STREAMS.items()):
        items = [(n, c, k) for s, n, c, k in legend if s == stream]
        x = 0.085 + i * 0.228
        fig.text(x, 0.235, stream, fontsize=19, fontweight="bold", ha="left", color=anchor)
        handles = [Patch(facecolor=c, edgecolor="#555", linewidth=.7, label=f"{n}  ({k})")
                   for n, c, k in items]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, 0.215),
                   frameon=False, fontsize=15, handlelength=1.5, handleheight=1.1,
                   labelspacing=.42, borderpad=0)

    out = PLOTS / "nsd_kastner_streams.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", out)
    print(f"{len(order)} maps, {int((relab > 0).sum())} vertices labelled")


if __name__ == "__main__":
    main()
