"""Manuscript-style NSD encoding figures (ICLR revision, 2026-09-23).

The rebuttal figures carried an embedded title, a long explanatory footer, and a small
canvas with small type. This rebuilds both at the scale and typography of the manuscript's
Figure 1 (plots/plot_fig_head_selection_corrected.py): large bold axis titles, 45-degree
rotated tick labels, white bar edges, black error bars, no baked-in title or caption. All
explanation lives in the LaTeX caption.

Usage:  python plot_nsd_manuscript_figs.py --head-rule cvbehav
Output: plots/wholebrain/nsd_wholebrain_manuscript_{rule}.png
        plots/regional/nsd_regional_manuscript_{rule}.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
import figstyle as FST

HERE = Path(__file__).parent
ENC = HERE / "encoding"
RES = HERE / "results"
PLOTS = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots")

MODELS = ["CNN_LSTM", "ViT_LSTM", "CNN_Transformer", "ViT_Transformer", "Molmo", "Qwen35"]
BASELINES = ["EdgeEnergy", "Luminance"]
COLORS = FST.ALL_COLORS
LABELS = {"CLIP": "CLIP (semantic)", "CNN_LSTM": "CNN-LSTM", "ViT_LSTM": "ViT-LSTM",
          "CNN_Transformer": "CNN-Transf.", "ViT_Transformer": "ViT-Transf.",
          "Molmo": "Molmo 7B-D", "Qwen35": "Qwen 9B-3.5",
          "EdgeEnergy": "Edge energy", "Luminance": "Luminance"}
TASKS = ["social", "describe"]
TASK_LABELS = {"social": "Social Task", "describe": "Describe Task"}
STREAMS = ["early", "ventral", "lateral", "parietal"]
STREAM_LABELS = {"early": "Early visual", "ventral": "Ventral",
                 "lateral": "Lateral", "parietal": "Parietal"}

# Typography lifted from the Figure 1 script.
FS_TITLE = FST.FS['title']; FS_TICK = FST.FS['tick']; FS_LABEL = FST.FS['label']
FS_LEGEND = FST.FS['legend']; FS_STAR = FST.FS['star']
BAR_W = 0.62
MIN_N = 3


def holm(p):
    p = np.asarray(p, float)
    adj = np.full(len(p), np.nan)
    fin = np.where(np.isfinite(p))[0]
    if not len(fin):
        return adj
    order = fin[np.argsort(p[fin])]
    m, run = len(fin), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def star(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def tidy(ax):
    FST.style_axes(ax)


# ───────────────────────── whole brain ─────────────────────────
def wholebrain(rule):
    rows = []
    for m in MODELS + ["Shuffled"]:
        for task in TASKS:
            for s in range(1, 9):
                f = ENC / f"r2_{m}_{task}_subj{s:02d}_{rule}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                rows.append(dict(model=m, task=task, subject=s,
                                 unique=float((d["r2_banded"] - d["r2_clip"]).mean())))
    df = pd.DataFrame(rows)

    sig = {}
    for task in TASKS:
        piv = df[df.task == task].pivot_table(index="subject", columns="model", values="unique")
        ps, recs = [], []
        for m in MODELS:
            pair = piv[[m, "Shuffled"]].dropna()
            t, p = stats.ttest_rel(pair[m], pair["Shuffled"])
            recs.append((m, pair[m].mean(), pair[m].sem())); ps.append(p)
        for (m, mu, se), ph in zip(recs, holm(ps)):
            sig[(task, m)] = (mu, se, ph)

    floor = df[df.model == "Shuffled"].groupby("task")["unique"].mean().to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(30, 11), sharey=True)
    x = np.arange(len(MODELS))
    for ax, task in zip(axes, TASKS):
        mus = [sig[(task, m)][0] for m in MODELS]
        ses = [sig[(task, m)][1] for m in MODELS]
        ax.bar(x, mus, BAR_W, zorder=3, **FST.bar_kwargs(MODELS))
        ax.errorbar(x, mus, yerr=ses, **FST.errorbar_kwargs())
        ax.axhline(floor[task], color="black", ls="--", lw=3.0, zorder=2)
        span = max(abs(v) + e for v, e in zip(mus, ses))
        for xi, m in zip(x, MODELS):
            mu, se, ph = sig[(task, m)]
            # Star sits at the end of its own bar: above a bar that clears the floor,
            # below one that falls under it, so the sign is never ambiguous.
            if mu >= floor[task]:
                y, va = max(mu + se, floor[task]) + 0.07 * span, "bottom"
            else:
                y, va = min(mu - se, 0.0) - 0.07 * span, "top"
            ax.text(xi, y, star(ph), ha="center", va=va, fontsize=FS_STAR,
                    fontweight="bold", color="black" if ph < .05 else "#888")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=FS_TICK,
                           rotation=45, ha="right")
        ax.set_title(TASK_LABELS[task], fontsize=FS_TITLE, fontweight="bold", pad=18)
        ax.axhline(0, color="black", lw=1.6, zorder=2)
        tidy(ax)
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo - 0.10 * (hi - lo), hi + 0.10 * (hi - lo))
    axes[0].set_ylabel("Unique attention $R^2$ over CLIP\n(whole brain)",
                       fontsize=FS_LABEL, labelpad=14)
    line = plt.Line2D([0], [0], color="black", ls="--", lw=3.0)
    axes[1].legend([line], ["Shuffled floor"], fontsize=FS_LEGEND, loc="upper right",
                   frameon=True, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    out = PLOTS / "wholebrain" / f"nsd_wholebrain_manuscript_{rule}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# ───────────────────────── regional ─────────────────────────
def regional(rule, tasks=None, letter=None, permodel=False, attn_only=False,
             kastner=False, ymax=None, stars=True, with_clip=False):
    # permodel: each model referenced to its OWN permutation floor (summarize_permodel_
    # floor.py). The shared CNN-LSTM floor it replaces was an outlier -- positive in
    # higher-order cortex where every other model's floor is negative.
    # attn_only: the encoding model fit WITHOUT the CLIP band, so the bars are each
    # model's total attention R^2 above its own floor rather than its unique variance.
    if with_clip:
        # same attention-only values plus a CLIP-only band, each above its own floor
        src = f"attnonly_withclip_per_subject_{rule}.csv"
    elif kastner:
        # Kastner2015 retinotopic stream masks; 'parietal' keys the dorsal stream
        src = f"kastner_attnonly_per_subject_{rule}.csv"
    elif attn_only:
        src = f"attnonly_per_subject_{rule}.csv"
    else:
        src = f"permodel_per_subject_{rule}.csv" if permodel else f"final_per_subject_{rule}.csv"
    per = pd.read_csv(RES / src)
    per = per[per.roi.isin(STREAMS)]
    lut = {}
    for (m, task), g in per.groupby(["model", "task"]):
        recs, ps = [], []
        for roi in STREAMS:
            gg = g[g.roi == roi]["attn_corrected"].dropna()
            if len(gg) < MIN_N:
                continue
            t, p = stats.ttest_1samp(gg, 0.0)
            recs.append((roi, gg.mean(), gg.sem())); ps.append(p)
        for (roi, mu, se), ph in zip(recs, holm(ps)):
            lut[(task, roi, m)] = (mu, se, ph)

    order = MODELS + BASELINES + (['CLIP'] if with_clip else [])
    tk = tasks or TASKS
    # A single-task panel is rendered at text width in the manuscript, so it needs a
    # narrower canvas than the two-row version or its tick labels scale down to ~5 pt.
    W = 34 if len(tk) == 2 else 23
    fig, axes = plt.subplots(len(tk), 4, figsize=(W, 10 * len(tk) * W / 34 + 1.5), squeeze=False)
    x = np.arange(len(order))
    for r, task in enumerate(tk):
        for c, roi in enumerate(STREAMS):
            ax = axes[r, c]
            mus = [lut.get((task, roi, m), (np.nan,) * 3)[0] for m in order]
            ses = [lut.get((task, roi, m), (np.nan,) * 3)[1] for m in order]
            bars = ax.bar(x, mus, BAR_W, zorder=3, **FST.bar_kwargs(order))
            FST.hatch_bars(bars, order)
            ax.errorbar(x, mus, yerr=ses, **FST.errorbar_kwargs())
            ax.axhline(0, color="black", ls="-" if permodel else "--",
                       lw=1.6 if permodel else 2.6, zorder=2)
            span = np.nanmax(np.abs(np.array(mus) + np.array(ses))) or 1e-3
            if stars:
                for xi, m in zip(x, order):
                    mu, se, ph = lut.get((task, roi, m), (np.nan, np.nan, np.nan))
                    if not np.isfinite(mu) or not np.isfinite(ph) or ph >= .05 or mu <= 0:
                        continue
                    ax.text(xi, mu + se + 0.04 * span, star(ph), ha="center", va="bottom",
                            fontsize=FS_STAR, fontweight="bold")
            ax.set_xticks(x)
            # With one row the four panels are narrow and eight rotated labels collide,
            # so the models are identified by a shared legend instead.
            if len(tk) == 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels([LABELS[m] for m in order] if r == len(tk) - 1 else [],
                                   fontsize=FS_TICK, rotation=45, ha="right")
            if r == 0:
                t_ = "Dorsal" if (kastner and roi == "parietal") else STREAM_LABELS[roi]
                ax.set_title(t_, fontsize=FS_TITLE, fontweight="bold", pad=18)
            if c == 0:
                base = ("$R^2$ above own floor" if with_clip else
                        "Test $R^2$ above floor" if (attn_only or kastner) else
                        "attention $R^2$" if permodel
                        else "attention $R^2$ over floor")
                lab = base if len(tk) == 1 else f"{TASK_LABELS[task]}\n{base}"
                ax.set_ylabel(lab, fontsize=FS_LABEL, labelpad=14)
            tidy(ax)
            if ymax is not None:
                # one shared scale across panels, with ticks every 0.02 at 2 dp
                ax.set_ylim(0, ymax)
                ticks = np.arange(0, ymax + 1e-9, 0.02)
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:.2f}" for t in ticks])
            else:
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.20 * (hi - lo))
    if len(tk) == 1:
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(facecolor=FST.color(m), edgecolor="black",
                                  hatch="//" if m in BASELINES else None, label=LABELS[m])
                   for m in order]
        # 9 series need 5 per row and a deeper reserve, or the key lands on the bars
        ncol = 5 if len(order) > 8 else 4
        fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=FS_TICK,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        # A panel letter needs a clear strip at the top or it lands on the
        # rotated y-axis label of the leftmost panel.
        bottom = 0.19 if len(order) > 8 else 0.13
        fig.tight_layout(rect=[0, bottom, 1, (0.92 if permodel else 0.95) if letter else 1])
    else:
        fig.tight_layout()
    if letter:
        FST.panel_letter(fig, letter)

    tag = "" if len(tk) == 2 else f"_{tk[0]}"
    stem = ("nsd_regional_withclip" if with_clip else
            "nsd_regional_kastner" if kastner else
            "nsd_regional_attnonly" if attn_only else
            "nsd_regional_permodel" if permodel else "nsd_regional_manuscript")
    out = PLOTS / "regional" / f"{stem}_{rule}{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def regional_rawfloor(rule, tasks=None, letter=None):
    """Regional panel on the RAW scale, with the Shuffled floor drawn as a line.

    The floor-subtracted version puts the floor at an implicit zero, so the reference the
    y-axis named was nowhere on the plot. Here each bar is the model's own unique
    attention R^2 over CLIP and the floor is an explicit dashed line, which is possible
    because the floor is one shared value per region x task (CNN-LSTM's maps with the
    image labels permuted), not a per-model quantity. Stars are unchanged: they still
    test each model's floor-subtracted value against zero.
    """
    per = pd.read_csv(RES / f"final_per_subject_{rule}.csv")
    per = per[per.roi.isin(STREAMS)]
    lut, floor = {}, {}
    for (m, task), g in per.groupby(["model", "task"]):
        recs, ps = [], []
        for roi in STREAMS:
            gg = g[g.roi == roi]
            raw = gg["attn_unique"].dropna()
            cor = gg["attn_corrected"].dropna()
            if len(cor) < MIN_N:
                continue
            t, p = stats.ttest_1samp(cor, 0.0)
            recs.append((roi, raw.mean(), raw.sem())); ps.append(p)
        for (roi, mu, se), ph in zip(recs, holm(ps)):
            lut[(task, roi, m)] = (mu, se, ph)
    for (task, roi), g in per[per.model == "Shuffled"].groupby(["task", "roi"]):
        floor[(task, roi)] = g["attn_unique"].mean()

    order = MODELS + BASELINES + (['CLIP'] if with_clip else [])
    tk = tasks or TASKS
    W = 34 if len(tk) == 2 else 23
    fig, axes = plt.subplots(len(tk), 4, figsize=(W, 10 * len(tk) * W / 34 + 1.5), squeeze=False)
    x = np.arange(len(order))
    for r, task in enumerate(tk):
        for c, roi in enumerate(STREAMS):
            ax = axes[r, c]
            mus = [lut.get((task, roi, m), (np.nan,) * 3)[0] for m in order]
            ses = [lut.get((task, roi, m), (np.nan,) * 3)[1] for m in order]
            bars = ax.bar(x, mus, BAR_W, zorder=3, **FST.bar_kwargs(order))
            FST.hatch_bars(bars, order)
            ax.errorbar(x, mus, yerr=ses, **FST.errorbar_kwargs())
            fl = floor[(task, roi)]
            ax.axhline(fl, color="black", ls="--", lw=2.6, zorder=4)
            ax.axhline(0, color="black", lw=1.6, zorder=2)
            span = np.nanmax(np.abs(np.array(mus) + np.array(ses))) or 1e-3
            for xi, m in zip(x, order):
                mu, se, ph = lut.get((task, roi, m), (np.nan,) * 3)
                if not np.isfinite(mu) or not np.isfinite(ph) or ph >= .05 or mu <= fl:
                    continue
                ax.text(xi, mu + se + 0.04 * span, star(ph), ha="center", va="bottom",
                        fontsize=FS_STAR, fontweight="bold")
            ax.set_xticks(x)
            if len(tk) == 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels([LABELS[m] for m in order] if r == len(tk) - 1 else [],
                                   fontsize=FS_TICK, rotation=45, ha="right")
            if r == 0:
                t_ = "Dorsal" if (kastner and roi == "parietal") else STREAM_LABELS[roi]
                ax.set_title(t_, fontsize=FS_TITLE, fontweight="bold", pad=18)
            if c == 0:
                lab = ("attention $R^2$" if len(tk) == 1
                       else f"{TASK_LABELS[task]}\nattention $R^2$")
                ax.set_ylabel(lab, fontsize=FS_LABEL, labelpad=14)
            tidy(ax)
            if ymax is not None:
                # one shared scale across panels, with ticks every 0.02 at 2 dp
                ax.set_ylim(0, ymax)
                ticks = np.arange(0, ymax + 1e-9, 0.02)
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:.2f}" for t in ticks])
            else:
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.20 * (hi - lo))
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=FST.color(m), edgecolor="black",
                              hatch="//" if m in BASELINES else None, label=LABELS[m])
               for m in order]
    handles.append(plt.Line2D([0], [0], color="black", ls="--", lw=2.6, label="Shuffled floor"))
    if len(tk) == 1:
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=FS_TICK,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.13, 1, (0.92 if permodel else 0.95) if letter else 1])
    else:
        fig.tight_layout()
    if letter:
        FST.panel_letter(fig, letter)
    tag = "" if len(tk) == 2 else f"_{tk[0]}"
    out = PLOTS / "regional" / f"nsd_regional_rawfloor_{rule}{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def task_difference(rule, letter=None):
    """Social minus Describe, per model and region, on the floor-corrected values.

    Asks how much each model's neural prediction retunes between tasks. Edge energy and
    luminance are computed from the image alone, so their difference is exactly zero by
    construction -- they are the control that shows any non-zero bar comes from the
    attention maps rather than from the pipeline.
    """
    per = pd.read_csv(RES / f"permodel_per_subject_{rule}.csv")
    per = per[per.roi.isin(STREAMS)]
    order = MODELS + BASELINES

    lut = {}
    for m, g in per.groupby("model"):
        cells, ps = [], []
        for roi in STREAMS:
            w = g[g.roi == roi].pivot_table(index="subject", columns="task",
                                            values="attn_corrected").dropna()
            if len(w) < MIN_N or "social" not in w or "describe" not in w:
                continue
            diff = w["social"] - w["describe"]
            t, pv = stats.ttest_rel(w["social"], w["describe"]) if diff.std(ddof=1) > 0 else (0.0, 1.0)
            cells.append((roi, diff.mean(), diff.sem())); ps.append(pv)
        for (roi, mu, se), ph in zip(cells, holm(ps)):
            lut[(roi, m)] = (mu, se, ph)

    W = 23
    fig, axes = plt.subplots(1, 4, figsize=(W, 10 * W / 34 + 1.5), squeeze=False)
    x = np.arange(len(order))
    for c, roi in enumerate(STREAMS):
        ax = axes[0, c]
        mus = [lut.get((roi, m), (np.nan,) * 3)[0] for m in order]
        ses = [lut.get((roi, m), (np.nan,) * 3)[1] for m in order]
        bars = ax.bar(x, mus, BAR_W, zorder=3, **FST.bar_kwargs(order))
        FST.hatch_bars(bars, order)
        ax.errorbar(x, mus, yerr=ses, **FST.errorbar_kwargs())
        ax.axhline(0, color="black", lw=1.6, zorder=2)
        span = np.nanmax(np.abs(np.array(mus) + np.array(ses))) or 1e-3
        for xi, m in zip(x, order):
            mu, se, ph = lut.get((roi, m), (np.nan,) * 3)
            if not np.isfinite(mu) or not np.isfinite(ph) or ph >= .05:
                continue
            up = mu >= 0
            y = (mu + se + 0.05 * span) if up else (mu - se - 0.05 * span)
            ax.text(xi, y, star(ph), ha="center", va="bottom" if up else "top",
                    fontsize=FS_STAR, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([])
        ax.set_title(STREAM_LABELS[roi], fontsize=FS_TITLE, fontweight="bold", pad=18)
        if c == 0:
            ax.set_ylabel("Social $-$ Describe\nattention $R^2$", fontsize=FS_LABEL, labelpad=14)
        tidy(ax)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.18 * (hi - lo))

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=FST.color(m), edgecolor="black",
                              hatch="//" if m in BASELINES else None, label=LABELS[m])
               for m in order]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=FS_TICK,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.13, 1, 0.92 if letter else 1])
    if letter:
        FST.panel_letter(fig, letter)
    out = PLOTS / "regional" / f"nsd_task_difference_{rule}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-rule", default="cvbehav")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="subset of tasks for the regional panel; default is both")
    ap.add_argument("--regional-only", action="store_true")
    ap.add_argument("--task-diff", action="store_true",
                    help="Social minus Describe difference plot")
    ap.add_argument("--with-clip", action="store_true",
                    help="add a CLIP-only bar per region")
    ap.add_argument("--no-stars", action="store_true",
                    help="omit the significance asterisks")
    ap.add_argument("--ymax", type=float, default=None,
                    help="shared y maximum across panels, ticks every 0.02")
    ap.add_argument("--kastner", action="store_true",
                    help="use the Kastner2015 retinotopic stream masks")
    ap.add_argument("--attn-only", action="store_true",
                    help="use the encoding model fit without the CLIP band")
    ap.add_argument("--permodel", action="store_true",
                    help="use the per-model permutation floors")
    ap.add_argument("--raw-floor", action="store_true",
                    help="regional panel on the raw scale with the floor drawn as a line")
    ap.add_argument("--panel-letter", default=None,
                    help="bare bold panel letter drawn top-left of the regional figure")
    a = ap.parse_args()
    if a.task_diff:
        task_difference(a.head_rule, a.panel_letter)
    elif a.raw_floor:
        regional_rawfloor(a.head_rule, a.tasks, a.panel_letter)
    else:
        if not a.regional_only:
            wholebrain(a.head_rule)
        regional(a.head_rule, a.tasks, a.panel_letter, permodel=a.permodel, attn_only=a.attn_only, kastner=a.kastner, ymax=a.ymax, stars=not a.no_stars, with_clip=a.with_clip)
