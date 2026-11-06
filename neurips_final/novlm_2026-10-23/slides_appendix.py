"""
slides_appendix.py -- 16:9 PowerPoint slides (13.333 x 7.5 in, 300 dpi) of Figures 3-5, 7 and
the Appendix B baselines figure of main_2026-10-23.tex.
  Native re-renders at slide font sizes, from the same data as the paper figures:
    fig4_metric_suite_slide.png          Describe and Social, five metrics, four trained models
                                         (follows qwen35_nothink/metric_suite_slide.py)
    baselines_alignment_slide.png        Appendix B panel A (fig_baselines_rsa.py data)
    baselines_map_similarity_slide.png   Appendix B panel B
    fig5_smoothness_{sweep,best}_slide.png   fig5_smoothness.py re-run without panel letters,
                                         split into the blur sweep (A-B) and best-blur bars (C)
  Panels cut from the paper PNGs (panel letters painted out), centred on a white 16:9 canvas:
    fig3_neural_{rois,social,describe,both}_slide.png
    fig7_nsd_baselines_{social,describe,both}_slide.png
Writes final_figures/manuscript_2026-10-23/slides/
"""
import os, sys, json, tempfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
FF = os.path.join(ROOT, 'neurips_final', 'final_figures')
MS = os.path.join(FF, 'manuscript_2026-10-23')
OUT = os.path.join(MS, 'slides')
os.makedirs(OUT, exist_ok=True)
W, H_IN, DPI = 13.333, 7.5, 300
PX_W, PX_H = round(W * DPI), round(H_IN * DPI)


def save(fig, name):
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=DPI, facecolor='white'); plt.close(fig)
    print('wrote', out, flush=True)


# ── metric suite (Figure 4) ───────────────────────────────────────────────────
src_path = os.path.join(ROOT, 'rebuttal', 'plots', 'plot_fig_metric_suite.py')
src = open(src_path).read()
old = "RES = os.path.join(REBUTTAL, 'task4_metric_suite', 'results')"
assert src.count(old) == 1
src = src.replace(old, f"RES = {os.path.join(FF, 'molmo_bilinear_pairwise', 'results')!r}")
ns = {'__file__': src_path, '__name__': 'metric_src'}
exec(compile(src[:src.index('def panel(')], src_path, 'exec'), ns)   # data, colors, normalize
summ, normalize, COLORS, TICK = ns['summ'], ns['normalize'], ns['COLORS'], ns['TICK']
ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer']
METRICS = ['CC', 'SIM', 'NSS', 'AUC', 'sAUC']   # as fig4_metric_suite.py
FS_TITLE, FS_TICK, FS_LABEL, FS_LEG = 26, 18, 20, 16
fig, axes = plt.subplots(1, 2, figsize=(W, H_IN), sharey=True)
BW = 0.8 / len(ORDER)
xm = np.arange(len(METRICS))
for ax, task in zip(axes, ['describe', 'social']):
    for k, m in enumerate(ORDER):
        vals, errs = [], []
        for metric in METRICS:
            sub = summ[(summ.metric == metric) & (summ.task == task)].set_index('model')
            p, e = normalize(metric, float(sub.loc[m, 'value']), float(sub.loc[m, 'sem']), float(sub['ceiling'].iloc[0]))
            vals.append(p); errs.append(e)
        off = (k - (len(ORDER) - 1) / 2) * BW
        ax.bar(xm + off, vals, BW, color=COLORS[m], label=TICK.get(m, m), edgecolor='black', linewidth=0.4, zorder=3)
        ax.errorbar(xm + off, vals, yerr=errs, fmt='none', color='black', capsize=2, linewidth=1.0, zorder=4)
    ax.axhline(100.0, color='black', ls='--', lw=2.0, zorder=2)
    ax.text(len(METRICS) - 0.55, 102, 'Human Upperbound', ha='right', va='bottom', fontsize=FS_LEG)
    ax.axhline(0, color='black', lw=0.8, zorder=2)
    ax.set_xticks(xm); ax.set_xticklabels(METRICS, fontsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(0, 115)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_title(ns['TASK_NAME'][task], fontsize=FS_TITLE, fontweight='bold', pad=10)
axes[0].set_ylabel('% of Human\nUpperbound', fontsize=FS_LABEL)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=FS_LEG, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.07, 1, 1])
save(fig, 'fig4_metric_suite_slide.png')

# ── baselines (Appendix B figure), one panel per slide ────────────────────────
bsrc_path = os.path.join(HERE, 'fig_baselines_rsa.py')
bsrc = open(bsrc_path).read()
bns = {'__file__': bsrc_path, '__name__': 'baselines_src'}
exec(compile(bsrc[:bsrc.index('fig = plt.figure(')], bsrc_path, 'exec'), bns)   # data, ITEMS, GROUP_ENDS
ITEMS, GROUP_ENDS, d, CEIL, rsa = bns['ITEMS'], bns['GROUP_ENDS'], bns['d'], bns['CEIL'], bns['rsa']
labels = [it[2] for it in ITEMS]
n = len(ITEMS)

BY_NAME = {it[0]: it for it in ITEMS}


def alignment_slide(groups, name, title):
    """Bars of % upperbound (mean of the Social and Describe percentages, as fig_baselines_rsa.py)
    for the predictors in `groups`, a list of name lists separated by dotted lines."""
    items = [BY_NAME[m] for grp in groups for m in grp]
    ends = np.cumsum([len(grp) for grp in groups])[:-1]
    fig = plt.figure(figsize=(W, H_IN), facecolor='white')
    ax = fig.add_axes([0.10, 0.27, 0.87, 0.62])
    x = np.arange(len(items), dtype=float)
    for b in ends:
        x[b:] += 0.6
    means, sems = [], []
    for it in items:
        pct, se = [], []
        for t in ('social', 'describe'):
            v = d[(d.task == t) & (d.model == it[0])].mean_z
            pct.append(v.mean() / CEIL[t] * 100); se.append(v.std(ddof=1) / np.sqrt(len(v)) / CEIL[t] * 100)
        means.append(np.mean(pct)); sems.append(np.sqrt(se[0] ** 2 + se[1] ** 2) / 2)
    width = 0.7 if len(items) > 8 else 0.6
    for xi, m, (_, _, _, c, h) in zip(x, means, items):
        ax.bar(xi, m, width, color=c, hatch=h, edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=4, linewidth=1.4, zorder=4)
    ax.axhline(100, color='black', linestyle='--', linewidth=2.0, zorder=2)
    ax.axhline(0, color='black', linewidth=1.0, zorder=2)
    for b in ends:
        ax.axvline((x[b - 1] + x[b]) / 2, color='gray', linewidth=1.0, linestyle=':', zorder=1)
    ax.text(0.99, 102, 'Human Upperbound', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=16)
    ax.set_xticks(x); ax.set_xticklabels([it[2] for it in items], rotation=45, ha='right', fontsize=18)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylim(-20 if min(means) < 0 else 0, 120); ax.set_yticks([0, 50, 100]); ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel('% of Human\nUpperbound', fontsize=20, labelpad=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=28, fontweight='bold', pad=12)
    save(fig, name)


TRAINED = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer']
alignment_slide([TRAINED, ['Itti-Koch', 'GBVS', 'Spectral residual', 'Edge energy', 'Luminance'],
                 ['Center prior', 'Person prior', 'Face prior'], ['Molmo', 'Qwen3.5']], 'baselines_alignment_slide.png',
                'Alignment with human gaze')
alignment_slide([TRAINED, ['Molmo', 'Qwen3.5']], 'baselines_alignment_vlms_slide.png', 'Alignment across tasks')
alignment_slide([TRAINED, ['Itti-Koch', 'GBVS', 'Spectral residual', 'Edge energy', 'Luminance']],
                'baselines_alignment_bottomup_slide.png', 'Alignment across tasks')
alignment_slide([TRAINED, ['Center prior', 'Person prior', 'Face prior']], 'baselines_alignment_priors_slide.png',
                'Alignment across tasks')

fig = plt.figure(figsize=(W, H_IN), facecolor='white')
ax = fig.add_axes([0.05, 0.20, 0.78, 0.71])
keys = [it[1] for it in ITEMS]
M = rsa.loc[keys, keys].values.astype(float); np.fill_diagonal(M, np.nan)
im = ax.imshow(M, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
ax.set_facecolor('#e6e6e6')
for i in range(n):
    for j in range(n):
        if i != j:
            ax.text(j, i, f'{M[i, j]:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=10,
                    color='white' if abs(M[i, j]) > 0.5 else '#222222')
for b in GROUP_ENDS:
    ax.axhline(b - 0.5, color='black', lw=1.2); ax.axvline(b - 0.5, color='black', lw=1.2)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=13)
ax.set_yticklabels(labels, fontsize=13)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title('Map-to-map similarity', fontsize=28, fontweight='bold', pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=12)
cb.set_label(r'Spearman $r$ between maps', fontsize=14)
save(fig, 'baselines_map_similarity_slide.png')


# ── panels cut from PNGs, centred on a 16:9 canvas ───────────────────────────
def blank(im, boxes):
    """Paint fractional boxes (x0, y0, x1, y1) white: removes panel letters."""
    a = np.array(im.convert('RGB'))
    H, Wd = a.shape[:2]
    for x0, y0, x1, y1 in boxes:
        a[int(y0 * H):int(y1 * H), int(x0 * Wd):int(x1 * Wd)] = 255
    return Image.fromarray(a)


def crop(im, y0, y1, x0=0.0, x1=1.0):
    Wd, H = im.size
    return im.crop((int(x0 * Wd), int(y0 * H), int(x1 * Wd), int(y1 * H)))


def stack(*parts):
    w = max(p.size[0] for p in parts)
    out = Image.new('RGB', (w, sum(p.size[1] for p in parts)), 'white')
    y = 0
    for p in parts:
        out.paste(p, ((w - p.size[0]) // 2, y)); y += p.size[1]
    return out


def place(im, name, margin=0.04):
    """Fit im inside the 16:9 canvas with a margin and centre it."""
    box_w, box_h = PX_W * (1 - 2 * margin), PX_H * (1 - 2 * margin)
    s = min(box_w / im.size[0], box_h / im.size[1])
    im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    can = Image.new('RGB', (PX_W, PX_H), 'white')
    can.paste(im, ((PX_W - im.size[0]) // 2, (PX_H - im.size[1]) // 2))
    out = os.path.join(OUT, name)
    can.save(out, dpi=(DPI, DPI))
    print('wrote', out, flush=True)


def white_rows(im, lo, hi, tol=0, x0=0.0):
    """Fractional y of the middle of the longest white row band between lo and hi
    (rows with at most tol non-white pixels, e.g. axis spines, count as white; columns left of x0 ignored)."""
    a = np.array(im.convert('L'))
    H = a.shape[0]
    white = (a[int(lo * H):int(hi * H), int(x0 * a.shape[1]):] <= 250).sum(axis=1) <= tol
    best, cur, start, bs = 0, 0, 0, 0
    for i, w in enumerate(white):
        cur = cur + 1 if w else 0
        if cur == 1: start = i
        if cur > best: best, bs = cur, start
    assert best > 0, 'no white band'
    return (int(lo * H) + bs + best / 2) / H


# Figure 3 (neural encoding): A brain ROIs, B Social, C Describe, shared legend at the bottom
im = blank(Image.open(os.path.join(MS, 'fig3_neural.png')),
           [(0, 0, 0.035, 0.055), (0, 0.355, 0.03, 0.405), (0, 0.66, 0.03, 0.705)])
leg = crop(im, 0.962, 1.0)
place(crop(im, 0.0, 0.345), 'fig3_neural_rois_slide.png')
place(stack(crop(im, 0.36, 0.66), leg), 'fig3_neural_social_slide.png')
place(stack(crop(im, 0.665, 0.962), leg), 'fig3_neural_describe_slide.png')
place(crop(im, 0.36, 1.0), 'fig3_neural_both_slide.png')

# Figure 7 (NSD, baseline VLMs and saliency models): A Social, B Describe, legend at the bottom
im = blank(Image.open(os.path.join(MS, 'fig7_nsd_vlms_saliency.png')), [(0, 0, 0.022, 0.045), (0, 0.45, 0.022, 0.495)])
cut = white_rows(im, 0.465, 0.50)
lcut = white_rows(im, 0.90, 0.95)
tcut = white_rows(im, 0.03, 0.052, tol=20, x0=0.06)   # ROI titles sit above row A only; x0 skips its tick labels
leg = crop(im, lcut, 1.0)
place(im, 'fig7_nsd_baselines_both_slide.png')
place(stack(crop(im, 0.0, cut), leg), 'fig7_nsd_baselines_social_slide.png')
place(stack(blank(crop(im, 0.0, tcut), [(0, 0, 0.075, 1)]), crop(im, cut, lcut), leg), 'fig7_nsd_baselines_describe_slide.png')

# Figure 5 (smoothness): fig5_smoothness.py without panel letters, then split into its two rows
w5 = open(os.path.join(HERE, 'fig5_smoothness.py')).read()
tmp = os.path.join(tempfile.mkdtemp(dir=OUT), 'fig5_noletters.png')
w5 = w5.replace("OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig5_smoothness.png')",
                f'OUT = {tmp!r}')
old = "    assert src.count(old) == 1, old\n    src = src.replace(old, new)\n"
assert w5.count(old) == 1
w5 = w5.replace(old, old + "src = src.replace(\"'AB'[k], transform\", \"'', transform\").replace(\"'C', transform\", \"'', transform\")\n")
exec(compile(w5, os.path.join(HERE, 'fig5_smoothness.py'), 'exec'), {'__file__': os.path.join(HERE, 'fig5_smoothness.py'), '__name__': '__main__'})
im = Image.open(tmp)
cut = white_rows(im, 0.40, 0.56)
place(crop(im, 0.0, cut), 'fig5_smoothness_sweep_slide.png')
place(crop(im, cut, 1.0), 'fig5_smoothness_best_slide.png')

# both rows on one slide: re-drawn at 16:9 proportions (wide, short) so the text stays legible,
# with two-line horizontal model labels on the bars
wide = (w5.replace(f'OUT = {tmp!r}', f"OUT = {tmp.replace('.png', '_wide.png')!r}")
          .replace("src = src.replace(\"'AB'[k], transform\"",
                   "src = (src.replace('figsize=(24, 20 if COMPACT else 21)', 'figsize=(36, 18.5)')\n"
                   "          .replace('hspace=0.22 if COMPACT else 0.38', 'hspace=0.30')\n"
                   "          .replace(\"[TICK.get(m, m) for m in ORDER], fontsize=FS_TICK, rotation=45, ha='right'\",\n"
                   "                   \"[TICK.get(m, m).replace('-', '-\\\\n', 1) for m in ORDER], fontsize=FS_TICK, rotation=0, ha='center'\"))\n"
                   "src = src.replace(\"'AB'[k], transform\""))
exec(compile(wide, os.path.join(HERE, 'fig5_smoothness.py'), 'exec'), {'__file__': os.path.join(HERE, 'fig5_smoothness.py'), '__name__': '__main__'})
place(Image.open(tmp.replace('.png', '_wide.png')), 'fig5_smoothness_both_slide.png', margin=0.03)
import shutil; shutil.rmtree(os.path.dirname(tmp))
