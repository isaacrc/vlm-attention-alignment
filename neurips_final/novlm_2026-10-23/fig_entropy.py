"""
fig_entropy.py -- the normalized-entropy preview (manuscript_2026-10-22/fig_entropy_preview.png) for
the 10-23 folder: the four trained models and the humans only, with the y-axis from 0 to 1.
Executes molmo_bilinear/plot_entropy_fig1style.py, the source of the 10-22 preview, with Molmo and
Qwen3.5 dropped and the axis changed. Same data (molmo_bilinear_pairwise/results/entropy_cvheads.csv).
Writes final_figures/manuscript_2026-10-23/fig_entropy_preview.png
"""
import os

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'plot_entropy_fig1style.py')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig_entropy_preview.png')

src = open(SRC).read()
for old, new in [("ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5', 'Human']",
                  "ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Human']"),
                 ("ax.set_ylim(0.5, 1.05); ax.set_yticks([0.5, 0.75, 1.0])",
                  "ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])"),
                 ("out = os.path.join(FF, 'fig_entropy_fig1style.png')", f"out = {OUT!r}")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
exec(compile(src, SRC, 'exec'), {'__file__': SRC, '__name__': '__main__'})

# 16:9 slide (13.333 x 7.5 in, 300 dpi): same figure, slide font sizes, no panel letters
SLIDE = os.path.join(os.path.dirname(OUT), 'slides', 'fig_entropy_slide.png')
s2 = src
for old, new in [("figsize=(26, 10)", "figsize=(13.333, 7.5)"),
                 ("FS_TITLE, FS_TICK, FS_LABEL = 56, 40, 42", "FS_TITLE, FS_TICK, FS_LABEL = 30, 20, 22"),
                 ("TITLE_PAD, BAR_W = 14, 0.55", "TITLE_PAD, BAR_W = 10, 0.55"),
                 ("capsize=6, linewidth=2.0", "capsize=4, linewidth=1.5"),
                 ("1.0, letter, transform", "1.0, '', transform"),
                 (f"out = {OUT!r}", f"out = {SLIDE!r}"),
                 ("fig.savefig(out, dpi=300, bbox_inches='tight')", "fig.savefig(out, dpi=300, facecolor='white')")]:
    assert s2.count(old) == 1, old
    s2 = s2.replace(old, new)
exec(compile(s2, SRC, 'exec'), {'__file__': SRC, '__name__': '__main__'})
