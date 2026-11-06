"""
fig5_smoothness.py -- Figure 5 (appendix smoothness) of main_2026-10-23.tex.
Executes molmo_bilinear/plot_fig_smoothness.py --compact, the source of the 10-22 Figure 5,
with three changes: Molmo and Qwen3.5 dropped; only the pairwise human upperbound drawn
(the leave-one-out line and its label removed); y-ranges tightened to the trained models
and the legend moved to the top right of panel B.
Writes final_figures/manuscript_2026-10-23/fig5_smoothness.png
"""
import os, sys

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'plot_fig_smoothness.py')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig5_smoothness.png')

src = open(SRC).read()
for old, new in [
        ("ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']",
         "ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer']"),
        ("for y, lab in ((PAIR[task], 'Human upperbound (pairwise)'), (LOO[task], 'Human upperbound (leave-one-out)')):",
         "for y, lab in ((PAIR[task], 'Human upperbound'),):"),
        ("YMIN, YMAX = 0.08, 0.40", "YMIN, YMAX = 0.08, 0.32"),
        ("loc='upper right', bbox_to_anchor=(1.0, (LOO['social'] - 0.008 - YMIN) / (YMAX - YMIN)),",
         "loc='upper right', bbox_to_anchor=(1.0, 1.0),"),
        ("YMAX_C = 0.42", "YMAX_C = 0.32"),
        ("ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])", "ax.set_yticks([0, 0.1, 0.2, 0.3])"),
        ("f = os.path.join(BS, 'fig_smoothness_compact.png' if COMPACT else 'fig_smoothness.png')", f"f = {OUT!r}")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
sys.argv = [SRC, '--compact']
exec(compile(src, SRC, 'exec'), {'__file__': SRC, '__name__': '__main__'})
