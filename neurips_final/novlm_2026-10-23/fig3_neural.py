"""
fig3_neural.py -- Figure 3 (neural encoding) of main_2026-10-23.tex with the trained models in the
order of the behavioral figures (CNN-LSTM, CNN-Transformer, ViT-LSTM, ViT-Transformer).
Executes molmo_bilinear/nsd_norm/plot_fig3_three_panel.py, the source of the 10-22 Figure 3, with
only the bar order changed. Same data, colours and layout.
Writes final_figures/manuscript_2026-10-23/fig3_neural.png
"""
import os, shutil, tempfile

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'nsd_norm', 'plot_fig3_three_panel.py')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig3_neural.png')
TMP = tempfile.mkdtemp(dir=os.path.dirname(OUT))

src = open(SRC).read()
for old, new in [
        ("PRED = [('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None),\n"
         "        ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None),\n"
         "        ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None),\n",
         "PRED = [('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None),\n"
         "        ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None),\n"
         "        ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None),\n"),
        ("OUTDIR = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'nsd_norm'", f"OUTDIR = Path({TMP!r})")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
exec(compile(src, SRC, 'exec'), {'__file__': SRC, '__name__': '__main__'})
shutil.copy2(os.path.join(TMP, 'fig3_neural_three_panel.png'), OUT)
shutil.rmtree(TMP)
print('wrote', OUT)
