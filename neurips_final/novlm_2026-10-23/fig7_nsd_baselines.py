"""
fig7_nsd_baselines.py -- Figure 7 (appendix NSD: baseline VLMs and saliency models) of
main_2026-10-23.tex in the order of the Appendix B figure (fig_baselines_rsa.py): trained models in the
behavioral order | Itti-Koch, GBVS, spectral residual, edge energy, luminance (one purple, cross-hatched
group, same shades as fig_baselines_rsa.py) | Molmo, Qwen3.5. Executes
molmo_bilinear/nsd_saliency/plot_nsd_saliency.py --letters, the source of the 10-22 Figure 7, with only
the bar order and baseline colours changed.
Writes final_figures/manuscript_2026-10-23/fig7_nsd_vlms_saliency.png
"""
import os, sys, shutil, tempfile

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'nsd_saliency', 'plot_nsd_saliency.py')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig7_nsd_vlms_saliency.png')
TMP = tempfile.mkdtemp(dir=os.path.dirname(OUT))

src = open(SRC).read()
OLD_PRED = src[src.index('PRED = ['):src.index(']\nrows = []') + 1]
NEW_PRED = """PRED = [  # (file stem, label, colour, hatch, folder)
    ('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None, 'encoding_norm'),
    ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None, 'encoding_norm'),
    ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None, 'encoding_norm'),
    ('ViT_Transformer', 'ViT-Transf.', FST.ALL_COLORS['ViT-Transformer'], None, 'encoding_norm'),
    ('SalItti', 'Itti-Koch', '#5e3c99', 'xx', 'encoding_norm_saliency'),
    ('SalGBVS', 'GBVS', '#7b5ea7', 'xx', 'encoding_norm_saliency'),
    ('SalSR', 'Spectral residual', '#a98bd0', 'xx', 'encoding_norm_saliency'),
    ('EdgeEnergy', 'Edge energy', '#c9b6e4', 'xx', 'encoding_norm'),
    ('Luminance', 'Luminance', '#e8dff5', 'xx', 'encoding_norm'),
    ('Molmo', 'Molmo 7B-D', FST.ALL_COLORS['Molmo'], None, 'encoding_norm'),
    ('Qwen35', 'Qwen 9B-3.5', FST.ALL_COLORS['Qwen3.5'], None, 'encoding_norm'),
]"""
for old, new in [
        (OLD_PRED, NEW_PRED),
        ("OUT = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'nsd_saliency'; OUT.mkdir(parents=True, exist_ok=True)",
         f"OUT = Path({TMP!r})")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
sys.argv = [SRC, '--letters']
exec(compile(src, SRC, 'exec'), {'__file__': SRC, '__name__': '__main__'})
shutil.copy2(os.path.join(TMP, 'fig_nsd_with_saliency_AB.png'), OUT)
shutil.rmtree(TMP)
print('wrote', OUT)
