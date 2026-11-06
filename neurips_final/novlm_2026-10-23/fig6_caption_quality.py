"""
fig6_caption_quality.py -- Figure 6 (appendix caption quality) of main_2026-10-23.tex: the
four trained models only. Runs the plotting part of scripts/fig_caption_cosine_slides.py, the
source of the 10-22 Figure 6, on its saved summary (final_figures/slides/caption_cosine_summary.csv;
no re-embedding), with Molmo and Qwen3.5 dropped.
Writes final_figures/manuscript_2026-10-23/fig6_caption_quality.png
"""
import os
import pandas as pd

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'neurips_final', 'scripts', 'fig_caption_cosine_slides.py')
SUMM = os.path.join(ROOT, 'neurips_final', 'final_figures', 'slides', 'caption_cosine_summary.csv')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig6_caption_quality.png')

src = open(SRC).read()
src = src[src.index('# ── plot'):]
old = "out = os.path.join(OUT, 'caption_cosine_describe_social_slide.png')"
assert src.count(old) == 1
src = src.replace(old, f'out = {OUT!r}')
summ = pd.read_csv(SUMM)
print(summ.to_string())
g = {'__file__': SRC, '__name__': '__main__', 'os': os, 'np': __import__('numpy'), 'pd': pd, 'ROOT': ROOT,
     'summ': summ, 'MODELS_6': ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer']}
exec(compile(src, SRC, 'exec'), g)
