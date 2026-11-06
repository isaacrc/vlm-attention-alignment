"""
fig4_metric_suite.py -- Figure 4 (appendix metric suite) of main_2026-10-23.tex: the four
trained models only. Executes qwen35_nothink/metric_overview_notitle.py, the source of the
10-22 Figure 4, on the molmo_bilinear_pairwise results with Molmo and Qwen3.5 dropped from
ORDER. The per-metric side figures the source also draws go to novlm_2026-10-23/metric_suite/.
Writes final_figures/manuscript_2026-10-23/fig4_metric_suite.png
"""
import os, shutil

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'neurips_final', 'qwen35_nothink', 'metric_overview_notitle.py')
FIG = os.path.join(HERE, 'metric_suite'); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig4_metric_suite.png')

os.environ.update(MOLMO_BILINEAR='1', MOLMO_BILINEAR_DIR='molmo_bilinear_pairwise')
src = open(SRC).read()
for old, new in [("RES, FIG = os.path.join(NT, 'results'), os.path.join(NT, 'appendix')",
                  f"RES, FIG = os.path.join(NT, 'results'), {FIG!r}"),
                 ("src = head + \"OV_METRICS",
                  "head = head.replace(\"ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']\",\n"
                  "                    \"ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer']\", 1)\n"
                  "assert \"'Molmo', 'Qwen3.5']\" not in head.split('ORDER = ')[1].split('\\n')[0]\n"
                  "src = head + \"OV_METRICS")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
exec(compile(src, SRC, 'exec'), {'__file__': SRC, '__name__': '__main__'})
shutil.copy2(os.path.join(FIG, 'fig_metric_overview_notitle.png'), OUT)
print('wrote', OUT)
