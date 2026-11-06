"""
fig1_alignment.py -- Figure 1 of main_2026-10-23.tex: the four trained models only.
Executes rebuttal/plots/plot_fig_head_selection_corrected.py on the molmo_bilinear_pairwise
results (the inputs of the 10-22 Figure 1, via scripts/fig3_slides.py) with Molmo and
Qwen3.5 dropped from every panel. Nothing else changes.
Writes final_figures/manuscript_2026-10-23/fig1_alignment.png
"""
import os

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'rebuttal', 'plots', 'plot_fig_head_selection_corrected.py')
RES = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig1_alignment.png')

src = open(SRC).read()
for old, new in [("os.path.join(DATA, 'alignment_summary_cv_honest.csv')", repr(os.path.join(RES, 'alignment_summary_cv_honest.csv'))),
                 ("os.path.join(T1, 'selected_heads.json')", repr(os.path.join(RES, 'selected_heads.json'))),
                 ("os.path.join(DATA, 'ceilings.json')", repr(os.path.join(RES, 'ceilings.json'))),
                 ("os.path.join(T2, 'model_between_task.csv')", repr(os.path.join(RES, 'model_between_task_fig1.csv'))),
                 ("os.path.join(HERE, 'head_selection', 'fig_head_selection_corrected.png')", repr(OUT)),
                 ("MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']",
                  "MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer']")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
exec(compile(src, SRC, 'exec'), {'__file__': SRC, '__name__': '__main__'})
