"""
make_figures.py
---------------
Every behavioural figure and table of the manuscript, rebuilt from the molmo_bilinear run
(rerun_molmo_bilinear.py, then the blur sweep), and copied into overleaf/figures/ under
new names so the previous versions stay untouched:

  fig_alignment_v5.png              Figure 1         scripts/fig3_slides.py
  fig_attn_by_task_v3.png           Figure 2         qwen35_nothink/final_fig5_entropy.py (+ entropy table)
  fig_layer_mean_vs_best_head_v2.png  appendix       rebuttal/plots/plot_fig_center_corrected.py
  fig_per_layer_v2.png              appendix         rebuttal/plots/plot_fig_per_layer.py
  fig_metric_overview_v6.png        appendix         qwen35_nothink/metric_overview_notitle.py
  fig_blur_sweep_v2.png             appendix         qwen35_nothink/blur_sweep.py (already run)
  fig_caption_cosine.png            appendix         scripts/fig_caption_cosine_slides.py (already run)
  table_baselines_v2.tex            appendix table   scripts/make_baseline_table.py
"""
import os, shutil, subprocess, sys

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
NF = os.path.join(ROOT, 'neurips_final')
MB = os.path.join(NF, 'final_figures', 'molmo_bilinear')
RES, APP = os.path.join(MB, 'results'), os.path.join(MB, 'appendix')
FIGS = os.path.join(NF, 'overleaf', 'figures')
os.makedirs(APP, exist_ok=True)
env = dict(os.environ, QWEN_NOTHINK='1', MOLMO_BILINEAR='1')


def run(script):
    print('$', script, flush=True)
    subprocess.run([sys.executable, script], env=env, check=True)


run(os.path.join(NF, 'scripts', 'fig3_slides.py'))
run(os.path.join(NF, 'qwen35_nothink', 'final_fig5_entropy.py'))
run(os.path.join(NF, 'qwen35_nothink', 'metric_overview_notitle.py'))
run(os.path.join(NF, 'scripts', 'make_baseline_table.py'))

# the two Transformer-layer figures read RES/OUT at module level; point them at this run
for name in ('plot_fig_center_corrected.py', 'plot_fig_per_layer.py'):
    path = os.path.join(ROOT, 'rebuttal', 'plots', name)
    src = open(path).read()
    for old, new in [("RES = os.path.join(REBUTTAL, 'task3_baselines', 'results')", f'RES = {RES!r}'),
                     ("OUT = os.path.join(HERE, 'head_selection')", f'OUT = {APP!r}')]:
        assert old in src, (name, old)
        src = src.replace(old, new)
    print('$', name, flush=True)
    exec(compile(src, path, 'exec'), {'__file__': path, '__name__': '__main__'})

copies = {
    os.path.join(MB, 'fig3_cvheads_qwen_nothink.png'): 'fig_alignment_v5.png',
    os.path.join(MB, 'fig5_cvheads_attn_by_task.png'): 'fig_attn_by_task_v3.png',
    os.path.join(APP, 'fig_layer_mean_vs_best_head_controlling_for_center_bias.png'): 'fig_layer_mean_vs_best_head_v2.png',
    os.path.join(APP, 'fig_per_layer.png'): 'fig_per_layer_v2.png',
    os.path.join(APP, 'fig_metric_overview_notitle.png'): 'fig_metric_overview_v6.png',
    os.path.join(MB, 'blur_sweep', 'fig_blur_sweep.png'): 'fig_blur_sweep_v2.png',
    os.path.join(NF, 'final_figures', 'slides', 'caption_cosine_describe_social_slide.png'): 'fig_caption_cosine.png',
}
for src, dst in copies.items():
    shutil.copy2(src, os.path.join(FIGS, dst))
    print('copied', dst, flush=True)
