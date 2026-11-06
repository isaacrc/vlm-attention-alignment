"""
make_figures_pairwise.py
------------------------
The behavioural figures and baselines table from the pairwise-ceiling bilinear run
(rerun_molmo_bilinear_pairwise.py), written to overleaf/figures/ under new names and,
with every other figure the manuscript uses, to final_figures/manuscript_2026-10-22/.
"""
import os, re, shutil, subprocess, sys

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
NF = os.path.join(ROOT, 'neurips_final')
MB = os.path.join(NF, 'final_figures', 'molmo_bilinear_pairwise')
FIGS = os.path.join(NF, 'overleaf', 'figures')
env = dict(os.environ, QWEN_NOTHINK='1', MOLMO_BILINEAR='1', MOLMO_BILINEAR_DIR='molmo_bilinear_pairwise')
for script in (os.path.join(NF, 'scripts', 'fig3_slides.py'),
               os.path.join(NF, 'qwen35_nothink', 'final_fig5_entropy.py'),
               os.path.join(NF, 'qwen35_nothink', 'metric_overview_notitle.py'),
               os.path.join(NF, 'qwen35_nothink', 'blur_sweep.py'),
               os.path.join(NF, 'scripts', 'make_baseline_table.py')):
    print('$', script, flush=True)
    subprocess.run([sys.executable, script], env=env, check=True)

copies = {
    os.path.join(MB, 'fig3_cvheads_qwen_nothink.png'): 'fig_alignment_v6.png',
    os.path.join(MB, 'fig5_cvheads_attn_by_task.png'): 'fig_attn_by_task_v4.png',
    os.path.join(MB, 'appendix', 'fig_metric_overview_notitle.png'): 'fig_metric_overview_v7.png',
    os.path.join(MB, 'blur_sweep', 'fig_blur_sweep.png'): 'fig_blur_sweep_v3.png',
}
for src, dst in copies.items():
    shutil.copy2(src, os.path.join(FIGS, dst))
    print('copied', dst, flush=True)
