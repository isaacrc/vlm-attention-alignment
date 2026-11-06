"""
nsd_molmo_finish.py
-------------------
After the Molmo / ShufMolmo refits (nsd_molmo_prepare.py + run_nsd_molmo_fit.slurm):
rebuild the cvbehav ROI tables, the per-subject attention table and the section 4.3
statistics exactly as qwen35_nothink/nsd_finish.py does, then draw the two NSD figures
on raw test R^2 (qwen35_nothink/nsd_raw_preview.py), the version the manuscript uses.

Outputs: final_figures/molmo_bilinear/nsd/ (tables, nsd_stats.txt, figures) and
overleaf/figures/fig_roi_and_social_testr2_v5.png, fig_nsd_regional_describe_testr2_v5.png
"""
import os, shutil, sys
from pathlib import Path
from PIL import Image

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
NT = ROOT / 'neurips_final' / 'qwen35_nothink'
OUT = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'nsd'
OUT.mkdir(parents=True, exist_ok=True)

src = (NT / 'nsd_finish.py').read_text()
old = "OUT = HERE.parent / 'final_figures' / 'qwen35_nothink' / 'nsd'"
assert old in src
src = src.replace(old, f'OUT = Path({str(OUT)!r})')
exec(compile(src, str(NT / 'nsd_finish.py'), 'exec'), {'__file__': str(NT / 'nsd_finish.py'), '__name__': '__main__'})

src = (NT / 'nsd_raw_preview.py').read_text()
old = "OUT = ROOT / 'neurips_final' / 'final_figures' / 'qwen35_nothink' / 'nsd' / 'raw_preview'"
assert old in src
src = src.replace(old, f'OUT = Path({str(OUT / "raw")!r})')
os.chdir(ROOT / 'rebuttal' / 'task8_nsd')
exec(compile(src, str(NT / 'nsd_raw_preview.py'), 'exec'), {'__file__': str(NT / 'nsd_raw_preview.py'), '__name__': '__main__'})

reg = OUT / 'raw' / 'regional'
A = Image.open(ROOT / 'rebuttal' / 'plots' / 'nsd_roi_atlas_grouped_A.png').convert('RGB')
B = Image.open(reg / 'nsd_regional_attnonly_cvbehav_social.png').convert('RGB')
canvas = Image.new('RGB', (max(A.width, B.width), 1311 + B.height), 'white')
canvas.paste(A, (0, 0)); canvas.paste(B, (0, 1311))
FIGS = ROOT / 'neurips_final' / 'overleaf' / 'figures'
canvas.save(FIGS / 'fig_roi_and_social_testr2_v5.png')
shutil.copy2(reg / 'nsd_regional_attnonly_cvbehav_describe.png', FIGS / 'fig_nsd_regional_describe_testr2_v5.png')
print('DONE nsd_molmo_finish', flush=True)
