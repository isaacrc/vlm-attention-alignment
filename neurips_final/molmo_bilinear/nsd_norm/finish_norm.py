"""
finish_norm.py
--------------
After run_fit_norm.slurm: the per-region tables, the section 4.3 statistics and the two
NSD figures, from the sum-to-1 fits (task8_nsd/encoding_norm/), built exactly as
qwen35_nothink/nsd_finish.py and nsd_raw_preview.py build the published ones (raw test
R^2, "Test R^2" axis). Everything goes to new locations:
  task8_nsd/results_norm/
  neurips_final/final_figures/molmo_bilinear/nsd_norm/   (tables, nsd_stats.txt, figures)
Nothing in the manuscript or overleaf/figures is touched.
"""
import subprocess, sys
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
NSD = ROOT / 'rebuttal' / 'task8_nsd'
NT = ROOT / 'neurips_final' / 'qwen35_nothink'
HERE = Path(__file__).resolve().parent
ENC, RES = NSD / 'encoding_norm', NSD / 'results_norm'
OUT = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'nsd_norm'
RES.mkdir(exist_ok=True); (OUT / 'regional').mkdir(parents=True, exist_ok=True)

n = len(list(ENC.glob('r2_*_cvbehav.npz')))
assert n == 8 * 8 * 2 * 2, f'{n} fits in encoding_norm, expected 256'

# 1. the two summarize scripts, pointed at encoding_norm / results_norm
for name in ('summarize_by_roi.py', 'summarize_permodel_floor.py'):
    src = (NSD / name).read_text()
    src = src.replace('HERE = Path(__file__).parent', f'HERE = Path({str(NSD)!r})')
    for old, new in (('ENC = HERE / "encoding"', f'ENC = Path({str(ENC)!r})'),
                     ('OUT = HERE / "results"', f'OUT = Path({str(RES)!r})')):
        if old in src:
            src = src.replace(old, new)
    assert f'OUT = Path({str(RES)!r})' in src, name
    (HERE / f'_{name}').write_text(src)
    subprocess.run([sys.executable, str(HERE / f'_{name}'), '--head-rule', 'cvbehav'], cwd=NSD, check=True)

# 2. attention-only table, contrasts and statistics: nsd_finish.py up to its figures
src = (NT / 'nsd_finish.py').read_text()
src = src[:src.index('# ── figures')]
for old, new in (("RES = NSD / 'results'", f'RES = Path({str(RES)!r})'),
                 ("OUT = HERE.parent / 'final_figures' / 'qwen35_nothink' / 'nsd'", f'OUT = Path({str(OUT)!r})'),
                 ("run(sys.executable, 'summarize_by_roi.py', '--head-rule', 'cvbehav')\n", ''),
                 ("run(sys.executable, 'summarize_permodel_floor.py', '--head-rule', 'cvbehav')\n", '')):
    assert old in src, old
    src = src.replace(old, new)
exec(compile(src, str(NT / 'nsd_finish.py'), 'exec'), {'__file__': str(NT / 'nsd_finish.py'), '__name__': '__main__'})

YMAX = 0.07   # top of the R^2 axis; the tallest bar + SEM is about .061

# 3. figures on raw test R^2, as nsd_raw_preview.py draws them
per = pd.read_csv(RES / 'attnonly_per_subject_cvbehav.csv')
per['attn_corrected'] = per['attn_unique']
per.to_csv(OUT / 'attnonly_per_subject_cvbehav.csv', index=False)
sys.path.insert(0, str(NSD))
src = (NSD / 'plot_nsd_manuscript_figs.py').read_text()
old = '"Test $R^2$ above floor" if (attn_only or kastner) else'
assert old in src
src = src.replace(old, '"Test $R^2$" if (attn_only or kastner) else')
ns = {'__file__': str(NSD / 'plot_nsd_manuscript_figs.py'), '__name__': 'nsd_figs'}
import os; os.chdir(NSD)
exec(compile(src, str(NSD / 'plot_nsd_manuscript_figs.py'), 'exec'), ns)
ns['regional'].__globals__['RES'] = OUT
ns['regional'].__globals__['PLOTS'] = OUT
ns['regional']('cvbehav', tasks=['social'], letter='B', attn_only=True, ymax=YMAX, stars=False)
ns['regional']('cvbehav', tasks=['describe'], attn_only=True, ymax=YMAX, stars=False)

A = Image.open(ROOT / 'rebuttal' / 'plots' / 'nsd_roi_atlas_grouped_A.png').convert('RGB')
B = Image.open(OUT / 'regional' / 'nsd_regional_attnonly_cvbehav_social.png').convert('RGB')
canvas = Image.new('RGB', (max(A.width, B.width), 1311 + B.height), 'white')
canvas.paste(A, (0, 0)); canvas.paste(B, (0, 1311))
canvas.save(OUT / 'fig3_neural_norm.png')
Image.open(OUT / 'regional' / 'nsd_regional_attnonly_cvbehav_describe.png').save(OUT / 'fig7_nsd_describe_norm.png')
print('DONE finish_norm', flush=True)
