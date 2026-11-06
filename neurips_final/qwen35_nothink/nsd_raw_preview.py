"""
nsd_raw_preview.py
------------------
Preview of the NSD regional figures (Fig 3B Social, appendix Describe) with RAW test R^2,
i.e. without subtracting each predictor's own permutation floor. Floors are |R^2| < 0.001
everywhere, so this should look the same as the floor-corrected figures.

Runs plot_nsd_manuscript_figs.regional() unchanged except: attn_corrected is replaced by
attn_unique (the raw R^2) in a scratch copy of the per-subject table, the y-label reads
"Test R^2", and output goes to final_figures/qwen35_nothink/nsd/raw_preview/.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
NSD = ROOT / 'rebuttal' / 'task8_nsd'
OUT = ROOT / 'neurips_final' / 'final_figures' / 'qwen35_nothink' / 'nsd' / 'raw_preview'
(OUT / 'regional').mkdir(parents=True, exist_ok=True)

per = pd.read_csv(NSD / 'results' / 'attnonly_per_subject_cvbehav.csv')
per['attn_corrected'] = per['attn_unique']
per.to_csv(OUT / 'attnonly_per_subject_cvbehav.csv', index=False)

sys.path.insert(0, str(NSD))
src_path = NSD / 'plot_nsd_manuscript_figs.py'
src = src_path.read_text()
old = '"Test $R^2$ above floor" if (attn_only or kastner) else'
assert old in src
src = src.replace(old, '"Test $R^2$" if (attn_only or kastner) else')
ns = {'__file__': str(src_path), '__name__': 'nsd_figs'}
exec(compile(src, str(src_path), 'exec'), ns)
ns['regional'].__globals__['RES'] = OUT
ns['regional'].__globals__['PLOTS'] = OUT
ns['regional']('cvbehav', tasks=['social'], letter='B', attn_only=True, ymax=0.08, stars=False)
ns['regional']('cvbehav', tasks=['describe'], attn_only=True, ymax=0.08, stars=False)
