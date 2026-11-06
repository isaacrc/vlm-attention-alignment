"""
nsd_finish.py
-------------
After the Qwen35 / ShufQwen35 refits: rebuild the cvbehav ROI tables, the attention-only
per-subject table that Figure 4 and §4.3 use, the statistics, and the two NSD figures.

The attention-only table is r2_attn(model) - r2_attn(Shuf{model}) per subject x region,
which reproduces the thinking-on attnonly_per_subject_cvbehav.csv to 1e-16 (checked
2026-10-20); there was no saved script for it.

Outputs:
  task8_nsd/results/{encoding_by_roi*,permodel_*,attnonly_per_subject,attnonly_contrasts}_cvbehav.csv
  neurips_final/final_figures/qwen35_nothink/nsd/  (copies + nsd_stats.txt + the two figures)
"""
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

NSD = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
PLOTS = NSD.parent / 'plots'
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'final_figures' / 'qwen35_nothink' / 'nsd'
OUT.mkdir(parents=True, exist_ok=True)
RES = NSD / 'results'
MODELS = ['CNN_LSTM', 'ViT_LSTM', 'CNN_Transformer', 'ViT_Transformer', 'Molmo', 'Qwen35',
          'EdgeEnergy', 'Luminance']
ROIS = ['early', 'ventral', 'lateral', 'parietal']


def run(*cmd, cwd=NSD):
    print('$', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


run(sys.executable, 'summarize_by_roi.py', '--head-rule', 'cvbehav')
run(sys.executable, 'summarize_permodel_floor.py', '--head-rule', 'cvbehav')

# ── attention-only per subject ────────────────────────────────────────────────
e = pd.read_csv(RES / 'encoding_by_roi_per_subject_cvbehav.csv')
key = ['task', 'subject', 'roi']
rows = []
for m in MODELS:
    real = e[e.model == m].set_index(key)['r2_attn'].rename('attn_unique')
    floor = e[e.model == f'Shuf{m}'].set_index(key)['r2_attn'].rename('floor')
    j = pd.concat([real, floor], axis=1, join='inner').reset_index()
    j['model'] = m
    j['attn_corrected'] = j['attn_unique'] - j['floor']
    rows.append(j)
ao = pd.concat(rows, ignore_index=True)
ao = ao[ao.roi.isin(ROIS)][['model', 'task', 'subject', 'roi', 'attn_corrected', 'attn_unique', 'floor']]
assert len(ao) == len(MODELS) * 2 * 8 * 4, len(ao)
ao.to_csv(RES / 'attnonly_per_subject_cvbehav.csv', index=False)

con = []
for (roi, task), g in ao.groupby(['roi', 'task']):
    w = g.pivot(index='subject', columns='model', values='attn_corrected')
    for a, b in itertools.combinations(MODELS, 2):
        d = (w[a] - w[b]).dropna()
        con.append(dict(roi=roi, task=task, a=a, b=b, diff=1000 * d.mean(),
                        dz=d.mean() / d.std(ddof=1), p=stats.ttest_rel(w[a], w[b]).pvalue))
pd.DataFrame(con).to_csv(RES / 'attnonly_contrasts_cvbehav.csv', index=False)

# ── statistics quoted in §4.3 ─────────────────────────────────────────────────
with open(OUT / 'nsd_stats.txt', 'w') as f:
    # statsmodels lives in cans-env, not tribe_enc
    subprocess.run(['/home/isaacrc/.conda/envs/cans-env/bin/python', str(HERE / 'nsd_stats.py'), str(RES / 'attnonly_per_subject_cvbehav.csv')],
                   stdout=f, check=True)
print(open(OUT / 'nsd_stats.txt').read())

# ── figures (same calls as slurm_fig4.sh / slurm_desc_app.sh) ────────────────
run(sys.executable, 'plot_nsd_manuscript_figs.py', '--head-rule', 'cvbehav', '--tasks', 'social',
    '--regional-only', '--attn-only', '--panel-letter', 'B', '--ymax', '0.08', '--no-stars')
soc = PLOTS / 'regional' / 'nsd_regional_attnonly_cvbehav_social.png'
run(sys.executable, 'plot_nsd_manuscript_figs.py', '--head-rule', 'cvbehav', '--tasks', 'describe',
    '--regional-only', '--attn-only', '--ymax', '0.08', '--no-stars')
des = PLOTS / 'regional' / 'nsd_regional_attnonly_cvbehav_describe.png'

# Figure 4 = ROI atlas (panel A, unchanged) over the Social bars, at the offsets of the
# published composite (A at the origin, B at y=1311, white 4587 x 2989 canvas).
A = Image.open(PLOTS / 'nsd_roi_atlas_grouped_A.png').convert('RGB')
B = Image.open(soc).convert('RGB')
canvas = Image.new('RGB', (max(A.width, B.width), 1311 + B.height), 'white')
canvas.paste(A, (0, 0)); canvas.paste(B, (0, 1311))
canvas.save(OUT / 'fig_roi_and_social_attnonly.png')
shutil.copy2(des, OUT / 'fig_nsd_regional_describe.png')
for name in ('attnonly_per_subject_cvbehav.csv', 'attnonly_contrasts_cvbehav.csv',
             'permodel_vs_floor_cvbehav.csv', 'encoding_by_roi_per_subject_cvbehav.csv'):
    shutil.copy2(RES / name, OUT / name)
print('DONE nsd_finish', flush=True)
