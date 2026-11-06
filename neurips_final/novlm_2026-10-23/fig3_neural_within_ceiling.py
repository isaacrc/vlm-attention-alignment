"""
fig3_neural_within_ceiling.py -- Figure 3 (neural encoding) with test R^2 expressed as % of each
participant's NSD noise ceiling, NC = ncsnr^2 / (ncsnr^2 + 1/3) averaged over the ROI's analysed vertices
(as nsd_within_ceiling_check.py and fig7_nsd_baselines_within_ceiling.py). Everything else is
fig3_neural.py: the same replacements are applied to molmo_bilinear/nsd_norm/plot_fig3_three_panel.py,
then each participant's region R^2 is divided by that participant's ceiling for the region before bars and
SEMs are taken. Panel A (ROI atlas) is unchanged.
Font: Helvetica Neue, falling back to Nimbus Sans where it is not installed (Della).
Writes final_figures/manuscript_2026-10-23/fig3_neural_withinNC.png (not in the manuscript)
"""
import os, shutil, tempfile
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans']
ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
NSD = os.path.join(ROOT, 'rebuttal', 'task8_nsd')
SRC = os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'nsd_norm', 'plot_fig3_three_panel.py')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig3_neural_withinNC.png')
TMP = tempfile.mkdtemp(dir=os.path.dirname(OUT))

nc = np.load(os.path.join(NSD, 'responses', 'noise_ceiling.npz'))
fit = (nc['ncsnr'] > 0.2) & nc['valid']
masks = np.load(os.path.join(NSD, 'rois', 'roi_masks_fs5.npz'))
NC = {}
for s in range(1, 9):
    sn = np.load(os.path.join(NSD, 'betas', f'ncsnr_subj{s:02d}.npy'))
    for roi in ('early', 'ventral', 'lateral', 'parietal'):
        v = sn[fit & masks[f'grp_{roi}'].astype(bool)]
        NC[s, roi] = float((v ** 2 / (v ** 2 + 1 / 3)).mean())

src = open(SRC).read()
f3 = open(os.path.join(HERE, 'fig3_neural.py')).read()
f3_reps = eval(f3[f3.index('for old, new in [') + len('for old, new in '):f3.index(']:\n    assert src.count(old) == 1') + 1]
               .replace('{TMP!r}', repr(TMP)).replace('f"OUTDIR', '"OUTDIR'))   # fig3_neural.py's bar-order and OUTDIR edits
for old, new in f3_reps + [
        ("d = pd.read_csv(ROOT / 'rebuttal' / 'task8_nsd' / 'results_norm' / 'attnonly_per_subject_cvbehav.csv')",
         "d = pd.read_csv(ROOT / 'rebuttal' / 'task8_nsd' / 'results_norm' / 'attnonly_per_subject_cvbehav.csv')\n"
         "d['attn_unique'] = 100 * d.attn_unique / d.apply(lambda x: NC[x.subject, x.roi], axis=1)"),
        ("ax.set_ylim(0, 0.07); ax.set_yticks([0, 0.02, 0.04, 0.06]); ax.set_xticks([])",
         "ax.set_ylim(0, 20); ax.set_yticks([0, 5, 10, 15, 20]); ax.set_xticks([])"),
        ("axes[r, 0].set_ylabel(f'{task.capitalize()} task\\nTest $R^2$', fontsize=FS_LABEL, labelpad=10)",
         "axes[r, 0].set_ylabel(f'{task.capitalize()} task\\n% of noise ceiling', fontsize=FS_LABEL, labelpad=10)")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
g = {'__file__': SRC, '__name__': '__main__', 'NC': NC}
exec(compile(src, SRC, 'exec'), g)
d = g['d']
top = d.groupby(['task', 'model', 'roi']).attn_unique.agg(lambda v: v.mean() + v.std(ddof=1) / np.sqrt(len(v))).max()
assert top <= 20, top
shutil.copy2(os.path.join(TMP, 'fig3_neural_three_panel.png'), OUT)
shutil.rmtree(TMP)
print('wrote', OUT)
