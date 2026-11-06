"""
fig7_nsd_baselines_within_ceiling.py -- Figure 7 (appendix NSD: baseline VLMs and saliency models) with
test R^2 expressed as % of each participant's NSD noise ceiling, NC = ncsnr^2 / (ncsnr^2 + 1/3) averaged
over the ROI's analysed vertices (as slides_fig3_noise_ceiling_compare.py). Everything else (data, bar
order, colours, layout) is fig7_nsd_baselines.py: its replacements are applied to
molmo_bilinear/nsd_saliency/plot_nsd_saliency.py, and then the R^2 of each participant and region is
divided by that participant's ceiling for the region before bars and SEMs are taken.
Font: Helvetica Neue, falling back to Nimbus Sans where it is not installed (Della).
Writes final_figures/manuscript_2026-10-23/fig7_nsd_vlms_saliency_withinNC.png (not in the manuscript)
"""
import os, sys, shutil, tempfile
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans']
ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
NSD = os.path.join(ROOT, 'rebuttal', 'task8_nsd')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig7_nsd_vlms_saliency_withinNC.png')
YMAX = 20

# within-participant noise ceiling per participant and region
nc = np.load(os.path.join(NSD, 'responses', 'noise_ceiling.npz'))
fit = (nc['ncsnr'] > 0.2) & nc['valid']
masks = np.load(os.path.join(NSD, 'rois', 'roi_masks_fs5.npz'))
NC = {}
for s in range(1, 9):
    sn = np.load(os.path.join(NSD, 'betas', f'ncsnr_subj{s:02d}.npy'))
    for roi in ('early', 'ventral', 'lateral', 'parietal'):
        v = sn[fit & masks[f'grp_{roi}'].astype(bool)]
        NC[s, roi] = float((v ** 2 / (v ** 2 + 1 / 3)).mean())

# the source as fig7_nsd_baselines.py modifies it, then the ceiling normalisation on top
f7 = open(os.path.join(HERE, 'fig7_nsd_baselines.py')).read()
ns = {}
exec(f7[f7.index('src = open(SRC).read()'):f7.index('sys.argv = [SRC')].replace('TMP', 'TMP_'),
     {'SRC': os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'nsd_saliency', 'plot_nsd_saliency.py'),
      'TMP_': (TMP := tempfile.mkdtemp(dir=os.path.dirname(OUT))), 'FST': None}, ns)
src = ns['src']
for old, new in [
        ("df = pd.DataFrame(rows); df.to_csv(OUT / 'nsd_with_saliency_per_subject.csv', index=False)",
         "df = pd.DataFrame(rows)\ndf['r2_attn'] = 100 * df.r2_attn / df.apply(lambda x: NC[x.subject, x.roi], axis=1)"),
        ("ax.set_xticks([]); ax.set_ylim(0, 0.07)", "ax.set_xticks([]); ax.set_ylim(0, YMAX)"),
        ("axes[r, 0].set_ylabel(f'{task.capitalize()} task\\nTest $R^2$', fontsize=20)",
         "axes[r, 0].set_ylabel(f'{task.capitalize()} task\\n% of noise ceiling', fontsize=20)")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
sys.argv = ['plot_nsd_saliency.py', '--letters']
g = {'__name__': '__main__', 'NC': NC, 'YMAX': YMAX}
exec(compile(src, 'plot_nsd_saliency.py', 'exec'), g)
top = g['df'].groupby(['task', 'model', 'roi']).r2_attn.agg(lambda v: v.mean() + v.std(ddof=1) / np.sqrt(len(v))).max()
assert top <= YMAX, top
shutil.copy2(os.path.join(TMP, 'fig_nsd_with_saliency_AB.png'), OUT)
shutil.rmtree(TMP)
print('wrote', OUT)
