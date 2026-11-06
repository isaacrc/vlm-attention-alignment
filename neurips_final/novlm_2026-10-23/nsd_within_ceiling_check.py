"""
nsd_within_ceiling_check.py -- re-runs every Section 4.3 neural statistic of main_2026-10-23.tex with each
participant's ROI R^2 divided by that participant's own NSD noise ceiling (NC = ncsnr^2 / (ncsnr^2 + 1/3),
averaged over the ROI's analysed vertices), next to the published raw version.
Published analysis (reconstructed here; it reproduces F(3,21) = 66.94, d_z 5.11/2.52/1.81, encoder
F = 43.45, decoder F = 0.03, interaction F = 41.24, interaction x task F = 0.11 exactly): floor-corrected
R^2 (attn_corrected, task8_nsd/results_norm/attnonly_per_subject_cvbehav.csv) of the four trained models,
averaged over the two tasks, except the encoder x decoder x region x task RM-ANOVA, which keeps task.
Also prints the appendix (Figure 7) cross-region pattern for Molmo, Qwen3.5 and the saliency models from
raw R^2 (molmo_bilinear/nsd_saliency/nsd_with_saliency_per_subject.csv), raw and as % of NC.
Writes nothing; prints a side-by-side table.
"""
import os, warnings
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
warnings.filterwarnings('ignore')

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
NSD = os.path.join(ROOT, 'rebuttal', 'task8_nsd')
ROIS = ['early', 'ventral', 'lateral', 'parietal']
KEYS = {r: f'grp_{r}' for r in ROIS}
T4 = {'CNN_LSTM': ('CNN', 'LSTM'), 'ViT_LSTM': ('ViT', 'LSTM'),
      'CNN_Transformer': ('CNN', 'Trans'), 'ViT_Transformer': ('ViT', 'Trans')}

nc = np.load(os.path.join(NSD, 'responses', 'noise_ceiling.npz'))
fit = (nc['ncsnr'] > 0.2) & nc['valid']
masks = np.load(os.path.join(NSD, 'rois', 'roi_masks_fs5.npz'))
NC = {}
for s in range(1, 9):
    sn = np.load(os.path.join(NSD, 'betas', f'ncsnr_subj{s:02d}.npy'))
    for r in ROIS:
        v = sn[fit & masks[KEYS[r]].astype(bool)]
        NC[s, r] = float((v ** 2 / (v ** 2 + 1 / 3)).mean())
ncs = lambda df: df.apply(lambda x: NC[x.subject, x.roi], axis=1)


def dz(x):
    x = np.asarray(x, float)
    return x.mean() / x.std(ddof=1), stats.ttest_1samp(x, 0).pvalue


def gg(wide):
    S = np.cov(wide.values.T); k = S.shape[0]; C = np.eye(k) - 1.0 / k; Sc = C @ S @ C
    return np.trace(Sc) ** 2 / ((k - 1) * np.sum(Sc ** 2))


d = pd.read_csv(os.path.join(NSD, 'results_norm', 'attnonly_per_subject_cvbehav.csv'))
d['raw'] = d.attn_corrected
d['nc'] = 100 * d.attn_corrected / ncs(d)
out = {}
for col in ('raw', 'nc'):
    R = {}
    t = d[d.model.isin(T4)].copy()
    t['enc'] = t.model.map(lambda m: T4[m][0]); t['dec'] = t.model.map(lambda m: T4[m][1])
    a = t.groupby(['subject', 'model', 'roi'])[col].mean().reset_index()           # task-averaged
    tab = AnovaRM(a, col, 'subject', within=['model', 'roi']).fit().anova_table
    F = tab.loc['roi', 'F Value']
    eps = gg(a.groupby(['subject', 'roi'])[col].mean().unstack()[ROIS])
    R['region F(3,21)'] = (F, stats.f.sf(F, 3 * eps, 21 * eps))
    m = a.groupby(['subject', 'roi'])[col].mean().unstack()
    for r in ('early', 'ventral', 'lateral'):
        R[f'parietal below {r} (dz)'] = dz(m[r] - m.parietal)
    R['mean by region (trained)'] = tuple(m[ROIS].mean().round(4 if col == 'raw' else 1))
    s = d.groupby(['subject', 'model', 'roi'])[col].mean().unstack('model')     # task-averaged, all predictors
    tm = s[list(T4)].mean(axis=1)
    for ctrl in ('EdgeEnergy', 'Luminance'):
        for r in ROIS:
            R[f'trained - {ctrl} {r} (dz)'] = dz((tm - s[ctrl]).xs(r, level='roi'))
    tab = AnovaRM(t, col, 'subject', within=['enc', 'dec', 'roi', 'task']).fit().anova_table
    for eff in ('enc', 'dec', 'enc:dec', 'enc:dec:task'):
        R[f'{eff} F(1,7)'] = (tab.loc[eff, 'F Value'], tab.loc[eff, 'Pr > F'])
    mm = t.groupby(['subject', 'model'])[col].mean().unstack()
    for x, y in (('CNN_LSTM', 'ViT_LSTM'), ('CNN_Transformer', 'ViT_Transformer'),
                 ('CNN_Transformer', 'CNN_LSTM'), ('ViT_LSTM', 'ViT_Transformer')):
        R[f'{x} > {y} (dz)'] = dz(mm[x] - mm[y])
    out[col] = R

print(f'{"statistic":42s} {"raw (published)":>22s} {"% of within-participant NC":>30s}')
for k in out['raw']:
    f = lambda v: (f'{v[0]:.2f}, p = {v[1]:.2g}' if isinstance(v[1], float) and len(v) == 2 and not isinstance(v[0], str)
                   else str(v))
    a, b = out['raw'][k], out['nc'][k]
    fa = str(a) if len(a) == 4 else f'{a[0]:7.2f}  p={a[1]:.2g}'
    fb = str(b) if len(b) == 4 else f'{b[0]:7.2f}  p={b[1]:.2g}'
    print(f'{k:42s} {fa:>22s} {fb:>30s}')

# appendix: cross-region pattern of the Figure 7 predictors (raw R^2, as Figure 7)
f7 = pd.read_csv(os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear', 'nsd_saliency',
                              'nsd_with_saliency_per_subject.csv'))
f7['nc'] = 100 * f7.r2_attn / ncs(f7)
print('\nFigure 7 predictors, mean over participants (raw R^2 | % of NC):')
for task in ('social', 'describe'):
    g = f7[f7.task == task].groupby(['model', 'roi'])[['r2_attn', 'nc']].mean().unstack('roi')
    for mdl in g.index:
        print(f'  {task:8s} {mdl:16s} ' + '  '.join(f'{r[:4]} {g.loc[mdl, ("r2_attn", r)]:.3f}|{g.loc[mdl, ("nc", r)]:4.1f}' for r in ROIS))
