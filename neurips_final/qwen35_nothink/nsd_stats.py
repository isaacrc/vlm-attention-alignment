"""
nsd_stats.py
------------
Every NSD statistic quoted in §4.3 (Social task, attention-only, per-model floors), from a
per-subject table with columns [model, task, subject, roi, attn_corrected].

Reproduces the published values when pointed at the thinking-on table
(rebuttal/task8_nsd/results/attnonly_per_subject_cvbehav.csv), which is how the model sets
below were identified:
  region effect        model(6) x region RM-ANOVA, Greenhouse-Geisser on region   -> F(3,21)=42.54
  parietal contrasts   six-model mean, paired across subjects                       -> dz 4.93/2.80/2.09
  low-level contrasts  four trained models' mean vs EdgeEnergy / Luminance
  architecture         encoder x decoder x region RM-ANOVA on the four trained models;
                       simple effects average each model over the four regions
  floor tests          each model vs 0 (already floor-corrected), Holm within model x task
                       across the four regions

Usage: python nsd_stats.py <per_subject.csv>
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

T4 = {'CNN_LSTM': ('CNN', 'LSTM'), 'ViT_LSTM': ('ViT', 'LSTM'),
      'CNN_Transformer': ('CNN', 'Trans'), 'ViT_Transformer': ('ViT', 'Trans')}
M6 = list(T4) + ['Molmo', 'Qwen35']
ROIS = ['early', 'ventral', 'lateral', 'parietal']


def dz(x):
    x = np.asarray(x, float)
    return x.mean() / x.std(ddof=1), stats.ttest_1samp(x, 0).pvalue


def holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def gg_epsilon(wide):
    S = np.cov(wide.values.T); k = S.shape[0]
    C = np.eye(k) - 1.0 / k; Sc = C @ S @ C
    return np.trace(Sc) ** 2 / ((k - 1) * np.sum(Sc ** 2))


def main(path):
    d = pd.read_csv(path)
    for task in ('social', 'describe'):
        t = d[d.task == task]
        print(f'\n================ {task.upper()} ================')
        s = t.pivot_table(index=['subject', 'roi'], columns='model', values='attn_corrected')

        # region effect: model(6) x region
        a6 = t[t.model.isin(M6)]
        r = AnovaRM(a6, 'attn_corrected', 'subject', within=['model', 'roi']).fit().anova_table
        F, df1, df2 = r.loc['roi', ['F Value', 'Num DF', 'Den DF']]
        eps = gg_epsilon(a6.groupby(['subject', 'roi']).attn_corrected.mean().unstack()[ROIS])
        p_gg = stats.f.sf(F, df1 * eps, df2 * eps)
        print(f'region (6 models): F({df1:.0f},{df2:.0f}) = {F:.2f}, GG eps = {eps:.3f}, p_GG = {p_gg:.2g}')

        m6 = s[M6].mean(axis=1).unstack('roi')
        for roi in ('early', 'ventral', 'lateral'):
            e, p = dz(m6[roi] - m6['parietal'])
            print(f'  parietal below {roi:8s}: dz = {e:.2f}, p = {p:.2g}')

        m4 = s[list(T4)].mean(axis=1).unstack('roi')
        for ctrl in ('EdgeEnergy', 'Luminance'):
            c = s[ctrl].unstack('roi')
            for roi in ROIS:
                e, p = dz(m4[roi] - c[roi])
                print(f'  trained mean - {ctrl:10s} {roi:8s}: dz = {e:+.2f}, p = {p:.2g}')

        tr = t[t.model.isin(T4)].copy()
        tr['enc'] = tr.model.map(lambda m: T4[m][0]); tr['dec'] = tr.model.map(lambda m: T4[m][1])
        r = AnovaRM(tr, 'attn_corrected', 'subject', within=['enc', 'dec', 'roi']).fit().anova_table
        for eff in ('enc', 'dec', 'enc:dec'):
            print(f'  {eff:8s} F(1,7) = {r.loc[eff, "F Value"]:.2f}, p = {r.loc[eff, "Pr > F"]:.2g}')
        mm = s[list(T4)].groupby('subject').mean()
        for a, b in (('CNN_LSTM', 'ViT_LSTM'), ('CNN_Transformer', 'ViT_Transformer'),
                     ('CNN_Transformer', 'CNN_LSTM'), ('ViT_LSTM', 'ViT_Transformer')):
            e, p = dz(mm[a] - mm[b])
            print(f'  {a} - {b}: dz = {e:.2f}, p = {p:.2g}')

        print('  floor tests (Holm within model across regions):')
        for m in M6 + ['EdgeEnergy', 'Luminance']:
            ps, es = [], []
            for roi in ROIS:
                x = s[m].xs(roi, level='roi'); e, p = dz(x); ps.append(p); es.append(e)
            ph = holm(ps)
            print(f'    {m:16s} ' + '  '.join(f'{roi}: {s[m].xs(roi, level="roi").mean():.4f} (pH={q:.2g})'
                                            for roi, q in zip(ROIS, ph)))


if __name__ == '__main__':
    main(sys.argv[1])
