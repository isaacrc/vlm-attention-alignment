"""
caption_tests.py -- statistics for Appendix G (caption quality), on caption_embed.py's output.

Test 1  Each trained model vs the human-human reference, paired across participants: for each
        participant, (model vs that participant's captions) - (other participants vs that
        participant's captions), averaged over images. Sign-flip permutation (10,000), Cohen's
        d_z, Holm across the four models within each task. Plus the decoder contrasts on caption
        similarity itself (per image, participant-averaged), as the alignment contrasts are tested.
Test 2  Does the decoder effect on alignment survive caption similarity? Per image i, for each
        encoder and task:  dA_i = alignment(LSTM)_i - alignment(Transformer)_i   (participant-
        averaged Fisher z, the Figure 1 data)  and  dC_i = caption similarity(LSTM)_i -
        caption similarity(Transformer)_i  (participant-averaged cosine). OLS dA ~ 1 + dC with HC3
        errors: the intercept is the decoder gap at equal caption similarity, the slope says whether
        images where the LSTM's caption is more human-like are images where its attention is.
        Secondary: all four models pooled, per image x model, alignment ~ decoder + encoder +
        caption similarity + image fixed effects.
Writes results/caption_tests.txt, caption_test1.csv, caption_test2.csv
"""
import os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TRAINED = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer']
PAIRS = [('CNN', 'CNN-LSTM', 'CNN-Transformer'), ('ViT', 'ViT-LSTM', 'ViT-Transformer')]
rng = np.random.default_rng(0)
log = []
def say(s=''):
    print(s, flush=True); log.append(s)

def signflip(d, n=10000):
    d = np.asarray(d, float); obs = d.mean()
    null = (rng.choice([-1, 1], size=(n, len(d))) * d).mean(1)
    return obs, obs / d.std(ddof=1), (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * p[i])); out[i] = run
    return out

cap = pd.read_csv(os.path.join(RES, 'caption_cosine_perimage.csv'))
cap['subject_id'] = cap.subject_id.str.replace('-', '', regex=False)
psub = cap.groupby(['task', 'model', 'subject_id'])['cos'].mean().unstack('model')

# ---- Test 1 ---------------------------------------------------------------
say('TEST 1a  model vs human-human caption similarity (paired across participants)')
t1 = []
for task in ('describe', 'social'):
    p = psub.loc[task]
    for m in TRAINED:
        obs, dz, pp = signflip(p[m] - p['Human'])
        t1.append(dict(test='model_vs_human', task=task, a=m, b='Human', n=len(p), mean_a=p[m].mean(),
                       mean_b=p['Human'].mean(), diff=obs, dz=dz, p_perm=pp))
t1 = pd.DataFrame(t1)
t1['p_holm'] = t1.groupby('task').p_perm.transform(lambda x: holm(x.values))
for r in t1.itertuples():
    say(f'  {r.task:8s} {r.a:16s} {r.mean_a:.3f} vs human {r.mean_b:.3f}  diff {r.diff:+.3f}  '
        f'dz {r.dz:+.2f}  p {r.p_perm:.4f}  pHolm {r.p_holm:.4f}  (n={r.n})')

say('\nTEST 1b  caption similarity, LSTM vs Transformer sharing the encoder (per image, participant-averaged)')
pim = cap.groupby(['task', 'model', 'image_id'])['cos'].mean().unstack('model')
t1b = []
for task in ('describe', 'social'):
    for enc_, l, t in PAIRS:
        x = pim.loc[task][[l, t]].dropna()
        obs, dz, pp = signflip(x[l] - x[t])
        t1b.append(dict(test='caption_decoder', task=task, a=l, b=t, n=len(x), mean_a=x[l].mean(),
                        mean_b=x[t].mean(), diff=obs, dz=dz, p_perm=pp))
t1b = pd.DataFrame(t1b)
for r in t1b.itertuples():
    say(f'  {r.task:8s} {r.a} {r.mean_a:.3f} vs {r.b} {r.mean_b:.3f}  diff {r.diff:+.4f}  dz {r.dz:+.2f}  p {r.p_perm:.4f}  (n={r.n} images)')
pd.concat([t1, t1b]).to_csv(os.path.join(RES, 'caption_test1.csv'), index=False)

# ---- Test 2 ---------------------------------------------------------------
al = pd.read_parquet(os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results',
                                  'alignment_long_cv_honest.parquet'))
al['image_id'] = al.image_id.str.replace('.jpg', '', regex=False).astype(int)
aim = al[al.model.isin(TRAINED)].groupby(['task', 'model', 'image_id'])['z'].mean().unstack('model')

say('\nTEST 2  per-image decoder gap in alignment (Fisher z), adjusted for the caption-similarity gap')
t2 = []
for task in ('describe', 'social'):
    for enc_, l, t in PAIRS:
        d = pd.DataFrame({'dA': aim.loc[task][l] - aim.loc[task][t],
                          'dC': pim.loc[task][l] - pim.loc[task][t]}).dropna()
        fit = smf.ols('dA ~ dC', d).fit(cov_type='HC3')
        raw_obs, raw_dz, raw_p = signflip(d.dA)
        t2.append(dict(task=task, encoder=enc_, n=len(d), raw_gap=d.dA.mean(), raw_dz=raw_dz, raw_p=raw_p,
                       mean_dC=d.dC.mean(), adj_gap=fit.params['Intercept'], adj_se=fit.bse['Intercept'],
                       adj_p=fit.pvalues['Intercept'], slope=fit.params['dC'], slope_se=fit.bse['dC'],
                       slope_p=fit.pvalues['dC'], r_dA_dC=np.corrcoef(d.dA, d.dC)[0, 1]))
        r = t2[-1]
        say(f'  {task:8s} {enc_}: raw gap {r["raw_gap"]:+.4f} (dz {r["raw_dz"]:.2f}, p {r["raw_p"]:.4f}); '
            f'caption gap {r["mean_dC"]:+.4f}; adjusted gap {r["adj_gap"]:+.4f} +- {r["adj_se"]:.4f} '
            f'(p {r["adj_p"]:.2g}, {100 * r["adj_gap"] / r["raw_gap"]:.0f}% of raw); '
            f'slope {r["slope"]:+.3f} (p {r["slope_p"]:.3f}); r(dA, dC) = {r["r_dA_dC"]:+.3f}')
pd.DataFrame(t2).to_csv(os.path.join(RES, 'caption_test2.csv'), index=False)

say('\nTEST 2b  pooled: alignment ~ decoder + encoder + caption similarity + image fixed effects (per image x model)')
for task in ('describe', 'social'):
    long = aim.loc[task].stack().rename('z').to_frame().join(pim.loc[task][TRAINED].stack().rename('cap')).reset_index()
    long['transformer'] = long.model.str.contains('Transformer').astype(int)
    long['vit'] = long.model.str.startswith('ViT').astype(int)
    for f in ('z ~ transformer + vit + C(image_id)', 'z ~ transformer + vit + cap + C(image_id)'):
        fit = smf.ols(f, long).fit(cov_type='cluster', cov_kwds={'groups': long.image_id})
        extra = f", caption b={fit.params['cap']:+.3f} (p {fit.pvalues['cap']:.3f})" if 'cap' in f else ''
        say(f'  {task:8s} {"with caption" if "cap" in f else "without    "}: decoder b={fit.params["transformer"]:+.4f} '
            f'(p {fit.pvalues["transformer"]:.2g}), encoder b={fit.params["vit"]:+.4f} (p {fit.pvalues["vit"]:.2g}){extra}')
open(os.path.join(RES, 'caption_tests.txt'), 'w').write('\n'.join(log) + '\n')
