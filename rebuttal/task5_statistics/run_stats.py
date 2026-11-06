"""
Task 5 — Inferential statistics on the architecture comparisons.
Primary (robust, assumption-light): per-image paired permutation tests for every
headline contrast. Secondary: crossed random-effects mixed model
z ~ encoder*decoder*task + (1|subject) + (1|image) via statsmodels.

Consumes a long table [model,encoder,decoder,task,subject_id,image_id,z].
Default = data/alignment_long_best_head.parquet; pass --table for CV-honest re-run.

Outputs (task5_statistics/results/): pairwise_contrasts.csv, lmm_summary.txt,
  effect_sizes.json
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__); RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
DATA = os.path.join(HERE, '..', 'data')
RNG = np.random.default_rng(0)

ap = argparse.ArgumentParser()
ap.add_argument('--table', default=os.path.join(DATA, 'alignment_long_best_head.parquet'))
ap.add_argument('--tag', default='best_head')
args = ap.parse_args()

long = pd.read_parquet(args.table)

# ── per-image subject-mean z (pairing unit = image within task) ──
pim = long.groupby(['model', 'task', 'image_id'])['z'].mean().reset_index()


def paired_perm(a_model, b_model, task, nperm=10000):
    """Paired permutation (sign-flip) on per-image differences a-b within task."""
    a = pim[(pim.model == a_model) & (pim.task == task)].set_index('image_id')['z']
    b = pim[(pim.model == b_model) & (pim.task == task)].set_index('image_id')['z']
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).values
    obs = d.mean()
    signs = RNG.choice([-1, 1], size=(nperm, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)
    # Cohen's dz
    dz = obs / (d.std(ddof=1) + 1e-12)
    return dict(a=a_model, b=b_model, task=task, mean_diff=float(obs),
                dz=float(dz), n_img=len(d), p_perm=float(p))


CONTRASTS = [
    ('CNN-LSTM', 'CNN-Transformer'), ('ViT-LSTM', 'ViT-Transformer'),
    ('CNN-LSTM', 'ViT-LSTM'), ('CNN-Transformer', 'ViT-Transformer'),
    ('CNN-LSTM', 'Molmo'), ('CNN-Transformer', 'Molmo'),
    ('ViT-Transformer', 'Qwen3.5'), ('Molmo', 'Qwen3.5'),
]

rows = []
for task in ['social', 'describe']:
    for a, b in CONTRASTS:
        rows.append(paired_perm(a, b, task))
cc = pd.DataFrame(rows)
# Holm correction within task
cc['p_holm'] = np.nan
for task in ['social', 'describe']:
    idx = cc.index[cc.task == task]
    ps = cc.loc[idx, 'p_perm'].values
    order = np.argsort(ps); m = len(ps); adj = np.empty(m)
    run = 0
    for rank, oi in enumerate(order):
        run = max(run, (m - rank) * ps[oi]); adj[oi] = min(run, 1.0)
    cc.loc[idx, 'p_holm'] = adj
cc.to_csv(os.path.join(RES, f'pairwise_contrasts_{args.tag}.csv'), index=False)
print(cc.to_string(index=False))

# ── mixed model (secondary) ──
lmm_txt = os.path.join(RES, f'lmm_summary_{args.tag}.txt')
try:
    import statsmodels.formula.api as smf
    # Restrict to the four trained models: the 2x2 encoder x decoder design.
    # Molmo/Qwen carry encoder=decoder='VLM', which leaves the interaction
    # rank-deficient (no VLM-encoder/LSTM-decoder cell) -> singular matrix.
    d = long[long.encoder.isin(['CNN', 'ViT'])].copy()
    # Subjects are nested in task (no subject does both), so subject is the
    # grouping factor and image is a crossed random effect within it.
    d['image_id'] = d['image_id'].astype(str)
    md = smf.mixedlm('z ~ C(encoder)*C(decoder)*C(task)', d,
                     groups=d['subject_id'],
                     vc_formula={'image': '0+C(image_id)'})
    mf = md.fit(method='lbfgs', maxiter=500)
    with open(lmm_txt, 'w') as f:
        f.write(str(mf.summary()))
        f.write('\n\nNote: fit on the four trained models only (2x2 design).\n'
                'Subjects nested in task; image as crossed random effect.\n')
    print('LMM converged; summary written.')
    print(mf.summary())
except Exception as e:
    # Fallback: subject-level random intercept only (no image RE).
    try:
        d2 = long[long.encoder.isin(['CNN', 'ViT'])].groupby(
            ['model', 'encoder', 'decoder', 'task', 'subject_id'])['z'].mean().reset_index()
        mf2 = smf.mixedlm('z ~ C(encoder)*C(decoder)*C(task)', d2,
                          groups=d2['subject_id']).fit(method='lbfgs', maxiter=500)
        with open(lmm_txt, 'w') as f:
            f.write(str(mf2.summary()))
            f.write(f'\n\nNote: crossed image RE omitted ({e}); '
                    'fit on subject-mean z, random intercept per subject.\n')
        print(f'LMM (crossed) failed: {e}\nFell back to subject-mean LMM; written.')
        print(mf2.summary())
    except Exception as e2:
        with open(lmm_txt, 'w') as f:
            f.write(f'LMM failed/omitted: {e} | fallback: {e2}\n'
                    'Primary inference = permutation tests above.')
        print(f'LMM skipped entirely: {e} | {e2}')

# ── effect sizes: decoder vs encoder (subject-mean z, both tasks pooled) ──
sm = long.groupby(['model', 'encoder', 'decoder', 'task', 'subject_id'])['z'].mean().reset_index()
trained = sm[sm.encoder.isin(['CNN', 'ViT'])]
def eta2(factor):
    grand = trained['z'].mean()
    ss_tot = ((trained['z'] - grand) ** 2).sum()
    ss_fac = 0.0
    for lvl, g in trained.groupby(factor):
        ss_fac += len(g) * (g['z'].mean() - grand) ** 2
    return float(ss_fac / ss_tot)
es = dict(eta2_decoder=eta2('decoder'), eta2_encoder=eta2('encoder'))
es['decoder_to_encoder_ratio'] = es['eta2_decoder'] / (es['eta2_encoder'] + 1e-12)
json.dump(es, open(os.path.join(RES, f'effect_sizes_{args.tag}.json'), 'w'), indent=2)
print('Effect sizes:', es)
