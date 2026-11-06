"""
rerun_molmo_bilinear.py
-----------------------
Re-run every behavioral analysis that scores Molmo, with Molmo's 28x28 token grid
bilinearly upsampled to 256x256 like the other five models (molmo_bilinear_maps.py).
Starts from the thinking-off Qwen results (neurips_final/final_figures/qwen35_nothink/
results). Differences from those: Molmo's interpolation; Molmo's head sweep restricted to
the canonical participants (the rebuttal sweep included the 3 excluded Describe
participants); and the noise ceiling, which is now leave-one-subject-out (each participant
against the rank-mean of the others; loo_ceiling.py) instead of mean pairwise, for both
the raw and the center-controlled metric.

  1. Molmo 784-head sweep (task1_head_selection/extract_molmo_alllayers.py, bilinear)
     -> replaces the Molmo rows of head_table_perimage.parquet
  2. CV per-task head selection + alignment tables (build_cvhonest_table.main);
     every non-Molmo row must reproduce the qwen35_nothink table exactly (checked)
  3. Paired permutation tests / LMM (task5_statistics/run_stats.py)
  4. Between-task: per-task CV heads, the one-head runs, and the Figure 1C rule
     (higher % of ceiling), Molmo recomputed, other rows copied
  5. Appendix: winner's curse, center-controlled alignment, metric suite, raw-metric
     contrasts against the saliency / semantic baselines (score_tierB_raw.py)
  6. Old (area) vs new (bilinear) comparison for Molmo

Output: neurips_final/final_figures/molmo_bilinear/{results,onehead,appendix}/
"""
import os, sys, json, shutil, subprocess, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
REB  = os.path.join(ROOT, 'rebuttal')
NT   = os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink')
OUT  = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear')
RES, OH, FIG = (os.path.join(OUT, d) for d in ('results', 'onehead', 'appendix'))
for d in (RES, OH, FIG):
    os.makedirs(d, exist_ok=True)
for sub in ('shared', 'task1_head_selection', 'task3_baselines', 'task4_metric_suite'):
    sys.path.insert(0, os.path.join(REB, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alignment_core as A
import vlm_maps as V
import extract_molmo_alllayers as EM
import molmo_bilinear_maps as MB
import loo_ceilings as LC

PROBE = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output'
V._QCFG = {t: dict(act=os.path.join(PROBE, d, 'qwen35_nothink', 'activations'),
                   mask=os.path.join(PROBE, d, 'qwen35_nothink', 'image_mask.pt'))
           for t, d in (('social', 'social_cap'), ('describe', 'coco'))}
MB.patch(V, EM)
LC.patch_sweep_exclusions(EM, A)
A._CEIL_CACHE.update(LC.LOO_Z)          # raw ceiling (A.ceiling) -> leave-one-out
TASKS = ['social', 'describe']
imgs = A.target_images()

# ── 1. Molmo sweep ────────────────────────────────────────────────────────────
EM.OUT_DIR = RES
EM.main()
old_pim = pd.read_parquet(os.path.join(NT, 'results', 'head_table_perimage.parquet'))
mol = []
for t in TASKS:
    md = pd.read_parquet(os.path.join(RES, f'molmo_allL_perimage_{t}.parquet'))[['task', 'layer', 'head', 'image_id', 'mean_z']].copy()
    md['model'] = 'Molmo'
    mol.append(md)
mol = pd.concat(mol, ignore_index=True)
new_pim = pd.concat([old_pim[old_pim.model != 'Molmo'], mol[old_pim.columns]], ignore_index=True)
new_pim.to_parquet(os.path.join(RES, 'head_table_perimage.parquet'))
print('head table: Molmo rows replaced', flush=True)

# ── 2. CV head selection + alignment tables ───────────────────────────────────
import build_cvhonest_table as B
json.dump(LC.LOO_Z, open(os.path.join(RES, 'ceilings.json'), 'w'), indent=2)
B.RES, B.DATA = RES, RES
B.main()
old_sum = pd.read_csv(os.path.join(NT, 'results', 'alignment_summary_cv_honest.csv'))
new_sum = pd.read_csv(os.path.join(RES, 'alignment_summary_cv_honest.csv'))
chk = old_sum.merge(new_sum, on=['model', 'task'], suffixes=('_old', '_new'))
other = chk[chk.model != 'Molmo']
assert np.allclose(other.mean_r_old, other.mean_r_new, atol=1e-9), \
    'non-Molmo rows changed:\n' + other.to_string()
print('Check passed: every non-Molmo alignment value matches qwen35_nothink.', flush=True)
sel = json.load(open(os.path.join(RES, 'selected_heads.json')))
V.MOLMO_BEST = {t: (sel[f'Molmo_{t}']['layer'], sel[f'Molmo_{t}']['head']) for t in TASKS}

# ── 3. Statistics ─────────────────────────────────────────────────────────────
tag = 'cv_honest_molmo_bilinear'
subprocess.run([sys.executable, 'run_stats.py', '--table', os.path.join(RES, 'alignment_long_cv_honest.parquet'),
                '--tag', tag], cwd=os.path.join(REB, 'task5_statistics'), check=True)
for f in (f'pairwise_contrasts_{tag}.csv', f'lmm_summary_{tag}.txt', f'effect_sizes_{tag}.json'):
    shutil.move(os.path.join(REB, 'task5_statistics', 'results', f), os.path.join(RES, f))

# ── 4. Between-task ───────────────────────────────────────────────────────────
def between(ms, md):
    rs = np.array([spearmanr(ms[im].ravel(), md[im].ravel())[0] for im in imgs
                   if im in ms and im in md and ms[im].std() > 1e-12 and md[im].std() > 1e-12])
    return rs.mean(), rs.std(ddof=1) / np.sqrt(len(rs)), len(rs)

def put(df, head, r):
    i = df.index[df.model == 'Molmo'][0]
    df.loc[i, ['between_r', 'sem', 'n_img']] = r
    df.loc[i, 'task_diff_index'] = 1 - r[0]
    if head is not None and 'head' in df.columns:
        df.loc[i, 'head'] = head
    return df

hs = {t: (sel[f'Molmo_{t}']['layer'], sel[f'Molmo_{t}']['head']) for t in TASKS}
cv = pd.read_csv(os.path.join(NT, 'results', 'model_between_task_cvheads.csv'))
put(cv, None, between(V.molmo_maps('social', *hs['social']), V.molmo_maps('describe', *hs['describe']))
    ).to_csv(os.path.join(RES, 'model_between_task_cvheads.csv'), index=False)
for anchor in TASKS:
    L, H = hs[anchor]
    bt = pd.read_csv(os.path.join(NT, 'onehead', f'model_between_task_{anchor}head.csv'))
    put(bt, f'L{L}H{H}', between(V.molmo_maps('social', L, H), V.molmo_maps('describe', L, H))
        ).to_csv(os.path.join(OH, f'model_between_task_{anchor}head.csv'), index=False)
summ = new_sum.set_index(['model', 'task'])
bt = {a: pd.read_csv(os.path.join(OH, f'model_between_task_{a}head.csv')).set_index('model') for a in TASKS}
rows = []
for m in bt['social'].index:
    anchor = max(TASKS, key=lambda t: summ.loc[(m, t), 'pct_ceiling'])
    r = bt[anchor].loc[m]
    rows.append(dict(model=m, between_r=r.between_r, sem=r['sem'], n_img=r.n_img,
                     task_diff_index=r.task_diff_index, head=r['head'],
                     head_task=anchor if pd.notna(r['head']) else None))
pd.DataFrame(rows).to_csv(os.path.join(RES, 'model_between_task_fig1.csv'), index=False)
for f in ('model_between_task.csv', 'human_group_between_task.json'):
    shutil.copy(os.path.join(NT, 'results', f), RES)
print(pd.DataFrame(rows).to_string(index=False), flush=True)

# ── 5. Appendix analyses ──────────────────────────────────────────────────────
import analyze_head_selection as AH
AH.RES, AH.FIG = RES, FIG
AH.main()
import run_headrule_center as HC
HC.SELECTED, HC.RES = sel, RES
HC.partial_ceiling = LC.loo_partial_ceiling
HC.main()
import score_tierB_raw as TB
shutil.copy(os.path.join(REB, 'task3_baselines', 'results', 'tierB_sigma_sweep.csv'), RES)
TB.RES = RES
TB.main()
import run_metric_suite_cv as MS
MS.SELECTED, MS.RES = sel, RES
MS.main()

# ── 6. Molmo: area (qwen35_nothink) vs bilinear ───────────────────────────────
old_sel = json.load(open(os.path.join(NT, 'results', 'selected_heads.json')))
cmp_rows = []
for t in TASKS:
    o = old_sum[(old_sum.model == 'Molmo') & (old_sum.task == t)].iloc[0]
    n = new_sum[(new_sum.model == 'Molmo') & (new_sum.task == t)].iloc[0]
    os_, ns_ = old_sel[f'Molmo_{t}'], sel[f'Molmo_{t}']
    cmp_rows.append(dict(metric=f'{t} alignment', old_head=f"L{os_['layer']}H{os_['head']} ({os_['fold_agreement']})",
                         new_head=f"L{ns_['layer']}H{ns_['head']} ({ns_['fold_agreement']})",
                         old_r=o.mean_r, new_r=n.mean_r, old_pct=o.pct_ceiling, new_pct=n.pct_ceiling))
f1o = pd.read_csv(os.path.join(NT, 'results', 'model_between_task_fig1.csv')).set_index('model').loc['Molmo']
f1n = pd.DataFrame(rows).set_index('model').loc['Molmo']
cmp_rows.append(dict(metric='between-task r (Fig 1C)', old_head=f1o['head'], new_head=f1n['head'],
                     old_r=f1o.between_r, new_r=f1n.between_r))
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(os.path.join(RES, 'molmo_area_vs_bilinear.csv'), index=False)
print('\n=== Molmo: area (current) vs bilinear ===')
print(cmp.round(4).to_string(index=False))
print('DONE rerun_molmo_bilinear', flush=True)
