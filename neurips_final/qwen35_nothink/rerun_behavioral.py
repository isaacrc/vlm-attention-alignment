"""
rerun_behavioral.py
-------------------
Recompute the behavioral (within-task alignment + between-task) results with Qwen3.5
attention re-extracted with thinking OFF (infer_qwen35_nothink.py), reusing the
rebuttal pipeline unchanged and writing everything to a separate directory so the
paper's results are not touched.

Steps (mirror rebuttal/task1_head_selection + task2_between_task + task5_statistics):
  1. Qwen head sweep (8 full-attention layers x 16 heads) on the new activations ->
     replaces only the Qwen rows of rebuttal head_table_perimage.parquet.
  2. build_cvhonest_table.main() on that table: CV per-task head selection and the
     long/summary alignment tables. Same RNG order as the original, so every
     non-Qwen model must reproduce the rebuttal numbers exactly (checked below).
  3. Between-task: Qwen recomputed (argmax head per task, as run_between_task.py
     does for VLMs); the other models' rows are copied from the rebuttal CSV.
  4. Paired permutation tests / LMM via task5_statistics/run_stats.py.

Output: neurips_final/final_figures/qwen35_nothink/results/
"""
import os, sys, json, shutil, subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
REB  = os.path.join(ROOT, 'rebuttal')
OUT  = os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink')
RES  = os.path.join(OUT, 'results'); os.makedirs(RES, exist_ok=True)
for sub in ('shared', 'task1_head_selection'):
    sys.path.insert(0, os.path.join(REB, sub))
import alignment_core as A
import vlm_maps as V

PROBE = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output'
V._QCFG = {t: dict(act=os.path.join(PROBE, d, 'qwen35_nothink', 'activations'),
                   mask=os.path.join(PROBE, d, 'qwen35_nothink', 'image_mask.pt'))
           for t, d in (('social', 'social_cap'), ('describe', 'coco'))}
QWEN_LAYERS, N_QHEAD = [3, 7, 11, 15, 19, 23, 27, 31], 16
TASKS = ['social', 'describe']

# ── 1. Qwen head sweep ────────────────────────────────────────────────────────
imgs = A.target_images()
rows = []
for task in TASKS:
    for L in QWEN_LAYERS:
        for H in range(1, N_QHEAD + 1):
            df = A.per_image_z(V.qwen_maps(task, L, H), task, imgs, 'Qwen3.5')
            pim = df.groupby('image_id')['z'].mean().reset_index().rename(columns={'z': 'mean_z'})
            pim['model'], pim['task'], pim['layer'], pim['head'] = 'Qwen3.5', task, L, H
            rows.append(pim)
        print(f'  sweep {task} layer {L}: done', flush=True)
qwen_pim = pd.concat(rows, ignore_index=True)
old_pim = pd.read_parquet(os.path.join(REB, 'task1_head_selection', 'results', 'head_table_perimage.parquet'))
new_pim = pd.concat([old_pim[old_pim.model != 'Qwen3.5'], qwen_pim[old_pim.columns]], ignore_index=True)
new_pim.to_parquet(os.path.join(RES, 'head_table_perimage.parquet'))

sweep = (qwen_pim.groupby(['task', 'layer', 'head'])['mean_z'].mean().reset_index())
sweep['mean_r'] = np.tanh(sweep['mean_z'])
sweep.to_csv(os.path.join(RES, 'qwen35_head_sweep.csv'), index=False)
QBEST = {t: tuple(int(v) for v in sweep[sweep.task == t].sort_values('mean_z').iloc[-1][['layer', 'head']])
         for t in TASKS}
V.qwen_best = lambda task: QBEST[task]
print('Qwen argmax heads (thinking off):', QBEST, flush=True)

# ── 2. CV-honest head selection + alignment tables ────────────────────────────
import build_cvhonest_table as B
shutil.copy(os.path.join(REB, 'data', 'ceilings.json'), RES)
B.RES, B.DATA = RES, RES
B.main()

old_sum = pd.read_csv(os.path.join(REB, 'data', 'alignment_summary_cv_honest.csv'))
new_sum = pd.read_csv(os.path.join(RES, 'alignment_summary_cv_honest.csv'))
chk = old_sum.merge(new_sum, on=['model', 'task'], suffixes=('_old', '_new'))
other = chk[chk.model != 'Qwen3.5']
assert np.allclose(other.mean_r_old, other.mean_r_new, atol=1e-9), \
    'non-Qwen rows changed: pipeline did not reproduce the rebuttal\n' + other.to_string()
print('Check passed: all non-Qwen alignment values identical to the rebuttal table.', flush=True)

# ── 3. Between-task ───────────────────────────────────────────────────────────
soc, des = V.vlm_maps('Qwen3.5', 'social'), V.vlm_maps('Qwen3.5', 'describe')
rs = np.array([spearmanr(soc[im].flatten(), des[im].flatten())[0] for im in imgs
               if im in soc and im in des and soc[im].std() > 1e-12 and des[im].std() > 1e-12])
bt = pd.read_csv(os.path.join(REB, 'task2_between_task', 'results', 'model_between_task.csv'))
qi = bt.index[bt.model == 'Qwen3.5'][0]
bt.loc[qi, ['between_r', 'sem', 'n_img']] = [rs.mean(), rs.std(ddof=1) / np.sqrt(len(rs)), len(rs)]
bt.loc[qi, 'task_diff_index'] = 1 - rs.mean()
bt.to_csv(os.path.join(RES, 'model_between_task.csv'), index=False)
shutil.copy(os.path.join(REB, 'task2_between_task', 'results', 'human_group_between_task.json'), RES)

# ── 4. Statistics ─────────────────────────────────────────────────────────────
tag = 'cv_honest_qwen_nothink'
subprocess.run([sys.executable, 'run_stats.py', '--table', os.path.join(RES, 'alignment_long_cv_honest.parquet'),
                '--tag', tag], cwd=os.path.join(REB, 'task5_statistics'), check=True)
for f in (f'pairwise_contrasts_{tag}.csv', f'lmm_summary_{tag}.txt', f'effect_sizes_{tag}.json'):
    shutil.move(os.path.join(REB, 'task5_statistics', 'results', f), os.path.join(RES, f))

# ── Old vs new summary for Qwen ───────────────────────────────────────────────
old_sel = json.load(open(os.path.join(REB, 'task1_head_selection', 'results', 'selected_heads.json')))
new_sel = json.load(open(os.path.join(RES, 'selected_heads.json')))
old_bt = pd.read_csv(os.path.join(REB, 'task2_between_task', 'results', 'model_between_task.csv'))
cmp_rows = []
for t in TASKS:
    o, n = old_sum[(old_sum.model == 'Qwen3.5') & (old_sum.task == t)].iloc[0], \
           new_sum[(new_sum.model == 'Qwen3.5') & (new_sum.task == t)].iloc[0]
    os_, ns_ = old_sel[f'Qwen3.5_{t}'], new_sel[f'Qwen3.5_{t}']
    cmp_rows.append(dict(metric=f'{t} alignment', old_head=f"L{os_['layer']}H{os_['head']}",
                         new_head=f"L{ns_['layer']}H{ns_['head']}", new_fold_agreement=ns_['fold_agreement'],
                         old_r=o.mean_r, new_r=n.mean_r, old_pct=o.pct_ceiling, new_pct=n.pct_ceiling))
cmp_rows.append(dict(metric='between-task r', old_r=float(old_bt[old_bt.model == 'Qwen3.5'].between_r.iloc[0]),
                     new_r=float(rs.mean())))
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(os.path.join(RES, 'qwen35_old_vs_new.csv'), index=False)
print('\n=== Qwen3.5: thinking on (paper) vs thinking off ===')
print(cmp.round(4).to_string(index=False))
