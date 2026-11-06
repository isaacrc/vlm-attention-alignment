"""
fig1_between_besthead.py
------------------------
Figure 1C rule: each model with heads is compared across tasks on ONE head, the one of its
two cross-validated task heads with the higher alignment as % of that task's noise
ceiling (ceilings differ between tasks, so raw r would favour the Social head). Panels A-B
and every other analysis keep the per-task heads. Reads the one-head runs of
fig1_onehead.py.

Output: final_figures/qwen35_nothink/results/model_between_task_fig1.csv (read by
scripts/fig3_slides.py with QWEN_NOTHINK=1)
"""
import os
import pandas as pd

NT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/qwen35_nothink'
OH, RES = os.path.join(NT, 'onehead'), os.path.join(NT, 'results')
summ = pd.read_csv(os.path.join(RES, 'alignment_summary_cv_honest.csv')).set_index(['model', 'task'])
bt = {a: pd.read_csv(os.path.join(OH, f'model_between_task_{a}head.csv')).set_index('model')
      for a in ('social', 'describe')}
rows = []
for m in bt['social'].index:
    anchor = max(('social', 'describe'), key=lambda t: summ.loc[(m, t), 'pct_ceiling'])
    r = bt[anchor].loc[m]
    rows.append(dict(model=m, between_r=r.between_r, sem=r['sem'], n_img=r.n_img,
                     task_diff_index=r.task_diff_index, head=r['head'],
                     head_task=anchor if pd.notna(r['head']) else ''))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, 'model_between_task_fig1.csv'), index=False)
print(out.round(4).to_string(index=False))
