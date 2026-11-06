"""
rerun_molmo_bilinear_pairwise.py
--------------------------------
rerun_molmo_bilinear.py with the original mean-pairwise noise ceiling (raw and
center-controlled) instead of the leave-one-out ceiling. Molmo's bilinear 784-head sweep
does not depend on the ceiling, so it is copied from final_figures/molmo_bilinear/results
rather than recomputed; everything downstream (CV heads, statistics, between-task,
center-controlled, metric suite) is rerun.

Output: neurips_final/final_figures/molmo_bilinear_pairwise/{results,onehead,appendix}/
"""
import os, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
FF = os.path.join(HERE, '..', 'final_figures')
SRC = os.path.join(HERE, 'rerun_molmo_bilinear.py')
src = open(SRC).read()
reps = [
    ("OUT  = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear')",
     "OUT  = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise')"),
    ("A._CEIL_CACHE.update(LC.LOO_Z)          # raw ceiling (A.ceiling) -> leave-one-out\n", ""),
    ("EM.OUT_DIR = RES\nEM.main()\n",
     "for t in TASKS:\n    for f in (f'molmo_allL_perimage_{t}.parquet', f'molmo_allL_summary_{t}.csv'):\n"
     "        shutil.copy(os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear', 'results', f), RES)\n"),
    ("json.dump(LC.LOO_Z, open(os.path.join(RES, 'ceilings.json'), 'w'), indent=2)",
     "shutil.copy(os.path.join(NT, 'results', 'ceilings.json'), RES)"),
    ("HC.partial_ceiling = LC.loo_partial_ceiling\n", ""),
]
for a, b in reps:
    assert a in src, a
    src = src.replace(a, b)
os.makedirs(os.path.join(FF, 'molmo_bilinear_pairwise', 'blur_sweep'), exist_ok=True)
# blur sweep per-image scores are ceiling-free; only the percentages are recomputed
shutil.copy(os.path.join(FF, 'molmo_bilinear', 'blur_sweep', 'blur_sweep_perimage.parquet'),
            os.path.join(FF, 'molmo_bilinear_pairwise', 'blur_sweep'))
exec(compile(src, SRC, 'exec'), {'__file__': SRC, '__name__': '__main__'})
