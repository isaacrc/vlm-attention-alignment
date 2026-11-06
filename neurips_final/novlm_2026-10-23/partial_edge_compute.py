"""
partial_edge_compute.py -- alignment with edge energy partialled out, for slides_partial_edge.py.
Same code path as partial_face_compute.py, with the control map swapped for the edge-energy map of the
Appendix B figure (Sobel magnitude, averaged into the 14x14 grid, score_baselines.cell_maps). All 200 images.
The face prior is flat on the 40 images without an annotated face; partial_rows drops those, so its raw and
partial scores cover 160 images.
  raw       Spearman r(map, participant's fixation map)                       (run_center_controlled.partial_rows)
  partial   Spearman r(map, participant's fixation map | edge energy)
  upperbound  mean pairwise participant-participant z under each metric on the same 160 images
Maps: trained models and VLMs at the cross-validated heads of molmo_bilinear_pairwise (Figure 1), loaded as
in molmo_bilinear/rerun_molmo_bilinear.py (bilinear Molmo, thinking-off Qwen3.5); baselines unblurred as in
the Appendix B figure (score_baselines.py). A sanity check first reproduces the center-controlled CNN-LSTM
value of headrule_center_summary.csv with the same code path.
Writes novlm_2026-10-23/results/partial_edge_200.csv (per participant: raw z, partial z) and
partial_edge_200_ceilings.json
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.join(ROOT, 'rebuttal')
for sub in ('shared', 'task1_head_selection', 'task3_baselines', 'task4_metric_suite'):
    sys.path.insert(0, os.path.join(REB, sub))
sys.path.insert(0, os.path.join(ROOT, 'neurips_final', 'molmo_bilinear'))
import alignment_core as A, vlm_maps as V, extract_molmo_alllayers as EM, molmo_bilinear_maps as MB
import tierB_common as T, run_center_controlled as CC, run_headrule_center as HC

PROBE = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output'
V._QCFG = {t: dict(act=os.path.join(PROBE, d, 'qwen35_nothink', 'activations'),
                   mask=os.path.join(PROBE, d, 'qwen35_nothink', 'image_mask.pt'))
           for t, d in (('social', 'social_cap'), ('describe', 'coco'))}
MB.patch(V, EM)
Q = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results')
HC.SELECTED = json.load(open(os.path.join(Q, 'selected_heads.json')))
OUT = os.path.join(HERE, 'results', 'partial_edge_200.csv')
TASKS = ('social', 'describe')

imgs_all = A.target_images()
center_rk_all = CC.loio_prior_ranked(imgs_all)

# sanity check: the shared code path reproduces the published center-controlled CNN-LSTM value
chk = CC.partial_rows(HC.get_maps('CNN-LSTM', 'na', 'social', imgs_all, None), 'social', imgs_all, center_rk_all, 'CNN-LSTM')
zs = chk.groupby('subject_id').z.mean()
pub = pd.read_csv(os.path.join(Q, 'headrule_center_summary.csv')).query("model == 'CNN-LSTM' and task == 'social'").r_partial.iloc[0]
print(f'check: CNN-LSTM social center-partial r = {np.tanh(zs.mean()):.4f} (published {pub:.4f})', flush=True)
assert abs(np.tanh(zs.mean()) - pub) < 1e-3

bs = open(os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'fig1_baselines', 'score_baselines.py')).read()
bns = {'__name__': 'baselines'}
exec(compile(bs[:bs.index('def main():')], 'score_baselines.py', 'exec'), bns)   # cell_maps
edge, lum = bns['cell_maps'](imgs_all)
edge_rk = A._rank(np.stack([np.asarray(edge[im], np.float32).reshape(-1) for im in imgs_all]), axis=1)
fixed = {'Itti-Koch': T.load_maps('itti_koch'), 'GBVS': T.load_maps('gbvs'),
         'Spectral residual': T.load_maps('spectral_residual'), 'Luminance': lum,
         'Center prior': {im: center_rk_all[k].reshape(A.OUT_SIZE, A.OUT_SIZE) for k, im in enumerate(imgs_all)},
         'Person prior': T.load_maps('coco_person'), 'Face prior': T.load_maps('coco_face')}
MODELS = [('CNN-LSTM', 'na'), ('CNN-Transformer', 'cv_best'), ('ViT-LSTM', 'na'), ('ViT-Transformer', 'cv_best'),
          ('Molmo', 'cv_best'), ('Qwen3.5', 'cv_best')]


def raw_ceiling(task, imgs):
    """A.ceiling on an image subset (A.ceiling caches the all-image value per task)."""
    ranked, valid, _ = A.load_human(task, imgs)
    per = []
    for j in range(len(imgs)):
        vi = np.where(valid[:, j])[0]
        if len(vi) < 2:
            continue
        Vc = ranked[vi, j] - ranked[vi, j].mean(axis=1, keepdims=True)
        Vn = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-10)
        per.append(np.arctanh(np.clip((Vn @ Vn.T)[np.triu_indices(len(vi), 1)], -0.999999, 0.999999)).mean())
    return float(np.mean(per))


rows, ceil = [], {}
for task in TASKS:
    ceil[task] = dict(raw_z=raw_ceiling(task, imgs_all), partial_z=CC.partial_ceiling(task, imgs_all, edge_rk))
    print(f'[{task}] upperbound on {len(imgs_all)} images: raw r = {np.tanh(ceil[task]["raw_z"]):.4f}, '
          f'edge-partial r = {np.tanh(ceil[task]["partial_z"]):.4f}', flush=True)
    preds = [(m, HC.get_maps(m, rule, task, imgs_all, None)) for m, rule in MODELS] + list(fixed.items())
    for name, maps in preds:
        df = CC.partial_rows(maps, task, imgs_all, edge_rk, name)
        df['z_raw'] = np.arctanh(np.clip(df.r_raw, -0.999999, 0.999999))
        g = df.groupby('subject_id')[['z_raw', 'z']].mean().reset_index().rename(columns={'z': 'z_partial'})
        g['model'], g['task'], g['n_images'] = name, task, df.image_id.nunique()
        rows.append(g)
        print(f'  {task:8s} {name:18s} raw {100 * g.z_raw.mean() / ceil[task]["raw_z"]:6.1f}%  '
              f'edge-partial {100 * g.z_partial.mean() / ceil[task]["partial_z"]:6.1f}%  ({df.image_id.nunique()} images)', flush=True)
pd.concat(rows).to_csv(OUT, index=False)
json.dump(ceil, open(OUT.replace('.csv', '_ceilings.json'), 'w'), indent=2)
print('wrote', OUT)
