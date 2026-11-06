"""
final_fig5_entropy.py
---------------------
Camera-ready Figure 5 (attention examples) and the §4.1.2 entropy table, from
archive/ICLR/scripts/regen_fig5_entropy_cvheads.py with two changes:

  * Qwen3.5: thinking-off activations and the thinking-off CV heads
    (results/selected_heads.json from rerun_behavioral.py).
  * Molmo: the 8 all-zero crop-overlap rows/cols of the 36x36 grid are dropped BEFORE
    anything else, leaving the 28x28 token grid, which is bilinearly upsampled with no
    smoothing. The archived loader Gaussian-blurred the 36x36 grid first, smearing the
    zero margins into the neighbouring cells (seams in the figure; inflated entropy).
    No model is smoothed now, matching the Appendix B post-processing text.

Everything else (trained-model loaders, human maps, entropy definition, layout) is the
archived code, executed as-is.

Outputs: neurips_final/final_figures/qwen35_nothink/{fig5_cvheads_attn_by_task.png, results/entropy_cvheads.csv}
"""
import os, json

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC  = os.path.join(ROOT, 'archive', 'ICLR', 'scripts', 'regen_fig5_entropy_cvheads.py')
# MOLMO_BILINEAR=1: heads from the Molmo-bilinear rerun, outputs to its directory
OUT  = os.path.join(ROOT, 'neurips_final', 'final_figures',
                    os.environ.get('MOLMO_BILINEAR_DIR', 'molmo_bilinear') if os.environ.get('MOLMO_BILINEAR') == '1' else 'qwen35_nothink')
SEL  = json.load(open(os.path.join(OUT, 'results', 'selected_heads.json')))

src = open(SRC).read()
cut = src.index('# ── 1. entropy')
g = {'__file__': SRC, '__name__': 'fig5_final'}
exec(compile(src[:cut], SRC, 'exec'), g)

g['OUT_CSV'] = os.path.join(OUT, 'results', 'entropy_cvheads.csv')
g['OUT_FIG'] = os.path.join(OUT, 'fig5_cvheads_attn_by_task.png')
for _m in ('Qwen3.5', 'Molmo'):
    g['HEADS'][_m] = {t: (SEL[f'{_m}_{t}']['layer'], SEL[f'{_m}_{t}']['head']) for t in ('social', 'describe')}
P = g['PROBING_DIR']
g['QCFG'] = {'social':   (os.path.join(P, 'output', 'social_cap', 'qwen35_nothink'),),
             'describe': (os.path.join(P, 'output', 'coco', 'qwen35_nothink'),)}

np, torch, cv2 = g['np'], g['torch'], g['cv2']

def molmo_maps(task, only=None):
    L, H = g['HEADS']['Molmo'][task]
    act_dir = os.path.join(g['MOLMO_DIR'], 'activations', task)
    grid36 = g['_molmo_grid'](act_dir)[0]
    vidx = np.where(grid36 > 0); pos = grid36[vidx]
    rows, cols = np.any(grid36 > 0, axis=1), np.any(grid36 > 0, axis=0)
    ids = only if only else [f[len('activations_img_'):-3] for f in os.listdir(act_dir)
                             if f.startswith('activations_img_') and f.endswith('.pt')]
    out = {}
    for im in ids:
        p = os.path.join(act_dir, f'activations_img_{im}.pt')
        if not os.path.exists(p):
            continue
        a = torch.load(p, map_location='cpu')[L - 1, H - 1].float().numpy()   # (ntok, seq)
        vol = np.zeros((a.shape[0],) + grid36.shape, np.float32)
        vol[:, vidx[0], vidx[1]] = a[:, pos]
        grid = vol.sum(0)[rows][:, cols]                                       # 28x28
        out[im] = cv2.resize(grid, (g['SIZE'], g['SIZE']), interpolation=cv2.INTER_LINEAR)
    return out
g['molmo_maps'] = molmo_maps

exec(compile(src[cut:], SRC, 'exec'), g)
