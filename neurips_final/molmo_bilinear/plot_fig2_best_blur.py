"""
plot_fig2_best_blur.py
----------------------
Figure 2 (attention maps for three images x two tasks) with every model's map blurred at
that model's best Gaussian sigma for the task, the sigma chosen by five-fold
cross-validation in the blur sweep (blur_sweep_summary.csv; the same sigmas as
plot_fig1_best_blur.py). Human maps are untouched. The maps, heads and layout are those of
qwen35_nothink/final_fig5_entropy.py (bilinear Molmo, thinking-off Qwen, heads of the
pairwise-ceiling run); only the figure is drawn, not the entropy table. Sigma is added to
each column header.
Writes final_figures/molmo_bilinear_pairwise/fig2_attention_maps_best_blur.png
"""
import os, json
from collections import Counter
import pandas as pd

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'archive', 'ICLR', 'scripts', 'regen_fig5_entropy_cvheads.py')
FF = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise')
SEL = json.load(open(os.path.join(FF, 'results', 'selected_heads.json')))
summ = pd.read_csv(os.path.join(FF, 'blur_sweep', 'blur_sweep_summary.csv')).set_index(['task', 'model'])
SIGMA = {(t, m): int(Counter(summ.loc[(t, m), 'cv_sigmas'].split('/')).most_common(1)[0][0])
         for t, m in summ.index}

src = open(SRC).read()
cut1, cut2 = src.index('# ── 1. entropy'), src.index('# ── 2. Fig 5 grid')
g = {'__file__': SRC, '__name__': 'fig2_best_blur'}
exec(compile(src[:cut1], SRC, 'exec'), g)
np, torch, cv2 = g['np'], g['torch'], g['cv2']

g['OUT_FIG'] = os.path.join(FF, 'fig2_attention_maps_best_blur.png')
for _m in ('CNN-Transformer', 'ViT-Transformer', 'Qwen3.5', 'Molmo'):
    g['HEADS'][_m] = {t: (SEL[f'{_m}_{t}']['layer'], SEL[f'{_m}_{t}']['head']) for t in ('social', 'describe')}
P = g['PROBING_DIR']
g['QCFG'] = {'social':   (os.path.join(P, 'output', 'social_cap', 'qwen35_nothink'),),
             'describe': (os.path.join(P, 'output', 'coco', 'qwen35_nothink'),)}


def molmo_maps(task, only=None):
    """28x28 token grid (crop-overlap rows/cols dropped) bilinearly upsampled, as in final_fig5_entropy.py."""
    L, H = g['HEADS']['Molmo'][task]
    act_dir = os.path.join(g['MOLMO_DIR'], 'activations', task)
    grid36 = g['_molmo_grid'](act_dir)[0]
    vidx = np.where(grid36 > 0); pos = grid36[vidx]
    rows, cols = np.any(grid36 > 0, axis=1), np.any(grid36 > 0, axis=0)
    out = {}
    for im in only:
        p = os.path.join(act_dir, f'activations_img_{im}.pt')
        if not os.path.exists(p):
            continue
        a = torch.load(p, map_location='cpu')[L - 1, H - 1].float().numpy()
        vol = np.zeros((a.shape[0],) + grid36.shape, np.float32)
        vol[:, vidx[0], vidx[1]] = a[:, pos]
        out[im] = cv2.resize(vol.sum(0)[rows][:, cols], (g['SIZE'], g['SIZE']), interpolation=cv2.INTER_LINEAR)
    return out
g['molmo_maps'] = molmo_maps


def blurred(loader, model):
    """Blur the 256x256 maps at the model's cross-validated sigma for the task."""
    def f(*args, **kw):
        task = args[-1] if model in ('Molmo', 'Qwen3.5') else args[1]
        s = SIGMA[(task, model)]
        return {k: cv2.GaussianBlur(np.asarray(v, np.float32), (0, 0), s) for k, v in loader(*args, **kw).items()}
    return f


_lstm, _trans, _molmo, _qwen = g['lstm_maps'], g['trans_maps'], g['molmo_maps'], g['qwen_maps']
g['lstm_maps'] = lambda m, t: blurred(_lstm, m)(m, t)
g['trans_maps'] = lambda m, t: blurred(_trans, m)(m, t)
g['molmo_maps'] = lambda t, only=None: blurred(_molmo, 'Molmo')(t, only=only)
g['qwen_maps'] = lambda t, only=None: blurred(_qwen, 'Qwen3.5')(t, only=only)

fig_src = src[cut2:]
# sigma under every model header
old_h = "def header(col, task):\n    if col == 'Human': return 'Human\\n(Mean)'\n"
assert old_h in fig_src
fig_src = fig_src.replace(old_h, "def header(col, task):\n    if col == 'Human': return 'Human\\n(Mean)'\n"
                          "    return _header(col, task) + f'\\n$\\\\sigma$={SIGMA[(task, col)]}'\n"
                          "def _header(col, task):\n")
# the extra sigma line makes the Describe headers taller: move that block further down
old_s = 'b.y0 - 0.035'
assert old_s in fig_src
fig_src = fig_src.replace(old_s, 'b.y0 - 0.06')
g['SIGMA'] = SIGMA
exec(compile(fig_src, SRC, 'exec'), g)
