"""
fig2_attention_maps.py -- Figure 2 of main_2026-10-23.tex: human, the four trained models, and
the two VLMs (Molmo 7B-D, Qwen3.5 9B) for comparison, the same seven columns as the 10-22 figure. Follows qwen35_nothink/final_fig5_entropy.py, the
wrapper that drew the 10-22 Figure 2: archive/ICLR/scripts/regen_fig5_entropy_cvheads.py executed
with the VLM heads from the molmo_bilinear_pairwise run, Qwen3.5 from the thinking-off run, and
Molmo's 28x28 token grid bilinearly upsampled (overlap margins removed first; no smoothing).
Same images, colour map and layout as the 10-22 figure.
The entropy table is skipped.
Writes final_figures/manuscript_2026-10-23/fig2_attention_maps.png
"""
import os, json

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'archive', 'ICLR', 'scripts', 'regen_fig5_entropy_cvheads.py')
SEL = json.load(open(os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results', 'selected_heads.json')))
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig2_attention_maps.png')

src = open(SRC).read()
for old, new in [("SKIP_ENTROPY = os.path.exists(OUT_CSV) and '--fig-only' in sys.argv", "SKIP_ENTROPY = True"),
                 ("COLS = ['Human', 'CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']",
                  "COLS = ['Human', 'CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
cut = src.index('# ── 1. entropy')
g = {'__file__': SRC, '__name__': 'fig2_1023'}
exec(compile(src[:cut], SRC, 'exec'), g)
g['OUT_FIG'] = OUT
for _m in ('Molmo', 'Qwen3.5'):
    g['HEADS'][_m] = {t: (SEL[f'{_m}_{t}']['layer'], SEL[f'{_m}_{t}']['head']) for t in ('social', 'describe')}
P = g['PROBING_DIR']   # thinking-off Qwen3.5 activations, as final_fig5_entropy.py
g['QCFG'] = {'social': (os.path.join(P, 'output', 'social_cap', 'qwen35_nothink'),),
             'describe': (os.path.join(P, 'output', 'coco', 'qwen35_nothink'),)}
np, torch, cv2 = g['np'], g['torch'], g['cv2']


def molmo_maps(task, only=None):   # as qwen35_nothink/final_fig5_entropy.py
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
        a = torch.load(p, map_location='cpu')[L - 1, H - 1].float().numpy()
        vol = np.zeros((a.shape[0],) + grid36.shape, np.float32)
        vol[:, vidx[0], vidx[1]] = a[:, pos]
        grid = vol.sum(0)[rows][:, cols]
        out[im] = cv2.resize(grid, (g['SIZE'], g['SIZE']), interpolation=cv2.INTER_LINEAR)
    return out


g['molmo_maps'] = molmo_maps
print('heads:', {m: g['HEADS'][m] for m in ('CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5')}, flush=True)
exec(compile(src[cut:], SRC, 'exec'), g)
