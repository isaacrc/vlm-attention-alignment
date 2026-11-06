"""
slides_fig2_saliency.py -- the layout of slides/fig2_describe_slide.png (same three images, same overlay)
with CNN-LSTM on the left for reference, then baselines. Two slides:
  fig2_describe_saliency_slide.png   Itti-Koch, GBVS, spectral residual, edge energy, luminance
  fig2_describe_priors_slide.png     center, person and face priors CNN-LSTM's map is the Describe map of Figure 2 (computed by slides_fig2.py's setup).
The baseline maps are the ones scored in the Appendix B figure (molmo_bilinear/fig1_baselines/score_baselines.py):
saliency maps from rebuttal tierB_maps (256x256, unblurred); edge energy and luminance per 14x14 cell, bilinearly upsampled
to 256x256. Person and face priors from tierB_maps. The center prior is the leave-one-image-out mean human
fixation map pooled over both tasks (run_center_controlled.loio_prior_ranked), shown before its rank transform.
Baselines see no task, so their maps are the same for both tasks.
"""
import os, sys

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))

# slides_fig2.py's setup (maps, images, overlay helpers), stopped before it draws any slide
s = open(os.path.join(HERE, 'slides_fig2.py')).read()
ns = {'__file__': os.path.join(HERE, 'slides_fig2.py'), '__name__': 'fig2_saliency_slide'}
exec(compile(s[:s.index("for task in ('social', 'describe'):")], ns['__file__'], 'exec'), ns)
np, plt, Image = ns['np'], ns['plt'], ns['Image']
maps, COL_COLORS, BLEND_IMGS, IMG_DIR, SIZE, ALPHA, CMAP, normalize = (
    ns[k] for k in ('maps', 'COL_COLORS', 'BLEND_IMGS', 'IMG_DIR', 'SIZE', 'ALPHA', 'CMAP', 'normalize'))
W, H_IN, CELL, GAP_X, GAP_Y, TITLE_H, HEAD_H, OUT = (ns[k] for k in ('W', 'H_IN', 'CELL', 'GAP_X', 'GAP_Y', 'TITLE_H', 'HEAD_H', 'OUT'))

sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'task3_baselines'))
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import inspect
import tierB_common as T, alignment_core as A, run_center_controlled as CC
bs = open(os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'fig1_baselines', 'score_baselines.py')).read()
bns = {'__name__': 'baselines'}
exec(compile(bs[:bs.index('def main():')], 'score_baselines.py', 'exec'), bns)   # cell_maps (edge energy, luminance)


def key(d, im):
    return next(k for k in d if k.replace('.jpg', '').endswith(im))


base = {'Itti-Koch': T.load_maps('itti_koch'), 'GBVS': T.load_maps('gbvs'), 'Spectral residual': T.load_maps('spectral_residual')}
img_keys = [key(base['GBVS'], im) for im in BLEND_IMGS]
edge, lum = bns['cell_maps'](img_keys)
base['Edge energy'], base['Luminance'] = dict(zip(img_keys, edge.values())), dict(zip(img_keys, lum.values()))
base['Person prior'], base['Face prior'] = T.load_maps('coco_person'), T.load_maps('coco_face')
# centre prior without the rank transform used for scoring (ranks would flatten it for display)
csrc = inspect.getsource(CC.loio_prior_ranked)
assert csrc.count('return A._rank(out, axis=1)') == 1
exec(csrc.replace('def loio_prior_ranked', 'def loio_prior_raw').replace('return A._rank(out, axis=1)', 'return out'), CC.__dict__)
all_imgs = A.target_images()
cen = CC.loio_prior_raw(all_imgs)
base['Center prior'] = {k: cen[all_imgs.index(k)].reshape(A.OUT_SIZE, A.OUT_SIZE) for k in img_keys}
HEAD = {'CNN-LSTM': 'CNN\nLSTM', 'Itti-Koch': 'Itti-Koch', 'GBVS': 'GBVS', 'Spectral residual': 'Spectral\nresidual',
        'Edge energy': 'Edge\nenergy', 'Luminance': 'Luminance', 'Center prior': 'Center\nprior', 'Person prior': 'Person\nprior', 'Face prior': 'Face\nprior'}
COLOR = {'CNN-LSTM': COL_COLORS['CNN-LSTM'], 'Itti-Koch': '#5e3c99', 'GBVS': '#7b5ea7',   # Appendix B shades,
         'Spectral residual': '#8a68bd', 'Edge energy': '#9b7ccb', 'Luminance': '#ab90d6',                          # darkened for legible text
         'Center prior': '#8c6d31', 'Person prior': '#a8892c', 'Face prior': '#c19a3c'}


def get_map(col, im):
    if col == 'CNN-LSTM':
        return maps['describe'][col][im]
    m = base[col][key(base[col], im)]
    return np.asarray(Image.fromarray(np.asarray(m, np.float32)).resize((SIZE, SIZE), Image.BILINEAR))


def slide(COLS, name):
    NC = len(COLS)
    x0 = (W - (NC * CELL + (NC - 1) * GAP_X)) / 2
    y_top = H_IN - TITLE_H - HEAD_H
    fig = plt.figure(figsize=(W, H_IN), dpi=300, facecolor='white')
    fig.text(0.5, 1 - 0.45 / H_IN, 'Describe Task', fontsize=30, fontweight='bold', ha='center', va='center', color='#2C3E50')
    for r, im in enumerate(BLEND_IMGS):
        img = np.array(Image.open(os.path.join(IMG_DIR, f'{im}.jpg')).convert('RGB').resize((SIZE, SIZE))).astype(np.float32) / 255.0
        for c, col in enumerate(COLS):
            left = x0 + c * (CELL + GAP_X) + (0.25 if c > 0 else 0)   # small extra gap after the reference column
            bottom = y_top - (r + 1) * CELL - r * GAP_Y
            ax = fig.add_axes([(left - 0.125) / W, bottom / H_IN, CELL / W, CELL / H_IN])
            m = get_map(col, im)
            heat = CMAP(normalize(m))[:, :, :3].astype(np.float32)
            ax.imshow(np.clip(img * (1 - ALPHA) + heat * ALPHA, 0, 1)); ax.axis('off')
            if np.ptp(m) == 0:   # face prior on an image with no annotated face (not scored there)
                ax.text(0.5, 0.5, 'No annotated\nface', transform=ax.transAxes, ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')
            if r == 0:
                ax.set_title(HEAD[col], fontsize=16, fontweight='bold', pad=6, color=COLOR[col])
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out)


slide(['CNN-LSTM', 'Itti-Koch', 'GBVS', 'Spectral residual', 'Edge energy', 'Luminance'], 'fig2_describe_saliency_slide.png')
slide(['CNN-LSTM', 'Center prior', 'Person prior', 'Face prior'], 'fig2_describe_priors_slide.png')
