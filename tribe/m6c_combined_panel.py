"""
m6c_combined_panel_m4d_top.py
------------------------------
Same as m6b_combined_panel_social.py but replaces Panel A with the
pre-rendered m4d_bar_attn_mean_noclip_2panel.png.

Panel A (top):    m4d_bar_attn_mean_noclip_2panel.png (loaded as image)
Panel B (bottom): Variance-partition brain maps, CNN-Transformer, Social task
                  (identical to m6b)

Output: manuscript/figures/m6c_combined_panel_m4d_top_blackyellow.png
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
black_yellow = LinearSegmentedColormap.from_list("black_yellow", ["black", "yellow"])
from pathlib import Path
from nilearn import plotting, datasets as nidatasets

BASE    = Path('/scratch/gpfs/GRAZIANO/CANS')
OUT_DIR = BASE / 'tribe/outputs'
FIG_DIR = OUT_DIR / 'figures/manuscript_v1_plots'

PANEL_LABEL_FS = 31

# ── Panel A: render from source data so font sizes match Panel B ──────────────
import matplotlib.patches as mpatches

ATTN_MODELS = [
    'CNN-LSTM', 'ViT-LSTM',
    'CNN-Transformer', 'ViT-Transformer',
    'Molmo', 'Qwen3.5',
    'Human',
]
TASKS = ['describe', 'social']
COLORS_BAR = {
    'Human':           '#1f77b4',
    'CNN-LSTM':        '#ff7f0e',
    'ViT-LSTM':        '#2ca02c',
    'CNN-Transformer': '#d62728',
    'ViT-Transformer': '#9467bd',
    'Molmo':           '#8c564b',
    'Qwen3.5':         '#17becf',
}
TASK_LABELS = {'social': 'Social task', 'describe': 'Describe task'}
DISPLAY_NAMES = {
    'Human':           'Human',
    'CNN-LSTM':        'CNN-LSTM',
    'ViT-LSTM':        'ViT-LSTM',
    'CNN-Transformer': 'CNN-Transformer',
    'ViT-Transformer': 'ViT-Transformer',
    'Molmo':           'Molmo 7B-D',
    'Qwen3.5':         'Qwen 9B-3.5',
}

r2_arrays_a = {}
for task in TASKS:
    p = OUT_DIR / f'r2_human_{task}.npy'
    if p.exists():
        r2_arrays_a[('Human', task)] = np.load(p)
    for model in [m for m in ATTN_MODELS if m != 'Human']:
        p = OUT_DIR / f'r2_{model}_{task}.npy'
        if p.exists():
            r2_arrays_a[(model, task)] = np.load(p)

vals_a      = {k: float(np.mean(v)) for k, v in r2_arrays_a.items()}
show_models = [m for m in ATTN_MODELS if any((m, t) in vals_a for t in TASKS)]
n_models    = len(show_models)
x_pos       = np.arange(n_models)
BAR_W       = 0.6

# ── font sizes matching Panel B (ROW_LABELS=20, tick_params=18) ───────────────
FS_TITLE = 20   # matches Panel B ROW_LABELS fontsize
FS_TICK  = 18   # matches Panel B tick_params labelsize
FS_LABEL = 19   # y-axis label

fig_a, axes_a = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

for ti, task in enumerate(TASKS):
    ax = axes_a[ti]
    y_top = max(vals_a.get((m, task), 0) for m in show_models)
    for xi, model in enumerate(show_models):
        if (model, task) not in vals_a:
            continue
        ax.bar(xi, vals_a[(model, task)], BAR_W,
               color=COLORS_BAR[model], edgecolor='white', linewidth=0.5)
    ax.set_ylim(0, y_top * 1.15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in show_models], fontsize=FS_TICK, rotation=30, ha='right')
    ax.set_title(TASK_LABELS[task], fontsize=FS_TITLE, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    if ti == 0:
        ax.set_ylabel('Mean R² (whole brain)', fontsize=FS_LABEL)

fig_a.suptitle(
    'Predicting Synthetic Brain Activity from Attention Maps',
    fontsize=23, fontweight='bold', y=1.01,
)
fig_a.text(0.005, 0.99, 'A', fontsize=PANEL_LABEL_FS, fontweight='bold',
           va='top', ha='left', transform=fig_a.transFigure)
fig_a.tight_layout()

buf_a = io.BytesIO()
fig_a.savefig(buf_a, format='png', dpi=150, bbox_inches='tight')
plt.close(fig_a)
buf_a.seek(0)
img_a = plt.imread(buf_a)
print(f'Panel A rendered: {img_a.shape}')


# ── Panel B: variance-partition brain maps (identical to m6b) ─────────────────
N_LH = 10242
fs5  = nidatasets.fetch_surf_fsaverage('fsaverage5')
MESH = {
    'left':  {'pial': fs5.pial_left,  'sulc': fs5.sulc_left},
    'right': {'pial': fs5.pial_right, 'sulc': fs5.sulc_right},
}

model = 'CNN-Transformer'
task  = 'social'
tag   = model.lower() if model == 'Human' else model

r2_clip_only = np.load(OUT_DIR / 'r2_clip_only.npy')
r2_attn      = np.load(OUT_DIR / f'r2_{tag}_{task}.npy')
r2_banded    = np.load(OUT_DIR / f'r2_banded_{tag}_{task}.npy')

unique_attn = np.clip(r2_banded - r2_clip_only, 0, None)
unique_clip = np.clip(r2_banded - r2_attn,      0, None)
shared      = np.clip(r2_attn + r2_clip_only - r2_banded, 0, None)

def _load_sig(preferred, fallback):
    p = OUT_DIR / preferred
    if p.exists():
        return np.load(p).astype(bool)
    f = OUT_DIR / fallback
    if f.exists():
        return np.load(f).astype(bool)
    return None

sig_unique_attn = _load_sig('sig_mask_unique_attn_clip_visual.npy',
                             f'sig_mask_{model}_{task}.npy')
sig_unique_clip = _load_sig('sig_mask_unique_clip_clip_visual.npy',
                             f'sig_mask_{model}_{task}.npy')
sig_shared      = _load_sig('sig_mask_shared_clip_visual.npy',
                             f'sig_mask_{model}_{task}.npy')

if sig_unique_attn is not None:
    unique_attn = np.where(sig_unique_attn, unique_attn, 0.0)
    print(f'Panel B unique_attn: {sig_unique_attn.sum()} sig vertices')
if sig_unique_clip is not None:
    unique_clip = np.where(sig_unique_clip, unique_clip, 0.0)
    print(f'Panel B unique_clip: {sig_unique_clip.sum()} sig vertices')
if sig_shared is not None:
    shared      = np.where(sig_shared,      shared,      0.0)
    print(f'Panel B shared:      {sig_shared.sum()} sig vertices')

rows = [unique_attn, unique_clip, shared]
ROW_LABELS = [
    'Unique Attn R²  ("Where")',
    'Unique CLIP R²  ("What")',
    'Shared R²',
]
VIEWS = [
    ('left',  'lateral'),
    ('left',  'medial'),
    ('right', 'lateral'),
    ('right', 'medial'),
]

fig_b, axes_b = plt.subplots(
    3, 4, figsize=(14, 10),
    subplot_kw={'projection': '3d'},
)
fig_b.suptitle(
    f'{model}  —  {task.capitalize()} task\n'
    f'Variance partition: Where  ·  What  ·  Shared',
    fontsize=23, fontweight='bold', y=0.99,
)
fig_b.subplots_adjust(
    left=0.02, right=0.99,
    top=0.96,  bottom=0.01,
    hspace=-0.4, wspace=0.01,
)

fig_b.text(0.005, 0.99, 'B', fontsize=PANEL_LABEL_FS, fontweight='bold',
           va='top', ha='left', transform=fig_b.transFigure)

for ri in range(3):
    pos0  = axes_b[ri, 0].get_position()
    pos3  = axes_b[ri, 3].get_position()
    x_mid = (pos0.x0 + pos3.x1) / 2
    y_top = pos0.y1 - 0.01
    fig_b.text(x_mid, y_top, ROW_LABELS[ri],
               ha='center', va='top', fontsize=20, fontweight='bold')

for ri, stat_map in enumerate(rows):
    pos  = stat_map[stat_map > 0]
    vmax = max(float(np.percentile(pos, 99)) if len(pos) > 10 else 1e-5, 1e-5)
    for ci, (hemi, view) in enumerate(VIEWS):
        ax        = axes_b[ri, ci]
        data      = stat_map[:N_LH] if hemi == 'left' else stat_map[N_LH:]
        show_cbar = (ci == 3)
        plotting.plot_surf_stat_map(
            surf_mesh=MESH[hemi]['pial'],
            stat_map=data,
            hemi=hemi, view=view,
            bg_map=MESH[hemi]['sulc'],
            colorbar=show_cbar,
            cmap=black_yellow, vmin=0, vmax=vmax,
            threshold=None,
            axes=ax,
        )
        ax.set_title('')

for ax in fig_b.axes:
    if not hasattr(ax, 'get_zlim'):
        ax.tick_params(labelsize=18)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: '0' if v == 0 else f'{v:.2f}')
        )

buf_b = io.BytesIO()
fig_b.savefig(buf_b, format='png', dpi=150, bbox_inches='tight')
plt.close(fig_b)
buf_b.seek(0)
img_b = plt.imread(buf_b)
print(f'Panel B rendered: {img_b.shape}')


# ── Composite ─────────────────────────────────────────────────────────────────
# Scale both panels to the same pixel width before compositing
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

target_w = max(w_a, w_b)

ratio_a = h_a / w_a
ratio_b = h_b / w_b

fig_width  = 14.0
fig_height = fig_width * (ratio_a + ratio_b)

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    figsize=(fig_width, fig_height),
    gridspec_kw={'height_ratios': [ratio_a, ratio_b]},
)

ax_top.imshow(img_a, aspect='auto')
ax_top.axis('off')

ax_bot.imshow(img_b, aspect='auto')
ax_bot.axis('off')

fig.subplots_adjust(hspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)

out = BASE / 'manuscript/figures/m6a_combined_panel_clip_visual_2.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')


# ── Big-text condensed variant ────────────────────────────────────────────────
# Panel A: no x-tick labels; color legend instead; scaled-up fonts; thin bars
BT_FS_TITLE = 28
BT_FS_TICK  = 26
BT_FS_LABEL = 27
BT_FS_SUPER = 30
BT_BAR_W    = 0.55
BT_PANEL_LABEL_FS = 38

fig_bt, axes_bt = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

legend_patches = []
for ti, task in enumerate(TASKS):
    ax = axes_bt[ti]
    y_top = max(vals_a.get((m, task), 0) for m in show_models)
    for xi, model in enumerate(show_models):
        if (model, task) not in vals_a:
            continue
        bar = ax.bar(xi, vals_a[(model, task)], BT_BAR_W,
                     color=COLORS_BAR[model], edgecolor='white', linewidth=0.5)
        if ti == 0:
            legend_patches.append(
                mpatches.Patch(color=COLORS_BAR[model], label=DISPLAY_NAMES.get(model, model))
            )
    ax.set_ylim(0, y_top * 1.18)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([], fontsize=0)   # hide x-tick labels — use legend
    ax.set_title(TASK_LABELS[task], fontsize=BT_FS_TITLE, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=BT_FS_TICK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    if ti == 0:
        ax.set_ylabel('Mean R² (whole brain)', fontsize=BT_FS_LABEL)

fig_bt.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=len(show_models),
    fontsize=BT_FS_TICK - 4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)
fig_bt.suptitle(
    'Predicting Synthetic Brain Activity from Attention Maps',
    fontsize=BT_FS_SUPER, fontweight='bold', y=1.01,
)
fig_bt.text(0.005, 0.99, 'A', fontsize=BT_PANEL_LABEL_FS, fontweight='bold',
            va='top', ha='left', transform=fig_bt.transFigure)
fig_bt.tight_layout()

buf_bt = io.BytesIO()
fig_bt.savefig(buf_bt, format='png', dpi=150, bbox_inches='tight')
plt.close(fig_bt)
buf_bt.seek(0)
img_bt = plt.imread(buf_bt)
print(f'Panel A (big-text) rendered: {img_bt.shape}')

# Composite with same Panel B
h_bt, w_bt = img_bt.shape[:2]
ratio_bt = h_bt / w_bt
ratio_b2 = h_b / w_b

fig2_width  = 14.0
fig2_height = fig2_width * (ratio_bt + ratio_b2)

fig2, (ax2_top, ax2_bot) = plt.subplots(
    2, 1,
    figsize=(fig2_width, fig2_height),
    gridspec_kw={'height_ratios': [ratio_bt, ratio_b2]},
)
ax2_top.imshow(img_bt, aspect='auto')
ax2_top.axis('off')
ax2_bot.imshow(img_b, aspect='auto')
ax2_bot.axis('off')
fig2.subplots_adjust(hspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)

out2 = BASE / 'manuscript/figures/m6c_combined_panel_m4d_top_blackyellow_bigtext.png'
fig2.savefig(out2, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f'Saved: {out2}')
