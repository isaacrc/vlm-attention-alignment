"""
plot_fig3_qwen35_lg_m6cstyle.py
---------------------------------
Same as plot_fig3_qwen35_lg.py but with font sizes matched to the m6c figure:
  subplot titles: 20 bold
  tick labels:    18
  axis labels:    19
  panel letters:  31

Output: fig3_qwen35_pct_lg_m6cstyle_m6colors.png
"""

import os, glob, json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec

ARCH_DIR  = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
GROUP_DIR = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
PROC_DIR  = os.path.join(GROUP_DIR, 'processed_correlations')
RNN_DIR   = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
OUT_DIR   = os.path.join(ARCH_DIR, 'manuscript_plots')

TARGET_CSV   = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
HUM_SOC_DIR  = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social')
HUM_DESC_DIR = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive')

BEST_HEAD  = {'CNN_Transformer': 5, 'ViT_Transformer': 8}
LAST_EPOCH = {
    'CNN_Transformer': {'phase2': 14, 'social_unfrozen': 3},
    'ViT_Transformer': {'phase2': 2,  'social_unfrozen': 1},
}

COLORS = {
    'Human':           '#1f77b4',
    'CNN-LSTM':        '#ff7f0e',
    'ViT-LSTM':        '#2ca02c',
    'CNN-Transformer': '#d62728',
    'ViT-Transformer': '#9467bd',
    'Molmo':           '#8c564b',
    'Qwen3.5':         '#17becf',
}
MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']

hum_soc_sp = pd.read_csv(os.path.join(PROC_DIR, 'HUM-HUM_cor_soc_soc_200_images_spearman.csv'))
hum_des_sp = pd.read_csv(os.path.join(PROC_DIR, 'HUM-HUM_cor_des_des_200_images_spearman.csv'))
CEIL_SOC = float(hum_soc_sp['mean_z_corr_within_dataset'].mean())
CEIL_DES = float(hum_des_sp['mean_z_corr_within_dataset'].mean())
print(f'Spatial ceilings: social={CEIL_SOC:.4f}, describe={CEIL_DES:.4f}')


def agg_subjects(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def get_spatial_social_lstm(model_name):
    if model_name == 'CNN-LSTM':
        df = pd.read_csv(os.path.join(ARCH_DIR, 'CNN_LSTM', 'correlations', 'spatial_corr.csv'))
        return agg_subjects(df[df['label'] == 'P3_E2']['fisher_z'])
    cond_key = model_name.replace('-', '_')
    df = pd.read_csv(os.path.join(ARCH_DIR, cond_key, 'correlations', 'spatial_corr_social_unfrozen.csv'))
    return agg_subjects(df[df['epoch'] == df['epoch'].max()]['fisher_z'])


def get_spatial_describe_lstm(model_name):
    if model_name == 'CNN-LSTM':
        df = pd.read_csv(os.path.join(PROC_DIR, 'all_models_all_tasks_correlations_final.csv'))
        by_sub = df[(df['model'] == 'Gated') & (df['task'] == 'describe')].groupby('subject_id')['spearman_z'].mean()
        return agg_subjects(by_sub.values)
    cond_key = model_name.replace('-', '_')
    df = pd.read_csv(os.path.join(ARCH_DIR, cond_key, 'correlations', 'spatial_corr_phase2.csv'))
    return agg_subjects(df[df['epoch'] == df['epoch'].max()]['fisher_z'])


def head_spatial_per_subject(hvec_npz_path, hm_dir, target_imgs, head_1idx):
    head_idx  = head_1idx - 1
    hvec_data = np.load(hvec_npz_path, allow_pickle=True)
    head_vecs = hvec_data['head_vecs']
    common_idx = hvec_data['common_idx']
    model_vecs  = head_vecs[head_idx]
    common_imgs = [target_imgs[int(i)] for i in common_idx]
    sub_means = []
    for fn in sorted(os.listdir(hm_dir)):
        if not fn.endswith('_heatmaps.npy'):
            continue
        sdat = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
        z_vals = []
        for ii, img_name in enumerate(common_imgs):
            hum = sdat.get(img_name)
            if hum is None:
                continue
            hum_r = np.array(Image.fromarray(
                np.nan_to_num(hum, nan=0.0).astype(np.float32)
            ).resize((256, 256), resample=Image.NEAREST)).flatten()
            model_flat = model_vecs[ii].astype(np.float64)
            if np.std(model_flat) < 1e-12 or np.std(hum_r) < 1e-12:
                continue
            r, _ = spearmanr(model_flat, hum_r)
            if not np.isnan(r):
                z_vals.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
        if z_vals:
            sub_means.append(float(np.tanh(np.mean(z_vals))))
    return agg_subjects(sub_means)


def head_between_task(des_hvec_path, soc_hvec_path, target_imgs, head_1idx):
    head_idx = head_1idx - 1
    d = np.load(des_hvec_path, allow_pickle=True)
    s = np.load(soc_hvec_path, allow_pickle=True)
    d_vecs = d['head_vecs'][head_idx]
    s_vecs = s['head_vecs'][head_idx]
    d_idx  = {int(i): ii for ii, i in enumerate(d['common_idx'])}
    s_idx  = {int(i): ii for ii, i in enumerate(s['common_idx'])}
    common_ti = sorted(set(d_idx) & set(s_idx))
    rs = []
    for ti in common_ti:
        dv = d_vecs[d_idx[ti]].astype(np.float64)
        sv = s_vecs[s_idx[ti]].astype(np.float64)
        if np.std(dv) < 1e-12 or np.std(sv) < 1e-12:
            continue
        r, _ = spearmanr(dv, sv)
        if not np.isnan(r):
            rs.append(r)
    rs = np.array(rs)
    return float(rs.mean()), float(rs.std(ddof=1) / np.sqrt(len(rs)))


_molmo_head_df = pd.read_csv(os.path.join(GROUP_DIR, 'molmo_head_attention_L9.csv'))
_qwen35_soc_df = pd.read_csv(os.path.join(GROUP_DIR, 'qwen35_head_attention_allL_social.csv'))
_qwen35_des_df = pd.read_csv(os.path.join(GROUP_DIR, 'qwen35_head_attention_allL_describe.csv'))

_q35_soc_best = _qwen35_soc_df.loc[_qwen35_soc_df['mean_r'].idxmax()]
_q35_des_best = _qwen35_des_df.loc[_qwen35_des_df['mean_r'].idxmax()]

VLM_BEST = {
    'Molmo':   {'social': {'layer': 9, 'head': 24}, 'describe': {'layer': 9, 'head': 27}},
    'Qwen3.5': {
        'social':   {'layer': int(_q35_soc_best['layer']), 'head': int(_q35_soc_best['head'])},
        'describe': {'layer': int(_q35_des_best['layer']), 'head': int(_q35_des_best['head'])},
    },
}
print(f"Qwen3.5 best: social=L{VLM_BEST['Qwen3.5']['social']['layer']}"
      f"H{VLM_BEST['Qwen3.5']['social']['head']}  "
      f"describe=L{VLM_BEST['Qwen3.5']['describe']['layer']}"
      f"H{VLM_BEST['Qwen3.5']['describe']['head']}")


def get_spatial_vlm(model_label, task):
    cfg = VLM_BEST[model_label][task]
    if model_label == 'Molmo':
        row = _molmo_head_df[(_molmo_head_df['task'] == task) &
                              (_molmo_head_df['head'] == cfg['head'])].iloc[0]
    elif model_label == 'Qwen3.5':
        df  = _qwen35_soc_df if task == 'social' else _qwen35_des_df
        row = df[(df['layer'] == cfg['layer']) & (df['head'] == cfg['head'])].iloc[0]
    else:
        raise ValueError(f'Unknown VLM: {model_label}')
    return float(row['mean_r']), float(row['sem_r'])


bt_spat    = pd.read_csv(os.path.join(OUT_DIR, 'between_task_spatial.csv'))
_vlm_bt_df = pd.read_csv(os.path.join(GROUP_DIR, 'vlm_between_task.csv'))


def get_between_task(model_name, trans_data):
    if model_name in ('CNN-Transformer', 'ViT-Transformer'):
        return trans_data[model_name.replace('-', '_')]
    if model_name in ('Molmo', 'Qwen3.5'):
        row = _vlm_bt_df[_vlm_bt_df['model'] == model_name].iloc[0]
        return float(row['mean_r']), float(row['sem_r'])
    if model_name == 'Human':
        row = bt_spat[bt_spat['Model'] == 'Human'].iloc[0]
        return float(row['Mean']), float(row['SEM'])
    row = bt_spat[bt_spat['Model'] == model_name.replace('-', '_')].iloc[0]
    return float(row['Mean']), float(row['SEM'])


target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()

def hvec_path(cond, phase, epoch, layer=3):
    return os.path.join(ARCH_DIR, cond, 'ckpt_cache',
                        f'{phase}_beam5_E{epoch:03d}_head_vecs_L{layer}.npz')

TRANS_CACHE = os.path.join(OUT_DIR, 'trans_values_cache.json')

def _load_trans_cache():
    with open(TRANS_CACHE) as f:
        raw = json.load(f)
    return {cond: {k: tuple(v) for k, v in tasks.items()} for cond, tasks in raw.items()}

def _save_trans_cache(tv):
    with open(TRANS_CACHE, 'w') as f:
        json.dump({cond: {k: list(v) for k, v in tasks.items()} for cond, tasks in tv.items()}, f)

if os.path.exists(TRANS_CACHE):
    print(f'Loading cached trans_values from {TRANS_CACHE}')
    trans_values = _load_trans_cache()
else:
    trans_values = {}
    for cond, best_h in BEST_HEAD.items():
        ep_soc = LAST_EPOCH[cond]['social_unfrozen']
        ep_des = LAST_EPOCH[cond]['phase2']
        soc_path = hvec_path(cond, 'social_unfrozen', ep_soc)
        des_path = hvec_path(cond, 'phase2', ep_des)
        trans_values.setdefault(cond, {})
        trans_values[cond]['social']   = head_spatial_per_subject(soc_path, HUM_SOC_DIR,  target_imgs, best_h)
        trans_values[cond]['describe'] = head_spatial_per_subject(des_path, HUM_DESC_DIR, target_imgs, best_h)
        trans_values[cond]['between']  = head_between_task(des_path, soc_path, target_imgs, best_h)
    _save_trans_cache(trans_values)

bar_data = {}
for m in MODELS_6:
    bar_data[m] = {'social': None, 'describe': None}
    if m in ('CNN-LSTM', 'ViT-LSTM'):
        bar_data[m]['social']   = get_spatial_social_lstm(m)
        bar_data[m]['describe'] = get_spatial_describe_lstm(m)
    elif m in ('CNN-Transformer', 'ViT-Transformer'):
        bar_data[m]['social']   = trans_values[m.replace('-','_')]['social']
        bar_data[m]['describe'] = trans_values[m.replace('-','_')]['describe']
    else:
        bar_data[m]['social']   = get_spatial_vlm(m, 'social')
        bar_data[m]['describe'] = get_spatial_vlm(m, 'describe')

trans_bt = {k: v['between'] for k, v in trans_values.items()}
for m in MODELS_6:
    bar_data[m]['between'] = get_between_task(m, trans_bt)
hum_bt = get_between_task('Human', trans_bt)


def make_tick(m):
    if m == 'CNN-Transformer': return 'CNN-Trans.'
    if m == 'ViT-Transformer': return 'ViT-Trans.'
    if m == 'CNN-LSTM':        return 'CNN-LSTM'
    if m == 'ViT-LSTM':        return 'ViT-LSTM'
    if m == 'Molmo':           return 'Molmo 7B-D'
    if m == 'Qwen3.5':         return 'Qwen 9B-3.5'
    return m


# ── Big-text font sizes + stumpy bars ────────────────────────────────────────
FS_TITLE  = 44   # subplot titles
FS_TICK   = 30   # tick labels
FS_LABEL  = 34   # axis labels
FS_LETTER = 56   # panel letter (A/B/C) — placed inside axes, no y-label collision
FS_LEGEND = 26
BAR_W     = 0.55  # wide stumpy bars


def draw_panel(ax, key, ceil, label, show_ylabel=False):
    if key == 'between':
        models_here = MODELS_6 + ['Human']
        scale = 1.0
        yticks = [0.0, 0.3, 0.6, 0.9]
    else:
        models_here = MODELS_6
        scale = 100.0 / ceil
        yticks = [0, 50, 100]

    x     = np.arange(len(models_here))
    means = np.array([bar_data[m][key][0] if m in bar_data else hum_bt[0]
                      for m in models_here]) * scale
    sems  = np.array([bar_data[m][key][1] if m in bar_data else hum_bt[1]
                      for m in models_here]) * scale
    cols  = [COLORS[m] for m in models_here]

    ax.bar(x, means, BAR_W, color=cols, edgecolor='white', linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='black',
                capsize=6, linewidth=2.0, zorder=4)

    line = None
    if ceil is not None:
        line = ax.axhline(100.0, color='black', linestyle='--', linewidth=2.5, zorder=2)

    tick_labels = [make_tick(m) for m in models_here]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=FS_TICK, rotation=45, ha='right')
    ax.margins(x=0.06)
    ax.tick_params(axis='x', pad=4)
    ax.set_yticks(yticks)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    if key == 'social':
        ax.tick_params(axis='y', labelleft=False)
    ax.set_ylim(bottom=0, top=yticks[-1] * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    title_map = {'describe': 'Describe Task', 'social': 'Social Task', 'between': 'Between-task'}
    ax.set_title(title_map[key], fontsize=FS_TITLE, fontweight='bold', pad=14)

    # Panel letter inside top-left corner — avoids y-label collision entirely
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=FS_LETTER, fontweight='bold', va='top', ha='left')

    if show_ylabel:
        ylabel_txt = ('Spatial Alignment (Spearman r)' if key == 'between'
                      else '% of Human Upperbound')
        ax.set_ylabel(ylabel_txt, fontsize=FS_LABEL, labelpad=12)
    return line


def add_ceil_legend(ax, line):
    ceil_frac = 100.0 / ax.get_ylim()[1]
    ax.legend(handles=[line], labels=['Human Upperbound'], fontsize=FS_LEGEND,
              bbox_to_anchor=(0.99, ceil_frac - 0.02), loc='upper right',
              bbox_transform=ax.transAxes,
              frameon=True, framealpha=0.9, edgecolor='none')


panel_configs = [
    ('describe', CEIL_DES, 'A', True),
    ('social',   CEIL_SOC, 'B', False),
    ('between',  None,     'C', True),
]

# Shorter figure height → stumpy bars; large bottom margin accommodates rotated labels
fig = plt.figure(figsize=(32, 12))
gs  = matplotlib.gridspec.GridSpec(
    1, 4, width_ratios=[6, 6, 0.7, 7],
    left=0.10, right=0.99, top=0.88, bottom=0.32, wspace=0.10,
    figure=fig
)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])]

lines = [draw_panel(ax, key, ceil, label, show_ylabel=show_y)
         for ax, (key, ceil, label, show_y) in zip(axes, panel_configs)]

y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
axes[0].set_ylim(0, y_max)
axes[1].set_ylim(0, y_max)

add_ceil_legend(axes[1], lines[1])

out = os.path.join(OUT_DIR, 'fig3_qwen35_pct_lg_m6cstyle_m6colors.png')
out2 = os.path.normpath(os.path.join(OUT_DIR, '../../manuscript/figures/fig3_qwen35_pct_lg_m6cstyle_m6colors.png'))
for p in [out, out2]:
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f'Saved: {p}')
plt.close(fig)
