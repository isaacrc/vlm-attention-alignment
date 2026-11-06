"""
slides_fig1.py -- Figure 1 of main_2026-10-23.tex as 16:9 PowerPoint slides (13.333 x 7.5 in, 300 dpi).
Follows scripts/fig3_slides.py: executes the data loading and panel drawing of
rebuttal/plots/plot_fig_head_selection_corrected.py on the molmo_bilinear_pairwise results
(as fig1_alignment.py), with the four trained models only and slide font sizes.
Writes final_figures/manuscript_2026-10-23/slides/
  fig1_describe_slide.png, fig1_social_slide.png, fig1_between_slide.png   one panel per slide
  fig1_describe_social_slide.png                                           Describe and Social side by side
"""
import os

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'rebuttal', 'plots', 'plot_fig_head_selection_corrected.py')
RES = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides')
os.makedirs(OUT, exist_ok=True)

src = open(SRC).read()
for old, new in [("os.path.join(DATA, 'alignment_summary_cv_honest.csv')", repr(os.path.join(RES, 'alignment_summary_cv_honest.csv'))),
                 ("os.path.join(T1, 'selected_heads.json')", repr(os.path.join(RES, 'selected_heads.json'))),
                 ("os.path.join(DATA, 'ceilings.json')", repr(os.path.join(RES, 'ceilings.json'))),
                 ("os.path.join(T2, 'model_between_task.csv')", repr(os.path.join(RES, 'model_between_task_fig1.csv'))),
                 ("MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']",
                  "MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer']")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
ns = {'__file__': SRC, '__name__': 'fig1_slides'}
exec(compile(src[:src.index('panel_configs = ')], SRC, 'exec'), ns)   # data, colors, draw_panel, draw_between_panel
plt = ns['plt']

W, H_IN = 13.333, 7.5
Y_MAX = 122   # shared y-range of panels A and B in the paper figure
TASKS = (('describe', ns['CEIL_DES']), ('social', ns['CEIL_SOC']))


def save(fig, name):
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out)


def new_fig(rect=(0.15, 0.27, 0.80, 0.62)):
    fig = plt.figure(figsize=(W, H_IN), dpi=300, facecolor='white')
    ns['fig'] = fig   # draw_* functions read the module-level `fig`
    return fig, fig.add_axes(rect)


# one panel per slide
ns.update(FS_TITLE=34, FS_TICK=22, FS_LABEL=24, FS_LEGEND=20, FS_LETTER=34, TITLE_PAD=12)
for task, ceil_z in TASKS:
    fig, ax = new_fig()
    line = ns['draw_panel'](ax, task, ceil_z, '', show_ylabel=True)
    ax.tick_params(axis='y', labelleft=True)
    ax.set_ylim(0, Y_MAX)
    ns['add_ceil_legend'](ax, line)
    save(fig, f'fig1_{task}_slide.png')

fig, ax = new_fig()
ns['draw_between_panel'](ax, '')
save(fig, 'fig1_between_slide.png')

# Describe and Social side by side, shared y-axis
ns.update(FS_TITLE=30, FS_TICK=20, FS_LABEL=22, FS_LEGEND=18)
fig, ax = new_fig((0.12, 0.27, 0.40, 0.62))
ns['fig'] = fig
axes = [ax, fig.add_axes([0.56, 0.27, 0.40, 0.62])]
for k, (a, (task, ceil_z)) in enumerate(zip(axes, TASKS)):
    line = ns['draw_panel'](a, task, ceil_z, '', show_ylabel=(k == 0))
    a.set_ylim(0, Y_MAX)
ns['add_ceil_legend'](axes[1], line)
save(fig, 'fig1_describe_social_slide.png')

# Describe and Social side by side in the bottom half of the slide, top half left empty
# (two-line horizontal tick labels, to save the height that rotated labels take)
ns.update(FS_TITLE=24, FS_TICK=16, FS_LABEL=17, FS_LEGEND=15, TITLE_PAD=8)
fig, ax = new_fig((0.10, 0.12, 0.41, 0.30))
axes = [ax, fig.add_axes([0.56, 0.12, 0.41, 0.30])]
for k, (a, (task, ceil_z)) in enumerate(zip(axes, TASKS)):
    line = ns['draw_panel'](a, task, ceil_z, '', show_ylabel=(k == 0))
    a.set_ylim(0, Y_MAX)
    a.set_xticklabels([ns['make_tick'](m).replace('-', '-\n', 1) for m in ns['MODELS_6']],
                      rotation=0, ha='center', fontsize=ns['FS_TICK'])
ns['add_ceil_legend'](axes[1], line)
save(fig, 'fig1_describe_social_bottom_slide.png')
