"""
slides_fig1_vlm.py -- Figure 1 as 16:9 slides (13.333 x 7.5 in, 300 dpi) with the four trained models
and the two pretrained VLMs (Molmo 7B-D, Qwen 9B-3.5): Describe and Social on one slide, Between-task on another.
Executes the data loading and panel drawing of rebuttal/plots/plot_fig_head_selection_corrected.py on the
molmo_bilinear_pairwise results, as slides_fig1.py does, but keeps all six models.
Writes final_figures/manuscript_2026-10-23/slides/fig1_vlm_describe_social_slide.png, fig1_vlm_between_slide.png
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
                 ("os.path.join(T2, 'model_between_task.csv')", repr(os.path.join(RES, 'model_between_task_fig1.csv')))]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
ns = {'__file__': SRC, '__name__': 'fig1_vlm_slide'}
exec(compile(src[:src.index('panel_configs = ')], SRC, 'exec'), ns)   # data, colors, draw_panel, draw_between_panel
plt = ns['plt']

Y_MAX = 122   # shared y-range of the two alignment panels, as in the paper figure


def save(fig, name):
    for a in fig.axes:
        a.axvline(3.5, color='gray', linewidth=1.0, linestyle=':', zorder=1)   # trained models | pretrained VLMs
        for c in a.containers:
            if hasattr(c, 'lines') and c.lines and c.lines[1]:   # errorbar caps sized for the slide
                for cap in c.lines[1]:
                    cap.set_markersize(5)
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out)


def new_fig():
    fig = plt.figure(figsize=(13.333, 7.5), dpi=300, facecolor='white')
    ns['fig'] = fig   # draw_* functions read the module-level `fig`
    return fig


# Describe and Social side by side, shared y-axis
ns.update(FS_TITLE=30, FS_TICK=20, FS_LABEL=22, FS_LEGEND=18, TITLE_PAD=12)
fig = new_fig()
axes = [fig.add_axes(r) for r in ([0.12, 0.27, 0.40, 0.62], [0.56, 0.27, 0.40, 0.62])]
for k, (a, task, ceil_z) in enumerate([(axes[0], 'describe', ns['CEIL_DES']), (axes[1], 'social', ns['CEIL_SOC'])]):
    line = ns['draw_panel'](a, task, ceil_z, '', show_ylabel=(k == 0))
    a.set_ylim(0, Y_MAX)
ns['add_ceil_legend'](axes[1], line)
save(fig, 'fig1_vlm_describe_social_slide.png')

# Between-task on its own slide
ns.update(FS_TITLE=34, FS_TICK=22, FS_LABEL=24)
fig = new_fig()
ns['draw_between_panel'](fig.add_axes([0.15, 0.27, 0.80, 0.62]), '')
save(fig, 'fig1_vlm_between_slide.png')
