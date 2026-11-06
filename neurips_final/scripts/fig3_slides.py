"""
fig3_slides.py
--------------
Split Fig 3 (alignment on CV heads) into one 16:9 PowerPoint slide per panel:
Describe Task, Social Task, Between-task. Reuses the data loading and panel
drawing from rebuttal/plots/plot_fig_head_selection_corrected.py (everything
above its figure assembly), with font sizes scaled for a 13.33 x 7.5 in slide.

Outputs:
  neurips_final/final_figures/slides/fig3_{describe,social,between}_slide.png

QWEN_NOTHINK=1 reads the thinking-off Qwen3.5 results (neurips_final/qwen35_nothink/
rerun_behavioral.py) instead, writes the slides to final_figures/qwen35_nothink/slides/,
and also renders the full paper-layout figure as final_figures/qwen35_nothink/fig3_cvheads_qwen_nothink.png.
"""
import os

ROOT    = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC     = os.path.join(ROOT, 'rebuttal', 'plots', 'plot_fig_head_selection_corrected.py')
NOTHINK = os.environ.get('QWEN_NOTHINK') == '1'
# MOLMO_BILINEAR=1 (with QWEN_NOTHINK=1): read the Molmo-bilinear rerun instead
NT_DIR  = os.path.join(ROOT, 'neurips_final', 'final_figures',
                       os.environ.get('MOLMO_BILINEAR_DIR', 'molmo_bilinear') if os.environ.get('MOLMO_BILINEAR') == '1' else 'qwen35_nothink')
OUT     = os.path.join(NT_DIR, 'slides') if NOTHINK else os.path.join(ROOT, 'neurips_final', 'final_figures', 'slides')
os.makedirs(OUT, exist_ok=True)

src = open(SRC).read()
if NOTHINK:
    NT_RES = os.path.join(NT_DIR, 'results')
    for old, new in [("os.path.join(DATA, 'alignment_summary_cv_honest.csv')", repr(os.path.join(NT_RES, 'alignment_summary_cv_honest.csv'))),
                     ("os.path.join(T1, 'selected_heads.json')", repr(os.path.join(NT_RES, 'selected_heads.json'))),
                     # ceiling of the run being plotted (leave-one-out in the molmo_bilinear run)
                     ("os.path.join(DATA, 'ceilings.json')", repr(os.path.join(NT_RES, 'ceilings.json'))),
                     # Panel C: trained Transformers on their cross-validated heads (between_task_cvheads.py)
                     ("os.path.join(T2, 'model_between_task.csv')", repr(os.path.join(NT_RES, 'model_between_task_fig1.csv'))),
                     ("os.path.join(HERE, 'head_selection', 'fig_head_selection_corrected.png')",
                      repr(os.path.join(NT_DIR, 'fig3_cvheads_qwen_nothink.png')))]:
        assert old in src, old
        src = src.replace(old, new)
    exec(src, {'__file__': SRC, '__name__': 'fig3_src_full'})   # paper-layout figure
ns = {'__file__': SRC, '__name__': 'fig3_src'}
exec(src[:src.index('panel_configs = ')], ns)   # data, colors, draw_panel, draw_between_panel
plt = ns['plt']

ns.update(FS_TITLE=34, FS_TICK=22, FS_LABEL=24, FS_LEGEND=20, FS_LETTER=34, TITLE_PAD=12)
W, H_IN = 13.333, 7.5
Y_MAX = 122   # shared y-range of panels A and B in the paper figure

def new_fig():
    fig = plt.figure(figsize=(W, H_IN), dpi=300, facecolor='white')
    ns['fig'] = fig   # draw_* functions read the module-level `fig`
    return fig, fig.add_axes([0.15, 0.27, 0.80, 0.62])

for task, ceil_z in (('describe', ns['CEIL_DES']), ('social', ns['CEIL_SOC'])):
    fig, ax = new_fig()
    line = ns['draw_panel'](ax, task, ceil_z, '', show_ylabel=True)
    ax.tick_params(axis='y', labelleft=True)
    ax.set_ylim(0, Y_MAX)
    ns['add_ceil_legend'](ax, line)
    out = os.path.join(OUT, f'fig3_{task}_slide.png')
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out)

fig, ax = new_fig()
ns['draw_between_panel'](ax, '')
out = os.path.join(OUT, 'fig3_between_slide.png')
fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
print('wrote', out)
