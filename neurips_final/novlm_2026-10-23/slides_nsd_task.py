"""
slides_nsd_task.py -- the NSD task, as an editable PowerPoint slide (13.333 x 7.5 in) in the style of
slides_encoding_methods.py. Trials run left to right on one row: each image is shown for 3 s on a grey screen with
a central fixation dot, followed by 1 s of blank screen, and participants press Yes or No to "Have you
seen this image before?" (NSD's continuous recognition task; Allen et al., 2022). The last trial repeats
the first image, so its answer is Yes. Facts box on the right: the NSD design and the part of it we used.
Images: three NSD shared-1000 stimuli (rebuttal/task8_nsd/stimuli), copied into slides/nsd_task/ so they
can be copied on their own. PNG preview drawn with matplotlib (no LibreOffice on Della).
Needs python-pptx (not in cans-env or lmw).
Writes slides/nsd_task/nsd_task_slide.{pptx,png}
"""
import os, shutil
from PIL import Image

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
D = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides', 'nsd_task')
STIM = os.path.join(ROOT, 'rebuttal', 'task8_nsd', 'stimuli')
INK, GREY, SCREEN, ACCENT = '#1f2a36', '#555555', '#808080', '#2b6cb0'
os.makedirs(D, exist_ok=True)
IMG = {}
for name, nsd in (('cat', 3181), ('frisbee', 3171), ('bear', 3489)):
    IMG[name] = os.path.join(D, f'nsd_{nsd:05d}_{name}.png')
    shutil.copy(os.path.join(STIM, f'{nsd:05d}.png'), IMG[name])

# layout, in inches: ('img', path, x, y, w) | ('rect', x, y, w, h, fill, edge, rounded)
#   ('dot', cx, cy, d, fill) | ('arrow', x0, y0, x1, y1, dashed) | ('text', s, x, y, w, h, size, bold, colour, align, rot)
E = [('text', 'The NSD task: continuous recognition', 0.5, 0.1, 12.33, 0.6, 30, True, INK, 'c', 0),
     ('text', 'On every trial: “Have you seen this image before, at any point in the experiment?”',
      0.5, 0.78, 12.33, 0.45, 18, False, INK, 'c', 0)]
F = 1.65                                # screen size
X0, Y0 = 0.4, 2.85                      # all trials on one row
XS = [0, 1.9, 3.8, 5.6, 7.25]            # x offset of each slot (the 'later' slot is text only)
TRIALS = [('cat', '3 s', 'No'), (None, '1 s', None), ('frisbee', '3 s', 'No'), ('later', None, None), ('cat', '3 s', 'Yes')]
for k, (img, dur, resp) in enumerate(TRIALS):
    x, y = X0 + XS[k], Y0
    if img == 'later':
        E.append(('text', '…\nweeks to\nmonths later', x + 0.03, y + F / 2 - 0.55, 1.3, 1.1, 14, True, GREY, 'c', 0))
        continue
    E.append(('rect', x, y, F, F, SCREEN, INK, False))
    if img:
        E.append(('img', IMG[img], x + 0.25, y + 0.25, F - 0.5))
    E.append(('dot', x + F / 2, y + F / 2, 0.09, '#d62728'))
    E.append(('text', dur, x + F - 0.6, y + F + 0.05, 0.6, 0.35, 15, True, INK, 'c', 0))
    if resp:
        on = resp == 'Yes'
        E.append(('rect', x, y + F + 0.07, 0.62, 0.32, ACCENT if on else '#ffffff', ACCENT, True))
        E.append(('text', resp, x, y + F + 0.07, 0.62, 0.32, 14, True, '#ffffff' if on else ACCENT, 'c', 0))
E.append(('arrow', X0, Y0 + F + 0.75, X0 + XS[4] + F, Y0 + F + 0.75, False))   # time
E.append(('text', 'time', X0 + XS[4] / 2, Y0 + F + 0.85, 1.6, 0.4, 16, True, INK, 'c', 0))
E.append(('text', 'Repeat of trial 1', X0 + XS[4], Y0 - 0.42, F, 0.35, 14, True, GREY, 'c', 0))

FX, FY = 9.6, 1.55                      # facts box
E.append(('rect', FX, FY, 3.6, 4.55, '#f4f6f8', INK, True))
E.append(('text', 'NSD design', FX + 0.2, FY + 0.15, 3.2, 0.45, 18, True, INK, 'l', 0))
E.append(('text', '• 8 participants, 7T fMRI\n• 30 to 40 sessions over a year\n• 750 images per session\n'
                  '• Each image shown 3 times\n• Fixate the central dot\n• Yes / No by button press',
          FX + 0.2, FY + 0.65, 3.2, 1.9, 15, False, INK, 'l', 0))
E.append(('text', 'What we used', FX + 0.2, FY + 2.7, 3.2, 0.45, 18, True, INK, 'l', 0))
E.append(('text', '• Shared images seen by all 8\n  (907 to 1,000 per participant)\n• Responses averaged over\n  the 3 repetitions',
          FX + 0.2, FY + 3.2, 3.2, 1.2, 15, False, INK, 'l', 0))


def to_pptx(path):
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    add_slide(prs)
    prs.save(path)


def add_slide(prs):
    """Draw E on a new blank slide of prs (also used by slides_nsd_methods_deck.py)."""
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.oxml import parse_xml
    rgb = lambda h: RGBColor.from_string(h.lstrip('#'))
    s = prs.slides.add_slide(prs.slide_layouts[6])
    for e in E:
        if e[0] == 'img':
            s.shapes.add_picture(e[1], Inches(e[2]), Inches(e[3]), width=Inches(e[4]))
        elif e[0] == 'rect':
            b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if e[7] else MSO_SHAPE.RECTANGLE, *(Inches(v) for v in e[1:5]))
            if e[7]:
                b.adjustments[0] = 0.15 if e[4] < 0.5 else 0.05
            b.fill.solid(); b.fill.fore_color.rgb = rgb(e[5])
            b.line.color.rgb = rgb(e[6]); b.line.width = Pt(1.5); b.shadow.inherit = False
        elif e[0] == 'dot':
            b = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(e[1] - e[3] / 2), Inches(e[2] - e[3] / 2), Inches(e[3]), Inches(e[3]))
            b.fill.solid(); b.fill.fore_color.rgb = rgb(e[4]); b.line.color.rgb = rgb('#000000'); b.line.width = Pt(0.75); b.shadow.inherit = False
        elif e[0] == 'arrow':
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, *(Inches(v) for v in e[1:5]))
            c.line.color.rgb = rgb(INK); c.line.width = Pt(2.5)
            ln = c.line._get_or_add_ln()
            if e[5]:
                ln.append(parse_xml('<a:prstDash xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="dash"/>'))
            ln.append(parse_xml('<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle" w="med" len="med"/>'))
        elif e[0] == 'text':
            _, txt, x, y, w, h, size, bold, col, align, rot = e
            t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
            t.rotation = rot
            tf = t.text_frame; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
                setattr(tf, m, 0)
            for i, line in enumerate(txt.split('\n')):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.alignment = {'c': PP_ALIGN.CENTER, 'l': PP_ALIGN.LEFT}[align]
                r = p.add_run(); r.text = line
                r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, rgb(col), 'Helvetica Neue'


def to_png(path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
    fig = plt.figure(figsize=(13.333, 7.5)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.333); ax.set_ylim(7.5, 0); ax.axis('off')
    for e in E:
        if e[0] == 'img':
            im = Image.open(e[1]); h = e[4] * im.height / im.width
            ax.imshow(im, extent=(e[2], e[2] + e[4], e[3] + h, e[3]), zorder=2)
        elif e[0] == 'rect':
            ax.add_patch(FancyBboxPatch((e[1], e[2]), e[3], e[4], boxstyle=f'round,pad=0,rounding_size={0.08 if e[7] else 0}',
                                        fc=e[5], ec=e[6], lw=1.5, zorder=1))
        elif e[0] == 'dot':
            ax.add_patch(Circle((e[1], e[2]), e[3] / 2, fc=e[4], ec='black', lw=0.75, zorder=3))
        elif e[0] == 'arrow':
            ax.annotate('', (e[3], e[4]), (e[1], e[2]), arrowprops=dict(arrowstyle='-|>', color=INK, lw=2.5, ls='--' if e[5] else '-', mutation_scale=20))
        elif e[0] == 'text':
            _, txt, x, y, w, h, size, bold, col, align, rot = e
            ax.text(x + (w / 2 if align == 'c' else 0), y + h / 2, txt, fontsize=size, fontweight='bold' if bold else 'normal', color=col,
                    ha='center' if align == 'c' else 'left', va='center', rotation=-rot % 360 if rot else 0, family=['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans'], zorder=4)
    fig.savefig(path, dpi=150, facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    to_pptx(os.path.join(D, 'nsd_task_slide.pptx'))
    to_png(os.path.join(D, 'nsd_task_slide.png'))
    print('wrote', D)
