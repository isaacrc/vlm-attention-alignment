"""
slides_encoding_methods.py -- methods slide for the NSD encoding analysis, as an editable PowerPoint slide
(13.333 x 7.5 in) after Isaac's sketch (slides/IMG_1325.HEIC). Top row: one subject's attention maps
trained and tested against brain responses by nested cross-validated ridge, giving held-out R^2, averaged
within region and then across subjects. Bottom row: the same fit with the image labels shuffled, giving
the permutation floor.
Images (all in slides/encoding_methods/, so they can be copied on their own):
  attn_map_{1,2,3}.png   the CNN-LSTM column of fig2_describe_slide.png (illustrative; the encoding model
                         is fit on maps of the NSD shared images, not these)
  brain_rh_lateral.png    the RH lateral panel of fig3_neural_rois_slide.png
Every shape and text box is native PowerPoint, so it can be edited. A PNG preview of the same layout is
drawn with matplotlib (no LibreOffice on Della), so check spacing in PowerPoint itself.
Needs python-pptx (not in cans-env or lmw): run with the scratch venv that has it.
Writes slides/encoding_methods/encoding_methods_slide.{pptx,png}
"""
import os
from PIL import Image

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
D = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides', 'encoding_methods')
ROI = [('Early visual', '#dfc27d'), ('Ventral', '#01665e'), ('Lateral', '#702c8c'), ('Parietal', '#b2182b')]
INK, GREY = '#1f2a36', '#555555'

# layout, in inches: ('img', path, x, y, w) | ('box', x, y, w, h) | ('arrow', x0, y0, x1, y1, dashed)
#                    ('text', s, x, y, w, h, size, bold, colour, align, rot) | ('line', x0, y0, x1, y1)
E = [('text', 'Regressing attention weights on NSD data', 0.5, 0.1, 12.33, 0.6, 30, True, INK, 'c', 0)]
W, BW = 1.45, 2.25                       # attention-map size, brain width
BH = BW * 570 / 780
for row, (y, label, order) in enumerate([(1.55, 'Sub 01', (1, 2, 3)), (4.85, 'Permutation floor', (3, 1, 2))]):
    yc = y + 0.2 + W / 2                 # centre line of the row
    E.append(('text', label, -0.45 if row else 0.0, yc - 0.25, 1.9 if row else 1.0, 0.5, 18, True, INK, 'c', 270))
    for k, m in enumerate(order):
        E.append(('img', os.path.join(D, f'attn_map_{m}.png'), 0.95 + 0.2 * k, y + 0.2 * k, W))
    bx = 5.15
    E.append(('arrow', 3.15, yc, 4.95, yc, False))
    E.append(('box', bx, yc - BH / 2 - 0.2, BW + 0.3, BH + 0.4))
    E.append(('img', os.path.join(D, 'brain_rh_lateral.png'), bx + 0.15, yc - BH / 2, BW))
    if row == 0:
        E.append(('text', 'Attention maps\n(14×14 = 196 features)', 0.8, y - 0.8, 2.4, 0.65, 15, True, INK, 'c', 0))
        E.append(('text', 'Train', 3.15, yc - 0.55, 1.8, 0.45, 20, True, INK, 'c', 0))
        E.append(('text', '• 5 outer folds\n• 4 inner folds\n• CV ridge penalty\n  (α = 10¹ to 10¹⁰)',
                  3.15, yc + 0.1, 2.0, 1.3, 15, False, INK, 'l', 0))
        E.append(('text', 'Brain responses\n(~800 training images)', bx, yc - BH / 2 - 0.95, BW + 0.3, 0.65, 15, True, INK, 'c', 0))
        E.append(('arrow', 7.9, yc, 9.25, yc, False))
        E.append(('text', 'Test', 7.9, yc - 0.55, 1.35, 0.45, 20, True, INK, 'c', 0))
        E.append(('text', 'held-out fold', 7.9, yc + 0.05, 1.35, 0.4, 14, False, GREY, 'c', 0))
        E.append(('box', 9.45, yc - BH / 2 - 0.2, BW + 0.3, BH + 0.4))
        E.append(('img', os.path.join(D, 'brain_rh_lateral.png'), 9.6, yc - BH / 2, BW))
        E.append(('text', 'Brain responses\n(~200 held-out images)', 9.45, yc - BH / 2 - 0.95, BW + 0.3, 0.65, 15, True, INK, 'c', 0))
        E.append(('text', 'R²', 12.1, yc - 0.4, 1.0, 0.8, 36, True, INK, 'c', 0))
        E.append(('arrow', 12.6, yc + 0.45, 12.6, 3.85, False))
        E.append(('text', 'Average within\neach region', 11.45, 3.9, 1.8, 0.65, 15, True, INK, 'c', 0))
        E.append(('arrow', 12.35, 4.6, 12.35, 5.0, False))
        E.append(('text', 'Average across\n8 subjects', 11.45, 5.05, 1.8, 0.65, 15, True, INK, 'c', 0))
        E.append(('text', '⋮', 1.55, 3.55, 0.7, 0.6, 26, True, GREY, 'c', 0))
        E.append(('text', 'Sub 08', 0.0, 3.45, 1.0, 0.5, 16, False, GREY, 'c', 270))
    else:
        E.append(('text', 'Shuffle image labels', 0.85, y + 0.4 + W + 0.05, 2.5, 0.4, 15, True, INK, 'c', 0))
        E.append(('text', 'same nested CV', 3.15, yc + 0.05, 1.8, 0.4, 14, False, GREY, 'c', 0))
        E.append(('arrow', 7.9, yc, 8.6, yc, False))
        E.append(('text', 'R²\npermutation\nfloor', 8.65, yc - 0.6, 2.2, 1.2, 18, True, INK, 'c', 0))
        E.append(('arrow', 10.85, yc, 11.45, yc, False))
        E.append(('arrow', 12.35, 5.75, 12.35, 6.1, False))
        E.append(('text', 'Above floor?\nPaired t-test, Holm\nacross regions', 11.25, 6.1, 2.2, 0.95, 14, False, INK, 'c', 0))
E.append(('line', 0.4, 4.3, 10.9, 4.3))
for k, (name, col) in enumerate(ROI):   # ROI legend under the bottom brain
    x = 4.85 + [0, 1.55, 2.95, 4.3][k]
    E.append(('swatch', x, 7.0, col))
    E.append(('text', name, x + 0.3, 6.88, 1.3, 0.4, 14, False, INK, 'l', 0))


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
        elif e[0] == 'box':
            b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, *(Inches(v) for v in e[1:5]))
            b.adjustments[0] = 0.06; b.fill.background(); b.line.color.rgb = rgb(INK); b.line.width = Pt(2); b.shadow.inherit = False
        elif e[0] == 'swatch':
            b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(e[1]), Inches(e[2]), Inches(0.22), Inches(0.16))
            b.fill.solid(); b.fill.fore_color.rgb = rgb(e[3]); b.line.color.rgb = rgb(INK); b.line.width = Pt(0.5); b.shadow.inherit = False
        elif e[0] in ('arrow', 'line'):
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, *(Inches(v) for v in e[1:5]))
            c.line.color.rgb = rgb(INK if e[0] == 'arrow' else '#999999')
            c.line.width = Pt(2.5 if e[0] == 'arrow' else 1)
            ln = c.line._get_or_add_ln()
            if e[0] == 'line' or e[5]:
                ln.append(parse_xml('<a:prstDash xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="dash"/>'))
            if e[0] == 'arrow':
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
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig = plt.figure(figsize=(13.333, 7.5)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.333); ax.set_ylim(7.5, 0); ax.axis('off')
    for e in E:
        if e[0] == 'img':
            im = Image.open(e[1]); h = e[4] * im.height / im.width
            ax.imshow(im, extent=(e[2], e[2] + e[4], e[3] + h, e[3]))
        elif e[0] == 'box':
            ax.add_patch(FancyBboxPatch((e[1], e[2]), e[3], e[4], boxstyle='round,pad=0,rounding_size=0.12', fill=False, ec=INK, lw=2))
        elif e[0] == 'swatch':
            ax.add_patch(Rectangle((e[1], e[2]), 0.22, 0.16, fc=e[3], ec=INK, lw=0.5))
        elif e[0] == 'arrow':
            ax.annotate('', (e[3], e[4]), (e[1], e[2]), arrowprops=dict(arrowstyle='-|>', color=INK, lw=2.5, ls='--' if e[5] else '-', mutation_scale=20))
        elif e[0] == 'line':
            ax.plot([e[1], e[3]], [e[2], e[4]], color='#999999', lw=1, ls='--')
        elif e[0] == 'text':
            _, txt, x, y, w, h, size, bold, col, align, rot = e
            ax.text(x + (w / 2 if align == 'c' else 0), y + h / 2, txt, fontsize=size, fontweight='bold' if bold else 'normal', color=col,
                    ha='center' if align == 'c' else 'left', va='center', rotation=-rot % 360 if rot else 0, family=['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans'])
    fig.savefig(path, dpi=150, facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    to_pptx(os.path.join(D, 'encoding_methods_slide.pptx'))
    to_png(os.path.join(D, 'encoding_methods_slide.png'))
    print('wrote', D)
