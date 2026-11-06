"""
build_deck.py -- training-dynamics deck from the CANS architecture-manipulation experiment,
framed against the NeurIPS 2026 camera-ready. Writes arch_training_dynamics.pptx (16:9).

Run: .venv/bin/python build_deck.py
"""
import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, 'img')
OUT = os.path.join(HERE, 'arch_training_dynamics.pptx')

FONT = 'Helvetica Neue'
INK = RGBColor(0x1C, 0x24, 0x30)
MUTED = RGBColor(0x5B, 0x66, 0x75)
ORANGE = RGBColor(0xD5, 0x5E, 0x00)
BLUE = RGBColor(0x00, 0x72, 0xB2)
GREEN = RGBColor(0x3A, 0x9A, 0x47)
TINT = RGBColor(0xF1, 0xF3, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x1C, 0x24, 0x30)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW = 13.333


def _run_style(run, size, color, bold=False, italic=False):
    f = run.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = color


def text(slide, x, y, w, h, paras, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=6, bullets=False, name=None):
    """paras: list of str or list of [(text, {opts})] runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after)
        runs = p if isinstance(p, list) else [(p, {})]
        if bullets:
            runs = [('•  ', {'color': MUTED})] + runs
        for t, o in runs:
            r = para.add_run()
            r.text = t
            _run_style(r, o.get('size', size), o.get('color', color),
                       o.get('bold', bold), o.get('italic', False))
    return tb


def title(slide, t, sub=None):
    text(slide, 0.6, 0.38, 12.1, 0.7, [t], size=30, bold=True, name='Title')
    if sub:
        text(slide, 0.6, 1.08, 12.1, 0.4, [sub], size=15, color=MUTED, name='Subtitle')


def footer(slide, n):
    text(slide, 0.6, 7.05, 8, 0.3, ['Architecture manipulation · training dynamics'],
         size=10, color=MUTED)
    text(slide, 11.73, 7.05, 1.0, 0.3, [str(n)], size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def image(slide, fname, x, y, w, h, align='center'):
    path = os.path.join(IMG, fname) if not os.path.isabs(fname) else fname
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    dx = x + (w - dw) / 2 if align == 'center' else x
    dy = y + (h - dh) / 2
    return slide.shapes.add_picture(path, Inches(dx), Inches(dy), Inches(dw), Inches(dh))


def box(slide, x, y, w, h, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def badge(slide, x, y, label, fill=INK, d=0.46):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    _run_style(r, 15, WHITE, bold=True)


def new_slide(dark=False):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = DARK if dark else WHITE
    return s


def fig_slide(n, t, fname, bullets, caption, notes, sub=None, img_w=8.4):
    """Figure on the left, takeaway column on the right."""
    s = new_slide()
    title(s, t, sub)
    top = 1.6 if sub else 1.35
    image(s, fname, 0.6, top, img_w, 6.85 - top - 0.55)
    cx = 0.6 + img_w + 0.35
    cw = SW - 0.6 - cx
    text(s, cx, top + 0.05, cw, 4.6, bullets, size=14, bullets=True, space_after=10)
    text(s, 0.6, 6.42, img_w, 0.55, [caption], size=10, color=MUTED)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = notes
    return s


# ── 1. Title ──────────────────────────────────────────────────────────────────
s = new_slide(dark=True)
text(s, 0.9, 2.2, 11.5, 1.6, ['How alignment with human gaze develops during training'],
     size=40, bold=True, color=WHITE)
text(s, 0.9, 3.85, 11.5, 0.9,
     ['Training dynamics of the four NeurIPS architectures, and what they add to the paper'],
     size=20, color=RGBColor(0xC9, 0xD2, 0xDE))
text(s, 0.9, 5.6, 11.5, 0.4, ['CANS architecture-manipulation experiment · 29 October 2026'],
     size=14, color=RGBColor(0x9A, 0xA6, 0xB5))
s.notes_slide.notes_text_frame.text = (
    'Source experiment: /scratch/gpfs/GRAZIANO/CANS/architecture_manipulation. Same four models, '
    'curriculum, 200 COCO images and gaze data as the NeurIPS 2026 paper (submission 29907).')

# ── 2. Takeaways ──────────────────────────────────────────────────────────────
s = new_slide()
title(s, 'Four things the training curves add to the paper')
cards = [
    ('1', 'Captioning skill is not the gap',
     'All four models reach 94–100% of the human caption-similarity ceiling by Social '
     'finetuning. Their gaze alignment still spans 57–89% of the gaze ceiling.'),
    ('2', 'LSTM alignment arrives early',
     'ViT-LSTM sits at r ≈ 0.20 after its first Describe epoch, with the encoder frozen, and '
     'barely moves. CNN-LSTM is the only model whose alignment climbs (0.11 → 0.24).'),
    ('3', 'Encoder training lifts Social alignment',
     'Unfrozen beats frozen at the end of Social finetuning for CNN-LSTM (0.24 vs. 0.19), '
     'CNN-Transformer (0.20 vs. 0.18) and ViT-Transformer (0.16 vs. 0.11). ViT-LSTM is unchanged.'),
    ('4', 'The best Transformer heads peak early',
     'CNN L4H4 reaches r = 0.23 on Describe gaze after one epoch, above both LSTMs at that point. '
     'Captioning training erodes it to 0.14; ViT L4H8 follows the same arc (0.19 → 0.09).'),
]
for i, (num, head, body) in enumerate(cards):
    cx = 0.6 + (i % 2) * 6.15
    cy = 1.45 + (i // 2) * 2.7
    box(s, cx, cy, 5.95, 2.45)
    badge(s, cx + 0.3, cy + 0.3, num)
    text(s, cx + 0.95, cy + 0.3, 4.8, 0.5, [head], size=18, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, cx + 0.95, cy + 0.9, 4.75, 1.45, [body], size=14, color=INK, name=f'card{num}')
footer(s, 2)
s.notes_slide.notes_text_frame.text = (
    'Numbers: caption cosine and gaze alignment at the checkpoints the paper analyzes (Social task). '
    'Gaze percentages are the camera-ready values (CNN-LSTM 88.8%, ViT-LSTM 81.3%, CNN-Transformer '
    '73.8%, ViT-Transformer 56.6%). Per-epoch values: CANS/architecture_manipulation/summary_table.csv '
    'and CANS/group_analysis/learning_to_attend/**/correlations*/.')

# ── 3. Design ─────────────────────────────────────────────────────────────────
s = new_slide()
title(s, 'Same 2 × 2 models and curriculum as the paper')
text(s, 0.6, 1.4, 5.2, 0.4, ['Architectures'], size=18, bold=True)
text(s, 2.0, 1.95, 1.8, 0.35, ['LSTM decoder'], size=13, color=MUTED, align=PP_ALIGN.CENTER)
text(s, 3.95, 1.95, 1.8, 0.35, ['Transformer decoder'], size=13, color=MUTED, align=PP_ALIGN.CENTER)
for r, enc in enumerate(['CNN\nResNet-101', 'ViT\nViT-B/16']):
    text(s, 0.6, 2.5 + r * 1.55, 1.3, 1.3, enc.split('\n'), size=13, color=MUTED,
         anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    for c, dec in enumerate(['LSTM', 'Transformer']):
        b = box(s, 2.0 + c * 1.95, 2.4 + r * 1.55, 1.8, 1.4,
                fill=RGBColor(0xE3, 0xEE, 0xF7) if c == 0 else TINT)
        lbl = f"{enc.split(chr(10))[0]}-{dec}"
        sub = '1 attention map' if c == 0 else '4 layers × 8 heads'
        text(s, 2.0 + c * 1.95, 2.55 + r * 1.55, 1.8, 1.1,
             [[(lbl, {'bold': True, 'size': 14})], [(sub, {'size': 12, 'color': MUTED})]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=2)
text(s, 0.6, 5.65, 5.2, 0.9,
     ['Gaze: 200 COCO images; Describe (N = 25 here, 22 in the paper) and Social (N = 27) tasks. '
      'Phases 1–2 are scored against Describe gaze, Social phases against Social gaze.'],
     size=12, color=MUTED)

text(s, 6.6, 1.4, 6.1, 0.4, ['Curriculum'], size=18, bold=True)
steps = [('Describe pretrain', 'COCO captions · encoder frozen', 2.0),
         ('Describe finetune', 'COCO captions · encoder unfrozen', 3.2)]
for lbl, sub, yy in steps:
    box(s, 6.6, yy, 3.0, 0.95, fill=RGBColor(0xFB, 0xEA, 0xDE))
    text(s, 6.7, yy + 0.08, 2.8, 0.8, [[(lbl, {'bold': True})], [(sub, {'size': 11, 'color': MUTED})]],
         size=14, anchor=MSO_ANCHOR.MIDDLE, space_after=2)
for yy in (2.95, ):
    a = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(7.95), Inches(yy), Inches(0.3), Inches(0.25))
    a.fill.solid(); a.fill.fore_color.rgb = MUTED; a.line.fill.background(); a.shadow.inherit = False
for i, (lbl, sub) in enumerate([('Social, encoder frozen', 'decoder only'),
                                ('Social, encoder unfrozen', 'paper’s Social checkpoint')]):
    yy = 4.55 + i * 1.15
    box(s, 10.0, yy, 2.73, 0.95, fill=RGBColor(0xDE, 0xEC, 0xF7))
    text(s, 10.08, yy + 0.08, 2.6, 0.8, [[(lbl, {'bold': True, 'size': 13})],
                                        [(sub, {'size': 11, 'color': MUTED})]],
         anchor=MSO_ANCHOR.MIDDLE, space_after=2)
    con = s.shapes.add_connector(1, Inches(9.6), Inches(3.67), Inches(10.0), Inches(yy + 0.47))
    con.line.color.rgb = MUTED
    con.line.width = Pt(1.5)
text(s, 6.6, 4.5, 3.2, 1.9,
     ['Both Social runs start from the best Describe-finetune checkpoint (training scripts load '
      'checkpoints/phase2). They are parallel branches, not sequential phases.'],
     size=12, color=MUTED)
footer(s, 3)
s.notes_slide.notes_text_frame.text = (
    'Verified in {CNN_Transformer,ViT_Transformer,ViT_LSTM}/train_social_{frozen,unfrozen}.py: '
    'both read P2_DIR = checkpoints/phase2. The camera-ready Methods (main.tex l.183-184) say '
    'Phase 4 (unfrozen) was initialized from Phase 3 (frozen) -- see the last slide.')

# ── 4. panel_combined ─────────────────────────────────────────────────────────
fig_slide(4, 'Captions improve everywhere; alignment mostly doesn\u2019t',
          'panel_combined.png',
          ['Green (caption cosine) climbs through every phase and ends at the human ceiling in all four '
           'models.',
           'Blue (gaze alignment) climbs only for CNN-LSTM, which nearly doubles from Describe pretrain to '
           'Social finetune.',
           'ViT-LSTM is flat near 0.20 from the first epoch; the Transformers wander around 0.10–0.17.',
           'Unfreezing the encoder clearly helps CNN-LSTM here; with the paper’s Social heads it helps '
           'the Transformers too (slide 10).'],
          'From the CANS draft (“Alignment During Training”). Transformer curves use the '
          'Describe heads L3H5 / L3H8 in every phase, so their Social segments understate the paper’s '
          'Social heads (L4H4 / L4H8). Dashed grey = human–human ceiling.',
          'Source: CANS/architecture_manipulation/plots/mega_figures/panel_combined.png (also '
          'CANS/manuscript/figures/). Generated by panel_combined.py on 2026-04-23. Transformer spatial '
          'r read from head_attention_analysis_L3.csv (the corrected 2026-04-15 recompute).',
          img_w=6.9)

# ── 5. native chart: caption vs gaze ──────────────────────────────────────────
s = new_slide()
title(s, 'Captioning is at ceiling; gaze alignment is not',
      'Social task, checkpoints analyzed in the paper, % of the human–human ceiling')
cd = CategoryChartData()
cd.categories = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer']
cd.add_series('Caption similarity', (99.6, 98.6, 98.4, 94.2))
cd.add_series('Gaze alignment', (88.8, 81.3, 73.8, 56.6))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.65),
                        Inches(8.2), Inches(5.0), cd)
ch = gf.chart
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(13)
ch.legend.font.name = FONT
ch.font.name = FONT
ch.font.size = Pt(12)
ch.font.color.rgb = MUTED
va = ch.value_axis
va.maximum_scale, va.minimum_scale = 110, 0
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor(0xE2, 0xE6, 0xEB)
va.format.line.fill.background()
va.tick_labels.font.size = Pt(11)
ch.category_axis.tick_labels.font.size = Pt(13)
ch.category_axis.tick_labels.font.color.rgb = INK
ch.category_axis.format.line.color.rgb = RGBColor(0xB0, 0xB8, 0xC2)
plot = ch.plots[0]
plot.gap_width = 70
plot.overlap = -10
for ser, col in zip(plot.series, (GREEN, BLUE)):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
    ser.data_labels.show_value = True
    ser.data_labels.number_format = '0'
    ser.data_labels.number_format_is_linked = False
    ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    ser.data_labels.font.size = Pt(12)
    ser.data_labels.font.color.rgb = INK
text(s, 9.2, 1.75, 3.5, 4.5,
     ['Caption similarity varies by 5 points across models; gaze alignment varies by 32.',
      'The weakest captioner (ViT-Transformer, 94%) is also the least aligned, but CNN-Transformer '
      'captions as well as the LSTMs and still trails them by 8–15 points.',
      'This supports reading the decoder effect as a difference in where the model looks, not in how '
      'well it captions.'],
     size=14, bullets=True, space_after=10)
text(s, 0.6, 6.62, 8.2, 0.35,
     ['Caption cosine vs. human Social captions (ceiling 0.507); gaze % from the camera-ready.'],
     size=10, color=MUTED)
footer(s, 5)
s.notes_slide.notes_text_frame.text = (
    'Caption cosine at paper checkpoints: CNN-LSTM 0.505 (CANS label, learning_to_attend final model), '
    'ViT-LSTM 0.500 (social_unfrozen E1), CNN-Transformer 0.499 (social_unfrozen E3), ViT-Transformer '
    '0.478 (social_unfrozen E1); divided by the human-human caption ceiling 0.5069. Gaze %: '
    'camera-ready main.tex l.236 (cross-validated Social heads for the Transformers).')

# ── 6. attention maps across phases ───────────────────────────────────────────
s = new_slide()
title(s, 'What the attention maps look like at each phase')
image(s, 'fig_traj_attn_human_combined.png', 0.6, 1.3, 12.13, 5.15)
text(s, 0.6, 6.5, 12.1, 0.5,
     ['Top: gaze alignment by epoch. Rows: attention at the end of each phase on three Social images, with '
      'Social-task human gaze at bottom. ViT-Transformer here is L3H1 (an earlier pick); the paper uses '
      'L3H8 (Describe) and L4H8 (Social).'],
     size=10, color=MUTED)
footer(s, 6)
s.notes_slide.notes_text_frame.text = (
    'Source: CANS/architecture_manipulation/manuscript_plots/fig_traj_attn_human_combined.png '
    '(plot_fig_traj_attn_human_combined.py, 2026-04-15). A ViT-H8 variant exists: '
    'fig_traj_attn_human_combined_ViT_H8.png. CNN-LSTM sharpens onto the person/hands/skateboard over '
    'training; CNN-Transformer L3H5 is already focal at Describe pretrain.')

# ── 7. Layer-3 heads ──────────────────────────────────────────────────────────
fig_slide(7, 'The Describe heads lead from the first epochs of pretraining',
          'head_attention_analysis_L3_stacked.png',
          [[('CNN-Transformer: ', {'bold': True}),
            ('Head 5 (the paper’s Describe head, L3H5) is the best Layer-3 head from epoch 5 of '
             'pretraining onward and stays on top through Social finetuning.', {})],
           [('ViT-Transformer: ', {'bold': True}),
            ('Head 1 leads in pretraining, then decays during Describe finetune; Head 8 (the paper’s '
             'L3H8) ends Describe finetune on top.', {})],
           'Each Layer-3 head keeps its rank across phases: alignment is a stable property of a head, '
           'consistent with the 5/5 fold agreement in cross-validated head selection.'],
          'Spatial Spearman r per Layer-3 head, all saved checkpoints. Describe gaze for Describe '
          'phases, Social gaze for Social phases. Corrected tie-ranking recompute (2026-04-15).',
          'Source: CANS/architecture_manipulation/manuscript_plots/head_attention_analysis_L3_stacked.png; '
          'data {CNN,ViT}_Transformer/head_attention_analysis_L3.csv. These Layer-3 values match the '
          'rebuttal head sweep (rebuttal/task1_head_selection) at the paper checkpoints.',
          img_w=6.6)

# ── 8. Layer-4 heads (rerun) ──────────────────────────────────────────────────
fig_slide(8, 'Layer-4 heads start out most aligned', 'head_attention_analysis_L4_stacked_corrected.png',
          [[('CNN-Transformer: ', {'bold': True}),
            ('Head 4 (the paper’s Social head, L4H4) leads Layer 4 at almost every checkpoint. On '
             'Describe gaze it falls from 0.226 after epoch 1 to 0.135 by the end of Describe finetune.', {})],
           [('ViT-Transformer: ', {'bold': True}),
            ('Head 8 (L4H8) leads throughout and falls from 0.19 to 0.09 over Describe training.', {})],
           'Social finetuning lifts both again (0.20 and 0.16 at the paper checkpoints). The leading head '
           'is nearly constant across 30 checkpoints, in line with the 5/5 fold agreement in head selection.'],
          'Spatial Spearman r per Layer-4 head. Recomputed for this deck with the corrected '
          'pipeline; the April Layer-4 plots in CANS predate the fix and should not be reused.',
          'Rerun: neurips_final/arch_training_deck/headL4_rerun (SLURM 16448705/16448833), a copy of '
          'CANS/architecture_manipulation/shared/head_attention_analysis.py with outputs redirected. '
          'The old CANS L4 CSVs (2026-04-10) predate the 2026-04-15 Spearman fix, which was applied to '
          'Layer 3 only; they disagree with the rebuttal head sweep (e.g. CNN L4 head ranking).',
          img_w=6.6)

# ── 9. paper heads summary ────────────────────────────────────────────────────
fig_slide(9, 'The paper’s heads over training', 'fig_neurips_head_trajectories.png',
          ['The Describe heads (orange) stay roughly flat. The Social heads (blue) start highest and '
           'decline during Describe training.',
           'After one epoch, CNN L4H4 is at r = 0.226, 86% of the Describe ceiling, ahead of CNN-LSTM '
           '(0.109) and ViT-LSTM (0.205) at the same point, while its captions are still poor '
           '(cosine 0.36).',
           'So the decoder gap at the paper’s Describe checkpoint partly reflects what captioning '
           'training does to the Transformer’s best head.'],
          'Coloured: the cross-validated heads used in the camera-ready. Grey: the other 14 heads in '
          'layers 3–4. Stars: the checkpoints the paper analyzes.',
          'Script: neurips_final/arch_training_deck/plot_neurips_head_trajectories.py. Layer 3 from the '
          'corrected CANS CSVs, Layer 4 from the rerun.',
          img_w=7.4)

# ── 10. encoder unfreezing stats ──────────────────────────────────────────────
s = new_slide()
title(s, 'Encoder finetuning helps three of four models',
      'Gaze alignment (Spearman r) at the end of each phase; Transformers read from the paper\u2019s heads')
cols = ['Describe pretrain\nenc. frozen', 'Describe finetune\nenc. unfrozen',
        'Social\nenc. frozen', 'Social\nenc. unfrozen']
for c, lbl in enumerate(cols):
    text(s, 3.3 + c * 2.4, 1.6, 2.2, 0.6, lbl.split('\n'), size=12, color=MUTED,
         align=PP_ALIGN.CENTER, space_after=0)
rows = [('CNN-LSTM', '', ['0.14', '0.22', '0.19', '0.24']),
        ('ViT-LSTM', '', ['0.20', '0.21', '0.22', '0.22']),
        ('CNN-Transformer', 'L3H5 | L4H4', ['0.14', '0.15', '0.18', '0.20']),
        ('ViT-Transformer', 'L3H8 | L4H8', ['0.11', '0.12', '0.11', '0.16'])]
for r, (name, heads, vals) in enumerate(rows):
    y = 2.3 + r * 1.05
    text(s, 0.6, y, 2.6, 0.9, [[(name, {'bold': True, 'size': 17})]] +
         ([[(heads, {'size': 11, 'color': MUTED})]] if heads else []),
         anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    for c, v in enumerate(vals):
        x = 3.3 + c * 2.4
        hi = c == 3 and r != 1
        box(s, x, y, 2.2, 0.9, fill=RGBColor(0xDE, 0xEC, 0xF7) if hi else TINT)
        text(s, x, y, 2.2, 0.9, [v], size=28, bold=True, color=BLUE if hi else INK,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
text(s, 0.6, 6.5, 12.1, 0.45,
     ['Describe columns are scored against Describe gaze, Social columns against Social gaze: compare '
      'frozen vs. unfrozen within the Social pair. Phase lengths differ (Social unfrozen: 2\u20135 epochs).'],
     size=11, color=MUTED)
footer(s, 10)
s.notes_slide.notes_text_frame.text = (
    'CNN-LSTM: P1 last 0.135, P2 last 0.219, Social frozen last 0.189 (range 0.163-0.189), Social '
    'unfrozen 0.237-0.241 (paper checkpoint 0.242). ViT-LSTM: P1 0.205 at E0, 0.196 at last; P2 last '
    '0.207; Social frozen last 0.223; Social unfrozen last 0.223 (paper checkpoint E1). Transformers: '
    'L3H5/L3H8 for Describe columns (CANS L3 CSVs), L4H4/L4H8 for Social columns (L4 rerun). Sources: '
    'learning_to_attend/correlations, final_replicated_model/correlations{,_frozen}, summary_table.csv.')

# ── 11. layer-wise maps ───────────────────────────────────────────────────────
s = new_slide()
title(s, 'Early decoder layers attend to fixed locations, not content',
      'Cross-attention averaged over heads and words, by decoder layer')
image(s, 'attn_layers_summary_CNN_Transformer.png', 0.6, 1.6, 4.6, 4.75)
image(s, 'attn_layers_summary_ViT_Transformer.png', 5.35, 1.6, 4.6, 4.75)
text(s, 0.6, 6.38, 4.6, 0.3, ['CNN-Transformer'], size=12, bold=True, align=PP_ALIGN.CENTER)
text(s, 5.35, 6.38, 4.6, 0.3, ['ViT-Transformer'], size=12, bold=True, align=PP_ALIGN.CENTER)
text(s, 10.2, 1.7, 2.55, 4.6,
     ['CNN layers 1–2 put their weight on a strip at the left image edge in every image.',
      'ViT layers 1–2 fire on a few scattered patches that do not track objects.',
      'Content-driven attention appears only in layers 3–4, which is where every cross-validated '
      'head lives and why layers 1–2 sit at the center prior in the rebuttal’s per-layer '
      'analysis.'],
     size=13, bullets=True, space_after=10)
footer(s, 11)
s.notes_slide.notes_text_frame.text = (
    'Source: CANS/architecture_manipulation/plots/attn_viz/attn_layers_summary_{CNN,ViT}_Transformer.png '
    '(2026-03-28). Per-layer alignment numbers: rebuttal/writeups/task3c_headrule_center.md and '
    'memory note on per-layer means (layers 1-2 at the center prior; CNN-Transformer L4 mean beats '
    'CNN-LSTM in Describe after center control).')

# ── 12. loss curves ───────────────────────────────────────────────────────────
s = new_slide()
title(s, 'Unfrozen Social runs overfit within a few epochs')
for i, (f, lbl) in enumerate([('loss_curve_CNN_Transformer.png', 'CNN-Transformer'),
                              ('loss_curve_ViT_Transformer.png', 'ViT-Transformer'),
                              ('loss_curve_ViT_LSTM.png', 'ViT-LSTM')]):
    x = 0.6 + (i % 2) * 6.15
    y = 1.35 + (i // 2) * 2.85
    image(s, f, x, y, 5.95, 2.75)
text(s, 6.75, 4.35, 5.95, 2.4,
     ['With the encoder unfrozen, train loss keeps falling (red) but validation BLEU peaks by epoch '
      '1–3 and then declines (CNN-Transformer 0.138 → 0.109). Frozen runs stay flat.',
      'Early stopping on BLEU picks those early checkpoints, so the paper’s Social models sit before '
      'the overfitting. CNN-LSTM curves are in the Project 1 directory.'],
     size=13, bullets=True, space_after=10)
footer(s, 12)
s.notes_slide.notes_text_frame.text = (
    'Source: CANS/architecture_manipulation/plots/loss_curves/. Epoch axis is cumulative across '
    'phases; Social unfrozen and Social frozen are plotted one after the other but both start from the '
    'Phase-2 checkpoint.')

# ── 13. implications + housekeeping ───────────────────────────────────────────
s = new_slide()
title(s, 'What this means for the camera-ready')
box(s, 0.6, 1.4, 5.95, 4.6)
text(s, 0.9, 1.6, 5.4, 0.45, ['Results that bear on the paper'], size=18, bold=True)
text(s, 0.9, 2.2, 5.4, 4.5,
     ['The decoder effect is not a captioning-skill effect: caption quality converges, gaze alignment '
      'does not (slide 5). A one-line appendix note would answer that question before a reader asks it.',
      'The Social heads (L4H4, L4H8) lead Layer 4 from the first epoch, a second line of evidence for '
      'the 5/5 fold agreement. The Describe heads take over Layer 3 only during training.',
      'The decoder ranking is not fixed from the start. After one epoch the best CNN-Transformer head '
      'beats both LSTMs on Describe gaze; the LSTM lead appears as captioning training erodes that '
      'head and CNN-LSTM finetunes its encoder.'],
     size=13, bullets=True, space_after=10)
box(s, 6.75, 1.4, 5.98, 4.6, fill=RGBColor(0xFB, 0xEA, 0xDE))
text(s, 7.05, 1.6, 5.4, 0.45, ['Check before reusing anything'], size=18, bold=True)
text(s, 7.05, 2.2, 5.45, 4.5,
     [[('Methods mismatch. ', {'bold': True}),
       ('Camera-ready says Phase 4 (unfrozen) starts from Phase 3 (frozen). Every training script '
        'starts both Social runs from Phase 2.', {})],
      [('Stale Layer-4 plots. ', {'bold': True}),
       ('CANS head_attention_analysis_*_L4 predate the Spearman fix. Use the rerun here.', {})],
      [('Different maps. ', {'bold': True}),
       ('mega_figure_*_Transformer and comparison_figure track the default attention map, not a '
        'selected head; their numbers do not match the paper.', {})],
      [('Old head picks. ', {'bold': True}),
       ('panel_combined uses L3 heads in Social; fig_traj uses ViT L3H1.', {})]],
     size=13, bullets=True, space_after=8)
footer(s, 13)
s.notes_slide.notes_text_frame.text = (
    'Methods text: neurips_final/overleaf/main.tex lines 183-184. Training scripts: '
    'CANS/architecture_manipulation/{CNN_Transformer,ViT_Transformer,ViT_LSTM}/train_social_*.py '
    '(P2_DIR = checkpoints/phase2). The paper analyzes the social_unfrozen checkpoint, so the numbers '
    'are unaffected; only the description of the initialization is wrong.')

prs.save(OUT)
print('saved', OUT)
