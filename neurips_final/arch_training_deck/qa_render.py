"""Rough PIL preview of a python-pptx deck (no LibreOffice on Della). Draws shapes, pictures,
wrapped text with Nimbus Sans at true point size, and flags text that overflows its box."""
import sys, os, glob
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Emu

DPI = 100
deck, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
FONTS = glob.glob('/usr/share/fonts/**/NimbusSans-Regular.otf', recursive=True) + \
    glob.glob('/usr/share/fonts/**/DejaVuSans.ttf', recursive=True)
BOLD = glob.glob('/usr/share/fonts/**/NimbusSans-Bold.otf', recursive=True) + \
    glob.glob('/usr/share/fonts/**/DejaVuSans-Bold.ttf', recursive=True)


def font(pt, bold):
    return ImageFont.truetype((BOLD if bold else FONTS)[0], max(1, int(pt * DPI / 72)))


def px(e):
    return int(Emu(e).inches * DPI)


prs = Presentation(deck)
W, H = px(prs.slide_width), px(prs.slide_height)
for n, slide in enumerate(prs.slides, 1):
    bg = slide.background.fill.fore_color.rgb if slide.background.fill.type == 1 else None
    im = Image.new('RGB', (W, H), '#' + str(bg) if bg else 'white')
    d = ImageDraw.Draw(im)
    for sh in slide.shapes:
        x, y, w, h = px(sh.left), px(sh.top), px(sh.width), px(sh.height)
        if sh.shape_type == 13:
            pic = Image.open(sh.image._blob if False else __import__('io').BytesIO(sh.image.blob)).convert('RGB')
            im.paste(pic.resize((max(w, 1), max(h, 1))), (x, y))
            continue
        if sh.has_chart:
            d.rectangle([x, y, x + w, y + h], outline='#888', width=2)
            d.text((x + 10, y + 10), 'CHART', fill='#888', font=font(14, True))
            continue
        try:
            if sh.fill.type == 1:
                d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill='#' + str(sh.fill.fore_color.rgb))
        except Exception:
            pass
        if not sh.has_text_frame or not sh.text_frame.text.strip():
            continue
        cy = y + 2
        for p in sh.text_frame.paragraphs:
            runs = [(r.text, r.font) for r in p.runs]
            if not runs:
                continue
            f0 = runs[0][1]
            size = max((r.size.pt for _, r in runs if r.size), default=14)
            words = []
            for t, f in runs:
                c = '#' + str(f.color.rgb) if f.color and f.color.type else '#000'
                for wd in t.replace('\n', ' \n ').split(' '):
                    words.append((wd, font(f.size.pt if f.size else 14, bool(f.bold)), c))
            line, lw = [], 0
            lines = []
            for wd, ft, c in words:
                ww = ft.getlength(wd + ' ')
                if (lw + ww > w - 8 and line) or wd == '\n':
                    lines.append(line); line, lw = [], 0
                    if wd == '\n':
                        continue
                line.append((wd, ft, c)); lw += ww
            lines.append(line)
            lh = int(size * DPI / 72 * 1.2)
            for ln in lines:
                tot = sum(ft.getlength(wd + ' ') for wd, ft, _ in ln)
                cx = x + 4 if p.alignment in (None, 1) else x + (w - tot) / 2 if p.alignment == 2 else x + w - tot - 4
                for wd, ft, c in ln:
                    d.text((cx, cy), wd, font=ft, fill=c)
                    cx += ft.getlength(wd + ' ')
                cy += lh
            cy += int((p.space_after.pt if p.space_after else 0) * DPI / 72)
        anchor_ok = cy <= y + h + 3
        if not anchor_ok:
            d.rectangle([x, y, x + w, y + h], outline='red', width=3)
            print(f'slide {n}: OVERFLOW in "{sh.text_frame.text[:50]}" ({cy - y - h}px)')
    im.save(os.path.join(outdir, f'slide-{n:02d}.png'))
print('rendered', len(prs.slides))
