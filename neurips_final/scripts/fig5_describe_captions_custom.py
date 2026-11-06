"""
fig5_describe_captions_custom.py
--------------------------------
Custom edit of fig5_describe_slide_captions.png (thinking-off version): same slide,
with hand-edited captions from CAPTION_EDITS. Reuses fig5_slides.py (cached maps)
but skips its map-only slides, so nothing else in slides/ is rewritten.

Output: neurips_final/final_figures/slides/fig5_describe_slide_captions_custom.png
"""
import os

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'neurips_final', 'scripts', 'fig5_slides.py')
CAPTION_EDITS = {('ViT-Transformer', '000000157601'): 'A woman is eating with a cup of coffee'}

os.environ['QWEN_NOTHINK'] = '1'
src = open(SRC).read()
head = src[:src.index("for task in ('social', 'describe'):\n    fig = plt.figure")]   # loaders, maps, layout constants
tail = src[src.index('# ── Describe slide with each agent'):]
tail = tail.replace("CELL_C, CAP_H = ",
                    "for (c_, im_), t_ in CAPTION_EDITS.items(): caps[c_][im_] = t_\nCELL_C, CAP_H = ", 1)
tail = tail.replace("'fig5_describe_slide_captions.png'", "'fig5_describe_slide_captions_custom.png'")
tail = tail.replace("OUT, 'fig5_describe", "SLIDES, 'fig5_describe")
SLIDES = os.path.join(ROOT, 'neurips_final', 'final_figures', 'slides')
exec(head + tail)
