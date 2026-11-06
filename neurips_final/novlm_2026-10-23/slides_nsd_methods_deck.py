"""
slides_nsd_methods_deck.py -- the two NSD methods slides in one PowerPoint file: the NSD task
(slides_nsd_task.py) and then the encoding analysis (slides_encoding_methods.py). Each slide is drawn by
its own script's add_slide, so editing either script and re-running this keeps the deck in step.
Needs python-pptx (not in cans-env or lmw).
Writes slides/nsd_methods_slides.pptx
"""
import os, importlib.util
from pptx import Presentation
from pptx.util import Inches

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join('/scratch/gpfs/GRAZIANO/vlm-attention-alignment', 'neurips_final', 'final_figures',
                   'manuscript_2026-10-23', 'slides', 'nsd_methods_slides.pptx')

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
for name in ('slides_nsd_task', 'slides_encoding_methods'):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.add_slide(prs)
prs.save(OUT)
print('wrote', OUT)
