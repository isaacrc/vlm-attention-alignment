"""Shared figure style for the ICLR manuscript (2026-09-24).

Every bar figure in the paper used the saturated matplotlib tab10 defaults, a grid, and
white bar edges, and each script carried its own copy of the colour map. This module is the
single source of truth so a model is the same colour in every figure.

Palette: Okabe-Ito pulled 15% toward grey, to sit closer to the muted reference style in
overleaf/example_plot_formatting.png while staying separable for six categories. The
reference itself has only three series; at that chroma six categories are not
distinguishable (sage and terracotta collapse to Delta-E 0.7 under protanopia). Validated
with the dataviz validator: lightness band PASS, chroma floor PASS, CVD separation PASS
(worst adjacent pair Delta-E 9.6 protan), normal-vision floor PASS (worst 18.7). The
contrast warning is answered by the legend and axis labels that every figure carries.

Style, matched to the reference: black bar edges, no grid, left and bottom spines only,
black error bars, an unframed legend, and an axis label about 1.6x the tick size.
"""

# ── colours ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    'CNN-LSTM':        '#c86313',   # vermillion
    'ViT-LSTM':        '#139a75',   # bluish green
    'CNN-Transformer': '#1374aa',   # blue
    'ViT-Transformer': '#c17aa1',   # reddish purple
    'Molmo':           '#d79a13',   # orange
    'Qwen3.5':         '#5cacd9',   # sky blue
}
# Underscored spellings used by the NSD scripts.
MODEL_COLORS.update({k.replace('-', '_'): v for k, v in MODEL_COLORS.items()})
MODEL_COLORS['Qwen35'] = MODEL_COLORS['Qwen3.5']

# Non-model series stay neutral so they never compete with a model for identity.
NEUTRAL = {
    'Human':        '#4d4d4d',
    # CLIP is neither a model nor a low-level control: a semantic reference band.
    # Slate, with its own cross-hatch, so it reads as a third kind of predictor.
    'CLIP':         '#4f5d6b',
    'CENTER_PRIOR': '#9e9e9e',
    'EdgeEnergy':   '#9e9e9e',
    'Luminance':    '#c7c7c7',
}
ALL_COLORS = {**MODEL_COLORS, **NEUTRAL}

HATCH_SERIES = ('EdgeEnergy', 'Luminance', 'CENTER_PRIOR')
XHATCH_SERIES = ('CLIP',)
BAR_EDGE = 'black'
BAR_EDGE_LW = 1.2
ERR_COLOR = 'black'


def color(name, default='#9e9e9e'):
    return ALL_COLORS.get(name, ALL_COLORS.get(name.replace('_', '-'), default))


# ── text proportions ──────────────────────────────────────────────────────────
# The reference sets its axis label at roughly 1.6x the tick label. Scale these
# together for a given canvas; the ratios are what matter.
def font_sizes(scale=1.0):
    return dict(tick=round(28 * scale), label=round(44 * scale),
                title=round(44 * scale), legend=round(28 * scale),
                star=round(34 * scale))


FS = font_sizes()


def style_axes(ax, fs=None, spines=('left', 'bottom')):
    """Reference styling: no grid, only the named spines, plain black ticks."""
    fs = fs or FS
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in spines)
        if s in spines:
            ax.spines[s].set_linewidth(1.8)
            ax.spines[s].set_color('black')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=fs['tick'], width=1.8, length=7, color='black')
    ax.tick_params(axis='x', labelsize=fs['tick'], length=0, pad=6)


def bar_kwargs(names):
    """facecolor/edge/hatch for a list of series names, in one call."""
    return dict(color=[color(n) for n in names], edgecolor=BAR_EDGE,
                linewidth=BAR_EDGE_LW)


def hatch_bars(bars, names):
    for b, n in zip(bars, names):
        if n in HATCH_SERIES:
            b.set_hatch('//')
        elif n in XHATCH_SERIES:
            b.set_hatch('xx')


def errorbar_kwargs(scale=1.0):
    return dict(fmt='none', color=ERR_COLOR, capsize=round(8 * scale),
                capthick=round(2.2 * scale), elinewidth=round(2.2 * scale), zorder=4)


# ── panel letters ─────────────────────────────────────────────────────────────
# Drawn into the image, bare and bold, as plot_fig_head_selection_corrected.py
# does it (an 80 pt letter on a 34 in canvas). Every panel is placed at the 5.5 in
# NeurIPS text width, so what has to match between figures is the apparent size
# after scaling, not the nominal one: hold 80 * 5.5/34 = 12.9 pt fixed and solve
# back for whatever canvas the figure uses. Keep this in step with FS_LETTER in
# plots/plot_fig_head_selection_corrected.py, which sets the reference.
LETTER_PT_PER_IN = 80 / 34


def panel_letter(fig, letter, x=0.005, y=0.995):
    """Bare bold panel letter at the top-left of a figure canvas."""
    fig.text(x, y, letter, fontsize=round(LETTER_PT_PER_IN * fig.get_size_inches()[0]),
             fontweight='bold', va='top', ha='left', color='black', zorder=10)
