"""Python port of the dataviz skill's palette validator (node is unavailable on Della).

Implements the computable checks with the same constants and the same
Machado-Oliveira-Fernandes (2009) severity-1.0 CVD transforms:
  lightness band, chroma floor, CVD separation (OKLab dE x100), normal-vision floor,
  WCAG contrast vs surface.

Used to check the manuscript's existing model palette rather than assume it is
colorblind-safe. The palette itself is fixed by the published figures, so a WARN here
obligates secondary encoding (direct labels / distinct mark styling), not a hue change.
"""

import itertools
import sys

import numpy as np

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def linear_to_oklab(lin):
    m = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                  [0.2119034982, 0.6806995451, 0.1073969566],
                  [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(m @ lin)
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return m2 @ lms


def oklab(hex_c):
    return linear_to_oklab(srgb_to_linear(hex_to_rgb(hex_c)))


def oklch(hex_c):
    L, a, b = oklab(hex_c)
    return L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360


def simulate(hex_c, kind):
    lin = srgb_to_linear(hex_to_rgb(hex_c))
    sim = np.array(MACHADO[kind]) @ lin
    return linear_to_oklab(np.clip(sim, 0, 1))


def de(a, b):
    return float(np.linalg.norm(a - b) * 100)


def rel_lum(hex_c):
    lin = srgb_to_linear(hex_to_rgb(hex_c))
    return float(np.dot([0.2126, 0.7152, 0.0722], lin))


def contrast(a, b):
    la, lb = rel_lum(a), rel_lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def main():
    palette = [c for c in sys.argv[1].split(",") if c.strip()]
    mode = "light"
    pairs_all = "--pairs" in sys.argv and "all" in sys.argv
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    surface = DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]

    print(f"palette: {len(palette)} colors | mode={mode} | surface={surface}")
    print(f"{'color':>9} {'L':>6} {'C':>6} {'contrast':>9}  status")
    fails = []
    for c in palette:
        L, C, _ = oklch(c)
        cr = contrast(c, surface)
        flags = []
        if not (lo <= L <= hi):
            flags.append(f"L OUT OF BAND [{lo},{hi}]")
        if C < CHROMA_FLOOR:
            flags.append("CHROMA LOW")
        if cr < CONTRAST_MIN:
            flags.append(f"CONTRAST WARN {cr:.2f}")
        status = "; ".join(flags) if flags else "ok"
        if flags:
            fails.append((c, status))
        print(f"{c:>9} {L:6.3f} {C:6.3f} {cr:9.2f}  {status}")

    idx = (list(itertools.combinations(range(len(palette)), 2)) if pairs_all
           else [(i, i + 1) for i in range(len(palette) - 1)])
    print(f"\n{'pair':>21} {'normal':>8} {'protan':>8} {'deutan':>8} {'tritan':>8}  status")
    worst_normal, worst_cvd = 1e9, 1e9
    for i, j in idx:
        a, b = palette[i], palette[j]
        n = de(oklab(a), oklab(b))
        p = de(simulate(a, "protan"), simulate(b, "protan"))
        d = de(simulate(a, "deutan"), simulate(b, "deutan"))
        t = de(simulate(a, "tritan"), simulate(b, "tritan"))
        m = min(p, d)
        worst_normal = min(worst_normal, n)
        worst_cvd = min(worst_cvd, m)
        st = "ok" if m >= CVD_TARGET else ("WARN (needs 2nd encoding)" if m >= CVD_FLOOR else "FAIL")
        if n < NORMAL_FLOOR:
            st += " | NORMAL-VISION FAIL"
        print(f"{a}/{b} {n:8.1f} {p:8.1f} {d:8.1f} {t:8.1f}  {st}")

    print(f"\nworst adjacent CVD dE = {worst_cvd:.1f} (target >={CVD_TARGET}, floor {CVD_FLOOR})")
    print(f"worst normal-vision dE = {worst_normal:.1f} (hard floor {NORMAL_FLOOR})")
    ok = worst_cvd >= CVD_FLOOR and worst_normal >= NORMAL_FLOOR
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
