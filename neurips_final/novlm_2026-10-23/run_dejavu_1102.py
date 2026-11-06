"""Re-run a figure script in DejaVu Sans (matplotlib's default, the font of Figure 1), writing to
manuscript_2026-11-02 instead of manuscript_2026-10-23. Usage: python run_dejavu_1102.py script.py"""
import sys, os
p = os.path.abspath(sys.argv[1]); s = open(p).read()
old = "plt.rcParams['font.family'] = ['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans']"
assert old in s, 'font line not found'
s = s.replace(old, "plt.rcParams['font.family'] = ['DejaVu Sans']")
s = s.replace("'manuscript_2026-10-23'", "'manuscript_2026-11-02'")
exec(compile(s, p, 'exec'), {'__file__': p, '__name__': '__main__'})
