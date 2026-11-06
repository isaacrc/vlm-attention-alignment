"""Re-run a figure script with its output moved from manuscript_2026-10-23 to manuscript_2026-11-02.
Font comes from MATPLOTLIBRC (Helvetica Neue -> Nimbus Sans, as Figure 3). Usage: python run_fonts_1102.py script.py"""
import sys, os
p = os.path.abspath(sys.argv[1]); s = open(p).read()
assert 'manuscript_2026-10-23' in s
s = s.replace("'manuscript_2026-10-23'", "'manuscript_2026-11-02'")
exec(compile(s, p, 'exec'), {'__file__': p, '__name__': '__main__'})
