"""Reviewable 80-digit CDF Taylor references, independent Hermite recurrence.

Requires optional mpmath; runtime tests only read the committed CSV.
"""
import csv
from pathlib import Path
import sys
import mpmath as mp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.complex_cdf.study_series import cdf_series


def main():
    mp.mp.dps = 80
    with (ROOT / 'testdata' / 'complex_cdf_golden.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['order', 'a', 'b', 'real', 'imag'])
        for order in range(2, 7):
            for a, b in [('-0.3', '.001'), ('.5', '-.005'), ('1.3', '.01'), ('0', '.002')]:
                result = cdf_series(mp.mpc(a, b), order)
                writer.writerow([order, a, b, mp.nstr(result.real, 50), mp.nstr(result.imag, 50)])


if __name__ == '__main__':
    main()
