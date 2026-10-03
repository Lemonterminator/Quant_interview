"""Regenerate fixed call/Delta/Gamma cases with independent 80-digit mpmath.

Optional development dependency: ``pip install mpmath``. Tests only read the
committed CSV; they never regenerate it. Review CSV changes before committing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import mpmath as mp


CASES = (
    ("atm_reference", "100", "100", "0", "0", "0.2", "1"),
    ("near_expiry_atm", "100", "100", "0", "0", "0.01", "0.0027397260273972603"),
    ("carry_itm", "120", "100", "0.03", "0.01", "0.25", "2"),
    ("negative_rate_otm", "80", "100", "-0.01", "0.04", "0.35", "0.75"),
    ("low_price_otm", "60", "100", "0.02", "0", "0.2", "0.25"),
    ("high_vol", "100", "110", "0.01", "0.03", "1.2", "4"),
    ("zero_vol_itm", "110", "100", "0.01", "0.02", "0", "1"),
    ("expiry_atm", "100", "100", "0", "0", "0.2", "0"),
)


def normal_cdf(x):
    return (1 + mp.erf(x / mp.sqrt(2))) / 2


def expected(row):
    _, *fields = row
    s, k, r, q, sigma, t = (mp.mpf(field) for field in fields)
    a = s * mp.exp(-q * t)
    b = k * mp.exp(-r * t)
    if sigma == 0 or t == 0:
        price = max(a - b, 0)
        delta = mp.exp(-q * t) * (1 if a > b else mp.mpf("0.5") if a == b else 0)
        return price, delta, mp.mpf("0")
    tau = sigma * mp.sqrt(t)
    d1 = (mp.log(s / k) + (r - q) * t + tau**2 / 2) / tau
    d2 = d1 - tau
    price = a * normal_cdf(d1) - b * normal_cdf(d2)
    delta = mp.exp(-q * t) * normal_cdf(d1)
    gamma = mp.exp(-q * t - d1**2 / 2) / (s * tau * mp.sqrt(2 * mp.pi))
    return price, delta, gamma


def main():
    mp.mp.dps = 80
    target = Path(__file__).with_name("bs_call_golden.csv")
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "S", "K", "r", "q", "sigma", "T", "price", "delta", "gamma"))
        for row in CASES:
            writer.writerow((*row, *(mp.nstr(value, 18) for value in expected(row))))
    print(target)


if __name__ == "__main__":
    main()
