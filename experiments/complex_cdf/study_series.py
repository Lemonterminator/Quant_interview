"""High-precision research experiment; does not change the pricing kernels.

Run from any directory: python experiments/complex_cdf/study_series.py
Optional research dependencies: mpmath and matplotlib.
"""
from pathlib import Path
import csv
import json

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "complex-cdf"
ORDERS = (2, 3, 4, 5, 6, 8)


def phi(z):
    return mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)


def cdf(z):
    return (1 + mp.erf(z / mp.sqrt(2))) / 2


def hermite(n, x):
    """Probabilists' Hermite polynomials He_n, not physicists' H_n."""
    previous, current = mp.mpf(1), x
    if n == 0:
        return previous
    for k in range(1, n):
        previous, current = current, x * current - k * previous
    return current


def cdf_series(z, order):
    a, b = mp.re(z), mp.im(z)
    return cdf(a) + phi(a) * mp.fsum(
        (-1) ** (n - 1) * hermite(n - 1, a) * (mp.j * b) ** n / mp.factorial(n)
        for n in range(1, order + 1)
    )


def call_price(z, params, normal_cdf):
    _, k, r, q, sigma, expiry = params
    tau = sigma * mp.sqrt(expiry)
    d1 = (mp.log(z / k) + (r - q + sigma * sigma / 2) * expiry) / tau
    return z * mp.exp(-q * expiry) * normal_cdf(d1) - k * mp.exp(-r * expiry) * normal_cdf(d1 - tau)


def second_45(function, x, h):
    omega = (1 + mp.j) / mp.sqrt(2)
    return mp.im(function(x + omega * h) + function(x - omega * h)) / (h * h)


def main():
    mp.mp.dps = 80
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = [("cdf_x_0.3", None), ("atm", ("100", "100", "0", "0", ".2", "1")),
                 ("near_expiry", ("100", "100", "0", "0", ".01", mp.mpf(1) / 365)),
                 ("carry_itm", ("120", "100", ".03", ".01", ".25", "2"))]
    rows = []
    for label, raw in scenarios:
        if raw is None:
            x, h0 = mp.mpf('.3'), mp.mpf('.04')
            reference = -x * phi(x)
            make_function = lambda P: P
        else:
            params = tuple(mp.mpf(v) for v in raw)
            x, k, r, q, sigma, expiry = params
            tau = sigma * mp.sqrt(expiry)
            d1 = (mp.log(x / k) + (r - q + sigma * sigma / 2) * expiry) / tau
            reference = mp.exp(-q * expiry) * phi(d1) / (x * tau)
            h0 = x * tau * mp.mpf('.01')
            make_function = lambda P: lambda z: call_price(z, params, P)
        for order in (*ORDERS, None):
            P = cdf if order is None else lambda z: cdf_series(z, order)
            function = make_function(P)
            previous_error = None
            for index in range(6):
                h = h0 / 2 ** index
                actual = second_45(function, x, h)
                error = abs(actual - reference)
                ratio = previous_error / error if previous_error is not None and error else None
                p = mp.log(ratio, 2) if ratio else None
                rows.append({"scenario": label, "method": "exact" if order is None else f"P{order}",
                             "n": index, "h": mp.nstr(h, 70), "actual": mp.nstr(actual, 70),
                             "reference": mp.nstr(reference, 70), "abs_error": mp.nstr(error, 70),
                             "previous_over_current": mp.nstr(ratio, 30) if ratio else "",
                             "observed_order": mp.nstr(p, 30) if p else ""})
                previous_error = error
    with (OUT / 'sequences.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'results.json').write_text(json.dumps({"precision_decimal_digits": 80, "rows": rows},
                                               indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, scenario in zip(axes, ('cdf_x_0.3', 'atm')):
        for method in ('P2', 'P3', 'P4', 'P5', 'exact'):
            selected = [row for row in rows if row['scenario'] == scenario and row['method'] == method]
            ax.loglog([float(row['h']) for row in selected], [float(row['abs_error']) for row in selected],
                      marker='o', label=method)
        ax.set(title=f'45-degree derivative: {scenario}', xlabel='absolute step h', ylabel='absolute error')
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([float(row['h']) for row in selected])
        ax.set_xticklabels([f"{float(row['h']):.3g}" for row in selected], rotation=30, ha='right')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=.25)
    fig.suptitle('80-digit arithmetic: truncation error (not float64 accuracy)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'series_errors.png', dpi=160)
    plt.close(fig)

    # Refresh only the generated numerical appendix; preserve the derivation.
    doc = ROOT / 'docs' / 'complex_cdf_series.md'
    marker = '<!-- generated numerical appendix -->'
    prefix = doc.read_text(encoding='utf-8').split(marker)[0].rstrip()
    appendix = [marker, '', '## 本次高精度数列', '',
                '由 `experiments/complex_cdf/study_series.py` 生成，mpmath 十进制精度为 80 位。',
                '研究实验模拟不同 CDF 截断阶数，没有修改生产定价函数。', '',
                '![误差序列](../artifacts/complex-cdf/series_errors.png)', '']
    for scenario in ('cdf_x_0.3', 'atm', 'near_expiry', 'carry_itm'):
        appendix += [f'### {scenario}', '', '| n | h | P2 误差 | P3 误差 | P4 误差 | P5 误差 | 完整 CDF 误差 |',
                     '| --- | --- | --- | --- | --- | --- | --- |']
        for n in range(6):
            matching = {row['method']: row for row in rows if row['scenario'] == scenario and row['n'] == n}
            values = [f"{float(matching[m]['abs_error']):.6e}" for m in ('P2', 'P3', 'P4', 'P5', 'exact')]
            appendix.append(f"| {n} | {float(matching['P2']['h']):.6e} | {' | '.join(values)} |")
        appendix += ['', '| CDF 阶数 | 最后一次减半的误差比 | 实测阶 p |', '| --- | --- | --- |']
        for method in ('P2', 'P3', 'P4', 'P5', 'P6', 'P8', 'exact'):
            row = next(row for row in rows if row['scenario'] == scenario and row['n'] == 5 and row['method'] == method)
            appendix.append(f"| {method} | {float(row['previous_over_current']):.6f} | {float(row['observed_order']):.6f} |")
        appendix += ['']
    appendix += ['原始数据：[CSV](../artifacts/complex-cdf/sequences.csv)、[JSON](../artifacts/complex-cdf/results.json)。',
                 '这些结果说明所测场景的局部截断误差，不能证明任意参数、任意步长或 float64 下均有同样精度。', '']
    doc.write_text(prefix + '\n\n' + '\n'.join(appendix), encoding='utf-8')
    print(f'Research report: {doc}')
    print(f'Sequence rows: {len(rows)}; precision: {mp.mp.dps} decimal digits')


if __name__ == '__main__':
    main()
