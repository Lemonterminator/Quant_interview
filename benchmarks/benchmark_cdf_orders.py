"""Measure float64 NumPy CDF truncation orders and the production Gamma path.

python benchmarks/benchmark_cdf_orders.py --repeats 15
The pricing function is cloned with a private CDF dispatch; production globals
are never patched. Inputs, correctness checks, plots and I/O are outside timing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import types

import numpy as np
import scipy
from scipy.special import erfc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Python_prototype'))
import black_scholes as bs

OUT = ROOT / 'artifacts' / 'cdf-performance'
METHODS = ('P2', 'P3', 'P4', 'P5', 'P6', 'complex_erfc')


def cdf_order(z, order):
    z = np.asarray(z)
    if not np.iscomplexobj(z):
        return bs.Phi_real(z)
    a, b = np.real(z), np.imag(z)
    pdf = bs.phi(a)
    b2 = b * b
    real = bs.Phi_real(a) + 0.5 * a * b2 * pdf
    imag = b * pdf
    if order >= 3:
        a2 = a * a
        b3 = b * b2
        imag += ((1 - a2) / 6) * b3 * pdf
    if order >= 4:
        b4 = b2 * b2
        real += ((3 - a2) * a / 24) * b4 * pdf
    if order >= 5:
        b5 = b3 * b2
        imag += (((a2 - 6) * a2 + 3) / 120) * b5 * pdf
    if order >= 6:
        b6 = b4 * b2
        real += (((a2 - 10) * a2 + 15) * a / 720) * b6 * pdf
    return real + 1j * imag


def dispatch(method):
    if method == 'complex_erfc':
        return lambda z, **kwargs: 0.5 * erfc(-np.asarray(z) * bs.INV_SQRT_2)
    return lambda z, **kwargs: bs.Phi_t(z, order=int(method[1:]))


def gamma_function(method, inputs, h):
    env = dict(bs.bs_price_call_t.__globals__)
    env['Phi_t'] = dispatch(method)
    price = types.FunctionType(bs.bs_price_call_t.__code__, env,
                               argdefs=bs.bs_price_call_t.__defaults__)
    price.__kwdefaults__ = bs.bs_price_call_t.__kwdefaults__.copy()
    s, *rest = inputs
    omega = (1 + 1j) / np.sqrt(2)

    def run():
        s0 = np.asarray(s, dtype=np.complex128)
        # Both perturbation construction and two pricing calls are timed.
        return np.imag(price(s0 + omega * h, *rest) + price(s0 - omega * h, *rest)) / (h * h)
    return run


def validate():
    import mpmath as mp
    from experiments.complex_cdf.study_series import cdf_series
    mp.mp.dps = 80
    zs = np.array([-.3 + .001j, .5 - .005j, 1.3 + .01j, 0 + .002j])
    np.testing.assert_array_equal(cdf_order(zs, 2), bs.Phi_t(zs, order=2))
    for order in range(2, 7):
        expected = np.array([complex(cdf_series(mp.mpc(float(z.real), float(z.imag)), order)) for z in zs])
        np.testing.assert_allclose(cdf_order(zs, order), expected, rtol=2e-15, atol=2e-16)
        np.testing.assert_allclose(bs.Phi_t(zs, order=order), expected, rtol=2e-15, atol=2e-16)
    args = (100., 100., 0., 0., .2, 1.)
    np.testing.assert_allclose(gamma_function('P2', args, .2)(), bs.gamma_45_deg(*args, h=.2, order=2), rtol=1e-13)
    return 'Explicit order=2 agrees with P2 CDF and Gamma; production P2-P6 agree with 80-digit expansions on four small-imaginary inputs.'


def measure(functions, repeats, seed):
    loops, samples = {}, {method: [] for method in functions}
    for method, fn in functions.items():
        fn()
        start = time.perf_counter_ns()
        for _ in range(3):
            value = fn()
        duration = (time.perf_counter_ns() - start) / 3e9
        if not np.isfinite(value).all():
            raise ArithmeticError(f'Nonfinite warmup: {method}')
        loops[method] = max(1, min(200, math.ceil(.005 / max(duration, 1e-9))))
    rng = random.Random(seed)
    for _ in range(repeats):
        order = list(functions)
        rng.shuffle(order)
        for method in order:
            fn, count = functions[method], loops[method]
            start = time.perf_counter_ns()
            for _ in range(count):
                value = fn()
            elapsed = (time.perf_counter_ns() - start) / count / 1e6
            samples[method].append(elapsed)
            if not np.isfinite(value).all():
                raise ArithmeticError(f'Nonfinite measurement: {method}')
    return samples, loops


def cpu_name():
    if os.name == 'nt':
        try:
            return subprocess.check_output(['powershell', '-NoProfile', '-Command',
                                            '(Get-CimInstance Win32_Processor).Name'],
                                           timeout=15, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or 'unknown'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1, 256, 4096, 65536, 262144])
    parser.add_argument('--repeats', type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 5 or any(n < 1 for n in args.sizes):
        parser.error('Require repeats>=5 and sizes>=1')
    sys.path.insert(0, str(ROOT))
    validation = validate()
    OUT.mkdir(parents=True, exist_ok=True)
    previous = None
    if (OUT / 'results.json').exists():
        shutil.copy2(OUT / 'results.json', OUT / 'previous-run.json')
        previous = json.loads((OUT / 'previous-run.json').read_text(encoding='utf-8'))
    rows, raw, accuracy = [], {}, []
    for n in args.sizes:
        rng = np.random.default_rng(20261003 + n)
        inputs = (rng.uniform(70, 130, n), rng.uniform(70, 130, n), rng.uniform(-.02, .05, n),
                  rng.uniform(0, .04, n), rng.uniform(.08, .7, n), rng.uniform(1 / 365, 3, n))
        h = .01 * inputs[0] * inputs[4] * np.sqrt(inputs[5])
        z = rng.uniform(-3, 3, n) + 1j * rng.uniform(-.01, .01, n)
        for workload in ('cdf', 'gamma45'):
            if workload == 'cdf':
                functions = {method: lambda method=method: dispatch(method)(z) for method in METHODS}
            else:
                functions = {method: gamma_function(method, inputs, h) for method in METHODS}
                reference = bs.Gamma(*inputs)
                for method, fn in functions.items():
                    error = np.abs(fn() - reference)
                    accuracy.append({'n': n, 'method': method, 'max_abs': float(np.max(error)),
                                     'median_abs': float(np.median(error))})
            samples, loops = measure(functions, args.repeats, 20261003 + n)
            baseline = statistics.median(samples['P2'])
            raw[f'{workload}:{n}'] = {'samples_ms': samples, 'inner_loops': loops}
            for method in METHODS:
                median = statistics.median(samples[method])
                rows.append({'workload': workload, 'n': n, 'method': method,
                             'median_ms': median, 'p25_ms': float(np.percentile(samples[method], 25)),
                             'p75_ms': float(np.percentile(samples[method], 75)),
                             'time_ratio_vs_P2': median / baseline,
                             'extra_time_percent': (median / baseline - 1) * 100,
                             'throughput_loss_percent': (1 - baseline / median) * 100,
                             'items_per_second': n * 1000 / median})
            print(f'{workload}: n={n} measured', flush=True)
    environment = {'cpu': cpu_name(), 'logical_cpus': os.cpu_count(), 'platform': platform.platform(),
                   'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                   'repeats': args.repeats, 'sizes': args.sizes, 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'thread_env': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}
    with (OUT / 'timings.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'results.json').write_text(json.dumps({'environment': environment, 'validation': validation,
                                                 'timings': rows, 'raw': raw, 'accuracy': accuracy}, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, workload in zip(axes, ('cdf', 'gamma45')):
        for method in METHODS:
            selected = [row for row in rows if row['workload'] == workload and row['method'] == method]
            ax.plot([row['n'] for row in selected], [row['time_ratio_vs_P2'] for row in selected], marker='o', label=method)
        ax.set(xscale='log', title=workload, xlabel='batch size', ylabel='time / P2 time')
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'slowdown.png', dpi=160)
    plt.close(fig)
    lines = ['# 高阶复数 CDF 的性能成本', '', f"实测时间：{environment['timestamp_utc']}。CPU：{environment['cpu']}。",
             f"Python {environment['python']} / NumPy {environment['numpy']} / SciPy {environment['scipy']}；逻辑 CPU 数 {environment['logical_cpus']}。", '',
             '## 运算量估计', '',
             '各阶共用一次实数 CDF 和一次 PDF；升阶不新增 erfc/exp。以下按本脚本表达式的逐元素算术计数，不计共同运算、复数结果组装、内存读写和 Python 调度；常数除法也列为一次操作，不能据此直接预测耗时。', '',
             '| 升阶 | 新增乘法 | 新增加减 | 新增常数除法 |', '| --- | --- | --- | --- |',
             '| P2 → P3 | 4 | 2 | 1 |', '| P3 → P4 | 4 | 2 | 1 |',
             '| P4 → P5 | 4 | 3 | 1 |', '| P5 → P6 | 5 | 3 | 1 |', '',
             'NumPy 的这些操作会形成额外数组遍历与临时数组。因此局部 CDF 的成本可能明显增加；完整 Gamma 还有 log、exp、广播、掩码等共同成本，其相对增幅通常更小。C++ 融合循环/SIMD 的成本需要另行实测。', '',
             '## 实测方法', '',
             f'每组 {args.repeats} 轮，顺序随机，先预热；每轮多次调用取每调用耗时，再报告中位数与四分位数。输入生成、检查、I/O 在计时外；结果分配、临时数组、Python 调度在计时内。无显式线程池，结果不代表多核或 C++ SIMD 性能。',
             'CDF 使用 a∈[-3,3]、b∈[-0.01,0.01]；Gamma 使用固定种子的随机期权参数与 h=0.01Sσ√T。Gamma 复用生产价格函数的代码，仅在私有函数环境中替换 CDF，生产全局状态未修改。',
             f'正确性检查：{validation}',
             '“额外耗时”=T/T₂−1；“吞吐损失”=1−T₂/T。这两个百分比不同。原始样本保留在 JSON；微小差异可能是噪声，不应推断稳定加速。', '',
             '![耗时比](../artifacts/cdf-performance/slowdown.png)', '']
    for workload in ('cdf', 'gamma45'):
        lines += [f'## {workload}', '', '| 批量 N | 方法 | 中位 ms | P25–P75 ms | 耗时倍数 | 额外耗时 | 吞吐损失 |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for row in rows:
            if row['workload'] == workload:
                lines.append(f"| {row['n']} | {row['method']} | {row['median_ms']:.6f} | {row['p25_ms']:.6f}–{row['p75_ms']:.6f} | {row['time_ratio_vs_P2']:.3f} | {row['extra_time_percent']:.1f}% | {row['throughput_loss_percent']:.1f}% |")
        lines += ['']
    largest = max(args.sizes)
    selected = {(x['workload'], x['method']): x for x in rows if x['n'] == largest}
    weight = 4 * selected['cdf', 'P2']['median_ms'] / selected['gamma45', 'P2']['median_ms']
    lines += [f'## 粗略总成本估计（N={largest}）', '',
              '45° Gamma 调用两次价格函数，每次有两个 CDF，因此可以建立 TΓ≈T其他+4TCDF 的模型。用独立 CDF 基准近似组件耗时，得到权重',
              f'ρ=4TCDF,P2/TΓ,P2≈{weight:.3f}；预计相对增幅≈ρ×(CDF 耗时倍数−1)。',
              '独立 CDF 的输入分布与价格函数内部不同，这只是成本估计，不是精确的函数剖析；内存和分配成本也可能不满足简单可加模型。', '',
              '| 方法 | 模型估计 Gamma 额外耗时 | 实测 Gamma 额外耗时 |', '| --- | --- | --- |']
    for method in METHODS[1:]:
        predicted = weight * (selected['cdf', method]['time_ratio_vs_P2'] - 1) * 100
        lines.append(f"| {method} | {predicted:.1f}% | {selected['gamma45', method]['extra_time_percent']:.1f}% |")
    if previous and previous['environment']['cpu'] == environment['cpu']:
        lines += ['', '## 与上一轮的比较', '', f"上一轮：{previous['environment']['timestamp_utc']}；本轮：{environment['timestamp_utc']}。", '',
                  '| 方法 | 上轮 Gamma 耗时比 | 本轮 Gamma 耗时比 |', '| --- | --- | --- |']
        for method in METHODS:
            old = next((x for x in previous['timings'] if x['n'] == largest and x['workload'] == 'gamma45' and x['method'] == method), None)
            if old:
                lines.append(f"| {method} | {old['time_ratio_vs_P2']:.3f} | {selected['gamma45', method]['time_ratio_vs_P2']:.3f} |")
        lines += ['', '上一轮原始记录：[previous-run.json](../artifacts/cdf-performance/previous-run.json)。']
    lines += ['', f'## 同一次实验的 float64 Gamma 误差（N={largest}）', '',
              '参考为同库解析 Gamma；该表是所采样参数与步长的诊断，不是独立高精度全域验收。高阶结果可能进入 float64 舍入平台。', '',
              '| 方法 | 最大绝对误差 | 中位绝对误差 |', '| --- | --- | --- |']
    lines += [f"| {x['method']} | {x['max_abs']:.6e} | {x['median_abs']:.6e} |" for x in accuracy if x['n'] == largest]
    lines += ['', '原始数据：[timings.csv](../artifacts/cdf-performance/timings.csv)、[results.json](../artifacts/cdf-performance/results.json)。',
              '重现：`python benchmarks/benchmark_cdf_orders.py --repeats 15`。额外研究依赖为 mpmath。生产接口与默认截断阶数未修改。', '']
    report = ROOT / 'docs' / 'cdf_performance.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report: {report}', flush=True)


if __name__ == '__main__':
    main()
