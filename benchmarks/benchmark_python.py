"""End-to-end Python call/Delta/Gamma throughput; no timing assertions.

Run from the repo root: python benchmarks/benchmark_python.py
Generated arrays and a warm thread pool are excluded from timed regions;
result allocation, task scheduling, and concatenation are included.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Python_prototype"))
from bs_batch import CallResult, evaluate_call  # noqa: E402


def make_inputs(n: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20261003)
    return (
        rng.uniform(70, 130, n),
        rng.uniform(70, 130, n),
        rng.uniform(-0.02, 0.05, n),
        rng.uniform(0, 0.04, n),
        rng.uniform(0.08, 0.7, n),
        rng.uniform(1 / 365, 3, n),
    )


def python_scalar(inputs: tuple[np.ndarray, ...]) -> CallResult:
    rows = [evaluate_call(*(float(value[i]) for value in inputs)) for i in range(len(inputs[0]))]
    return CallResult(*(np.array([float(getattr(row, name)) for row in rows])
                        for name in ("price", "delta", "gamma")))


def threaded(inputs: tuple[np.ndarray, ...], pool: ThreadPoolExecutor, workers: int) -> CallResult:
    n = len(inputs[0])
    cuts = np.linspace(0, n, workers + 1, dtype=int)
    futures = [pool.submit(evaluate_call, *(value[cuts[i]:cuts[i + 1]] for value in inputs))
               for i in range(workers)]
    parts = [future.result() for future in futures]
    return CallResult(*(np.concatenate([getattr(part, name) for part in parts])
                        for name in ("price", "delta", "gamma")))


def verify(reference: CallResult, candidate: CallResult):
    for field in ("price", "delta", "gamma"):
        np.testing.assert_allclose(getattr(candidate, field), getattr(reference, field), rtol=5e-11, atol=2e-11)


def measure(fn, repeats: int) -> tuple[float, float, float]:
    fn()  # allocation, library dispatch and thread startup warmup
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        result = fn()
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        if not np.isfinite(result.price).all():
            raise ArithmeticError("Nonfinite benchmark output")
        samples.append(elapsed_ms)
    return (statistics.median(samples), float(np.percentile(samples, 25)), float(np.percentile(samples, 75)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 4096, 32768])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--scalar-max", type=int, default=4096)
    args = parser.parse_args()
    if args.repeats < 3 or any(n <= 0 for n in args.sizes) or any(w < 2 for w in args.workers):
        parser.error("Require repeats>=3, positive sizes and workers>=2")

    print("backend,n,workers,median_ms,p25_ms,p75_ms,options_per_s,speedup_vs_numpy,worker_efficiency")
    for n in args.sizes:
        inputs = make_inputs(n)
        reference = evaluate_call(*inputs)
        numpy_ms, lo, hi = measure(lambda: evaluate_call(*inputs), args.repeats)
        print(f"numpy,{n},1,{numpy_ms:.6f},{lo:.6f},{hi:.6f},{n * 1000 / numpy_ms:.1f},1.000,na")
        if n <= args.scalar_max:
            verify(reference, python_scalar(inputs))
            ms, lo, hi = measure(lambda: python_scalar(inputs), args.repeats)
            print(f"python_loop,{n},1,{ms:.6f},{lo:.6f},{hi:.6f},{n * 1000 / ms:.1f},{numpy_ms / ms:.3f},na")
        for workers in args.workers:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fn = lambda: threaded(inputs, pool, workers)
                verify(reference, fn())
                ms, lo, hi = measure(fn, args.repeats)
                speedup = numpy_ms / ms
                print(f"numpy_threads,{n},{workers},{ms:.6f},{lo:.6f},{hi:.6f},"
                      f"{n * 1000 / ms:.1f},{speedup:.3f},{speedup / workers:.3f}")


if __name__ == "__main__":
    main()
