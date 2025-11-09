"""
Delta/Gamma accuracy sweeps for the Quant Black-Scholes Greeks assignment.

The script follows the instruction set from the PDF: build a logarithmic grid
of relative step sizes h_rel ∈ [1e-16, 1e-4], scale them by the spot (h=S⋅h_rel),
and compare analytic Greeks with their finite-difference and complex-step
calculations. Detailed CSVs, plots, and terminal summaries demonstrate the
expertise applied to both Δ and Γ.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import time

from black_scholes import (
    Gamma,
    delta_call,
    delta_cs,
    delta_forward,
    gamma_45_deg,
    gamma_cs_real,
    gamma_forward,
)


HREL_MIN = 1e-16
HREL_MAX = 1e-4
HREL_POINTS = 24
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FLOAT_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class Scenario:
    """Container for a single set of option inputs."""

    name: str
    S: float
    K: float
    r: float
    q: float
    sigma: float
    T: float


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(
        name="ATM reference S=100 K=100 r=0 q=0 sigma=20% T=1y",
        S=100.0,
        K=100.0,
        r=0.0,
        q=0.0,
        sigma=0.20,
        T=1.0,
    ),
    Scenario(
        name="Near-expiry low-vol ATM S=100 K=100 r=0 q=0 sigma=1% T=1/365",
        S=100.0,
        K=100.0,
        r=0.0,
        q=0.0,
        sigma=0.01,
        T=1.0 / 365.0,
    ),
    # Additional scenarios from the assignment PDF can be appended here.
)


def slugify(text: str) -> str:
    """Create a filesystem-friendly identifier."""

    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = re.sub(r"_{2,}", "_", slug)
    return slug.strip("_")


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Robust relative error with an epsilon floor on the denominator."""

    denom = np.maximum(np.abs(reference), FLOAT_EPS)
    return np.abs(estimate - reference) / denom


def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    """Return max, median, and p99 absolute errors."""

    return {
        "max": float(np.max(errors)),
        "median": float(np.median(errors)),
        "p99": float(np.percentile(errors, 99)),
    }


def run_delta_sweep(scenario: Scenario, hrel_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute analytic and numerical deltas for every h in the grid."""

    h_values = hrel_grid * scenario.S
    d_ref = float(delta_call(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T))
    d_ref_vec = np.full_like(hrel_grid, d_ref, dtype=np.float64)

    d_fd = np.array(
        [float(delta_forward(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h=float(h))) for h in h_values],
        dtype=np.float64,
    )
    d_cs = np.array(
        [float(delta_cs(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h=float(h))) for h in h_values],
        dtype=np.float64,
    )

    errors = {
        "delta_fd_abs_err": np.abs(d_fd - d_ref_vec),
        "delta_cs_abs_err": np.abs(d_cs - d_ref_vec),
        "delta_fd_rel_err": relative_error(d_fd, d_ref_vec),
        "delta_cs_rel_err": relative_error(d_cs, d_ref_vec),
    }

    return {
        "hrel": hrel_grid,
        "h": h_values,
        "delta_ref": d_ref_vec,
        "delta_fd": d_fd,
        "delta_cs": d_cs,
        **errors,
    }


def run_gamma_sweep(scenario: Scenario, hrel_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute analytic and numerical gammas for every h in the grid."""

    h_values = hrel_grid * scenario.S
    g_ref = float(Gamma(scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T))
    g_ref_vec = np.full_like(hrel_grid, g_ref, dtype=np.float64)

    g_fd = np.array(
        [
            float(
                gamma_forward(
                    scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h=float(h)
                )
            )
            for h in h_values
        ],
        dtype=np.float64,
    )

    g_cs = np.array(
        [
            float(
                gamma_cs_real(
                    scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h=float(h)
                )
            )
            for h in h_values
        ],
        dtype=np.float64,
    )

    g_45 = np.array(
        [
            float(
                gamma_45_deg(
                    scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T, h=float(h)
                )
            )
            for h in h_values
        ],
        dtype=np.float64,
    )

    errors = {
        "gamma_fd_abs_err": np.abs(g_fd - g_ref_vec),
        "gamma_cs_abs_err": np.abs(g_cs - g_ref_vec),
        "gamma_45_deg_abs_err": np.abs(g_45 - g_ref_vec),
        "gamma_fd_rel_err": relative_error(g_fd, g_ref_vec),
        "gamma_cs_rel_err": relative_error(g_cs, g_ref_vec),
        "gamma_45_deg_rel_err": relative_error(g_45, g_ref_vec),
    }

    return {
        "hrel": hrel_grid,
        "h": h_values,
        "gamma_ref": g_ref_vec,
        "gamma_fd": g_fd,
        "gamma_cs_real": g_cs,
        "gamma_45_deg": g_45,
        **errors,
    }


def scenario_dir(slug: str) -> Path:
    """Create a per-scenario folder so CSVs/plots stay grouped."""

    directory = RESULTS_DIR / slug
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_greeks_csv(path: Path, delta_data: Dict[str, np.ndarray], gamma_data: Dict[str, np.ndarray]) -> None:
    """Persist the combined delta / gamma sweep per assignment spec."""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "h_rel",
        "h",
        "Delta_analytic",
        "Delta_fd",
        "Delta_cs",
        "err_D_fd",
        "err_D_cs",
        "Gamma_analytic",
        "Gamma_fd",
        "Gamma_cs_real",
        "Gamma_cs_45",
        "err_G_fd",
        "err_G_cs_real",
        "err_G_cs_45",
    ]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        rows = len(delta_data["hrel"])
        for idx in range(rows):
            writer.writerow(
                [
                    delta_data["hrel"][idx],
                    delta_data["h"][idx],
                    delta_data["delta_ref"][idx],
                    delta_data["delta_fd"][idx],
                    delta_data["delta_cs"][idx],
                    delta_data["delta_fd_abs_err"][idx],
                    delta_data["delta_cs_abs_err"][idx],
                    gamma_data["gamma_ref"][idx],
                    gamma_data["gamma_fd"][idx],
                    gamma_data["gamma_cs_real"][idx],
                    gamma_data["gamma_45_deg"][idx],
                    gamma_data["gamma_fd_abs_err"][idx],
                    gamma_data["gamma_cs_abs_err"][idx],
                    gamma_data["gamma_45_deg_abs_err"][idx],
                ]
            )


def plot_error_curves(
    path: Path,
    hrel: np.ndarray,
    curves: Tuple[Tuple[str, np.ndarray], ...],
    title: str,
    ylabel: str,
) -> None:
    """Save a log-log error plot for quick visual inspection."""

    markers = ["o", "s", "^", "D", "v"]
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, (label, values) in enumerate(curves):
        marker = markers[idx % len(markers)]
        ax.loglog(hrel, values, marker=marker, label=label)

    ax.set_xlabel("h_rel")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_error_summary(title: str, method_errors: Dict[str, np.ndarray]) -> None:
    """Pretty-print absolute error statistics for multiple methods."""

    header = "method              max_abs_err     median_abs_err      p99_abs_err"
    print(f"{title}")
    print(header)
    print("-" * len(header))

    for method, values in method_errors.items():
        stats = summarize_errors(values)
        print(
            f"{method:<18} {stats['max']:>12.3e}   {stats['median']:>15.3e}   {stats['p99']:>12.3e}"
        )


def generate_stress_test_dataset(n, seed=1234):
    """Assemble heterogeneous market regimes to probe numerical robustness."""

    def mix_concat(arrs):  # 快速拼接
        return np.concatenate(arrs, axis=0)
    
    rng = np.random.default_rng(seed)
    

    # Regular input batch
    S_reg = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    K_reg = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    r_reg = rng.uniform(-0.01, 0.05, size=n)
    q_reg = rng.uniform(0.00, 0.03, size=n)
    sig_reg = rng.uniform(0.05, 0.60, size=n)
    T_reg = rng.uniform(1 / 365, 2.0, size=n)

    # --- Near_expiriry & small voltality, small tau, ATM-oriented
    m = n//4
    S_ne = rng.lognormal(np.log(100.0), 0.05, m)
    K_ne = S_ne * rng.lognormal(0.0, 0.001, m)              # close to ATM
    T_ne = rng.uniform(1/3650, 3/365, m)                    # 0.1~3 days
    sig_ne = rng.uniform(1e-3, 8e-3, m)                     # 0.1%~0.8%
    r_ne = rng.uniform(-0.01, 0.03, m); 
    q_ne = rng.uniform(0.0, 0.02, m)

    # --- Deep ITM/OTM
    k = n//4
    S_deep = rng.lognormal(np.log(100.0), 0.25, k)
    lm = rng.choice([-1, 1], size=k) * rng.uniform(1.0, 2.5, k)  # |logM|∈[1,2.5]
    K_deep = S_deep * np.exp(-lm)                                # far from S_deep
    T_deep = rng.uniform(5/365, 1.5, k)
    sig_deep = rng.uniform(0.10, 0.60, k)
    r_deep = rng.uniform(-0.01, 0.05, k); q_deep = rng.uniform(0.0, 0.03, k)

    # high voltality + long durations
    h = n//6
    S_hv = rng.lognormal(np.log(100.0), 0.25, h)
    K_hv = rng.lognormal(np.log(100.0), 0.25, h)
    T_hv = rng.uniform(1.0, 5.0, h)
    sig_hv = rng.uniform(0.6, 1.2, h)                            # 60%~120%
    r_hv = rng.uniform(-0.005, 0.04, h); q_hv = rng.uniform(0.0, 0.03, h)

    # --- 点刺：严格 ATM、T→0、σ→0（用于验证平台/爆炸）
    p = 128
    S_pt = np.full(p, 100.0); K_pt = np.full(p, 100.0)
    T_pt = np.geomspace(1e-6, 1e-3, p)                           # 秒级到分钟级（以年计）
    sig_pt = np.geomspace(1e-4, 1e-2, p)                         # 0.01%~1%
    r_pt = np.zeros(p); q_pt = np.zeros(p)

    # 合并：every bucket is concatenated to form the final stress tensor
    S = mix_concat([S_reg, S_ne, S_deep, S_hv, S_pt])
    K = mix_concat([K_reg, K_ne, K_deep, K_hv, K_pt])
    r = mix_concat([r_reg, r_ne, r_deep, r_hv, r_pt])
    q = mix_concat([q_reg, q_ne, q_deep, q_hv, q_pt])
    sigma = mix_concat([sig_reg, sig_ne, sig_deep, sig_hv, sig_pt])
    T = mix_concat([T_reg, T_ne, T_deep, T_hv, T_pt])

    return S, K, r, q, sigma, T


def route_hrel(S: np.ndarray, K: np.ndarray, sigma: np.ndarray, T: np.ndarray) -> Dict[str, Any]:
    """
    Heuristic router that chooses h_rel per input using intermediate indicators.

    The rules mimic practitioner intuition:
      * Very small τ = σ√T => aggressive shrinkage of h to avoid cancellation.
      * Deep ITM/OTM (|log(S/K)| large) => grow h to fight subtraction error.
      * Otherwise stick to the well-behaved 1e-8 baseline from single-scenario sweeps.
    """

    S_safe = np.maximum(S, 1e-12)
    K_safe = np.maximum(K, 1e-12)
    tau = sigma * np.sqrt(np.maximum(T, 0.0))
    moneyness = np.abs(np.log(S_safe / K_safe))
    atm_mask = moneyness <= 1e-3

    h_rel = np.full_like(S_safe, 1e-8)

    ultra_short_tau = tau < 1e-4
    short_tau = (tau >= 1e-4) & (tau < 1e-3)
    large_tau = tau > 0.5
    deep_otm_itm = moneyness > 0.5
    extreme_otm_itm = moneyness > 1.0

    h_rel[short_tau] = 5e-11
    h_rel[ultra_short_tau] = 5e-12
    h_rel[large_tau] = np.maximum(h_rel[large_tau], 2e-8)
    h_rel[deep_otm_itm] = np.maximum(h_rel[deep_otm_itm], 5e-8)
    h_rel[extreme_otm_itm] = np.maximum(h_rel[extreme_otm_itm], 2e-7)

    # Sub-cent ATM nodes with tiny σ benefit from the smallest step we permit
    atm_tiny_sigma = atm_mask & (sigma < 2e-3)
    h_rel[atm_tiny_sigma] = np.minimum(h_rel[atm_tiny_sigma], 1e-12)

    h_rel = np.clip(h_rel, HREL_MIN, 5e-5)

    return {
        "h_rel": h_rel,
        "tau": tau,
        "moneyness": moneyness,
        "atm_mask": atm_mask,
        "segments": {
            "ultra_short_tau": ultra_short_tau,
            "short_tau": short_tau & ~ultra_short_tau,
            "large_tau": large_tau,
            "deep_otm_itm": deep_otm_itm,
            "extreme_otm_itm": extreme_otm_itm,
            "atm_tiny_sigma": atm_tiny_sigma,
        },
    }


def print_router_diagnostics(h_rel: np.ndarray, indicators: Dict[str, np.ndarray]) -> None:
    """Summarize how the router partitioned the batch."""

    tau = indicators["tau"]
    moneyness = indicators["moneyness"]
    atm_mask = indicators["atm_mask"]
    segments = indicators["segments"]

    total = h_rel.size
    quantiles = np.quantile(h_rel, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(
        "Router h_rel stats:"
        f" min={quantiles[0]:.1e}, q25={quantiles[1]:.1e}, median={quantiles[2]:.1e},"
        f" q75={quantiles[3]:.1e}, max={quantiles[4]:.1e}"
    )
    print(
        f"tau range [{tau.min():.2e}, {tau.max():.2e}], "
        f"moneyness range [{moneyness.min():.2e}, {moneyness.max():.2e}], "
        f"ATM share {atm_mask.mean():.1%}"
    )

    for name, mask in segments.items():
        if mask is None:
            continue
        pct = mask.mean()
        if pct < 1e-4:
            continue
        print(f"  segment {name:<18} count={mask.sum():>6} ({pct:>6.2%}), h_rel~{np.median(h_rel[mask]):.1e}")


def run_parallel_stress_test(batch_size: int = 40000) -> None:
    """
    Exercise the complex-step delta + 45° gamma estimators without fallbacks.

    A router allocates h_rel per path, then both Greeks are evaluated with the
    routed absolute step h = h_rel * S. This emulates a production-style batch
    where different regimes share the same kernel launch but retain numerical
    stability.
    """

    S, K, r, q, sigma, T = generate_stress_test_dataset(batch_size)
    routing = route_hrel(S, K, sigma, T)
    h_rel = routing["h_rel"]
    h_abs = h_rel * np.maximum(S, 1e-12)

    start = time.perf_counter()
    delta = delta_cs(S, K, r, q, sigma, T, h=h_abs)
    gamma = gamma_45_deg(S, K, r, q, sigma, T, h=h_abs)
    elapsed = time.perf_counter() - start

    print("\n-------------------------------------------------------------")
    print(f"Stress batch size {batch_size:,} evaluated in {elapsed:.3f} seconds.")
    print("-------------------------------------------------------------\n")
    

    print_router_diagnostics(h_rel, routing)
    print(
        f"delta stats -> min {delta.min():.3e}, median {np.median(delta):.3e}, max {delta.max():.3e}; "
        f"gamma stats -> min {gamma.min():.3e}, median {np.median(gamma):.3e}, max {gamma.max():.3e}"
    )


def main() -> None:
    hrel_grid = np.logspace(math.log10(HREL_MIN), math.log10(HREL_MAX), num=HREL_POINTS, dtype=np.float64)

    for scenario in SCENARIOS:
        slug = slugify(scenario.name)
        out_dir = scenario_dir(slug)

        print(f"\nScenario: {scenario.name}")

        # Delta sweep
        delta_data = run_delta_sweep(scenario, hrel_grid)
        delta_plot = out_dir / "delta_errors.png"

        plot_error_curves(
            delta_plot,
            delta_data["hrel"],
            (
                ("delta forward", delta_data["delta_fd_abs_err"]),
                ("delta complex-step", delta_data["delta_cs_abs_err"]),
            ),
            "Delta absolute error vs h_rel (Python)",
            "|delta_numeric - delta_analytic|",
        )

        print_error_summary(
            "Delta absolute error summary",
            {
                "delta forward": delta_data["delta_fd_abs_err"],
                "delta complex-step": delta_data["delta_cs_abs_err"],
            },
        )
        print(f"Saved delta plot to {delta_plot}")

        # Gamma sweep
        gamma_data = run_gamma_sweep(scenario, hrel_grid)
        gamma_plot = out_dir / "gamma_errors.png"

        plot_error_curves(
            gamma_plot,
            gamma_data["hrel"],
            (
                ("gamma forward", gamma_data["gamma_fd_abs_err"]),
                ("gamma cs real", gamma_data["gamma_cs_abs_err"]),
                ("gamma 45deg", gamma_data["gamma_45_deg_abs_err"]),
            ),
            "Gamma absolute error vs h_rel (Python)",
            "|gamma_numeric - gamma_analytic|",
        )

        print_error_summary(
            "Gamma absolute error summary",
            {
                "gamma forward": gamma_data["gamma_fd_abs_err"],
                "gamma cs real": gamma_data["gamma_cs_abs_err"],
                "gamma 45deg": gamma_data["gamma_45_deg_abs_err"],
            },
        )
        print(f"Saved gamma plot to {gamma_plot}")

        combined_csv = out_dir / "bs_fd_vs_complex.csv"
        write_greeks_csv(combined_csv, delta_data, gamma_data)
        print(f"Saved combined CSV to {combined_csv}")

    # Parallel stress harness uses routed h_rel with delta CS + gamma 45° only
    run_parallel_stress_test(batch_size=40_000)
if __name__ == "__main__":
    main()
