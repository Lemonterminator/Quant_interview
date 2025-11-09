Quant Black–Scholes Greeks: Analytics, FD, and Complex-Step (Python + C++)

This repository implements analytic and numerical Greeks for Black–Scholes in both Python and C++, validates them across two scenarios, emits assignment‑format CSVs, and produces error plots.

Repository layout

- `Python_prototype/`: Analytic Greeks, FD, complex-step, sweeps, plots, and stress harness.
- `BS_cpp/`: C++ core implementations and a test harness that mirrors the Python sweeps.
- `results/`, `results_cpp/`: Per‑scenario outputs (CSVs + plots).

Quick start

- Python (3.10+)
  - Install dependencies: `pip install -r requirements.txt`
  - Run sweeps + plots (writes per‑scenario CSV + PNGs under `Python_prototype/results/`):
    - `python Python_prototype/black_scholes_test.py`
  - Plot C++ CSVs (optional): `python Python_prototype/plot_cpp_results.py`

- C++ (CMake 3.15+, MSVC/Clang/GCC)
  - PowerShell one‑liner (build + run + plot C++ CSVs): `./Make_cpp.ps1`
  - Manual steps:
    - Configure: `cmake -S BS_cpp -B BS_cpp/build`
    - Build: `cmake --build BS_cpp/build --config Release`
    - Run: `BS_cpp/build/Release/bs_greeks_test.exe`
    - Plot C++ CSVs: `python Python_prototype/plot_cpp_results.py`

CSV schema (assignment format)

- File per scenario: `bs_fd_vs_complex.csv`
- Header:
  - `h_rel,h,Delta_analytic,Delta_fd,Delta_cs,err_D_fd,err_D_cs,`
  - `Gamma_analytic,Gamma_fd,Gamma_cs_real,Gamma_cs_45,err_G_fd,err_G_cs_real,err_G_cs_45`
- Meaning:
  - `h_rel`, `h`: relative/absolute steps; `h = h_rel * S`.
  - `Delta_*`: analytic, forward FD, and complex‑step Delta, with absolute errors.
  - `Gamma_*`: analytic, forward FD, complex‑step (real‑part), and 45° complex‑step Gamma, with absolute errors.

Validation report

- Scenarios
  - S1 (ATM reference): `S=K=100, r=q=0, σ=20%, T=1`
  - S2 (Near‑expiry, low‑vol): `S=K=100, r=q=0, σ=1%, T=1/365`

- Error summaries (max/median/p99 across `h_rel`)

  - Delta (S1)
    - FD: max 1.707e−01, median 5.839e−06, p99 1.573e−01
    - Complex‑step: max 4.962e−09, median 2.220e−16, p99 3.924e−09

  - Gamma (S1)
    - FD: max 7.105e+13, median 1.987e−01, p99 5.511e+13 (unstable)
    - CS real: max 3.300e−01, median 1.985e−02, p99 2.714e−01
    - CS 45°: max 1.985e−02, median 9.335e−09, p99 1.605e−02

  - Delta (S2)
    - FD: max 9.210e−01, median 1.506e−04, p99 7.259e−01
    - Complex‑step: max 1.975e−06, median 6.212e−14, p99 1.561e−06

  - Gamma (S2)
    - FD: max 1.421e+14, median 1.069e+01, p99 1.109e+14 (unstable)
    - CS real: max 7.622e+00, median 7.622e+00, p99 7.622e+00
    - CS 45°: max 7.622e+00, median 1.323e−04, p99 6.133e+00

Observations

- Accuracy ranking
  - Delta: complex‑step ≪ forward FD.
  - Gamma: 45° complex‑step < CS real ≪ forward FD.

- Step‑size behavior
  - FD shows the classic U‑shape error vs. `h`: truncation (↑ with large `h`) vs. roundoff (↑ with small `h`).
  - Delta CS remains flat until ≈ `h_rel≈1e−8`, then increases with the expected O(h²) trend.
  - Gamma CS 45° exhibits the smallest errors, consistent with higher‑order truncation (O(h⁴)).

- Scenario differences
  - S2 has tiny `τ = σ√T` (≈5.2e−4), making Γ much larger (≈7.6) and numerical FD far more fragile.
  - Across both scenarios the method ranking is unchanged; S2 simply magnifies errors.

Stability issues and mitigations

- Analytic/log terms
  - Stable `log(F/K)` via a `log1p` branch near `F≈K` in both Python and C++.

- Complex‑step pricing preconditions
  - Require `σ√T > 0` for complex‑step pricing (Python raises, C++ throws). This avoids degenerate divisions.

- Guarded divisions and error metrics
  - Denominators are clamped (`max(…, ε)`) in error computations; Gamma uses masked branches for `S≤0`, `σ≤0`, or `T≤0`.

- FD Gamma instability
  - The three‑point stencil suffers catastrophic cancellation at small `h`; we keep it for comparison only and recommend CS alternatives.

Reasoning: truncation vs. roundoff (and why CS helps)

- FD truncation error scales like O(h) or O(h²) depending on stencil; roundoff grows like O(ε/h) from subtractive cancellation.
- The CS method perturbs in the imaginary direction, extracting derivatives from the imaginary part, which preserves O(h) accuracy without real‑axis cancellation; the real‑part and 45° constructions yield higher‑order curvature estimates (smaller truncation).

Practical recommendations

- Delta (all scenarios)
  - Use complex‑step with `h_rel≈1e−8` baseline; shrink to `1e−11…1e−12` when `τ` is ultra‑small.

- Gamma (all scenarios)
  - Prefer 45° complex‑step; start at `h_rel≈1e−8` and shrink for tiny `τ`.
  - Avoid forward FD for Γ in production unless you tune `h` per‑path and accept higher error.

- Router (batch guidance)
  - Python `route_hrel` and C++ `route_hrel` segment by `τ`, moneyness, and ATM‑tiny‑σ to pick `h_rel` in `[1e−12, 2e−7]`:
    - Ultra‑short `τ<1e−4`: `h_rel≈5e−12`.
    - Short `τ<1e−3`: `h_rel≈5e−11`.
    - Deep/extreme ITM/OTM (`|log(S/K)|>0.5/1.0`): grow `h_rel` up to `2e−7`.
    - ATM with tiny `σ<2e−3`: clamp down to `1e−12`.

Parallel and vectorized execution

- Python (NumPy vectorization)
  - All functions accept scalars or `numpy.ndarray`; broadcasting promotes scalars to arrays.
  - Math ops call NumPy ufuncs implemented in C, so a single call runs fast element‑wise loops.
  - Complex‑step paths use complex dtypes end‑to‑end, keeping the imaginary component accurate.

- C++ (scalar → vector → parallel vector)
  - Scalar API: `double delta_call(...)`, `double Gamma(...)` are the ground‑truth building blocks.
  - Vector wrappers: `void delta_call(const std::vector<double>&,...,std::vector<double>& out)` loop over scalars for easy verification.
  - Parallel+SIMD: `*_par` variants use `std::for_each(std::execution::par_unseq, ...)` and raw‑pointer indexing.
    - `par_unseq` allows the implementation to run iterations concurrently (multi‑core) and unsequenced (vectorized/SIMD).
    - Each iteration is independent (no aliasing, no synchronization), so it is safe to reorder and parallelize.
    - We precompute per‑element terms (e.g., `d1`, discount) and then apply `Phi_real` over a vector with a separate parallel pass to help auto‑vectorizers.
  - Benefits: scales across cores and uses vector units; overhead amortizes well on large batches (e.g., 40k+ stress set).

Repro tips

- Python outputs: `Python_prototype/results/<slug>/`
  - `delta_errors.png`, `gamma_errors.png`, `bs_fd_vs_complex.csv`
- C++ outputs: `results_cpp/<slug>/`
  - `bs_fd_vs_complex.csv` (+ optional `*_cpp.png` via `plot_cpp_results.py`)

Math in README

- GitHub renders LaTeX in Markdown. Examples: inline `$d_1 = \frac{\ln(F/K) + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}$`, display `$$\Gamma = e^{-qT}\,\frac{\phi(d_1)}{S\sigma\sqrt{T}}$$`.
