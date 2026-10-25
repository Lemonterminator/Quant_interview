# Repository map

Run commands below from the repository root.

| Directory | Responsibility |
| --- | --- |
| `Python_prototype/` | Python scalar and NumPy pricing APIs, plus legacy script entry points. |
| `BS_cpp/` | C++ scalar/batch APIs, CMake targets, contract tests, and benchmark. |
| `experiments/assignment/` | Assignment sweeps, C++ result plotting, and numerical-method demos. |
| `artifacts/assignment/python/` | Historical and newly generated Python CSVs and plots. |
| `artifacts/assignment/cpp/` | Historical and newly generated C++ CSVs and plots. |
| `notebooks/CAPM/` | Standalone CAPM notebooks and their images. |
| `notebooks/covariance/` | Covariance notebook and its local `data/ticker_map.csv`. |
| `tests/`, `testdata/` | Python contracts and shared Python/C++ golden cases. |
| `benchmarks/` | Python throughput benchmark. |
| `docs/` | Assignment plan and batch testing notes. |
| `strategies/gold_qqq_cash/` | Cash / gold / QQQ sub-strategy research: data pipeline (Yahoo + FRED), studies, reports and artifacts. See its README. |

The pricing imports and C++ target names stay stable. The former Python experiment paths remain thin wrappers, so existing commands such as `python Python_prototype/black_scholes_test.py` still work. New work should use the canonical scripts in `experiments/assignment/`.

`python experiments/assignment/black_scholes_sweep.py` writes under `artifacts/assignment/python/`. `bs_greeks_test` writes under `artifacts/assignment/cpp/` when launched from the repository root. `python experiments/assignment/plot_cpp_results.py` reads the C++ CSVs there. Both sweeps default to P3, accept `--order 2..6`, and write new results into `p{order}/<slug>/` subdirectories. Public C++ complex kernels live in `BS_cpp/complex_step.h/.cpp`; see [API configuration](complex_step_api.md). `Make_cpp.ps1` builds in `out/bs_cpp/` and runs both C++ sweep and plotting.

The notebook data is kept beside its notebook so `DATA_DIR = Path("data")` resolves when the notebook is opened with `notebooks/covariance/` as its working directory. The notebook itself was moved without changing its cells.

`graphify-out/` is a local generated knowledge graph and is ignored by Git. Rebuild it after future structural changes with the Graphify skill.

The old `BS_cpp/build_ninja/` build files are no longer part of the source tree. A local copy is retained under the ignored `out/legacy-build_ninja/` directory.
