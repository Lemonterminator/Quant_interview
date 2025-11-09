from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


RESULTS_CPP = Path("results_cpp")


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read the greeks CSV produced by the C++ harness."""

    with path.open() as f:
        reader = csv.DictReader(f)
        columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                if value is None:
                    continue
                columns[key].append(float(value))

    return {key: np.array(values, dtype=np.float64) for key, values in columns.items()}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plot(
    path: Path,
    hrel: np.ndarray,
    curves: Tuple[Tuple[str, np.ndarray], ...],
    title: str,
    ylabel: str,
) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = ("o", "s", "^", "D", "v")
    for idx, (label, series) in enumerate(curves):
        ax.loglog(hrel, series, marker=markers[idx % len(markers)], label=label)
    ax.set_xlabel("h_rel")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_for_csv(csv_path: Path) -> None:
    data = load_csv(csv_path)
    hrel = data["h_rel"]
    scenario_dir = csv_path.parent

    delta_plot = scenario_dir / "delta_errors_cpp.png"
    save_plot(
        delta_plot,
        hrel,
        (
            ("delta forward (C++)", data["err_D_fd"]),
            ("delta complex-step (C++)", data["err_D_cs"]),
        ),
        "Delta absolute error vs h_rel (C++)",
        "|delta_numeric - delta_analytic|",
    )

    gamma_plot = scenario_dir / "gamma_errors_cpp.png"
    save_plot(
        gamma_plot,
        hrel,
        (
            ("gamma forward (C++)", data["err_G_fd"]),
            ("gamma CS real (C++)", data["err_G_cs_real"]),
            ("gamma 45deg (C++)", data["err_G_cs_45"]),
        ),
        "Gamma absolute error vs h_rel (C++)",
        "|gamma_numeric - gamma_analytic|",
    )

    print(f"Rendered delta/gamma plots for {scenario_dir.name}")


def main() -> None:
    if not RESULTS_CPP.exists():
        raise FileNotFoundError(f"{RESULTS_CPP} not found; run the C++ harness first.")

    for csv_path in RESULTS_CPP.glob("*/bs_fd_vs_complex.csv"):
        plot_for_csv(csv_path)


if __name__ == "__main__":
    main()
