"""Legacy entry point; the C++ plotting script now lives in experiments/."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.assignment.plot_cpp_results import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
