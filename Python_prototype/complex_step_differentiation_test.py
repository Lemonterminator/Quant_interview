"""Legacy entry point; complex-step demonstrations live in experiments/."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.assignment.complex_step_demo import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
