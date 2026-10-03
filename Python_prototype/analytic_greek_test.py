"""Legacy entry point; analytic Greek demonstrations now live in experiments/."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.assignment.analytic_greek_demo import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
