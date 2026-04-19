from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit.png_plots import generate_run_plots, plotting_available


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the curated research PNG plot set for an Orbit benchmark run"
    )
    parser.add_argument("run_dir", help="benchmark output directory")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="do not descend into seed-* subdirectories",
    )
    args = parser.parse_args()

    if not plotting_available():
        print("seaborn plotting dependencies are unavailable", file=sys.stderr)
        return 1

    run_dir = Path(args.run_dir).resolve()
    for plot_path in generate_run_plots(run_dir, recursive=not args.no_recursive):
        print(plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
