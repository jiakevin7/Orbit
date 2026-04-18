from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit.png_plots import generate_run_plots, plotting_available
from orbit.visualizer import generate_reports, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an HTML report for an Orbit benchmark run")
    parser.add_argument("run_dir", help="benchmark output directory")
    parser.add_argument("--output", help="optional output HTML path for a single report")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="do not descend into seed-* subdirectories",
    )
    parser.add_argument(
        "--skip-pngs",
        action="store_true",
        help="generate HTML only and skip seaborn PNG plot generation",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not args.skip_pngs:
        if plotting_available():
            for plot_path in generate_run_plots(run_dir, recursive=not args.no_recursive):
                print(plot_path)
        else:
            print("warning: seaborn plotting dependencies are unavailable; skipping PNG plots", file=sys.stderr)
    if args.output:
        report_path = write_report(run_dir, args.output)
        print(report_path)
        return 0

    report_paths = generate_reports(run_dir, recursive=not args.no_recursive)
    for report_path in report_paths:
        print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
