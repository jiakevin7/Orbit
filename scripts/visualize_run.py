import argparse
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from orbit.png_plots import generate_run_plots, plotting_available

def main():
    parser = argparse.ArgumentParser(description='Generate PNG plots for an Orbit benchmark run')
    parser.add_argument('run_dir', help='benchmark output directory')
    parser.add_argument('--no-recursive', action='store_true', help='do not descend into seed-* subdirectories')
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not plotting_available():
        print('warning: seaborn plotting dependencies are unavailable; skipping PNG plots', file=sys.stderr)
        return 0
    for plot_path in generate_run_plots(run_dir, recursive=not args.no_recursive):
        print(plot_path)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
