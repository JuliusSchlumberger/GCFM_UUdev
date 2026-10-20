"""Render Snakemake rule graphs and job DAGs for pipeline targets to figs/dag/.

Run from the project root (`python plot_dag.py`); needs Graphviz `dot`
(conda install graphviz -c conda-forge). Writes <target>_rulegraph.<fmt> and
<target>_dag.<fmt> (one node per job; large for many basins).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def render_graph(
    target: str, graph_flag: str, suffix: str, out_dir: Path, fmt: str
) -> None:
    """Render one Snakemake graph of target through Graphviz; warn on failure."""
    out_file = out_dir / f"{target}_{suffix}.{fmt}"
    print(f"  {graph_flag}  →  {out_file}")

    # --forceall shows every rule even when outputs exist; DOT is on stdout.
    result = subprocess.run(
        ["snakemake", target, graph_flag, "--forceall", "--quiet"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [warning] snakemake exited with code {result.returncode} — skipping.")
        return

    dot_result = subprocess.run(
        ["dot", f"-T{fmt}", "-o", str(out_file)],
        input=result.stdout,
        capture_output=True,
        text=True,
    )
    if dot_result.returncode != 0:
        print(f"  [warning] dot rendering failed:\n{dot_result.stderr.strip()}")
    else:
        print(f"  Written: {out_file}")


def main() -> None:
    """Render the graphs for every requested target."""
    parser = argparse.ArgumentParser(
        description="Plot Snakemake DAGs for pipeline targets."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["preprocess", "build"],
        metavar="TARGET",
        help="Snakemake targets to visualise (default: preprocess build)",
    )
    parser.add_argument(
        "--out-dir",
        default="figs/dag",
        metavar="DIR",
        help="Output directory (default: figs/dag)",
    )
    parser.add_argument(
        "--format",
        default="svg",
        choices=["svg", "png", "pdf"],
        metavar="FMT",
        help="Output format: svg, png, or pdf (default: svg)",
    )
    parser.add_argument(
        "--no-dag",
        action="store_true",
        help="Skip the full per-job DAG (only generate rulegraph)",
    )
    args = parser.parse_args()

    if not shutil.which("snakemake"):
        sys.exit("Error: 'snakemake' not found. Activate your conda environment first.")
    if not shutil.which("dot"):
        sys.exit(
            "Error: Graphviz 'dot' not found.\n"
            "Install it with:  conda install graphviz -c conda-forge"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        print(f"\nTarget: {target}")
        render_graph(target, "--rulegraph", "rulegraph", out_dir, args.format)
        if not args.no_dag:
            render_graph(target, "--dag", "dag", out_dir, args.format)

    print(f"\nDone. Open the .{args.format} files in {out_dir}/")


if __name__ == "__main__":
    main()
