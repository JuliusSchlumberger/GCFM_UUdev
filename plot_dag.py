"""
plot_dag.py — Generate DAG visualisations for the preprocess and build targets.

Run from the project root:
    python plot_dag.py

Requirements:
    Graphviz (dot command) — install via:
        conda install graphviz -c conda-forge

Outputs (written to figs/dag/):
    *_rulegraph.svg  — rule-level graph  (compact, best for understanding structure)
    *_dag.svg        — full job-level DAG (one node per job; large for many basins)

--forceall makes Snakemake show every rule even when outputs already exist.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def render_graph(
    target: str, graph_flag: str, suffix: str, out_dir: Path, fmt: str
) -> None:
    out_file = out_dir / f"{target}_{suffix}.{fmt}"
    print(f"  {graph_flag}  →  {out_file}")

    # Run snakemake and capture DOT output on stdout; discard its own logging (stderr).
    result = subprocess.run(
        ["snakemake", target, graph_flag, "--forceall", "--quiet"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [warning] snakemake exited with code {result.returncode} — skipping.")
        return

    # Pipe the DOT text into graphviz dot.
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

    # ── check dependencies ────────────────────────────────────────────────────
    if not shutil.which("snakemake"):
        sys.exit("Error: 'snakemake' not found. Activate your conda environment first.")
    if not shutil.which("dot"):
        sys.exit(
            "Error: Graphviz 'dot' not found.\n"
            "Install it with:  conda install graphviz -c conda-forge"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── generate graphs ───────────────────────────────────────────────────────
    for target in args.targets:
        print(f"\nTarget: {target}")
        render_graph(target, "--rulegraph", "rulegraph", out_dir, args.format)
        if not args.no_dag:
            render_graph(target, "--dag", "dag", out_dir, args.format)

    print(f"\nDone. Open the .{args.format} files in {out_dir}/")


if __name__ == "__main__":
    main()
