"""
_snakemake_script_runner.py — executes one workflow/scripts/*.py file
(written as a Snakemake `script:` target, i.e. it reads a global `snakemake`
object for input/output/params/log) standalone, outside of Snakemake.

Used by tests/test_grid_resolution_benchmark.py to reuse 13_build_sfincs.py /
14_run_spinup.py / 16_run_event.py verbatim for many different parameter
combinations, without re-running Snakemake's rule graph (which would tie
every combination's build to the same results_dir as its expensive,
resolution-independent preprocessing outputs).

Run as its own subprocess per script call (rather than importing/exec'ing
in-process) so that logging.basicConfig() -- a one-time no-op after its
first call in a process -- gets a fresh interpreter each time, and so that
no state (matplotlib figures, xarray/rasterio file handles, hydromt caches)
can leak between one combination's build/run and the next across a sweep
that may span many hours.

Usage:
    python _snakemake_script_runner.py <script_name> <config_json_path>

<config_json_path> is a JSON file with keys "input", "output", "params"
(each a flat dict of the snakemake.{input,output,params}.<name> values this
script needs) and "log" (a single string path).
"""

import json
import runpy
import sys
import types
from pathlib import Path


class _SnakemakeMock:
    """Minimal stand-in for Snakemake's own injected `snakemake` object."""

    def __init__(self, input_d: dict, output_d: dict, params_d: dict, log_path: str):
        self.input = types.SimpleNamespace(**input_d)
        self.output = types.SimpleNamespace(**output_d)
        self.params = types.SimpleNamespace(**params_d)
        self.log = [log_path]


def main() -> None:
    script_name = sys.argv[1]
    config_json_path = sys.argv[2]

    with open(config_json_path) as fh:
        cfg = json.load(fh)

    mock = _SnakemakeMock(cfg["input"], cfg["output"], cfg["params"], cfg["log"])

    workflow_dir = Path(__file__).resolve().parents[1] / "workflow"
    # Snakemake's `script:` directive puts the Snakefile's own directory
    # (workflow/) on sys.path, which is how e.g. `from src.quadtree_refinement
    # import ...` resolves normally -- replicate that here since we run
    # scripts standalone via runpy instead of through Snakemake.
    sys.path.insert(0, str(workflow_dir))
    script_path = workflow_dir / "scripts" / script_name
    runpy.run_path(
        str(script_path), init_globals={"snakemake": mock}, run_name="__main__"
    )


if __name__ == "__main__":
    main()
