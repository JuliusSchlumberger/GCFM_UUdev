"""Print the resolved adaptation parameters per strategy and scenario.

Runs the strategy expansion (src.adaptation_strategies) and the resolver
(src.adaptation_scaling.resolve_forcing_params) on an existing results folder
without Snakemake, printing the crest/discharge values an 18a/18b job would use
and the resolver's log lines.

Usage (repo root, project env activated):
    python analysis/check_adaptation_scaling.py <results_dir>/2433835 coast_500 river_500
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from _common import REPO_ROOT as REPO
from _common import script_parser

sys.path.insert(0, str(REPO / "workflow"))
from src.adaptation_scaling import resolve_forcing_params  # noqa: E402
from src.adaptation_strategies import (  # noqa: E402
    expand_strategies,
    resolve_locations,
)
from src.io import site_setting  # noqa: E402


def load_strategy_expansion():
    """Load the strategy, measure and scenario configs and expand the strategies."""
    files = {
        "strategies_file": "config/adaptation_strategies.yml",
        "measures_file": "config/measures.yml",
        "scenarios_file": "config/scenarios.yml",
    }
    strategies_raw, measures_raw, scenario_defs = (
        yaml.safe_load((REPO / f).read_text(encoding="utf-8")) or {}
        for f in files.values()
    )
    return expand_strategies(strategies_raw, measures_raw, scenario_defs, **files)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = script_parser(__doc__)
    parser.add_argument(
        "results", help="basin results folder, <results_dir>/<basin_id>"
    )
    parser.add_argument("scenarios", nargs="+", help="scenario name(s)")
    return parser


def main(results, scenarios):
    """Print the resolved measure parameters for every strategy x scenario."""
    results = Path(results)
    adaptation_data_root = site_setting("adaptation_data_root")
    expansion = load_strategy_expansion()
    logging.basicConfig(
        level=logging.INFO, format="      %(message)s", stream=sys.stdout
    )
    log = logging.getLogger("check_adaptation_scaling")

    for sc in scenarios:
        root = results / "runs" / sc / "sfincs"
        print(f"\n===== {sc} (event={expansion.scenario_defs[sc].get('event')}) =====")
        for st in expansion.defs:
            print(f"  {st}")
            resolved = resolve_forcing_params(
                resolve_locations(
                    expansion.strategy_def_for(st, sc),
                    results / "adaptation_lines",
                    results.name,
                ),
                baseline_sfincs_root=root,
                baseline_map_nc=root / "sfincs_map.nc",
                skeleton_weir=results / "sfincs_skeleton/sfincs.weir",
                adaptation_root=adaptation_data_root,
                log=log,
            )
            for m, params in resolved["measures"].items():
                for p, v in params.items():
                    if p not in ("locations", "merge", "target_code", "min_elevation"):
                        print(f"    -> {m}.{p} = {v}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args.results, args.scenarios)
