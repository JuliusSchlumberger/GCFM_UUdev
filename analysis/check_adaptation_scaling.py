"""Print the resolved adaptation parameters per strategy and scenario.

Runs the strategy expansion (00_common.smk: strategy_def_for) and the resolver
(src.adaptation_scaling.resolve_forcing_params) on an existing results folder
without Snakemake, printing the crest/discharge values an 18a/18b job would use
and the resolver's log lines.

Usage (repo root, project env activated):
    python analysis/check_adaptation_scaling.py <results_dir>/2433835 coast_500 river_500
"""

import logging
import re
import sys
from pathlib import Path

import yaml
from _common import REPO_ROOT as REPO

sys.path.insert(0, str(REPO / "workflow"))
from src.adaptation_scaling import resolve_forcing_params  # noqa: E402
from src.io import require_keys, site_setting  # noqa: E402

RAW_DATA_ROOT = site_setting("raw_data_root")


def load_strategy_expansion():
    """Exec the strategy block of 00_common.smk and return its namespace.

    The block is cut out by two marker strings in 00_common.smk.
    """
    src = (REPO / "workflow/rules/00_common.smk").read_text(encoding="utf-8")
    block = src[
        src.index("# ---- Adding adaptation wildcard") : src.index(
            "def adaptation_input_path(rel):"
        )
    ]
    with open(REPO / "config/scenarios.yml") as f:
        scenario_defs = yaml.safe_load(f)
    ns = {
        "_yaml": yaml,
        "_re": re,
        "Path": Path,
        "require_keys": require_keys,
        "SCENARIO_DEFS": scenario_defs,
        "CATALOGUE": {"meta": {"root": RAW_DATA_ROOT}},
        "config": {
            "adaptation": {
                "strategies_file": str(REPO / "config/adaptation_strategies.yml"),
                "measures_file": str(REPO / "config/measures.yml"),
            },
            "scenarios_file": "config/scenarios.yml",
        },
    }
    exec(block, ns)
    return ns


def main(results, scenarios):
    """Print the resolved measure parameters for every strategy x scenario."""
    results = Path(results)
    ns = load_strategy_expansion()
    logging.basicConfig(
        level=logging.INFO, format="      %(message)s", stream=sys.stdout
    )
    log = logging.getLogger("check_adaptation_scaling")

    for sc in scenarios:
        root = results / "runs" / sc / "sfincs"
        print(f"\n===== {sc} (event={ns['SCENARIO_DEFS'][sc].get('event')}) =====")
        for st in ns["STRATEGY_DEFS"]:
            print(f"  {st}")
            resolved = resolve_forcing_params(
                ns["strategy_def_for"](st, sc),
                baseline_sfincs_root=root,
                baseline_map_nc=root / "sfincs_map.nc",
                skeleton_weir=results / "sfincs_skeleton/sfincs.weir",
                adaptation_root=RAW_DATA_ROOT,
                log=log,
            )
            for m, params in resolved["measures"].items():
                for p, v in params.items():
                    if p not in ("locations", "merge", "target_code", "min_elevation"):
                        print(f"    -> {m}.{p} = {v}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
