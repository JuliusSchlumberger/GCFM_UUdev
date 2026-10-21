"""
Check the scale-1 strategy expansion (00_common.smk: strategy_def_for) and
the run-time resolver (src/adaptation_scaling.py: resolve_forcing_params)
on an existing results folder, without running Snakemake.

Prints, for every strategy x scenario, the concrete crest/discharge values a
real 18a/18b job would use, plus the resolver's own log lines (matched weir
segments, where the river maximum sits, ...).

Usage (repo root, hmt_sfincs_dev env):
    python tests/KL_check_adaptation_scaling.py D:/GCFM_UU/results/2433835_V5_bigpoly_manual coast_100 river_100
"""

import logging
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workflow"))
from src.adaptation_scaling import resolve_forcing_params  # noqa: E402

RAW_DATA_ROOT = "D:/GCFM_UU/raw_data"


def load_strategy_expansion():
    """Execute only the strategy block of 00_common.smk against the real
    config files -> its namespace (STRATEGY_DEFS, strategy_def_for, ...).
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
