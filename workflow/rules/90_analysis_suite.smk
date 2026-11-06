# Rules: the analysis suite (target analysis_suite), cross-delta figures from
# finished production runs, set by config/analysis_suite.yml. Each rule runs a
# script in analysis/; the depth-sensitivity runs are opt-in.

import os as _os

import yaml as _yaml

with open(config.get("analysis_suite_file", "config/analysis_suite.yml"), encoding="utf-8") as _f:
    SUITE = _yaml.safe_load(_f)

_S_BASINS = [str(b) for b in SUITE["basins"]]
_S_COAST = SUITE["coast_scenario"]
_S_RIVER = SUITE["river_scenario"]
_S_ARGS = " ".join(_S_BASINS)
_S_DIR = results_path("figures/analysis")
_S_DEPTH = SUITE["depth_sensitivity"]
_S_DEPTH_ROOT = _S_DEPTH.get("runs_root") or _os.path.join(
    _os.path.dirname(_os.path.normpath(RESULTS_DIR)), "depth_sensitivity_runs"
)
_S_SCALES = [f"{float(s):g}" for s in _S_DEPTH["scales"]]
_S_VMAX = bool(config["sfincs"]["simulation"]["storevelmax"])


def _s_event(scenario):
    return results_path(f"{{basin_id}}/runs/{scenario}/sfincs/sfincs_map.nc")


def _s_gvf_basin_outputs(basin_id):
    names = ["nodes.csv", "surface_options.csv"]
    if _S_VMAX:
        names.append(f"froude_sfincs_{_S_RIVER}.npz")
    return [f"{_S_DIR}/gvf_setup/{basin_id}/{n}" for n in names]


def _s_figures(folder, names):
    return [f"{_S_DIR}/{folder}/{n}{ext}" for n in names for ext in (".png", ".txt")]


rule suite_gvf_setup_basin:
    input:
        depth_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
        spinup_map    = results_path("{basin_id}/spin_up/sfincs_map.nc"),
        river_map     = _s_event(_S_RIVER),
    output:
        [f"{_S_DIR}/gvf_setup/{{basin_id}}/{n}" for n in ["nodes.csv", "surface_options.csv"]]
        + ([f"{_S_DIR}/gvf_setup/{{basin_id}}/froude_sfincs_{_S_RIVER}.npz"] if _S_VMAX else []),
    log: "logs/analysis_suite/gvf_setup_{basin_id}.log"
    shell: f"python analysis/gvf_setup.py compute {{wildcards.basin_id}} --river {_S_RIVER} > {{log}} 2>&1"


rule suite_gvf_setup_figures:
    input:
        [o for b in _S_BASINS for o in _s_gvf_basin_outputs(b)],
    output:
        _s_figures("gvf_setup", ["concept", "froude_1d", f"froude_sfincs_{_S_RIVER}", "bed_alternatives"]
                   + [f"long_profile_{b}" for b in _S_BASINS]),
    log: "logs/analysis_suite/gvf_setup_figures.log"
    shell:
        f"python analysis/gvf_setup.py figures {_S_ARGS} --river {_S_RIVER} "
        f"--fr-ok {SUITE['froude']['acceptable']} --fr-limit {SUITE['froude']['limit']} > {{log}} 2>&1"


rule suite_travel_time:
    input:
        [results_path(f"{b}/runs/{s}/sfincs/sfincs_map.nc") for b in _S_BASINS for s in (_S_RIVER, _S_COAST)],
        [results_path(f"{b}/preprocessing_inputs/domain/{b}_river_network_depth_estimated.gpkg") for b in _S_BASINS],
    output:
        summary = f"{_S_DIR}/travel_time/summary.csv",
        figures = _s_figures("travel_time", ["travel_time"]),
    log: "logs/analysis_suite/travel_time.log"
    shell: f"python analysis/travel_time.py {_S_ARGS} --coast {_S_COAST} --river {_S_RIVER} > {{log}} 2>&1"


rule suite_upstream_boundary:
    input:
        [results_path(f"{b}/runs/{_S_COAST}/sfincs/sfincs_map.nc") for b in _S_BASINS],
        [results_path(f"{b}/spin_up/sfincs_map.nc") for b in _S_BASINS],
    output:
        summary = f"{_S_DIR}/upstream_boundary/summary.csv",
        figures = _s_figures("upstream_boundary", ["upstream_boundary"]),
    log: "logs/analysis_suite/upstream_boundary.log"
    shell: f"python analysis/upstream_boundary.py {_S_ARGS} --coast {_S_COAST} > {{log}} 2>&1"


rule suite_depth_run:
    input:
        depth_network = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_river_network_depth_estimated.gpkg"),
    output:
        [_os.path.join(_S_DEPTH_ROOT, "depth_{scale}", "{basin_id}", "runs", s, "metrics", "flood_metrics.csv")
         for s in (_S_COAST, _S_RIVER)],
    wildcard_constraints:
        scale = r"[0-9.]+",
    threads: workflow.cores
    log: "logs/analysis_suite/depth_run_{basin_id}_{scale}.log"
    shell:
        f"python analysis/depth_sensitivity.py run {{wildcards.basin_id}} --scale {{wildcards.scale}} "
        f"--scenarios {_S_COAST} {_S_RIVER} --cores {{threads}} --runs-root {_S_DEPTH_ROOT} > {{log}} 2>&1"


rule suite_depth_figure:
    input:
        [results_path(f"{b}/runs/{s}/metrics/flood_metrics.csv") for b in _S_BASINS for s in (_S_COAST, _S_RIVER)],
        [_os.path.join(_S_DEPTH_ROOT, f"depth_{k}", b, "runs", s, "metrics", "flood_metrics.csv")
         for b in _S_BASINS for k in _S_SCALES for s in (_S_COAST, _S_RIVER)],
    output:
        summary = f"{_S_DIR}/depth_sensitivity/summary.csv",
        figures = _s_figures("depth_sensitivity", ["depth_sensitivity"]),
    log: "logs/analysis_suite/depth_sensitivity.log"
    shell:
        f"(python analysis/depth_sensitivity.py summary {_S_ARGS} --scales {' '.join(_S_SCALES)} "
        f"--scenarios {_S_COAST} {_S_RIVER} --runs-root {_S_DEPTH_ROOT} "
        f"&& python analysis/depth_sensitivity.py figure) > {{log}} 2>&1"


rule analysis_suite:
    """
    Cross-delta analysis figures from finished production runs (config/analysis_suite.yml).

    Usage:  snakemake analysis_suite --cores N
    """
    input:
        rules.suite_gvf_setup_figures.output,
        rules.suite_travel_time.output,
        rules.suite_upstream_boundary.output,
        rules.suite_depth_figure.output if _S_DEPTH["enabled"] else [],
