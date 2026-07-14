"""
test_grid_resolution_benchmark.py — benchmarks SFINCS BUILD time and
full-timeseries RUN time (spin-up + main event, compound forcing mode)
across combinations of main grid resolution, subgrid (on/off, n_cells), and
quadtree (on/off), for two basins (2444235, 4267691).

Reuse strategy: rather than reimplementing hydromt_sfincs/sfincs.exe
invocation, or driving full Snakemake per combination (which would tie
every combination's build to its own results_dir, this script executes workflow/scripts/
13_build_sfincs.py / 14_run_spinup.py / 16_run_event.py VERBATIM, once per
combination, via _snakemake_script_runner.py -- a small helper that
constructs the same snakemake.input/output/params/log object Snakemake
itself would inject, run as an independent subprocess per script call (so
Python's logging.basicConfig(), matplotlib figure state, and any
hydromt/xarray/rasterio caches never leak between one combination's
build/run and the next across a sweep that may span many hours to days).

Combination matrix: for each main grid resolution R, and each
(subgrid n_cells, quadtree on/off) pairing, the minimum resulting grid cell
size is:

    min_cell_size = R / quadtree_factor / n_cells      (subgrid enabled)
    min_cell_size = R / quadtree_factor                (subgrid disabled;
                                                         always quadtree=off)

where quadtree_factor = 2 ** river_refinement_level when quadtree is
enabled (config: sfincs.grid.quadtree.river_refinement_level; coastal
refinement is currently disabled in config so it is not folded in here --
extend to 2**max(river_level, coastal_level) if that changes). Combinations
with min_cell_size below MIN_CELL_FLOOR_M are excluded.

Usage:
    conda run -n hmt_sfincs_dev python tests/test_grid_resolution_benchmark.py            # pilot (default, 5 combos x 2 basins)
    conda run -n hmt_sfincs_dev python tests/test_grid_resolution_benchmark.py --full      # full sweep (31 combos x 2 basins)

Resumable: combinations already recorded in the run's CSV
(figs/grid_resolution_benchmark/grid_resolution_benchmark_{pilot,full}.csv)
are skipped on a subsequent run, so a crash partway through only costs the
one combination that was in flight -- rerun the same command to pick up
where it left off. Pass --restart to ignore the existing CSV and rerun
every combination from scratch (e.g. after a code fix invalidates prior
results).
"""

import json
import logging
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = Path(__file__).resolve().parent / "_snakemake_script_runner.py"
FIGS_DIR = REPO_ROOT / "figs" / "grid_resolution_benchmark"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = Path("D:/GCFM_UU/experiments/grid_resolution_benchmark")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)

RESULTS_DIR = Path(config["results_dir"])
BASIN_IDS = ["2444235", "4267691"]
MIN_CELL_FLOOR_M = 30.0

FULL_RESOLUTIONS = [900, 600, 300, 120, 60]
FULL_N_CELLS = [2, 4, 6, 10]

PILOT_RAW_COMBOS = [
    dict(resolution=900, subgrid_enabled=False, n_cells=None, quadtree_enabled=False),
    dict(resolution=900, subgrid_enabled=True, n_cells=4, quadtree_enabled=False),
    dict(resolution=300, subgrid_enabled=True, n_cells=4, quadtree_enabled=False),
    dict(resolution=120, subgrid_enabled=True, n_cells=2, quadtree_enabled=False),
    dict(resolution=60, subgrid_enabled=True, n_cells=2, quadtree_enabled=False),
]

RIVER_REFINEMENT_LEVEL = int(
    config["sfincs"]["grid"]["quadtree"]["river_refinement_level"]
)
COASTAL_REFINEMENT_ENABLED = bool(
    config["sfincs"]["grid"]["quadtree"]["coastal_refinement_enabled"]
)
COASTAL_REFINEMENT_LEVEL = int(
    config["sfincs"]["grid"]["quadtree"]["coastal_refinement_level"]
)


def _quadtree_factor(quadtree_enabled: bool) -> int:
    if not quadtree_enabled:
        return 1
    level = RIVER_REFINEMENT_LEVEL
    if COASTAL_REFINEMENT_ENABLED:
        level = max(level, COASTAL_REFINEMENT_LEVEL)
    return 2**level


def _finalize_combo(
    resolution: int, subgrid_enabled: bool, n_cells: int | None, quadtree_enabled: bool
) -> dict:
    qt_factor = _quadtree_factor(quadtree_enabled)
    if subgrid_enabled:
        min_cell = resolution / qt_factor / n_cells
        label = f"R{resolution}_sg{n_cells}_qt{'on' if quadtree_enabled else 'off'}"
        group = f"n={n_cells}, quadtree={'on' if quadtree_enabled else 'off'}"
    else:
        min_cell = resolution / qt_factor
        label = f"R{resolution}_nosubgrid"
        group = "no subgrid"
    return dict(
        resolution=resolution,
        subgrid_enabled=subgrid_enabled,
        n_cells=n_cells,
        quadtree_enabled=quadtree_enabled,
        min_cell_size_m=min_cell,
        label=label,
        group=group,
    )


def generate_full_combinations() -> list[dict]:
    combos = []
    for R in FULL_RESOLUTIONS:
        combos.append(_finalize_combo(R, False, None, False))
        for quadtree_enabled in (False, True):
            for n in FULL_N_CELLS:
                c = _finalize_combo(R, True, n, quadtree_enabled)
                if c["min_cell_size_m"] < MIN_CELL_FLOOR_M:
                    continue
                combos.append(c)
    return combos


def generate_pilot_combinations() -> list[dict]:
    return [
        _finalize_combo(
            c["resolution"], c["subgrid_enabled"], c["n_cells"], c["quadtree_enabled"]
        )
        for c in PILOT_RAW_COMBOS
    ]


def _run_mock_script(
    script_name: str, input_d: dict, output_d: dict, params_d: dict, log_path: Path
) -> float:
    """Run one workflow/scripts/*.py file as an isolated subprocess via
    _snakemake_script_runner.py, timing its wall-clock duration."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for v in output_d.values():
        if isinstance(v, str):
            Path(v).parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "input": input_d,
        "output": output_d,
        "params": params_d,
        "log": str(log_path),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cfg, fh)
        cfg_path = fh.name
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(RUNNER_SCRIPT), script_name, cfg_path],
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            tail_out = "\n".join(proc.stdout.splitlines()[-40:])
            tail_err = "\n".join(proc.stderr.splitlines()[-40:])
            raise RuntimeError(
                f"{script_name} failed (exit {proc.returncode}).\n"
                f"--- stdout (tail) ---\n{tail_out}\n"
                f"--- stderr (tail) ---\n{tail_err}"
            )
        return elapsed
    finally:
        Path(cfg_path).unlink(missing_ok=True)


def _mock_config_13(
    basin_id: str, combo: dict, sfincs_root: Path
) -> tuple[dict, dict, dict]:
    basin_inputs = RESULTS_DIR / basin_id / "inputs"
    conditioning_enabled = bool(config["river_processing"]["conditioning"]["enabled"])
    burn_rivers_enabled = bool(config["river_processing"]["burn_rivers"]["enabled"])
    combo_visuals = sfincs_root.parent / "visuals" / "sfincs_build"

    input_d = dict(
        domain_gpkg=str(basin_inputs / "domain" / f"{basin_id}_domain.gpkg"),
        elevation_merged=str(
            basin_inputs
            / "domain"
            / (
                f"{basin_id}_elevation_conditioned.tif"
                if conditioning_enabled
                else f"{basin_id}_elevation_merged.tif"
            )
        ),
        zbed_anchors=(
            str(basin_inputs / "domain" / f"{basin_id}_zbed_anchors.gpkg")
            if conditioning_enabled
            else []
        ),
        river_burned_dem=(
            str(basin_inputs / "domain" / f"{basin_id}_river_burned_dem.tif")
            if burn_rivers_enabled
            else []
        ),
        roughness=str(basin_inputs / "domain" / f"{basin_id}_roughness.tif"),
        land_polygons=str(basin_inputs / "domain" / f"{basin_id}_land_polygons.gpkg"),
        river_network=str(
            basin_inputs / "domain" / f"{basin_id}_river_network_estuarine.gpkg"
        ),
        delta_outflow_points=str(
            basin_inputs / "domain" / f"{basin_id}_delta_outflow_points.gpkg"
        ),
        zsini=str(basin_inputs / "domain" / f"{basin_id}_zsini.tif"),
        surge_forcing=str(basin_inputs / "forcing" / "surge_forcing.nc"),
        river_forcing=str(basin_inputs / "forcing" / "river_forcing.nc"),
    )
    output_d = dict(
        sfincs_inp=str(sfincs_root / "sfincs.inp"),
        sfincs_subgrid=str(sfincs_root / "sfincs_subgrid.nc"),
        plot_grid=str(combo_visuals / "01_grid.png"),
        plot_elevation=str(combo_visuals / "02_elevation.png"),
        plot_mask=str(combo_visuals / "03_mask.png"),
        plot_roughness=str(combo_visuals / "04_roughness.png"),
    )
    if combo["quadtree_enabled"]:
        output_d["refinement_polygons"] = str(
            sfincs_root / f"{basin_id}_refinement_polygons.gpkg"
        )
        output_d["plot_refinement"] = str(combo_visuals / "01b_refinement_zones.png")

    params_d = dict(
        preburn_enabled=conditioning_enabled,
        burn_rivers_enabled=burn_rivers_enabled,
        resolution=combo["resolution"],
        include_subgrid=combo["subgrid_enabled"],
        include_rstart=bool(config["sfincs"]["spinup"]["enabled"]),
        spinup_days=config["sfincs"]["spinup"]["spinup_days"],
        nr_subgrid_pixels=(
            combo["n_cells"]
            if combo["subgrid_enabled"]
            else config["sfincs"]["subgrid"]["nr_subgrid_pixels"]
        ),
        nr_levels=config["sfincs"]["subgrid"]["nr_levels"],
        nrmax=config["sfincs"]["subgrid"]["nrmax"],
        tref=config["sfincs"]["simulation"]["tref"],
        dtmapout=config["sfincs"]["simulation"]["dtmapout"],
        dtmaxout=config["sfincs"]["simulation"]["dtmaxout"],
        dthisout=config["sfincs"]["simulation"]["dthisout"],
        storevelmax=config["sfincs"]["simulation"]["storevelmax"],
        storetwet=config["sfincs"]["simulation"]["storetwet"],
        forcing_mode="compound",  # forced regardless of the current config.yml value
        compound_lag_hr=config["boundary_setup"]["compound"]["lag_hr"],
        flat_boundary_point_spacing_m=config["boundary_setup"][
            "flat_boundary_point_spacing_m"
        ],
        waterlevel_buffer_m=config["boundary_setup"]["waterlevel_buffer_m"],
        outflow_buffer_m=config["boundary_setup"]["outflow_buffer_m"],
        n_top_crossings=config["sfincs"]["observation_points"]["n_top_crossings"],
        n_per_crossing=config["sfincs"]["observation_points"]["n_per_crossing"],
        max_downstream_hops=config["sfincs"]["observation_points"][
            "max_downstream_hops"
        ],
        inputs_dir=str(basin_inputs),
        sfincs_root=str(sfincs_root),
        quadtree_enabled=combo["quadtree_enabled"],
        river_refinement_level=RIVER_REFINEMENT_LEVEL,
        river_buffer_factor=config["sfincs"]["grid"]["quadtree"]["river_buffer_factor"],
        coastal_refinement_enabled=COASTAL_REFINEMENT_ENABLED,
        coastal_refinement_level=COASTAL_REFINEMENT_LEVEL,
        coastal_buffer_m=config["sfincs"]["grid"]["quadtree"]["coastal_buffer_m"],
    )
    return input_d, output_d, params_d


def _rst_fname() -> str:
    tref = datetime.strptime(
        config["sfincs"]["simulation"]["tref"], "%Y-%m-%d %H:%M:%S"
    )
    spinup_end = tref + timedelta(days=config["sfincs"]["spinup"]["spinup_days"])
    return f"sfincs.{spinup_end.strftime('%Y%m%d.%H%M%S')}.rst"


def _mock_config_14(
    basin_id: str, combo: dict, sfincs_root: Path, rst_fname: str
) -> tuple[dict, dict, dict]:
    basin_inputs = RESULTS_DIR / basin_id / "inputs"
    combo_visuals = sfincs_root.parent / "visuals" / "model_runs" / "spinup"

    input_d = dict(
        sfincs_inp=str(sfincs_root / "sfincs.inp"),
        land_polygons=str(basin_inputs / "domain" / f"{basin_id}_land_polygons.gpkg"),
        landuse=str(basin_inputs / "domain" / f"{basin_id}_landuse.tif"),
        domain_gpkg=str(basin_inputs / "domain" / f"{basin_id}_domain.gpkg"),
        clean_river_network=str(
            basin_inputs / "domain" / f"{basin_id}_river_network_clean.gpkg"
        ),
    )
    output_d = dict(
        rstart=str(sfincs_root / "spinup" / rst_fname),
        sfincs_map_nc=str(sfincs_root / "spinup" / "sfincs_map.nc"),
        plot_spinup=str(combo_visuals / "validation_spinup.png"),
        plot_max_inundation=str(combo_visuals / "validation_max_inundation.png"),
    )
    params_d = dict(
        sfincs_root=str(sfincs_root),
        spinup_days=config["sfincs"]["spinup"]["spinup_days"],
        sfincs_exe=config["sfincs"]["simulation"]["sfincs_exe"],
        rst_fname=rst_fname,
        dtmapout_s=config["sfincs"]["spinup"]["dtmapout_s"],
        dthisout_s=config["sfincs"]["spinup"]["dthisout_s"],
        include_subgrid=combo["subgrid_enabled"],
        timeout_s=config["sfincs"]["spinup"]["timeout_s"],
    )
    return input_d, output_d, params_d


def _mock_config_16(
    basin_id: str, combo: dict, sfincs_root: Path, rst_fname: str
) -> tuple[dict, dict, dict]:
    basin_inputs = RESULTS_DIR / basin_id / "inputs"
    combo_visuals = sfincs_root.parent / "visuals" / "model_runs" / "main_run"

    input_d = dict(
        sfincs_inp=str(sfincs_root / "sfincs.inp"),
        rstart=str(sfincs_root / "spinup" / rst_fname),
        land_polygons=str(basin_inputs / "domain" / f"{basin_id}_land_polygons.gpkg"),
        landuse=str(basin_inputs / "domain" / f"{basin_id}_landuse.tif"),
        domain_gpkg=str(basin_inputs / "domain" / f"{basin_id}_domain.gpkg"),
        clean_river_network=str(
            basin_inputs / "domain" / f"{basin_id}_river_network_clean.gpkg"
        ),
    )
    output_d = dict(
        sfincs_map_nc=str(sfincs_root / "sfincs_map.nc"),
        plot_inundation_ratio=str(combo_visuals / "01_inundation_ratio.png"),
        animation_flood_progress=str(combo_visuals / "02_flood_animation.mp4"),
        flood_timeseries_csv=str(combo_visuals / "flood_timeseries.csv"),
    )
    params_d = dict(
        sfincs_root=str(sfincs_root),
        sfincs_exe=config["sfincs"]["simulation"]["sfincs_exe"],
        timeout_s=config["sfincs"]["event"]["timeout_s"],
        min_inundation_depth_m=config["sfincs"]["sanity_checks"][
            "min_inundation_depth_m"
        ],
        include_subgrid=combo["subgrid_enabled"],
        animation_fps=config["sfincs"]["sanity_checks"]["animation_fps"],
    )
    return input_d, output_d, params_d


def run_one_combination(basin_id: str, combo: dict) -> dict:
    combo_dir = EXPERIMENTS_DIR / basin_id / combo["label"]
    sfincs_root = combo_dir / "sfincs"
    logs_dir = combo_dir / "logs"

    log.info(
        f"[{basin_id}/{combo['label']}] BUILD (resolution={combo['resolution']} m, "
        f"subgrid={combo['subgrid_enabled']}, n_cells={combo['n_cells']}, "
        f"quadtree={combo['quadtree_enabled']}, min_cell={combo['min_cell_size_m']:.1f} m)"
    )
    input_d, output_d, params_d = _mock_config_13(basin_id, combo, sfincs_root)
    build_time_s = _run_mock_script(
        "13_build_sfincs.py",
        input_d,
        output_d,
        params_d,
        logs_dir / "13_build_sfincs.log",
    )
    log.info(f"[{basin_id}/{combo['label']}] build done in {build_time_s:.1f} s")

    rst_fname = _rst_fname()
    log.info(f"[{basin_id}/{combo['label']}] RUN: spin-up")
    input_d, output_d, params_d = _mock_config_14(
        basin_id, combo, sfincs_root, rst_fname
    )
    spinup_time_s = _run_mock_script(
        "14_run_spinup.py", input_d, output_d, params_d, logs_dir / "14_run_spinup.log"
    )

    log.info(f"[{basin_id}/{combo['label']}] RUN: main event")
    input_d, output_d, params_d = _mock_config_16(
        basin_id, combo, sfincs_root, rst_fname
    )
    event_time_s = _run_mock_script(
        "16_run_event.py", input_d, output_d, params_d, logs_dir / "16_run_event.log"
    )

    run_time_s = spinup_time_s + event_time_s
    log.info(
        f"[{basin_id}/{combo['label']}] run done in {run_time_s:.1f} s "
        f"(spin-up {spinup_time_s:.1f} s + event {event_time_s:.1f} s)"
    )

    return dict(
        basin_id=basin_id,
        label=combo["label"],
        group=combo["group"],
        resolution=combo["resolution"],
        subgrid_enabled=combo["subgrid_enabled"],
        n_cells=combo["n_cells"],
        quadtree_enabled=combo["quadtree_enabled"],
        min_cell_size_m=combo["min_cell_size_m"],
        build_time_s=build_time_s,
        spinup_time_s=spinup_time_s,
        event_time_s=event_time_s,
        run_time_s=run_time_s,
    )


def make_plot(
    df: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, len(BASIN_IDS), figsize=(9 * len(BASIN_IDS), 7))
    if len(BASIN_IDS) == 1:
        axes = [axes]

    groups = sorted(df["group"].unique())
    cmap = plt.get_cmap("tab10")
    linestyles = ["-", "--", "-.", ":"]
    style_by_group = {
        g: (cmap(i % 10), linestyles[(i // 10) % len(linestyles)])
        for i, g in enumerate(groups)
    }

    for ax, basin_id in zip(axes, BASIN_IDS):
        sub_basin = df[df["basin_id"] == basin_id]
        for g in groups:
            sub = sub_basin[sub_basin["group"] == g].sort_values(
                "resolution", ascending=False
            )
            if sub.empty:
                continue
            color, ls = style_by_group[g]
            # min_cell_size scales linearly with resolution for a fixed
            # (n_cells, quadtree) combination -- state it as a factor of R
            # (e.g. "min cell = R/4") rather than a single number, since it
            # varies across the resolution sweep for the same combination.
            if sub["n_cells"].iloc[0] is None or pd.isna(sub["n_cells"].iloc[0]):
                min_cell_label = "min cell = R"
            else:
                qt_factor = _quadtree_factor(bool(sub["quadtree_enabled"].iloc[0]))
                denom = sub["n_cells"].iloc[0] * qt_factor
                min_cell_label = f"min cell = R/{denom:g}"
            ax.plot(
                sub["resolution"],
                sub[metric],
                color=color,
                linestyle=ls,
                marker="o",
                label=f"{g} ({min_cell_label})",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(FULL_RESOLUTIONS)
        ax.set_xticklabels([str(r) for r in FULL_RESOLUTIONS])
        ax.set_xlabel("Main grid resolution R (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Basin {basin_id}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Plot written: {out_path}")


def main() -> None:
    full_mode = "--full" in sys.argv[1:]
    restart = "--restart" in sys.argv[1:]
    combos = (
        generate_full_combinations() if full_mode else generate_pilot_combinations()
    )
    log.info(
        f"Mode: {'FULL' if full_mode else 'PILOT'} -- {len(combos)} combination(s) per basin, "
        f"{len(combos) * len(BASIN_IDS)} total"
    )

    csv_path = (
        FIGS_DIR / f"grid_resolution_benchmark_{'full' if full_mode else 'pilot'}.csv"
    )
    results: list[dict] = []
    completed: set[tuple[str, str]] = set()
    if csv_path.exists() and not restart:
        existing_df = pd.read_csv(csv_path, dtype={"basin_id": str})
        results = existing_df.to_dict("records")
        completed = set(zip(existing_df["basin_id"], existing_df["label"]))
        log.info(
            f"Resuming from {csv_path}: {len(completed)} combination(s) already completed, will be skipped "
            f"(pass --restart to ignore and rerun everything)"
        )

    for basin_id in BASIN_IDS:
        basin_inputs = RESULTS_DIR / basin_id / "inputs"
        if not basin_inputs.exists():
            log.warning(f"Basin {basin_id}: {basin_inputs} not found -- skipping")
            continue
        for combo in combos:
            if (basin_id, combo["label"]) in completed:
                log.info(f"[{basin_id}/{combo['label']}] already completed -- skipping")
                continue
            try:
                row = run_one_combination(basin_id, combo)
                results.append(row)
                completed.add((basin_id, combo["label"]))
            except Exception:
                log.exception(f"[{basin_id}/{combo['label']}] failed -- skipping")
            pd.DataFrame(results).to_csv(csv_path, index=False)

    if not results:
        log.warning("No successful combinations -- nothing to plot")
        return

    df = pd.DataFrame(results)
    log.info(f"CSV written: {csv_path}")

    suffix = "full" if full_mode else "pilot"
    make_plot(
        df,
        "build_time_s",
        "SFINCS build time vs. grid resolution / subgrid / quadtree",
        "Build time (s, log scale)",
        FIGS_DIR / f"grid_resolution_benchmark_{suffix}_build_time.png",
    )
    make_plot(
        df,
        "run_time_s",
        "SFINCS run time (spin-up + full event, compound mode) vs. grid resolution / subgrid / quadtree",
        "Run time (s, log scale)",
        FIGS_DIR / f"grid_resolution_benchmark_{suffix}_run_time.png",
    )


if __name__ == "__main__":
    main()
