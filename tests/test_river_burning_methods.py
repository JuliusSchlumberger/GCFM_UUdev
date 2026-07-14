"""
test_river_burning_methods.py — compares how three different river-bed/
burning methods affect total flood extent, across several discharge return
periods, as one line plot (x=return period, y=flooded km², one line per
method).

Three mutually-exclusive methods, controlled by two existing config.yml
booleans (`river_processing.conditioning.enabled`,
`river_processing.burn_rivers.enabled`) -- traced directly from
workflow/rules/13_build_sfincs.smk + workflow/scripts/13_build_sfincs.py:

  1. "dem_burning" (conditioning=True, burn_rivers=True): rule 11b burns a
     channel-only DEM (river_burned_dem.tif), fed to hydromt_sfincs as a
     higher-priority elevation source; hydromt_sfincs's own burn_river_rect
     is skipped entirely (river_list=[]).
  2. "zb_levels" (conditioning=True, burn_rivers=False): rule 11 computes
     zbed_anchors.gpkg (absolute bed elevation points, rivbed =
     DEM_conditioned - rivdph), passed to subgrid_component.create(
     river_list=[{"centerlines": rivers, "gdf_zb": zbed_gdf}]).
  3. "water_depth" (conditioning=False, burn_rivers=False): only
     {"centerlines": rivers} (the rivers gdf's own rivdph column) is passed,
     no gdf_zb -- hydromt_sfincs computes bed level internally as the
     UNCONDITIONED DEM minus depth.

KNOWN, ACCEPTED CONFOUND: method 3 necessarily also skips DEM conditioning
(rule 10) -- rule 11b (DEM burning) also requires conditioning=True, so
conditioning is a prerequisite for methods 1+2, not an independent axis.
Method 3's results therefore reflect "no burning" AND "no conditioning" at
once; these are the three states the pipeline already supports, not
engineered to isolate burning technique alone.

Rules 01-09b are independent of both flags (confirmed: neither script reads
either flag) -- results/{basin_id}/inputs/ through rule 09b is reused
unchanged across all three methods. Methods 1+2 both need conditioning=True,
so elevation_conditioned.tif/zbed_anchors.gpkg are identical between them
(same inputs, deterministic) and are computed ONCE, shared (see
ensure_shared_conditioning).

Reuse strategy, following tests/test_grid_resolution_benchmark.py: rules 10/
11/11b/13/14 are executed VERBATIM via tests/_snakemake_script_runner.py (a
mock snakemake.input/output/params/log object, run as an isolated
subprocess per script call), building into an isolated experiment directory
while reading real preprocessing outputs read-only -- never touching the
real results/{basin_id}/sfincs/. The return-period discharge rescaling +
event-run + flood-extent computation reuses tests/
test_discharge_return_period_response.py's exact pattern (GPD return-value
fit stored in river_forcing.nc, forwarding the built sfincs.inp unchanged
except disfile), parameterized per method instead of a single shared
production build.

Usage:
    conda run -n hmt_sfincs_dev python tests/test_river_burning_methods.py                # default: RP 45, 50, 100 yr
    conda run -n hmt_sfincs_dev python tests/test_river_burning_methods.py 10 25 50        # only these return periods (yr)
    conda run -n hmt_sfincs_dev python tests/test_river_burning_methods.py --restart       # ignore existing CSV, rerun everything

Resumable at two levels (keyed against this run's own CSV,
figs/river_burning_methods/river_burning_methods_{basin_id}.csv, rewritten
after every successful (method, RP)): a method's build+spin-up is skipped if
its sfincs.inp + spin-up restart already exist on disk; a (method, RP) run
is skipped if it's already a row in the CSV. Pass --restart to ignore all of
this and rerun from scratch.
"""

import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.io import load_catalogue, raw_input_path
from src.postprocessing import compute_max_inundation
from src.river_forcing import build_design_discharge_matrix

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    logging.getLogger(_name).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = Path(__file__).resolve().parent / "_snakemake_script_runner.py"

BASIN_ID = "4267691"
RETURN_PERIODS_YR = [45.0, 50.0, 100.0]  # default sweep (years)

METHODS = {
    "dem_burning": dict(conditioning_enabled=True, burn_rivers_enabled=True),
    "zb_levels": dict(conditioning_enabled=True, burn_rivers_enabled=False),
    "water_depth": dict(conditioning_enabled=False, burn_rivers_enabled=False),
}

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
catalogue = load_catalogue(REPO_ROOT / config["data_catalogue"])

RESULTS_DIR = Path(config["results_dir"])
EXPERIMENTS_DIR = Path("D:/GCFM_UU/experiments/river_burning_methods") / BASIN_ID
FIGS_DIR = REPO_ROOT / "figs" / "river_burning_methods"
SHARED_DIR = EXPERIMENTS_DIR / "_shared_conditioning"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

sfincs_cfg = config["sfincs"]
sfincs_exe = Path(sfincs_cfg["simulation"]["sfincs_exe"]).resolve()
event_timeout_s = sfincs_cfg["event"]["timeout_s"]
include_subgrid = sfincs_cfg["subgrid"]["enabled"]
min_inundation_depth_m = sfincs_cfg["sanity_checks"]["min_inundation_depth_m"]
resolution_m = float(sfincs_cfg["grid"]["resolution"])

BASIN_INPUTS = RESULTS_DIR / BASIN_ID / "inputs"
LANDUSE_PATH = BASIN_INPUTS / "domain" / f"{BASIN_ID}_landuse.tif"
RIVER_FORCING_PATH = BASIN_INPUTS / "forcing" / "river_forcing.nc"


def ensure_preprocessing_fresh(basin_id: str) -> None:
    """Ensure rules 01-09b (everything rule 13 needs except the conditioning/
    burn_rivers-specific elevation_conditioned/zbed_anchors/river_burned_dem,
    which this script computes itself below) are up to date -- targets the
    EXPLICIT unconditional file paths rather than the `preprocess` named
    rule, since that rule's own target list is itself conditional on the
    CURRENT config.yml conditioning/burn_rivers values and would redundantly
    (possibly expensively, e.g. rule 11b's 3 raw data-catalogue sources)
    rebuild real results/{basin_id}/ conditioning outputs this script
    doesn't use. Snakemake's own dependency tracking makes this a fast
    no-op once everything is already fresh."""
    domain_dir = RESULTS_DIR / basin_id / "inputs" / "domain"
    forcing_dir = RESULTS_DIR / basin_id / "inputs" / "forcing"
    targets = [
        str(domain_dir / f"{basin_id}_domain.gpkg"),
        str(domain_dir / f"{basin_id}_elevation_merged.tif"),
        str(domain_dir / f"{basin_id}_roughness.tif"),
        str(domain_dir / f"{basin_id}_land_polygons.gpkg"),
        str(domain_dir / f"{basin_id}_landuse.tif"),
        str(domain_dir / f"{basin_id}_river_network_processed.gpkg"),
        str(domain_dir / f"{basin_id}_river_network_estuarine.gpkg"),
        str(domain_dir / f"{basin_id}_river_network_clean.gpkg"),
        str(domain_dir / f"{basin_id}_delta_outflow_points.gpkg"),
        str(domain_dir / f"{basin_id}_zsini.tif"),
        str(forcing_dir / "surge_forcing.nc"),
        str(forcing_dir / "river_forcing.nc"),
    ]
    log.info(
        f"Ensuring rules 01-09b are up to date for basin {basin_id} (snakemake)..."
    )
    proc = subprocess.run(
        # --rerun-incomplete: a file interrupted mid-write by some earlier,
        # unrelated killed process (this session has killed several
        # snakemake/sfincs.exe processes) leaves Snakemake's own metadata
        # flagging it incomplete on every subsequent run until it's either
        # regenerated or explicitly marked complete -- regenerating is the
        # safe default (Snakemake's own suggested remedy) rather than
        # trusting a file that may genuinely be truncated.
        [
            "snakemake",
            *targets,
            "--cores",
            "all",
            "--rerun-incomplete",
            "--config",
            f"target_basins=[{basin_id}]",
        ],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"snakemake preprocessing failed for basin {basin_id} (exit code {proc.returncode}) "
            f"-- see output above for details."
        )
    log.info(f"Preprocessing up to date for basin {basin_id}.")


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


def _rst_fname() -> str:
    tref = datetime.strptime(
        config["sfincs"]["simulation"]["tref"], "%Y-%m-%d %H:%M:%S"
    )
    spinup_end = tref + timedelta(days=config["sfincs"]["spinup"]["spinup_days"])
    return f"sfincs.{spinup_end.strftime('%Y%m%d.%H%M%S')}.rst"


def ensure_shared_conditioning() -> tuple[Path | None, Path | None, Path | None]:
    """Compute elevation_conditioned.tif / zbed_anchors.gpkg (rules 10/11,
    shared between methods 1+2, deterministic given the same real basin
    inputs) and river_burned_dem.tif (rule 11b, only for dem_burning) ONCE.
    Skips any step whose output already exists (resume support)."""
    needs_conditioning = any(m["conditioning_enabled"] for m in METHODS.values())
    needs_burn = "dem_burning" in METHODS

    conditioned_path = SHARED_DIR / f"{BASIN_ID}_elevation_conditioned.tif"
    zbed_path = SHARED_DIR / f"{BASIN_ID}_zbed_anchors.gpkg"
    burned_dem_path = SHARED_DIR / f"{BASIN_ID}_river_burned_dem.tif"
    visuals_dir = SHARED_DIR / "visuals"
    logs_dir = SHARED_DIR / "logs"

    if needs_conditioning:
        if conditioned_path.exists():
            log.info(
                "Shared conditioning: elevation_conditioned.tif already exists -- skipping rule 10"
            )
        else:
            log.info("Shared conditioning: running rule 10 (condition_elevation)...")
            input_d = dict(
                elevation_merged=str(
                    BASIN_INPUTS / "domain" / f"{BASIN_ID}_elevation_merged.tif"
                ),
                river_network=str(
                    BASIN_INPUTS / "domain" / f"{BASIN_ID}_river_network_processed.gpkg"
                ),
            )
            output_d = dict(
                elevation_conditioned=str(conditioned_path),
                plot_conditioning=str(visuals_dir / "10_condition_elevation.png"),
            )
            _run_mock_script(
                "10_condition_elevation.py",
                input_d,
                output_d,
                {},
                logs_dir / "10_condition_elevation.log",
            )

        if zbed_path.exists():
            log.info(
                "Shared conditioning: zbed_anchors.gpkg already exists -- skipping rule 11"
            )
        else:
            log.info("Shared conditioning: running rule 11 (river_preburn)...")
            input_d = dict(
                elevation_conditioned=str(conditioned_path),
                river_network=str(
                    BASIN_INPUTS / "domain" / f"{BASIN_ID}_river_network_estuarine.gpkg"
                ),
            )
            output_d = dict(
                zbed_anchors=str(zbed_path),
                plot_preburn=str(visuals_dir / "11_river_preburn.png"),
            )
            _run_mock_script(
                "11_river_preburn.py",
                input_d,
                output_d,
                {},
                logs_dir / "11_river_preburn.log",
            )
    else:
        conditioned_path = None
        zbed_path = None

    if needs_burn:
        if burned_dem_path.exists():
            log.info(
                "Shared conditioning: river_burned_dem.tif already exists -- skipping rule 11b"
            )
        else:
            log.info("Shared conditioning: running rule 11b (burn_river_dem)...")
            input_d = dict(
                zbed_anchors=str(zbed_path),
                river_network=str(
                    BASIN_INPUTS / "domain" / f"{BASIN_ID}_river_network_estuarine.gpkg"
                ),
                domain_gpkg=str(BASIN_INPUTS / "domain" / f"{BASIN_ID}_domain.gpkg"),
                elevation_merged=str(
                    BASIN_INPUTS / "domain" / f"{BASIN_ID}_elevation_merged.tif"
                ),
                global_topography_tiles=raw_input_path(catalogue, "fathomdem"),
                goco06s_gfc=raw_input_path(catalogue, "goco06s"),
                egm2008_gfc=raw_input_path(catalogue, "egm2008_geoid"),
            )
            output_d = dict(
                river_burned_dem=str(burned_dem_path),
                plot_river_burn=str(visuals_dir / "11b_river_burn.png"),
            )
            params_d = dict()
            _run_mock_script(
                "11b_burn_river_dem.py",
                input_d,
                output_d,
                params_d,
                logs_dir / "11b_burn_river_dem.log",
            )
    else:
        burned_dem_path = None

    return conditioned_path, zbed_path, burned_dem_path


def _mock_config_13_method(
    method_name: str,
    conditioning_enabled: bool,
    burn_rivers_enabled: bool,
    conditioned_path: Path | None,
    zbed_path: Path | None,
    burned_dem_path: Path | None,
    sfincs_root: Path,
) -> tuple[dict, dict, dict]:
    combo_visuals = sfincs_root.parent / "visuals" / "sfincs_build"
    input_d = dict(
        domain_gpkg=str(BASIN_INPUTS / "domain" / f"{BASIN_ID}_domain.gpkg"),
        elevation_merged=(
            str(conditioned_path)
            if conditioning_enabled
            else str(BASIN_INPUTS / "domain" / f"{BASIN_ID}_elevation_merged.tif")
        ),
        zbed_anchors=(str(zbed_path) if conditioning_enabled else []),
        river_burned_dem=(str(burned_dem_path) if burn_rivers_enabled else []),
        roughness=str(BASIN_INPUTS / "domain" / f"{BASIN_ID}_roughness.tif"),
        land_polygons=str(BASIN_INPUTS / "domain" / f"{BASIN_ID}_land_polygons.gpkg"),
        river_network=str(
            BASIN_INPUTS / "domain" / f"{BASIN_ID}_river_network_estuarine.gpkg"
        ),
        delta_outflow_points=str(
            BASIN_INPUTS / "domain" / f"{BASIN_ID}_delta_outflow_points.gpkg"
        ),
        zsini=str(BASIN_INPUTS / "domain" / f"{BASIN_ID}_zsini.tif"),
        surge_forcing=str(BASIN_INPUTS / "forcing" / "surge_forcing.nc"),
        river_forcing=str(BASIN_INPUTS / "forcing" / "river_forcing.nc"),
    )
    output_d = dict(
        sfincs_inp=str(sfincs_root / "sfincs.inp"),
        sfincs_subgrid=str(sfincs_root / "sfincs_subgrid.nc"),
        plot_grid=str(combo_visuals / "01_grid.png"),
        plot_elevation=str(combo_visuals / "02_elevation.png"),
        plot_mask=str(combo_visuals / "03_mask.png"),
        plot_roughness=str(combo_visuals / "04_roughness.png"),
    )
    quadtree_enabled = bool(sfincs_cfg["grid"]["quadtree"]["enabled"])
    if quadtree_enabled:
        output_d["refinement_polygons"] = str(
            sfincs_root / f"{BASIN_ID}_refinement_polygons.gpkg"
        )
        output_d["plot_refinement"] = str(combo_visuals / "01b_refinement_zones.png")

    params_d = dict(
        preburn_enabled=conditioning_enabled,
        burn_rivers_enabled=burn_rivers_enabled,
        resolution=resolution_m,
        include_subgrid=include_subgrid,
        include_rstart=bool(sfincs_cfg["spinup"]["enabled"]),
        spinup_days=sfincs_cfg["spinup"]["spinup_days"],
        nr_subgrid_pixels=sfincs_cfg["subgrid"]["nr_subgrid_pixels"],
        nr_levels=sfincs_cfg["subgrid"]["nr_levels"],
        nrmax=sfincs_cfg["subgrid"]["nrmax"],
        tref=sfincs_cfg["simulation"]["tref"],
        dtmapout=sfincs_cfg["simulation"]["dtmapout"],
        dtmaxout=sfincs_cfg["simulation"]["dtmaxout"],
        dthisout=sfincs_cfg["simulation"]["dthisout"],
        storevelmax=sfincs_cfg["simulation"]["storevelmax"],
        storetwet=sfincs_cfg["simulation"]["storetwet"],
        # Hardcoded regardless of the current config.yml value -- this
        # script fully controls its own build (same convention
        # test_grid_resolution_benchmark.py uses for forcing_mode
        # "compound"), rather than depending on ambient config state.
        # river_only isolates flood-extent differences to the river's own
        # contribution, uncontaminated by coastal/surge variability.
        forcing_mode="river_only",
        compound_lag_hr=config["boundary_setup"]["compound"]["lag_hr"],
        flat_boundary_point_spacing_m=config["boundary_setup"][
            "flat_boundary_point_spacing_m"
        ],
        waterlevel_buffer_m=config["boundary_setup"]["waterlevel_buffer_m"],
        outflow_buffer_m=config["boundary_setup"]["outflow_buffer_m"],
        n_top_crossings=sfincs_cfg["observation_points"]["n_top_crossings"],
        n_per_crossing=sfincs_cfg["observation_points"]["n_per_crossing"],
        max_downstream_hops=sfincs_cfg["observation_points"]["max_downstream_hops"],
        # MUST be the real basin inputs dir, not the experiment/shared dir:
        # 13_build_sfincs.py computes zsini_path.relative_to(inputs_dir)
        # when baseline_m == 0.0, which raises unless this is exactly right.
        inputs_dir=str(BASIN_INPUTS),
        sfincs_root=str(sfincs_root),
        quadtree_enabled=quadtree_enabled,
        river_refinement_level=sfincs_cfg["grid"]["quadtree"]["river_refinement_level"],
        river_buffer_factor=sfincs_cfg["grid"]["quadtree"]["river_buffer_factor"],
        coastal_refinement_enabled=sfincs_cfg["grid"]["quadtree"][
            "coastal_refinement_enabled"
        ],
        coastal_refinement_level=sfincs_cfg["grid"]["quadtree"][
            "coastal_refinement_level"
        ],
        coastal_buffer_m=sfincs_cfg["grid"]["quadtree"]["coastal_buffer_m"],
    )
    return input_d, output_d, params_d


def _mock_config_14_method(
    sfincs_root: Path, rst_fname: str
) -> tuple[dict, dict, dict]:
    combo_visuals = sfincs_root.parent / "visuals" / "model_runs" / "spinup"
    input_d = dict(
        sfincs_inp=str(sfincs_root / "sfincs.inp"),
        land_polygons=str(BASIN_INPUTS / "domain" / f"{BASIN_ID}_land_polygons.gpkg"),
        landuse=str(LANDUSE_PATH),
        domain_gpkg=str(BASIN_INPUTS / "domain" / f"{BASIN_ID}_domain.gpkg"),
        clean_river_network=str(
            BASIN_INPUTS / "domain" / f"{BASIN_ID}_river_network_clean.gpkg"
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
        spinup_days=sfincs_cfg["spinup"]["spinup_days"],
        sfincs_exe=sfincs_cfg["simulation"]["sfincs_exe"],
        rst_fname=rst_fname,
        dtmapout_s=sfincs_cfg["spinup"]["dtmapout_s"],
        dthisout_s=sfincs_cfg["spinup"]["dthisout_s"],
        include_subgrid=include_subgrid,
        timeout_s=sfincs_cfg["spinup"]["timeout_s"],
    )
    return input_d, output_d, params_d


# ── shared (method-independent) discharge-forcing state, loaded once ─────────
river_ds = xr.open_dataset(RIVER_FORCING_PATH, decode_times=False)
has_glofas = river_ds["has_glofas"].values.astype(bool)
active_indices = np.where(has_glofas)[0]


def build_scaled_dis_table(
    dis_table: np.ndarray,
    n_points: int,
    col_to_crossing: dict[int, int],
    return_period_yr: float,
) -> np.ndarray:
    """Build the discharge table for return_period_yr directly from
    river_forcing.nc's discharge_rp_table via build_design_discharge_matrix
    (same function rule 13 uses at build time), mapped back onto sfincs.dis's
    column layout via col_to_crossing."""
    scaled = dis_table.copy()
    design_matrix = build_design_discharge_matrix(
        river_ds, has_glofas, return_period_yr
    )
    active_to_row = {int(idx): k for k, idx in enumerate(active_indices)}
    for j in range(n_points):
        crossing_idx = col_to_crossing.get(j)
        row = active_to_row.get(crossing_idx) if crossing_idx is not None else None
        if row is None:
            log.warning(f"  column {j}: no crossing mapping -- left unscaled")
            continue
        q_col = design_matrix[row]
        if len(q_col) != scaled.shape[0]:
            log.warning(
                f"  column {j} (crossing {crossing_idx}): time-axis length "
                f"mismatch ({len(q_col)} vs {scaled.shape[0]}) -- left unscaled"
            )
            continue
        scaled[:, 1 + j] = q_col
        log.info(
            f"  column {j} (crossing {crossing_idx}): RP={return_period_yr:g} yr "
            f"-> peak Q={np.nanmax(q_col):.1f} m3/s"
        )
    return scaled


# Only disfile is handled explicitly (swapped per return period); every
# other key -- including tstart/tstop/rstfile -- is forwarded from the
# method's own built sfincs.inp unchanged.
_HANDLED_KEYS = {"disfile"}


def write_event_run(
    prod_cfg: dict[str, str],
    prod_sfincs_root: Path,
    scaled_table: np.ndarray,
    run_dir: Path,
) -> None:
    """Write a rescaled sfincs.dis + this method's built sfincs.inp into
    run_dir. Every input file OTHER than disfile is referenced by its
    absolute path directly from prod_sfincs_root -- that build is never
    copied, modified, or opened for writing."""
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(run_dir / "sfincs.dis", scaled_table, fmt="%14.3f")

    lines = []
    for key, value in prod_cfg.items():
        if key in _HANDLED_KEYS:
            continue
        if key.endswith("file"):
            fpath = prod_sfincs_root / value
            if fpath.exists() and fpath.stat().st_size > 0:
                lines.append(f"{key:<20} = {fpath.resolve().as_posix()}")
            continue
        lines.append(f"{key:<20} = {value}")
    lines.append(f"{'disfile':<20} = sfincs.dis")

    with open(run_dir / "sfincs.inp", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def run_sfincs(run_dir: Path) -> None:
    """Execute the SFINCS solver with cwd=run_dir so its relative disfile
    reference resolves correctly (every other input is an absolute path)."""
    proc = subprocess.Popen(
        [str(sfincs_exe)],
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _forward(pipe, prefix):
        for line in pipe:
            line = line.rstrip()
            if line:
                print(f"{prefix} {line}", file=sys.stderr, flush=True)

    t_out = threading.Thread(target=_forward, args=(proc.stdout, "[sfincs out]"))
    t_err = threading.Thread(target=_forward, args=(proc.stderr, "[sfincs err]"))
    t_out.start()
    t_err.start()

    # proc.wait() must run BEFORE joining the reader threads -- see the same
    # fix in test_discharge_return_period_response.py/14_run_spinup.py/
    # 16_run_event.py: joining first blocks unconditionally until the
    # process's own pipes close (i.e. until it already exited), silently
    # defeating the timeout.
    try:
        proc.wait(timeout=event_timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        t_out.join()
        t_err.join()
        raise RuntimeError(f"SFINCS run exceeded {event_timeout_s}s timeout")

    t_out.join()
    t_err.join()
    if proc.returncode != 0:
        raise RuntimeError(f"SFINCS run failed with exit code {proc.returncode}")


def run_method(
    method_name: str,
    conditioning_enabled: bool,
    burn_rivers_enabled: bool,
    conditioned_path: Path | None,
    zbed_path: Path | None,
    burned_dem_path: Path | None,
    csv_path: Path,
    completed: set[tuple[str, float]],
    results: list[dict],
) -> None:
    method_dir = EXPERIMENTS_DIR / method_name
    sfincs_root = method_dir / "sfincs"
    logs_dir = method_dir / "logs"
    rst_fname = _rst_fname()
    rst_path = sfincs_root / "spinup" / rst_fname
    inp_path = sfincs_root / "sfincs.inp"

    if inp_path.exists() and rst_path.exists() and rst_path.stat().st_size > 0:
        log.info(f"[{method_name}] build+spin-up already done -- skipping")
    else:
        log.info(
            f"[{method_name}] BUILD (conditioning={conditioning_enabled}, "
            f"burn_rivers={burn_rivers_enabled})"
        )
        input_d, output_d, params_d = _mock_config_13_method(
            method_name,
            conditioning_enabled,
            burn_rivers_enabled,
            conditioned_path,
            zbed_path,
            burned_dem_path,
            sfincs_root,
        )
        build_time_s = _run_mock_script(
            "13_build_sfincs.py",
            input_d,
            output_d,
            params_d,
            logs_dir / "13_build_sfincs.log",
        )
        log.info(f"[{method_name}] build done in {build_time_s:.1f}s")

        log.info(f"[{method_name}] RUN: spin-up")
        input_d, output_d, params_d = _mock_config_14_method(sfincs_root, rst_fname)
        spinup_time_s = _run_mock_script(
            "14_run_spinup.py",
            input_d,
            output_d,
            params_d,
            logs_dir / "14_run_spinup.log",
        )
        log.info(f"[{method_name}] spin-up done in {spinup_time_s:.1f}s")

    # ── parse this method's own built sfincs.inp (event-mode, per rule 13's
    # convention: tstart=spin-up end, tstop=full timeseries end, rstfile=
    # this method's own spin-up restart) -- all LOCAL to this method, never
    # module globals, since each method has its own independent build. ──────
    prod_sfincs_root = sfincs_root
    prod_cfg: dict[str, str] = {}
    with open(inp_path) as fh:
        for line in fh:
            line = line.strip()
            if "=" in line and not line.startswith("!"):
                key, _, val = line.partition("=")
                prod_cfg[key.strip().lower()] = val.strip()

    if "rstfile" not in prod_cfg:
        raise ValueError(f"[{method_name}] {inp_path} has no rstfile entry")
    prod_rst_path = prod_sfincs_root / prod_cfg["rstfile"]
    if not prod_rst_path.exists() or prod_rst_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"[{method_name}] spin-up restart not found/empty: {prod_rst_path}"
        )

    prod_dis_path = prod_sfincs_root / prod_cfg["disfile"]
    dis_table = np.loadtxt(prod_dis_path)
    if dis_table.ndim == 1:
        dis_table = dis_table[np.newaxis, :]
    n_points = dis_table.shape[1] - 1
    log.info(
        f"[{method_name}] loaded discharge: {prod_dis_path} "
        f"({dis_table.shape[0]} steps, {n_points} point(s))"
    )

    if len(active_indices) == n_points:
        col_to_crossing = {j: int(active_indices[j]) for j in range(n_points)}
    else:
        col_to_crossing = {}
        log.warning(
            f"[{method_name}] sfincs.dis has {n_points} column(s) but river_forcing.nc "
            f"has {len(active_indices)} has_glofas=1 crossing(s) -- discharge will not be scaled"
        )

    for rp in RETURN_PERIODS_YR:
        key = (method_name, rp)
        if key in completed:
            log.info(f"[{method_name}/RP={rp:g}] already completed -- skipping")
            continue
        tag = f"rp_{rp:g}"
        run_dir = method_dir / tag
        log.info(f"[{method_name}/RP={rp:g}] === run -> {run_dir} ===")

        try:
            scaled_table = build_scaled_dis_table(
                dis_table, n_points, col_to_crossing, rp
            )
            write_event_run(prod_cfg, prod_sfincs_root, scaled_table, run_dir)
            run_sfincs(run_dir)
            da_hmax, da_dep = compute_max_inundation(
                run_dir,
                prod_sfincs_root,
                LANDUSE_PATH,
                hmin=min_inundation_depth_m,
                include_subgrid=include_subgrid,
            )
            if da_hmax is None or da_dep is None:
                log.warning(
                    f"[{method_name}/RP={rp:g}] no zsmax/bed level available -- skipping"
                )
                continue

            try:
                res_x, res_y = da_dep.rio.resolution()
                pixel_area_m2 = abs(res_x * res_y)
            except Exception:
                pixel_area_m2 = np.nan

            n_land = int(da_dep.notnull().sum().item())
            n_flooded = int(da_hmax.notnull().sum().item())
            frac = n_flooded / n_land if n_land > 0 else 0.0
            flooded_km2 = n_flooded * pixel_area_m2 / 1e6
            land_km2 = n_land * pixel_area_m2 / 1e6
            log.info(
                f"[{method_name}/RP={rp:g}] {flooded_km2:.2f}/{land_km2:.2f} km² flooded "
                f"({frac:.2%}, {n_flooded:,} cells)"
            )
            results.append(
                {
                    "method": method_name,
                    "conditioning_enabled": conditioning_enabled,
                    "burn_rivers_enabled": burn_rivers_enabled,
                    "return_period_yr": rp,
                    "flooded_km2": flooded_km2,
                    "land_km2": land_km2,
                    "frac_flooded": frac,
                    "n_flooded": n_flooded,
                    "n_land": n_land,
                }
            )
            completed.add(key)
        except Exception:
            log.exception(f"[{method_name}/RP={rp:g}] failed -- skipping")

        pd.DataFrame(results).to_csv(csv_path, index=False)


def main() -> None:
    restart = "--restart" in sys.argv[1:]
    cli_rps = [float(a) for a in sys.argv[1:] if not a.startswith("--")]
    global RETURN_PERIODS_YR
    if cli_rps:
        RETURN_PERIODS_YR = cli_rps
    log.info(f"Return periods to test: {RETURN_PERIODS_YR} yr")
    log.info(f"Methods: {list(METHODS.keys())}")

    ensure_preprocessing_fresh(BASIN_ID)
    conditioned_path, zbed_path, burned_dem_path = ensure_shared_conditioning()

    csv_path = FIGS_DIR / f"river_burning_methods_{BASIN_ID}.csv"
    results: list[dict] = []
    completed: set[tuple[str, float]] = set()
    if csv_path.exists() and not restart:
        existing_df = pd.read_csv(csv_path)
        results = existing_df.to_dict("records")
        completed = set(zip(existing_df["method"], existing_df["return_period_yr"]))
        log.info(
            f"Resuming from {csv_path}: {len(completed)} (method, RP) combination(s) "
            f"already completed, will be skipped (pass --restart to ignore)"
        )

    for method_name, flags in METHODS.items():
        try:
            run_method(
                method_name,
                flags["conditioning_enabled"],
                flags["burn_rivers_enabled"],
                conditioned_path,
                zbed_path,
                burned_dem_path,
                csv_path,
                completed,
                results,
            )
        except Exception:
            log.exception(
                f"[{method_name}] method failed -- skipping its remaining return periods"
            )

    if not results:
        log.warning("No successful runs -- nothing to plot")
        return

    df = pd.DataFrame(results)
    log.info(f"CSV written: {csv_path}")

    # land_km2 derives only from the landuse raster + grid geometry, both
    # method-independent -- a meaningful mismatch across methods would
    # indicate an accidental params divergence between the three mock-13
    # calls rather than a real methodological difference.
    land_by_method = df.groupby("method")["land_km2"].mean()
    if (
        len(land_by_method) > 1
        and land_by_method.max() - land_by_method.min() > 0.01 * land_by_method.mean()
    ):
        log.warning(
            f"land_km2 differs meaningfully across methods: {land_by_method.to_dict()}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for method_name in METHODS:
        sub = df[df["method"] == method_name].sort_values("return_period_yr")
        if sub.empty:
            continue
        ax.plot(
            sub["return_period_yr"], sub["flooded_km2"], marker="o", label=method_name
        )
    ax.set_xlabel("River discharge return period (yr)")
    ax.set_ylabel("Flooded area (km²)")
    ax.set_title(
        f"Basin {BASIN_ID} — flooded area vs. return period, by river-burning method\n"
        f"(water_depth also uses the unconditioned DEM -- not burning technique alone)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = FIGS_DIR / f"river_burning_methods_{BASIN_ID}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info(f"Plot written: {plot_path}")


if __name__ == "__main__":
    main()
