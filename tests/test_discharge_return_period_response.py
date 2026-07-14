"""
test_discharge_return_period_response.py — Reuse the basin's production
SFINCS model and identify how flooded area during the main event responds
to river discharge at DIFFERENT RETURN PERIODS.

Before doing anything else, runs `snakemake build --config target_basins=
[...]` for this basin (see run_full_pipeline below) so results/{basin_id}/
is guaranteed up to date with the current code/config.

Like tests/test_discharge_step_response.py, this reuses the basin's actual
production model at results/{basin_id}/sfincs/ UNCHANGED -- grid,
elevation, roughness, mask, subgrid, initial conditions, and boundary
files are all referenced directly from the production build. The ONLY thing
that varies per run is the river discharge series (sfincs.dis).

Unlike the step-response test (which scales the whole hydrograph by a single,
basin-wide, physically-arbitrary factor), this test builds a PHYSICALLY
MEANINGFUL discharge hydrograph per boundary crossing directly from
river_forcing.nc's discharge_rp_table (per-crossing GPD return-value table,
log-RP interpolated) via src.river_forcing.build_design_discharge_matrix --
the same function rule 13 (build_sfincs) now uses at SFINCS-build time, so
each return period's discharge exactly matches what the production pipeline
would build at that design return period.

This test requires the production build to be in forcing_mode="river_only"
(config: boundary_setup.mode) -- the real surge/tide boundary is replaced by
a flat constant there, so flooded-area differences across return periods
reflect the river's own contribution only, uncontaminated by coastal
variability.

Mapping sfincs.dis columns back to river_forcing.nc's "crossing" dimension:
13_build_sfincs.py builds the discharge DataFrame from only the
has_glofas=1 crossings (in their original order -- boolean masking preserves
order), then further drops any that fall outside the active SFINCS region.
When none are dropped by that second filter (verified for basin 4267691: all
3 has_glofas crossings survive, matching sfincs.dis's 3 columns and their
bankfull_discharge values exactly), dis column j corresponds exactly to
active_indices[j] where active_indices = np.where(has_glofas)[0]. If the
counts don't match for some other basin, this script cannot safely infer
which crossings were dropped and falls back to leaving that basin's
discharge unscaled (scale factor 1.0) with a clear warning, rather than
guessing.

Runs the actual MAIN EVENT (matching 16_run_event.py), not a cold-start
spin-up: results/{basin_id}/sfincs/sfincs.inp is already configured by rule
13 exactly as rule 16 uses it (tstart = spin-up end, tstop = end of the
full forcing timeseries, rstfile pointing at rule 14's production spin-up
restart), so every key other than disfile is forwarded unchanged --
including tstart/tstop/rstfile -- rather than constructing a separate
cold-start window. This means each run picks up from the SAME real,
already-spun-up initial condition and covers the full event duration, only
the river discharge magnitude differs between return periods.

Usage:
    conda run -n hmt_sfincs_dev python tests/test_discharge_return_period_response.py                # default: 2, 10, 50, 100 yr
    conda run -n hmt_sfincs_dev python tests/test_discharge_return_period_response.py 5 25 100 200    # only these return periods (yr)
"""

import logging
import subprocess
import sys
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.plots import reproject_max_for_plot
from src.postprocessing import compute_max_inundation
from src.river_forcing import build_design_discharge_matrix

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    logging.getLogger(_name).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIN_ID = "2433835"
RETURN_PERIODS_YR = [45.0, 50.0, 55.0, 150.0]  # default sweep (years)
N_PANELS = 4

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)

RESULTS_DIR = Path(config["results_dir"])
EXPERIMENTS_DIR = (
    Path("D:/GCFM_UU/experiments/discharge_return_period_response") / BASIN_ID
)
FIGS_DIR = REPO_ROOT / "figs" / "discharge_return_period_response"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

boundary_mode = config["boundary_setup"]["mode"]
if boundary_mode != "river_only":
    raise ValueError(
        f"boundary_setup.mode={boundary_mode!r} but this test requires the "
        f"production build to already be in 'river_only' mode (flat coastal "
        f"boundary), so that flooded-area differences reflect only the "
        f"river's own contribution. Set boundary_setup.mode: river_only in "
        f"config.yml and rebuild the model before running this test."
    )


def run_full_pipeline(basin_id: str) -> None:
    """Ensure the full Snakemake pipeline (rules 01-16) is up to date for
    this basin before reusing its results/{basin_id}/ outputs -- they may
    be stale relative to the current code/config (e.g. after a data-source
    swap or a discharge-formula change), and this test otherwise trusts
    them unconditionally. Snakemake's own dependency tracking means this
    only rebuilds whatever actually changed; it's a fast no-op once
    everything is already fresh."""
    log.info(
        f"Ensuring the full pipeline is up to date for basin {basin_id} (snakemake build)..."
    )
    proc = subprocess.run(
        [
            "snakemake",
            "build",
            "--cores",
            "all",
            "--config",
            f"target_basins=[{basin_id}]",
        ],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"snakemake build failed for basin {basin_id} (exit code {proc.returncode}) "
            f"-- see output above for details."
        )
    log.info(f"Pipeline up to date for basin {basin_id}.")


run_full_pipeline(BASIN_ID)

sfincs_cfg = config["sfincs"]
resolution_m = float(sfincs_cfg["grid"]["resolution"])
event_timeout_s = sfincs_cfg["event"]["timeout_s"]
sfincs_exe = Path(sfincs_cfg["simulation"]["sfincs_exe"]).resolve()
include_subgrid = sfincs_cfg["subgrid"]["enabled"]
min_inundation_depth_m = sfincs_cfg["sanity_checks"]["min_inundation_depth_m"]

prod_sfincs_root = RESULTS_DIR / BASIN_ID / "sfincs"
prod_inp_path = prod_sfincs_root / "sfincs.inp"
landuse_path = RESULTS_DIR / BASIN_ID / "inputs" / "domain" / f"{BASIN_ID}_landuse.tif"
river_forcing_path = RESULTS_DIR / BASIN_ID / "inputs" / "forcing" / "river_forcing.nc"

# ── parse the production sfincs.inp verbatim ──────────────────────────────────
# This is the same sfincs.inp rule 16 (run_event) uses directly: rule 13
# (build_sfincs) already writes it configured for the main event (tstart =
# spin-up end, tstop = end of the full forcing timeseries, rstfile pointing
# at rule 14's spin-up restart) -- so every field parsed here, including
# tstart/tstop/rstfile, is forwarded to each run unchanged (see
# write_event_run below); only disfile is swapped per return period.
prod_cfg: dict[str, str] = {}
with open(prod_inp_path) as fh:
    for line in fh:
        line = line.strip()
        if "=" in line and not line.startswith("!"):
            key, _, val = line.partition("=")
            prod_cfg[key.strip().lower()] = val.strip()

if "rstfile" not in prod_cfg:
    raise ValueError(
        f"{prod_inp_path} has no rstfile entry -- expected rule 13 to have "
        f"configured it to start from rule 14's spin-up restart (see rule "
        f"16, run_event). Rebuild the model (rules 13/14) before running "
        f"this test."
    )
prod_rst_path = prod_sfincs_root / prod_cfg["rstfile"]
if not prod_rst_path.exists() or prod_rst_path.stat().st_size == 0:
    raise FileNotFoundError(
        f"Production spin-up restart file not found or empty: {prod_rst_path} "
        f"-- run the spin_up rule for basin {BASIN_ID} first."
    )

# ── load the production discharge series (sfincs.dis) once ───────────────────
prod_dis_path = prod_sfincs_root / prod_cfg["disfile"]
dis_table = np.loadtxt(prod_dis_path)
if dis_table.ndim == 1:
    dis_table = dis_table[np.newaxis, :]
n_points = dis_table.shape[1] - 1
log.info(
    f"Loaded production discharge: {prod_dis_path} "
    f"({dis_table.shape[0]} steps, {n_points} point(s), "
    f"max Q = {dis_table[:, 1:].max():.1f} m3/s)"
)

# ── map sfincs.dis columns back to river_forcing.nc crossings ────────────────
river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
has_glofas = river_ds["has_glofas"].values.astype(bool)
active_indices = np.where(has_glofas)[0]

if len(active_indices) == n_points:
    col_to_crossing = {j: int(active_indices[j]) for j in range(n_points)}
    log.info(
        f"Mapped {n_points} sfincs.dis column(s) to river_forcing.nc crossing(s) "
        f"{list(col_to_crossing.values())} (has_glofas count matches exactly)"
    )
else:
    col_to_crossing = {}
    log.warning(
        f"sfincs.dis has {n_points} column(s) but river_forcing.nc has "
        f"{len(active_indices)} has_glofas=1 crossing(s) -- cannot reliably "
        f"map columns to crossings (some active crossings were likely "
        f"dropped as outside the SFINCS region during the build). "
        f"Return-period discharge will NOT be applied; every run will reuse "
        f"the unscaled production hydrograph."
    )


def build_scaled_dis_table(return_period_yr: float) -> np.ndarray:
    """Build the discharge table for return_period_yr directly from
    river_forcing.nc's discharge_rp_table via build_design_discharge_matrix
    (same function rule 13 uses at build time), mapped back onto sfincs.dis's
    column layout via col_to_crossing."""
    scaled = dis_table.copy()
    design_matrix = build_design_discharge_matrix(
        river_ds, has_glofas, return_period_yr
    )
    # design_matrix's rows are in has_glofas/active_indices order.
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


cli_rps = [float(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
RETURN_PERIODS_YR = cli_rps or RETURN_PERIODS_YR
log.info(f"Return periods to test: {RETURN_PERIODS_YR} yr")

_panel_targets = np.linspace(min(RETURN_PERIODS_YR), max(RETURN_PERIODS_YR), N_PANELS)
panel_rps = sorted(
    {min(RETURN_PERIODS_YR, key=lambda r: abs(r - t)) for t in _panel_targets}
)
log.info(f"Return periods selected for the panel figure: {panel_rps}")

# Only disfile is handled explicitly (swapped per return period); every
# other key -- including tstart/tstop/rstfile -- is forwarded from the
# production event sfincs.inp unchanged (see module docstring).
_HANDLED_KEYS = {"disfile"}


def write_event_run(scaled_table: np.ndarray, run_dir: Path) -> None:
    """
    Write a rescaled sfincs.dis + the production event's sfincs.inp into run_dir.

    Every production input file OTHER than disfile is referenced by its
    absolute path directly from prod_sfincs_root -- the production build
    itself is never copied, modified, or opened for writing, so re-running
    this test can never corrupt it.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(run_dir / "sfincs.dis", scaled_table, fmt="%14.3f")

    lines = []
    for key, value in prod_cfg.items():
        if key in _HANDLED_KEYS:
            continue
        if key.endswith("file"):
            fpath = prod_sfincs_root / value
            if fpath.exists() and fpath.stat().st_size > 0:
                # SFINCS's own ASCII .inp parser mishandles backslashes in
                # absolute Windows paths (observed: "indexfile" truncated to
                # just the drive+first path segment, "Index file "D:\GCFM_UU"
                # not found!"). Forward slashes are accepted identically by
                # Windows file APIs and avoid the issue entirely -- the same
                # convention hydromt_sfincs itself never needs to worry about
                # since production sfincs.inp only ever writes plain relative
                # filenames (run with cwd=sfincs_root), never absolute paths.
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

    # proc.wait() must run BEFORE joining the reader threads: t_out.join()/
    # t_err.join() block unconditionally until SFINCS's own stdout/stderr
    # pipes close, which only happens once it exits on its own -- so calling
    # them first would make the timeout below unreachable until the process
    # had already finished, silently defeating it. Killing the process here
    # closes its pipes, which is what lets the reader threads finish and
    # join() return.
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


# ── main sweep ─────────────────────────────────────────────────────────────────
_tag = f"{BASIN_ID}_res{resolution_m:.0f}m"
csv_path = FIGS_DIR / f"discharge_return_period_response_{_tag}.csv"
results: list[dict] = []
hmax_by_rp: dict[float, object] = {}

for rp in RETURN_PERIODS_YR:
    tag = f"rp_{rp:g}"
    run_dir = EXPERIMENTS_DIR / tag
    log.info(f"=== return period={rp:g} yr -> {run_dir} ===")

    try:
        scaled_table = build_scaled_dis_table(rp)
        write_event_run(scaled_table, run_dir)
        run_sfincs(run_dir)
        da_hmax, da_dep = compute_max_inundation(
            run_dir,
            prod_sfincs_root,
            landuse_path,
            hmin=min_inundation_depth_m,
            include_subgrid=include_subgrid,
        )
        if da_hmax is None or da_dep is None:
            log.warning(f"RP={rp:g} yr: no zsmax/bed level available -- skipping")
            continue

        if rp in panel_rps:
            hmax_by_rp[rp] = da_hmax

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
            f"RP={rp:g} yr: {flooded_km2:.2f}/{land_km2:.2f} km² flooded "
            f"({frac:.2%}, {n_flooded:,} cells)"
        )
        results.append(
            {
                "return_period_yr": rp,
                "flooded_km2": flooded_km2,
                "land_km2": land_km2,
                "frac_flooded": frac,
                "n_flooded": n_flooded,
                "n_land": n_land,
            }
        )
    except Exception:
        log.exception(f"RP={rp:g} yr failed -- skipping")
        continue

    pd.DataFrame(results).to_csv(csv_path, index=False)

if results:
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["return_period_yr"], df["flooded_km2"], marker="o")
    ax.set_xlabel("River discharge return period (yr)")
    ax.set_ylabel("Flooded area (km²)")
    ax.set_title(
        f"Basin {BASIN_ID} — main event flooded area vs. river discharge return period\n"
        f"(grid resolution: {resolution_m:.0f} m; production build reused as-is, "
        f"forcing_mode=river_only)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = FIGS_DIR / f"discharge_return_period_response_{_tag}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info(f"Plot written: {plot_path}")
    log.info(f"CSV written: {csv_path}")
else:
    log.warning("No successful runs -- nothing to plot")

if hmax_by_rp:
    valid_vals = np.concatenate(
        [da.values[~np.isnan(da.values)].ravel() for da in hmax_by_rp.values()]
    )
    vmax_panel = float(np.percentile(valid_vals, 99)) if valid_vals.size else 1.0
    vmax_panel = max(vmax_panel, 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    im = None
    for ax, rp in zip(axes.flat, panel_rps):
        da_hmax_rp = hmax_by_rp.get(rp)
        if da_hmax_rp is None:
            ax.set_title(f"RP={rp:g} yr (run failed)")
            ax.axis("off")
            continue

        # Reproject directly to a coarse resolution with max-value resampling
        # (see src.plots.reproject_max_for_plot) -- reprojecting compute_max_
        # inundation's native subgrid-resolution array (e.g. ~25 m with a
        # 20x20 subgrid) at full resolution produces an array with tens of
        # millions of pixels, which blows up matplotlib's imshow/RGBA
        # rendering (observed: an 11627x19827 array -> a 6.9 GiB float64
        # allocation and a crash).
        da_wgs = reproject_max_for_plot(da_hmax_rp.squeeze())
        arr = da_wgs.values.astype(np.float32)
        left, bottom, right, top = da_wgs.rio.bounds()
        im = ax.imshow(
            arr,
            cmap="Blues",
            vmin=0,
            vmax=vmax_panel,
            extent=(left, right, bottom, top),
            origin="upper",
            aspect="auto",
        )
        n_flooded_panel = int(np.isfinite(arr).sum())
        ax.set_title(f"RP={rp:g} yr  ({n_flooded_panel:,} flooded cells)")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

    for ax in axes.flat[len(panel_rps) :]:
        ax.axis("off")

    if im is not None:
        fig.colorbar(
            im, ax=axes, shrink=0.8, extend="max", label="Max inundation depth (m)"
        )
    fig.suptitle(
        f"Basin {BASIN_ID} — max inundation depth on land across the return-period sweep\n"
        f"(grid resolution: {resolution_m:.0f} m; production build reused as-is)"
    )
    panels_path = FIGS_DIR / f"discharge_return_period_response_{_tag}_panels.png"
    fig.savefig(panels_path, dpi=150)
    plt.close(fig)
    log.info(f"Panel figure written: {panels_path}")
else:
    log.warning(
        "No successful runs among the panel return periods -- skipping panel figure"
    )
