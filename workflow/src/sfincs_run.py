"""Run the SFINCS binary, hand-write sfincs.inp files and place discharge points.

Used by every rule that executes SFINCS. sfincs.inp files that borrow geometry
from another model directory (14_build_sfincs.py, 13a_run_spinup.py) are written
by hand because HydroMT's get_set_file_variable absolutizes any file reference
outside the model root instead of keeping a relative ../ path.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from src.geometry import snap_points_into_region
from src.raster import restrict_water_level_boundary_to_sea
from src.river_burn import snap_points_to_centerline_cells

logger = logging.getLogger(__name__)


def event_timeline(
    start_hr: float,
    surge_ramp_hr: float,
    surge_window_hr: float,
    surge_travel_hr: float,
    river_lead_hr: float,
    river_tail_hr: float,
    lag_hr: float,
    surge_event: bool,
    river_event: bool,
) -> tuple[float, float, float]:
    """Return (surge peak, river peak, end) in hours of an event run's forcing.

    start_hr: the restart. Surge part: the boundary blends from the calm level
    over surge_ramp_hr, then surge_window_hr each side of the peak, then
    surge_travel_hr for that peak to reach the main mouth. River part:
    river_lead_hr from the start of its ramp from the mean to its peak at the
    inflow, river_tail_hr from that peak until the event has passed the mouth
    (falling limb + travel). With both events lag_hr = river peak - surge peak,
    so both peaks reach the main mouth together; the part that needs more time
    before its peak starts at the restart, the other is padded with mean
    conditions (mean discharge; tide). The run ends when the later part ends.
    Without either event the surge part's timing holds (tide, mean discharge).
    """
    surge_lead_hr = surge_ramp_hr + surge_window_hr
    if surge_event and river_event:
        surge_hr = start_hr + max(surge_lead_hr, river_lead_hr - lag_hr)
        river_hr = surge_hr + lag_hr
    elif river_event:
        river_hr = surge_hr = start_hr + river_lead_hr
    else:
        river_hr = surge_hr = start_hr + surge_lead_hr
    end_hr = start_hr
    if surge_event or not river_event:
        end_hr = surge_hr + surge_window_hr + surge_travel_hr
    if river_event:
        end_hr = max(end_hr, river_hr + river_tail_hr)
    return surge_hr, river_hr, end_hr


def parse_sfincs_inp(path: str | Path) -> dict[str, str]:
    """Parse a sfincs.inp file into a lowercased {key: value} dict.

    The format is plain ``key = value`` lines; comments start with ``!``.
    """
    cfg: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if "=" in line and not line.startswith("!"):
                key, _, val = line.partition("=")
                cfg[key.strip().lower()] = val.strip()
    return cfg


def read_boundary_forcing(sfincs_root: str | Path) -> dict:
    """Return the water-level and discharge boundary timeseries a SFINCS run reads.

    Keys: wl_times, wl_m (point, time), q_times, q_m3s (point, time), tstart.
    Times are datetimes counted from tref; a run without a bzsfile or disfile
    gets empty arrays for it.
    """
    root = Path(sfincs_root)
    cfg = parse_sfincs_inp(root / "sfincs.inp")
    tref = pd.to_datetime(cfg["tref"], format="%Y%m%d %H%M%S")

    def _series(key: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
        if not cfg.get(key):
            return pd.DatetimeIndex([]), np.empty((0, 0))
        table = np.loadtxt(root / cfg[key], ndmin=2)
        return tref + pd.to_timedelta(table[:, 0], unit="s"), table[:, 1:].T

    wl_times, wl_m = _series("bzsfile")
    q_times, q_m3s = _series("disfile")
    return {
        "wl_times": wl_times,
        "wl_m": wl_m,
        "q_times": q_times,
        "q_m3s": q_m3s,
        "tstart": pd.to_datetime(cfg["tstart"], format="%Y%m%d %H%M%S"),
    }


def forward_geometry_files(
    cfg: dict[str, str],
    source_root: str | Path,
    dest_root: str | Path,
    exclude: frozenset[str] = frozenset(),
) -> list[str]:
    """Build ``"key = <relative path>"`` lines forwarding ``cfg``'s ``*file`` entries.

    ``cfg`` is ``source_root``'s parsed sfincs.inp; paths are made relative to
    ``dest_root`` with os.path.relpath. Skips keys in ``exclude`` (typically
    {"rstfile"}) and entries whose target is missing or a 0-byte placeholder
    (e.g. sfincs_subgrid.nc/sfincs.weir when that feature is disabled).
    """
    source_root = Path(source_root)
    dest_root = Path(dest_root)
    lines = []
    for key, value in cfg.items():
        if not key.endswith("file") or key in exclude:
            continue
        fpath = source_root / value
        if fpath.exists() and fpath.stat().st_size > 0:
            rel = os.path.relpath(fpath, start=dest_root)
            lines.append(f"{key:<20} = {rel}")
    return lines


FORCING_FILE_KEYS = ("bndfile", "bzsfile", "srcfile", "disfile", "netsrcdisfile")


def write_sfincs_inp(
    path: str | Path,
    skeleton_cfg: dict[str, str],
    skeleton_root: str | Path,
    settings: dict[str, object],
    files: dict[str, object],
    exclude_files: frozenset[str] = frozenset(),
) -> Path:
    """Write sfincs.inp: skeleton scalars, skeleton geometry files, ``settings``, ``files``.

    Hand-written because HydroMT makes paths outside the model root absolute.
    Skeleton scalars are copied unless ``settings`` sets them; skeleton ``*file``
    entries are re-pointed relative to ``path``'s directory, except rstfile, the
    forcing files and ``exclude_files``. ``files`` holds the file entries the
    caller wrote next to ``path`` (forcing, adapted components); falsy values
    are skipped.
    """
    path = Path(path)
    exclude = frozenset({"rstfile", *FORCING_FILE_KEYS}) | exclude_files
    lines = [
        f"{k:<20} = {v}"
        for k, v in skeleton_cfg.items()
        if not k.endswith("file") and k not in settings
    ]
    lines += forward_geometry_files(
        skeleton_cfg, skeleton_root, path.parent, exclude=exclude
    )
    lines += [f"{k:<20} = {v}" for k, v in settings.items()]
    lines += [f"{k:<20} = {v}" for k, v in files.items() if v]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def place_discharge_points(
    sf,
    crossings: gpd.GeoDataFrame,
    discharge: pd.DataFrame,
    centerline_cells: gpd.GeoDataFrame,
    reach_ids: Sequence[str | None],
    resolution_m: float,
) -> gpd.GeoDataFrame:
    """Snap river crossings onto the channel and the active region; add them to ``sf``.

    ``crossings`` is EPSG:4326 with an 'index' column; ``discharge`` (m3/s) has
    one column per crossing row, in row order. Each crossing snaps to a
    centreline cell of its own reach (``centerline_cells`` from
    river_burn.build_centerline_cells_regular on the model grid); crossings
    farther than one cell outside the active region are dropped, near-misses
    are nudged inside, because hydromt drops points outside the unbuffered
    region. Returns the kept crossings (EPSG:4326); raises RuntimeError when
    none remain.
    """
    snapped = snap_points_to_centerline_cells(
        crossings.to_crs(sf.crs),
        centerline_cells,
        reach_ids=reach_ids,
        resolution_m=resolution_m,
    ).to_crs("EPSG:4326")
    region_wgs84 = sf.region.to_crs("EPSG:4326").geometry.union_all()
    kept, keep_mask = snap_points_into_region(
        snapped, region_wgs84, float(resolution_m) / 111_000.0
    )
    n_outside = int((~keep_mask).sum())
    if n_outside:
        logger.warning(
            f"{n_outside}/{len(crossings)} discharge crossing(s) outside the active SFINCS region -- skipped"
        )
    if kept.empty:
        raise RuntimeError(
            "All discharge crossings fall outside the active SFINCS region"
        )
    timeseries = pd.DataFrame(
        discharge.to_numpy()[:, keep_mask],
        index=discharge.index,
        columns=kept["index"].to_numpy(),
    )
    sf.discharge_points.create(timeseries=timeseries, locations=kept)
    return kept


# Crest (and par1) format of sfincs.weir: 1 cm, the precision every crest is
# rounded to (src.surge.ceil_crest).
WEIR_Z_FMT = "%.2f"


def read_weir_file(path: str | Path) -> gpd.GeoDataFrame:
    """Read an ASCII sfincs.weir into one row per segment; `elevation` = highest crest.

    Every segment this repo writes has the same crest on both vertices.
    No CRS is set; the caller sets it.
    """
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    geoms, crests, i = [], [], 0
    while i < len(lines):
        n_pts = int(lines[i + 1].split()[0])
        pts = np.array(
            [[float(v) for v in ln.split()] for ln in lines[i + 2 : i + 2 + n_pts]]
        )
        geoms.append(LineString(pts[:, :2]))
        crests.append(float(pts[:, 2].max()))
        i += 2 + n_pts
    return gpd.GeoDataFrame({"elevation": crests}, geometry=geoms)


def read_face_zmin(subgrid_path: str | Path, mask) -> tuple[np.ndarray, np.ndarray]:
    """Return SFINCS's subgrid face minimum levels (uv_zmin) on the grid, NaN where no face.

    u[r, c] is the face between cells (r, c) and (r, c + 1), v[r, c] the face
    between (r, c) and (r + 1, c), in the array order of mask (the model's
    mask DataArray); same mapping as hydromt-sfincs' subgrid reader.
    """
    import xarray as xr
    from hydromt_sfincs import utils

    _, index_mu1, index_nu1 = utils.find_uv_indices(mask)
    shape = mask.shape
    with xr.open_dataset(subgrid_path) as ds:
        uv_zmin = ds["uv_zmin"].values.ravel()
    faces = []
    for index in (index_mu1, index_nu1):
        out = np.full(shape, np.nan, dtype=np.float32)
        active = np.flatnonzero(index > -1)
        out[np.unravel_index(active, shape, order="F")] = uv_zmin[index[active]]
        faces.append(out)
    return faces[0], faces[1]


def write_weir_file(sf) -> Path | None:
    """Rewrite ``sf``'s sfincs.weir with crests at 1 cm instead of hydromt's 0.1 m.

    ``sf.weirs.write()`` calls ``hydromt_sfincs.utils.write_geoms`` without
    ``fmt_z`` (default "%.1f", round to nearest) and offers no way to change it.
    SFINCS honours crests to the millimetre (one weir in a flat basin: crests of
    0.430 and 0.437 m start overtopping 7 mm of water level apart). Call AFTER
    ``sf.write()`` / ``sf.weirs.write()``; overwrites their file with the same
    writer and x/y format. Returns None when the model has no weirs.
    """
    from hydromt_sfincs import utils

    if sf.weirs.data is None or sf.weirs.data.empty:
        return None
    path = sf.config.get_set_file_variable(
        key="weirfile", value=None, default="sfincs.weir"
    )
    fmt_xy = "%11.6f" if sf.crs.is_geographic else "%11.1f"
    utils.write_geoms(
        path,
        utils.gdf2linestring(sf.weirs.data),
        stype="weir",
        fmt=fmt_xy,
        fmt_z=WEIR_Z_FMT,
    )
    return Path(path)


def create_active_mask(
    sf,
    include_polygon: gpd.GeoDataFrame,
    river_elevation_max_m: float | None,
    elevation_buffer_m: float,
) -> None:
    """Set the active mask to cells inside include_polygon, optionally below a ceiling.

    With river_elevation_max_m given, cells above river_elevation_max_m +
    elevation_buffer_m (m, same datum as the model elevation) stay inactive.
    """
    active_mask_kwargs = {}
    if river_elevation_max_m is not None:
        clip_elevation_m = river_elevation_max_m + elevation_buffer_m
        active_mask_kwargs = {"include_zmax": clip_elevation_m}
    sf.mask.create_active(include_polygon=include_polygon, **active_mask_kwargs)
    logger.info(
        "Active mask created: cells within delta polygon"
        + (
            f" AND below {clip_elevation_m:.1f} m (river max {river_elevation_max_m:.1f} m + "
            f"{elevation_buffer_m:.1f} m buffer)"
            if active_mask_kwargs
            else ""
        )
    )


def set_sea_and_outflow_boundaries(
    sf,
    landuse_on_grid: np.ndarray,
    n_outflow_points: int,
    outflow_buffer_m: float,
) -> None:
    """Set water-level boundary cells on open sea, then the delta outflow boundary.

    Water-level cells (mask=2) are kept only where landuse_on_grid == 200;
    raises ValueError when none remain. With n_outflow_points > 0, cells within
    outflow_buffer_m (m) of the data-catalogue source
    "local_delta_outflow_points" become outflow cells (mask=3).
    """
    # reset_bounds=False: create_active just built a fresh 0/1 mask, and hydromt's
    # create_boundary with reset_bounds=True and no polygon/elevation filter only
    # resets and returns -- it would set no boundary cells at all.
    sf.mask.create_boundary(btype="waterlevel", reset_bounds=False)
    bnd_mask, n_bnd_on_land = restrict_water_level_boundary_to_sea(
        sf.grid.data["mask"].values, landuse_on_grid
    )
    if not (bnd_mask == 2).any():
        raise ValueError(
            "No water-level boundary cell on open sea (landuse_on_grid == 200) at the "
            "active-domain edge -- check the active mask and landuse_on_grid.tif."
        )
    sf.grid.data["mask"].values[:] = bnd_mask
    logger.info(
        f"Waterlevel boundary set: {int((bnd_mask == 2).sum())} edge cell(s) on open sea -> mask=2 "
        f"({n_bnd_on_land} edge cell(s) on land use != 200 left as normal active cells)"
    )

    # reset_bounds=False overrides cells the water-level step claimed: an inland
    # truncation must not be forced to the coastal baseline. The buffer makes each
    # point capture at least one cell.
    if n_outflow_points:
        sf.mask.create_boundary(
            btype="outflow",
            include_polygon="local_delta_outflow_points",
            include_polygon_buffer=outflow_buffer_m,
            reset_bounds=False,
        )
        logger.info(
            f"Outflow boundary set: {n_outflow_points} delta-outline "
            f"crossing(s), {outflow_buffer_m:.0f} m buffer -> mask=3"
        )


def run_sfincs_subprocess(
    sfincs_exe: str | Path,
    cwd: str | Path,
    timeout_s: float,
    log,
    label: str = "SFINCS run",
    n_threads: int | None = None,
) -> None:
    """Run the SFINCS binary in ``cwd`` with a wall-clock timeout, streaming its output.

    stdout/stderr go line by line to ``log`` (info/warning) and the terminal.
    ``cwd`` is where SFINCS resolves its relative file references. ``label`` is
    used only in error messages. ``n_threads`` sets OMP_NUM_THREADS for the
    subprocess only; SFINCS defaults to 1 thread when it is unset, so pass
    ``snakemake.threads`` (rule threads from config ``resources``).
    None/0 leaves the environment unchanged.
    Raises RuntimeError on timeout or non-zero exit code.
    """
    sfincs_exe = Path(sfincs_exe).resolve()
    log.info(f"Running {label}: {sfincs_exe}")

    env = None
    if n_threads:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(n_threads))
        log.info(f"{label}: OMP_NUM_THREADS={n_threads}")

    proc = subprocess.Popen(
        [str(sfincs_exe)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
        env=env,
    )

    def _forward(pipe, log_fn):
        for line in pipe:
            line = line.rstrip()
            if line:
                log_fn(f"[sfincs] {line}")
                print(f"[sfincs] {line}", file=sys.stderr, flush=True)

    t_out = threading.Thread(target=_forward, args=(proc.stdout, log.info))
    t_err = threading.Thread(target=_forward, args=(proc.stderr, log.warning))
    t_out.start()
    t_err.start()

    # Wait before joining the reader threads: join() blocks until SFINCS closes
    # its pipes, which would make the timeout unreachable. Killing the process
    # closes its pipes, so the threads finish and join() returns.
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        t_out.join()
        t_err.join()
        raise RuntimeError(f"{label} exceeded {timeout_s}s timeout")

    t_out.join()
    t_err.join()

    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")


def level_reversal_m(zs: np.ndarray) -> np.ndarray:
    """Return per cell the largest water-level step undone by the next output step (m).

    zs is (time, cell). A step up followed by a step down (or the reverse)
    counts with the smaller of the two, so a steady rise or fall, such as a
    flood front filling the channel, counts 0. A cell whose level turns
    non-finite after a finite start gets inf.
    """
    zs = np.asarray(zs, dtype=float)
    d = np.diff(zs, axis=0)
    flip = np.sign(d[1:]) * np.sign(d[:-1]) < 0
    rev = np.where(flip, np.minimum(np.abs(d[1:]), np.abs(d[:-1])), 0.0)
    out = np.nanmax(rev, axis=0, initial=0.0)
    broke = np.isfinite(zs[0]) & ~np.isfinite(zs).all(axis=0)
    return np.where(broke, np.inf, out)


def river_cells_on_map(
    x: np.ndarray, y: np.ndarray, rivers: gpd.GeoDataFrame, spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (n, m) indices of the map cells the river lines run through.

    x, y are sfincs_map.nc's cell-centre coordinates (n, m), in the CRS of
    ``rivers``. Lines are sampled every ``spacing_m``; a sample further than
    one cell from any centre is dropped.
    """
    from scipy.spatial import cKDTree

    pts = [
        line.interpolate(d).coords[0]
        for line in rivers.geometry
        if line is not None and not line.is_empty
        for d in np.arange(0.0, line.length, spacing_m)
    ]
    if not pts:
        raise ValueError("river_cells_on_map: no river line inside the map")
    centres = np.column_stack([np.ravel(x), np.ravel(y)])
    ok = np.isfinite(centres).all(axis=1)
    tree = cKDTree(centres[ok])
    dist, k = tree.query(np.asarray(pts))
    cell_m = float(np.nanmedian(np.abs(np.diff(x, axis=1))))
    flat = np.unique(np.flatnonzero(ok)[k[dist < cell_m]])
    return np.unravel_index(flat, np.shape(x))


def river_network_stability(
    run_dir: str | Path,
    rivers: gpd.GeoDataFrame,
    reversal_max_m: float,
    spacing_m: float = 50.0,
) -> dict:
    """Return the hourly level-swing check of one finished run on the river-network cells.

    Keys: alpha (from the run's sfincs.inp), max_swing_m, cells_over (cells
    whose swing exceeds reversal_max_m), cells, x/y of the worst cell (model
    CRS), unstable. Reads only the river cells of sfincs_map.nc, which must
    hold hourly maps (sfincs.simulation.dtmapout 3600).
    """
    import xarray as xr

    run_dir = Path(run_dir)
    inp = parse_sfincs_inp(run_dir / "sfincs.inp")
    rivers = rivers.to_crs(epsg=int(inp["epsg"]))
    with xr.open_dataset(run_dir / "sfincs_map.nc") as ds:
        n, m = river_cells_on_map(ds["x"].values, ds["y"].values, rivers, spacing_m)
        zs = (
            ds["zs"]
            .isel(n=xr.DataArray(n, dims="cell"), m=xr.DataArray(m, dims="cell"))
            .values
        )
        xs, ys = ds["x"].values[n, m], ds["y"].values[n, m]
    reversal = level_reversal_m(zs)
    worst = int(np.argmax(np.nan_to_num(reversal, nan=-1.0)))
    cells_over = int((reversal > reversal_max_m).sum())
    return {
        "alpha": float(inp["alpha"]),
        "max_swing_m": float(reversal[worst]),
        "cells_over": cells_over,
        "cells": int(reversal.size),
        "x": float(xs[worst]),
        "y": float(ys[worst]),
        "unstable": bool(cells_over > 0),
    }


def river_network_steady(
    run_dir: str | Path,
    rivers: gpd.GeoDataFrame,
    window_hr: float,
    tol_m: float,
    spacing_m: float = 50.0,
) -> dict:
    """Return whether the river-network levels of a finished run have settled at its end.

    Per river-network cell the window_hr mean level at the run's end is
    compared with the window_hr mean just before it (rule 11's steady_check,
    in both directions); a change above tol_m in any cell is not steady.
    Keys: max_level_change_m, cells_changing, steady. The map must span at
    least 2 x window_hr.
    """
    import xarray as xr

    run_dir = Path(run_dir)
    inp = parse_sfincs_inp(run_dir / "sfincs.inp")
    rivers = rivers.to_crs(epsg=int(inp["epsg"]))
    with xr.open_dataset(run_dir / "sfincs_map.nc") as ds:
        n, m = river_cells_on_map(ds["x"].values, ds["y"].values, rivers, spacing_m)
        zs = (
            ds["zs"]
            .isel(n=xr.DataArray(n, dims="cell"), m=xr.DataArray(m, dims="cell"))
            .values
        )
        t_hr = (ds["time"].values - ds["time"].values[0]) / np.timedelta64(1, "h")
    end = t_hr[-1]
    last = t_hr > end - window_hr
    before = (t_hr > end - 2 * window_hr) & ~last
    if not before.any():
        raise ValueError(
            f"river_network_steady: run of {end:g} h is shorter than 2 x window_hr ({window_hr:g} h)"
        )
    change = np.abs(np.nanmean(zs[last], axis=0) - np.nanmean(zs[before], axis=0))
    cells = int(np.nansum(change > tol_m))
    return {
        "max_level_change_m": float(np.nanmax(change)),
        "cells_changing": cells,
        "steady": cells == 0,
    }


def stability_note(row: dict) -> str | None:
    """Return the one-line warning stamped on figures of an unstable run, None if stable.

    row: a stability.csv row (river_network_stability), optionally with
    spinup_unstable, the verdict of the spin-up the run restarts from.
    """
    notes = []
    if row.get("unstable", True):
        notes.append(
            f"hourly level swing {row['max_swing_m']:.2f} m on the river network "
            f"({row['cells_over']} cells, alpha {row['alpha']:g})"
        )
    if not row.get("steady", True):
        notes.append(f"spin-up not steady after {row['spinup_hr']:.0f} h")
    if row.get("spinup_unstable", False):
        notes.append("spin-up unstable or not steady")
    return "Check stability: " + " | ".join(notes) if notes else None


def read_stability(path: str | Path) -> dict:
    """Return the single row of a run's stability.csv (river_network_stability)."""
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"{path}: expected one row, found {len(df)}")
    row = df.iloc[0].to_dict()
    for key in ("unstable", "steady", "spinup_unstable"):
        if key in row:
            row[key] = bool(row[key])
    return row


def spinup_flagged(spinup_stability_csv: str | Path) -> bool:
    """Return whether a spin-up is unstable or did not settle (its stability.csv)."""
    row = read_stability(spinup_stability_csv)
    return row["unstable"] or not row.get("steady", True)


def stability_metrics(path: str | Path) -> dict:
    """Return the flood_metrics.csv columns of a run's stability.csv.

    unstable is true when the run or the spin-up it restarts from is unstable.
    """
    row = read_stability(path)
    spinup = bool(row.get("spinup_unstable", False))
    return {
        "unstable": bool(row["unstable"]) or spinup,
        "spinup_unstable": spinup,
        "max_swing_m": float(row["max_swing_m"]),
        "alpha": float(row["alpha"]),
    }


def read_observation_levels(run_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (hours since tref, water level (n_time, n_points)) of a run's sfincs_his.nc."""
    import xarray as xr

    run_dir = Path(run_dir)
    tref = pd.to_datetime(
        parse_sfincs_inp(run_dir / "sfincs.inp")["tref"], format="%Y%m%d %H%M%S"
    )
    with xr.open_dataset(run_dir / "sfincs_his.nc") as ds:
        t_hr = (pd.to_datetime(ds["time"].values) - tref) / pd.Timedelta(hours=1)
        zs = ds["point_zs"].transpose("time", ...).values
    return np.asarray(t_hr, dtype=float), zs.reshape(len(t_hr), -1)


def selected_alpha(alpha_check_csv: str | Path) -> float:
    """Return the alpha rule 11's calibration round kept (row with selected true)."""
    df = pd.read_csv(alpha_check_csv)
    row = df[df["selected"].astype(bool)]
    if len(row) != 1:
        raise ValueError(
            f"{alpha_check_csv}: expected one selected alpha, found {len(row)}"
        )
    return float(row["alpha"].iloc[0])
