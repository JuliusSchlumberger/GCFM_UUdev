"""Shared inputs of the Phase 1 diagnostic scripts (analysis/phase1_*.py): basin paths and a rule-11 round-0 bed."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from _common import REPO_ROOT, site_path

sys.path.insert(0, str(REPO_ROOT / "workflow"))

PHASE1_BASINS = ["2433835", "3279946"]  # Ebro first, then Chao Phraya


def load_config() -> dict:
    """Return config/config.yml as a dict."""
    return yaml.safe_load(
        (REPO_ROOT / "config" / "config.yml").read_text(encoding="utf-8")
    )


def basin_paths(basin_id: str) -> dict[str, Path]:
    """Return the rule 06-11 outputs of a basin the Phase 1 scripts read."""
    root = site_path("results_dir") / basin_id
    prep = root / "preprocessing_inputs"
    dom, frc = prep / "domain", prep / "forcing"
    return {
        "root": root,
        "prep": prep,
        "diagnostics": prep / "diagnostics",
        "sfincs_grid": dom / f"{basin_id}_sfincs_grid.json",
        "domain_bbox": dom / "domain_bbox.json",
        "domain_gpkg": dom / f"{basin_id}_domain.gpkg",
        "elevation_merged": dom / f"{basin_id}_elevation_merged.tif",
        "rivers_clean": dom / f"{basin_id}_river_network_clean.gpkg",
        "rivers_depth": dom / f"{basin_id}_river_network_depth_estimated.gpkg",
        "water_surface": dom / f"{basin_id}_river_water_surface.gpkg",
        "burned_dem_grid": dom / f"{basin_id}_river_burned_dem_sfincs_grid.tif",
        "protection_levels": dom / "protection_levels.json",
        "river_forcing": frc / "river_forcing.nc",
        "surge_forcing": frc / "surge_forcing.nc",
        "node_table": prep
        / "visuals"
        / "11_calibration"
        / "river_bed"
        / f"{basin_id}_river_bed_nodes.csv",
        "round1": prep / "depth_crest_calibration" / "round1",
        "calibration_rounds": prep
        / "diagnostics"
        / f"{basin_id}_calibration_rounds.json",
        "water_level_r1": prep
        / "diagnostics"
        / f"{basin_id}_calibration_water_level_r1.csv",
        "spin_up": root / "spin_up",
        "skeleton": root / "sfincs_skeleton",
    }


def calm_sea_level_m(basin_id: str) -> float:
    """Return the basin's calm sea level (surge_forcing.nc), as rule 11 uses it."""
    import xarray as xr
    from src.surge import read_calm_sea_level_m

    with xr.open_dataset(
        basin_paths(basin_id)["surge_forcing"], decode_times=False
    ) as ds:
        return float(read_calm_sea_level_m(ds))


def round0(basin_id: str, cfg: dict, **bed_overrides):
    """Re-run rule 11's round 0 on the stored rule 08-10 outputs; return (bed, rivers, q_design, q_mean).

    bed_overrides replace keys of river_depth_modelling.bed (e.g.
    design_discharge="mean", slope_floor=5e-6). Same call as rule 11
    (src.river_bed.estimate_river_bed_from_files), on the stored water surface.
    """
    import geopandas as gpd
    from src.domain import load_domain
    from src.grid import load_grid_def
    from src.io import catalogue_entry, load_catalogue, raw_input_path
    from src.river_bed import estimate_river_bed_from_files

    p = basin_paths(basin_id)
    rp = cfg["river_processing"]
    bed_cfg = {**rp["river_depth_modelling"]["bed"], **bed_overrides}
    cat = load_catalogue(REPO_ROOT / cfg["data_catalogue"])
    sword = [
        raw_input_path(cat, "river_network").format(region=r)
        for r in catalogue_entry(cat, "river_network")["regions"]
    ]
    crs = load_grid_def(p["sfincs_grid"])["crs"]
    wgs84_bounds, _, _ = load_domain(p["domain_bbox"], p["domain_gpkg"])
    rivers = gpd.read_file(p["rivers_clean"])
    bed, q_design, _seeds, q_mean = estimate_river_bed_from_files(
        rivers,
        p["river_forcing"],
        p["water_surface"],
        sword,
        wgs84_bounds,
        crs,
        sea_level_m=calm_sea_level_m(basin_id),
        gebco_min_depth_m=cfg["terrain"]["gebco_min_depth_m"],
        manning_n=rp["channel_manning_n"],
        surface_cfg=rp["water_surface"],
        cfg=bed_cfg,
    )
    return bed, rivers, q_design, q_mean


def node_class(bed, basin_id: str) -> np.ndarray:
    """Return rule 10's node class per node of the bed's network."""
    from src.river_bed import surface_on_network
    from src.water_surface import read_water_surface

    nodes, _ = read_water_surface(basin_paths(basin_id)["water_surface"])
    return surface_on_network(bed.net, nodes)["node_class"].to_numpy(object)


def network_end_levels(bed) -> dict[int, float]:
    """Return rule 10's surface level at every downstream network end (as src.river_bed does)."""
    net = bed.net
    return {
        int(net.reach_nodes[r][-1]): float(bed.level_m[net.reach_nodes[r][-1]])
        for r in net.reach_ids
        if not net.downstream[r]
    }


def reach_q_dict(rivers, q: np.ndarray) -> dict[str, float]:
    """Return {reach_id: Q} for a per-reach discharge array of rivers."""
    from src.river_network import normalize_reach_id

    return dict(
        zip(rivers["reach_id"].map(normalize_reach_id), np.asarray(q, dtype=float))
    )


def git_hash() -> str:
    """Return the short git hash for figure titles."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def pct(values: np.ndarray, q: float) -> float:
    """Return the q-th percentile of the finite values, NaN when there are none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def write_summary(rows: list[dict], path: Path) -> pd.DataFrame:
    """Write the cross-basin summary rows as CSV and return them."""
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4g")
    return df


def read_json(path: Path) -> dict:
    """Return a JSON file as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def calibration_sea_level_m(basin_id: str) -> float:
    """Return rule 11's calibration sea level: the highest station storm tide at the FLOPROS coastal RP."""
    import xarray as xr
    from src.surge import storm_tide_at_rp_interpolated

    rp = read_json(basin_paths(basin_id)["protection_levels"]).get(
        "surge_protection_rp_yr"
    )
    rp = float(rp) if rp is not None and np.isfinite(float(rp)) else None
    if rp is None:
        return calm_sea_level_m(basin_id)
    with xr.open_dataset(
        basin_paths(basin_id)["surge_forcing"], decode_times=False
    ) as ds:
        return float(np.max(storm_tide_at_rp_interpolated(ds, rp)))


def map_cells(map_path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, zs[time, cell], time_s) of a regular SFINCS sfincs_map.nc, cells flattened."""
    import xarray as xr

    with xr.open_dataset(map_path, decode_times=False) as ds:
        x = ds["x"].values.ravel()
        y = ds["y"].values.ravel()
        zs = ds["zs"].values.reshape(ds["zs"].shape[0], -1).astype(float)
        t = ds["time"].values.astype(float)
    zs[zs < -1000.0] = np.nan  # SFINCS fill value of inactive cells
    return x, y, zs, t


def raster_at(path, x: np.ndarray, y: np.ndarray, crs=None) -> np.ndarray:
    """Sample band 1 of a raster at points (in crs, default the raster's); NaN for nodata."""
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as src:
        if crs is not None and src.crs != crs:
            x, y = Transformer.from_crs(crs, src.crs, always_xy=True).transform(x, y)
        v = np.array([s[0] for s in src.sample(zip(x, y))], dtype=float)
        if src.nodata is not None:
            v[v == src.nodata] = np.nan
    return v
