"""
test_subgrid_river_depth_profile.py — Diagnostic: what river bed elevation
actually ended up burned into each basin's *built* SFINCS subgrid table,
along the main river arms from each active boundary-forcing crossing (seed
reach) down to wherever that mainstem run ends (mouth/outlet)?

Unlike tests/test_river_burning_sfincs.py (which recomputes the bed-level
estimation standalone, on the whole unblocked raster, via
hydromt_sfincs.workflows.bathymetry.burn_river_rect -- useful for
diagnosing the block-tiling crash but not necessarily identical to what a
real build actually wrote), this reads the real, final product: each
basin's sfincs/subgrid/dep_subgrid_lev*.tif (quadtree) or dep_subgrid.tif
(regular grid). No standalone re-burning logic here at all -- just
sampling what's actually on disk.

Samples via rasterio's windowed point .sample() (finest quadtree level
first, falling back to coarser levels where the finest has no data) rather
than src.postprocessing.get_bed_level/_mosaic_quadtree_dep_levels, which
materializes a full-domain mosaic at finest-subgrid resolution -- several
GiB and an easy MemoryError for a large basin, when all that's actually
needed here is a handful of narrow river corridors.

For every basin under results_dir with a built subgrid (sfincs/subgrid/
contains dep_subgrid*.tif), traces the downstream mainstem path from every
active river boundary-forcing crossing (is_seed == True;
src.river_network.trace_seed_mainstem_paths) and plots, along that path's
cumulative distance from the seed:
  - the original (unburned) DEM elevation near the channel
    (src.river_network.sample_dem_near_river),
  - the burned bed elevation actually present in the built subgrid.

Outputs (figs/subgrid_river_depth_profile/): one PNG + one CSV per (basin,
seed crossing).

Usage:
    conda run -n hmt_sfincs_dev python tests/test_subgrid_river_depth_profile.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.river_network import (
    _as_linestring,
    compute_seed_path_offsets,
    normalize_reach_id,
    sample_dem_near_river,
    trace_seed_mainstem_paths,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    logging.getLogger(_name).setLevel(logging.WARNING)

plt.ioff()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = REPO_ROOT / "figs" / "subgrid_river_depth_profile"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
RESULTS_DIR = Path(config["results_dir"])
CONDITIONING_ENABLED = bool(config["river_processing"]["conditioning"]["enabled"])


def list_basins_with_subgrid() -> list[str]:
    """Basins under results_dir with a built sfincs/subgrid/dep_subgrid*.tif."""
    basin_ids = []
    for subgrid_dir in sorted(RESULTS_DIR.glob("*/sfincs/subgrid")):
        if any(subgrid_dir.glob("dep_subgrid*.tif")):
            basin_ids.append(subgrid_dir.parents[1].name)
    return basin_ids


def find_subgrid_dep_paths(sfincs_root: Path) -> list[Path]:
    """
    Dep-elevation raster(s) for a built model, finest-resolution first.
    Quadtree builds write one dep_subgrid_lev*.tif per refinement level
    (each NaN/nodata outside that level's own cells); a regular grid
    writes a single dep_subgrid.tif. Resolution is read from each file's
    own metadata only (rasterio.open, no data read) to decide priority.
    """
    subgrid_dir = sfincs_root / "subgrid"
    level_paths = list(subgrid_dir.glob("dep_subgrid_lev*.tif"))
    if level_paths:
        return sorted(level_paths, key=lambda p: abs(rasterio.open(p).res[0]))
    single_path = subgrid_dir / "dep_subgrid.tif"
    return [single_path] if single_path.exists() else []


def sample_bed_along_line(
    line: LineString, dep_paths_finest_first: list[Path], step_m: float
) -> list[dict]:
    """
    Sample bed elevation along `line` at `step_m` intervals using
    rasterio's windowed point .sample() (reads only the blocks touched by
    the requested points, never the full raster). Tries each raster in
    `dep_paths_finest_first` order, keeping the first valid (non-nodata,
    finite) value found per point -- mirrors
    src.postprocessing._mosaic_quadtree_dep_levels' finest-takes-priority
    convention without ever materializing a full-domain mosaic.
    """
    length = line.length
    n = 1 if length == 0 else max(2, int(np.ceil(length / step_m)) + 1)
    distances = np.linspace(0.0, length, n)
    xys = [(p.x, p.y) for p in (line.interpolate(d) for d in distances)]

    readers = [rasterio.open(p) for p in dep_paths_finest_first]
    try:
        samples_per_level = [list(r.sample(xys)) for r in readers]
        nodatas = [r.nodata for r in readers]
    finally:
        for r in readers:
            r.close()

    results: list[dict] = []
    for i, d in enumerate(distances):
        z = np.nan
        for level_samples, nodata in zip(samples_per_level, nodatas):
            val = float(level_samples[i][0])
            if nodata is not None and val == nodata:
                continue
            if np.isfinite(val):
                z = val
                break
        if np.isfinite(z):
            results.append({"along_m": float(d), "bed_elevation_m": z})
    return results


def plot_seed_profile(
    basin_id: str,
    seed: str,
    old_pixels: pd.DataFrame,
    burned: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if not old_pixels.empty:
        ax.scatter(
            old_pixels["distance_from_seed_m"],
            old_pixels["elevation_m"],
            s=3,
            color="grey",
            alpha=0.3,
            label="DEM (original)",
        )
    if not burned.empty:
        ax.plot(
            burned["distance_from_seed_m"],
            burned["bed_elevation_m"],
            color="steelblue",
            linewidth=1.2,
            label="Burned subgrid bed elevation",
        )

    ax.set_title(f"Basin {basin_id}, seed reach {seed}: subgrid bed elevation profile")
    ax.set_xlabel("Distance from seed (active boundary-forcing crossing), m")
    ax.set_ylabel("Elevation (m)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def process_basin(basin_id: str) -> None:
    basin_dir = RESULTS_DIR / basin_id / "inputs"
    sfincs_root = RESULTS_DIR / basin_id / "sfincs"
    river_path = basin_dir / "domain" / f"{basin_id}_river_network_processed.gpkg"
    elevation_path = (
        basin_dir
        / "domain"
        / (
            f"{basin_id}_elevation_conditioned.tif"
            if CONDITIONING_ENABLED
            else f"{basin_id}_elevation_merged.tif"
        )
    )
    if not river_path.exists() or not elevation_path.exists():
        log.warning(
            f"basin {basin_id}: missing river network or elevation raster, skipping"
        )
        return

    rivers = gpd.read_file(river_path)
    required = {
        "reach_id",
        "rch_id_dn",
        "main_path_id",
        "is_mainstem_edge",
        "is_seed",
        "width",
        "dist_out",
    }
    missing = required - set(rivers.columns)
    if missing:
        log.warning(f"basin {basin_id}: missing column(s) {sorted(missing)}, skipping")
        return

    seed_paths = trace_seed_mainstem_paths(rivers)
    if not seed_paths:
        log.warning(
            f"basin {basin_id}: no active boundary-forcing (is_seed) reach, skipping"
        )
        return

    dep_paths = find_subgrid_dep_paths(sfincs_root)
    if not dep_paths:
        log.warning(f"basin {basin_id}: no bed level available, skipping")
        return
    with rasterio.open(dep_paths[0]) as _src:
        bed_crs = _src.crs
        step_m = abs(_src.res[0])

    rivers_proj = rivers.to_crs(bed_crs)
    rivers_proj["reach_id_norm"] = rivers_proj["reach_id"].apply(normalize_reach_id)

    offsets_by_seed = {
        seed: compute_seed_path_offsets(rivers_proj, path_rids)
        for seed, path_rids in seed_paths.items()
    }

    for seed, path_rids in seed_paths.items():
        offsets = offsets_by_seed[seed]
        rivers_path_gdf = rivers[
            rivers["reach_id"].apply(normalize_reach_id).isin(path_rids)
        ]
        rivers_path_proj = rivers_proj[rivers_proj["reach_id_norm"].isin(path_rids)]

        old_pixels = sample_dem_near_river(
            rivers_path_gdf, elevation_path, width_column="width"
        )
        if not old_pixels.empty:
            old_pixels["distance_from_seed_m"] = (
                old_pixels["reach_id"].map(offsets) + old_pixels["along_m"]
            )

        burned_rows: list[dict] = []
        for row in rivers_path_proj.itertuples():
            rid = row.reach_id_norm
            line = _as_linestring(row.geometry)
            if line is None or line.length == 0:
                continue
            offset = offsets.get(rid, 0.0)
            for s in sample_bed_along_line(line, dep_paths, step_m):
                burned_rows.append(
                    {
                        "reach_id": rid,
                        "along_m": s["along_m"],
                        "distance_from_seed_m": offset + s["along_m"],
                        "bed_elevation_m": s["bed_elevation_m"],
                    }
                )
        burned = (
            pd.DataFrame(burned_rows).sort_values("distance_from_seed_m")
            if burned_rows
            else pd.DataFrame()
        )

        csv_path = FIGS_DIR / f"{basin_id}_seed{seed}_subgrid_depth.csv"
        pd.concat(
            [
                old_pixels.assign(series="old_dem")
                if not old_pixels.empty
                else old_pixels,
                burned.assign(series="burned_subgrid") if not burned.empty else burned,
            ],
            ignore_index=True,
        ).to_csv(csv_path, index=False)

        png_path = FIGS_DIR / f"{basin_id}_seed{seed}_subgrid_depth.png"
        plot_seed_profile(basin_id, seed, old_pixels, burned, png_path)
        log.info(f"basin {basin_id} seed {seed}: wrote {png_path}")


def main() -> None:
    basin_ids = list_basins_with_subgrid()
    log.info(f"{len(basin_ids)} basin(s) with a built subgrid: {basin_ids}")
    for basin_id in basin_ids:
        process_basin(basin_id)


if __name__ == "__main__":
    main()
