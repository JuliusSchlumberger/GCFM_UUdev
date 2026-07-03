"""Standalone inspection of GRDC-vs-GloFAS correlation across delta basins.

For every active river crossing (has_glofas) in each basin's river_forcing.nc,
independently finds the nearest GRDC station (regardless of whether
boundary_forcings.bias_correction.enabled was set when the pipeline last ran)
and computes the raw overlap correlation. Writes one overview PNG per matched
crossing -- domain map, raw scatter, seasonal scatter, overlap time series --
to figs/grdc_overview/<basin_id>/. Never writes to results/ and never applies
the bias correction itself.

Basins are taken from the delta_polygons dataset in the data catalogue (the
same set as BASINS in the Snakemake workflow). Basins whose rule-04 outputs
don't exist yet (river_forcing.nc missing) are skipped.

Usage:
    python tests/inspect_grdc_correlation.py              # all delta basins
    python tests/inspect_grdc_correlation.py 4267691 ...  # only these basins
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.domain import load_domain
from src.extreme_values import compute_grdc_correlation, plot_grdc_overview
from src.io import catalogue_entry, load_catalogue, raw_input_path, read_geometry
from src.river_forcing import (
    find_nearest_grdc_station,
    load_grdc_series,
    load_grdc_stations,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
GRDC_PATH = Path("D:/GCFM_UU/raw_data/GRDC/GRDC-Daily.nc")

with open(REPO_ROOT / "config" / "config.yml") as f:
    config = yaml.safe_load(f)
river_cfg = config["boundary_forcings"]["river"]
glofas_variable = river_cfg["glofas_variable"]
grdc_radius_m = float(river_cfg["grdc_search_radius_km"]) * 1000.0
min_overlap_days = int(river_cfg["bias_correction"]["min_overlap_days"])
results_dir_root = Path(config["results_dir"])

catalogue = load_catalogue(REPO_ROOT / "config" / "data_catalogue.yml")
delta_attr = catalogue_entry(catalogue, "delta_polygons")["attributes"][0]["name"]
deltas = read_geometry(raw_input_path(catalogue, "delta_polygons"))
basin_ids = sorted(deltas[delta_attr].astype(int).to_list())

if len(sys.argv) > 1:
    requested = {int(a) for a in sys.argv[1:]}
    basin_ids = [b for b in basin_ids if b in requested]

grdc_stations = load_grdc_stations(GRDC_PATH)

n_written_total = 0
n_no_station_total = 0
n_skipped_total = 0
n_basins_skipped = 0

for basin_id in basin_ids:
    results_dir = results_dir_root / str(basin_id)
    river_forcing_path = results_dir / "inputs/forcing/river_forcing.nc"
    if not river_forcing_path.exists():
        print(f"basin {basin_id}: no river_forcing.nc yet, skipping")
        n_basins_skipped += 1
        continue

    domain_dir = results_dir / "inputs/domain"
    _, domain_crs, domain_poly = load_domain(
        domain_dir / "domain_bbox.json", domain_dir / f"{basin_id}_domain.gpkg"
    )
    river_gdf = gpd.read_file(domain_dir / f"{basin_id}_river_network.gpkg")

    river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
    glofas_clip = xr.open_dataset(results_dir / "inputs/forcing/glofas_clip.nc")

    lat_dim = "latitude" if "latitude" in glofas_clip.dims else "lat"
    lon_dim = "longitude" if "longitude" in glofas_clip.dims else "lon"
    time_dim = "valid_time" if "valid_time" in glofas_clip.dims else "time"
    lat_arr = glofas_clip[lat_dim].values
    lon_arr = glofas_clip[lon_dim].values
    times_arr = glofas_clip[time_dim].values

    crossings_gdf = gpd.GeoDataFrame(
        {"has_glofas": river_ds["has_glofas"].values.astype(bool)},
        geometry=gpd.points_from_xy(
            river_ds["longitude"].values, river_ds["latitude"].values
        ),
        crs="EPSG:4326",
    )

    out_dir = REPO_ROOT / "figs" / "grdc_overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_no_station = 0
    n_skipped = 0
    seen: set[tuple] = set()
    for i in range(river_ds.sizes["crossing"]):
        if not bool(river_ds["has_glofas"].values[i]):
            continue

        pt_lon = float(river_ds["longitude"].values[i])
        pt_lat = float(river_ds["latitude"].values[i])
        grdc_id = find_nearest_grdc_station(
            grdc_stations, pt_lon, pt_lat, grdc_radius_m, utm_crs=domain_crs
        )
        if grdc_id is None:
            n_no_station += 1
            continue

        cell_lon = float(river_ds["glofas_cell_lon"].values[i])
        cell_lat = float(river_ds["glofas_cell_lat"].values[i])
        i_lat = int(np.argmin(np.abs(lat_arr - cell_lat)))
        i_lon = int(np.argmin(np.abs(lon_arr - cell_lon)))

        key = (i_lat, i_lon, grdc_id)
        if key in seen:
            continue
        seen.add(key)

        glofas_values = (
            glofas_clip[glofas_variable]
            .isel({lat_dim: i_lat, lon_dim: i_lon})
            .values.astype(float)
        )
        grdc_times, grdc_values = load_grdc_series(GRDC_PATH, grdc_id)

        label = f"basin_{basin_id}_crossing_{i}_cell_({cell_lat:.3f},{cell_lon:.3f})"
        diagnostics = compute_grdc_correlation(
            times_arr,
            glofas_values,
            grdc_times,
            grdc_values,
            min_overlap_days,
            label=label,
        )
        if diagnostics is None:
            print(
                f"{label}: GRDC station {grdc_id} found but overlap < {min_overlap_days} d"
            )
            n_skipped += 1
            continue

        plot_grdc_overview(
            domain_poly=domain_poly,
            osm_land_path=domain_dir / f"{basin_id}_land_polygons.gpkg",
            river_gdf=river_gdf,
            crossings_gdf=crossings_gdf,
            grdc_stations=grdc_stations,
            highlight_crossing_idx=i,
            highlight_station_id=grdc_id,
            diagnostics=diagnostics,
            output_path=out_dir / f"grdc_overview_{label}.png",
            label=label,
        )
        n_written += 1

    print(
        f"basin {basin_id}: {n_written} plot(s) written to {out_dir}, "
        f"{n_skipped} match(es) with overlap < {min_overlap_days} d, "
        f"{n_no_station} crossing(s) with no GRDC station within {grdc_radius_m / 1000:.1f} km"
    )
    n_written_total += n_written
    n_no_station_total += n_no_station
    n_skipped_total += n_skipped

print(
    f"\n{n_written_total} plot(s) written across {len(basin_ids) - n_basins_skipped} basin(s) "
    f"({n_basins_skipped} basin(s) skipped, no river_forcing.nc yet)\n"
    f"{n_skipped_total} crossing(s) had a GRDC match but overlap < {min_overlap_days} d\n"
    f"{n_no_station_total} crossing(s) had no GRDC station within {grdc_radius_m / 1000:.1f} km"
)
