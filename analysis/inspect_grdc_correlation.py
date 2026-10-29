"""Plot GRDC-vs-GloFAS correlation for every active river crossing per delta basin.

Finds the nearest GRDC station for each has_glofas crossing in river_forcing.nc
regardless of boundary_forcings.bias_correction.enabled, and writes one overview
PNG per matched crossing to <results_dir>/<basin_id>/figures/analysis/
(90_inspect_grdc_correlation_crossing_<i>_cell_(<lat>,<lon>).png). Never
applies the bias correction. Basins come from the catalogue's
delta_polygons (as BASINS in Snakemake); basins without river_forcing.nc
(rule 04) are skipped.

Usage:
    python analysis/inspect_grdc_correlation.py              # all delta basins
    python analysis/inspect_grdc_correlation.py 4267691 ...  # only these basins
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from _common import (  # noqa: E402
    REPO_ROOT,
    analysis_dir,
    analysis_path,
    script_parser,
    site_path,
)
from shapely.geometry import Polygon

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.domain import load_domain
from src.extreme_values import (
    _plot_overlap_timeseries,
    _plot_raw_scatter,
    _plot_seasonal_scatter,
    compute_grdc_correlation,
)
from src.io import catalogue_entry, load_catalogue, raw_input_path, read_geometry
from src.plots import map_background, save_figure
from src.river_forcing import (
    find_nearest_grdc_station,
    load_grdc_series,
    load_grdc_stations,
)

log = logging.getLogger(__name__)


def plot_grdc_overview(
    domain_poly: Polygon,
    land_polygons_path: str,
    river_gdf: gpd.GeoDataFrame,
    crossings_gdf: gpd.GeoDataFrame,
    grdc_stations: gpd.GeoDataFrame,
    highlight_crossing_idx: int,
    highlight_station_id: int,
    diagnostics: dict,
    output_path: str | Path,
    label: str = "",
) -> None:
    """Render a 2x2 GRDC-vs-GloFAS correlation overview figure (raw overlap only).

    diagnostics comes from compute_grdc_correlation, so no bias correction is
    needed. All geometries WGS84; crossings_gdf needs a bool has_glofas
    column; land_polygons_path is the land mask (land use != 200) background.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g_ov = np.asarray(diagnostics["glofas_overlap"], dtype=float)
    r_ov = np.asarray(diagnostics["grdc_overlap"], dtype=float)
    t_ov = pd.to_datetime(diagnostics["glofas_overlap_times"])
    tail_mask = diagnostics["tail_mask"]
    tail_pct = diagnostics["tail_percentile"]
    r_raw = diagnostics["correlation_raw"]
    r_raw_full = diagnostics["correlation_full_raw"]
    n_overlap = diagnostics["grdc_overlap_days"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"GRDC-GloFAS correlation — {label}  (station {highlight_station_id}, {n_overlap} d overlap)",
        fontsize=13,
        y=0.98,
    )

    ax = axes[0, 0]
    station_pt = grdc_stations.loc[
        grdc_stations["id"] == highlight_station_id, "geometry"
    ].iloc[0]
    crossing_pt = crossings_gdf.geometry.iloc[highlight_crossing_idx]

    map_background(ax, domain_poly, land_polygons_path, margin_frac=0.15)
    bx, by = domain_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=2, zorder=2, label="Model domain")

    # Widen the view if the matched GRDC station falls outside the domain bbox.
    xlim = (min(ax.get_xlim()[0], station_pt.x), max(ax.get_xlim()[1], station_pt.x))
    ylim = (min(ax.get_ylim()[0], station_pt.y), max(ax.get_ylim()[1], station_pt.y))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    map_bounds = (xlim[0], ylim[0], xlim[1], ylim[1])

    river_gdf.plot(ax=ax, color="steelblue", linewidth=0.8, zorder=1)

    if not crossings_gdf.empty:
        active = crossings_gdf["has_glofas"].astype(bool)
        if active.any():
            crossings_gdf[active].plot(
                ax=ax,
                color="limegreen",
                markersize=40,
                zorder=3,
                label="Crossing (active)",
            )
        if (~active).any():
            crossings_gdf[~active].plot(
                ax=ax,
                color="grey",
                markersize=40,
                marker="x",
                zorder=3,
                label="Crossing (inactive)",
            )

    grdc_in_view = grdc_stations.cx[
        map_bounds[0] : map_bounds[2], map_bounds[1] : map_bounds[3]
    ]
    if not grdc_in_view.empty:
        grdc_in_view.plot(
            ax=ax,
            color="purple",
            markersize=70,
            marker="*",
            zorder=3,
            alpha=0.7,
            label="GRDC station",
        )

    ax.scatter(
        [crossing_pt.x],
        [crossing_pt.y],
        s=180,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        zorder=4,
        label="This crossing",
    )
    ax.scatter(
        [station_pt.x],
        [station_pt.y],
        s=180,
        facecolors="none",
        edgecolors="orange",
        linewidths=2,
        zorder=4,
        label="Matched GRDC station",
    )

    ax.set_title("Domain context")
    ax.legend(loc="best", fontsize=7)

    lims = (0.0, float(max(g_ov.max(), r_ov.max())) * 1.05)
    _plot_raw_scatter(
        axes[0, 1], g_ov, r_ov, r_raw, r_raw_full, tail_mask, tail_pct, lims
    )
    _plot_seasonal_scatter(axes[1, 0], g_ov, r_ov, t_ov, lims)
    _plot_overlap_timeseries(axes[1, 1], t_ov, g_ov, r_ov, r_raw)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path, dpi=130)
    log.info(f"Wrote GRDC overview plot: {output_path}")


def build_parser():
    """Return the command-line parser."""
    parser = script_parser(__doc__)
    parser.add_argument(
        "basin_ids",
        nargs="*",
        type=int,
        help="only these delta basin ids (default: all delta basins)",
    )
    return parser


def main():
    """Run the GRDC overview plots for the requested delta basins."""
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # raw_data_root is a site setting; see config/site.example.yml.
    GRDC_PATH = site_path("raw_data_root") / "GRDC/GRDC-Daily.nc"

    with open(REPO_ROOT / "config" / "config.yml") as f:
        config = yaml.safe_load(f)
    river_cfg = config["boundary_forcings"]["river"]
    glofas_variable = (
        "dis24"  # sole variable of the catalogue's river_discharge (GloFAS v4) source
    )
    grdc_radius_m = (
        float(river_cfg["bias_correction"]["grdc_search_radius_km"]) * 1000.0
    )
    min_overlap_days = int(river_cfg["bias_correction"]["min_overlap_days"])
    # results_dir is a site setting; see config/site.example.yml.
    results_dir_root = site_path("results_dir")

    catalogue = load_catalogue(REPO_ROOT / "config" / "data_catalogue.yml")
    delta_attr = catalogue_entry(catalogue, "delta_polygons")["attributes"][0]["name"]
    deltas = read_geometry(raw_input_path(catalogue, "delta_polygons"))
    basin_ids = sorted(deltas[delta_attr].astype(int).to_list())

    if args.basin_ids:
        requested = set(args.basin_ids)
        basin_ids = [b for b in basin_ids if b in requested]

    grdc_stations = load_grdc_stations(GRDC_PATH)

    n_written_total = 0
    n_no_station_total = 0
    n_skipped_total = 0
    n_basins_skipped = 0

    for basin_id in basin_ids:
        results_dir = results_dir_root / str(basin_id)
        river_forcing_path = (
            results_dir / "preprocessing_inputs/forcing/river_forcing.nc"
        )
        if not river_forcing_path.exists():
            print(f"basin {basin_id}: no river_forcing.nc yet, skipping")
            n_basins_skipped += 1
            continue

        domain_dir = results_dir / "preprocessing_inputs/domain"
        _, domain_crs, domain_poly = load_domain(
            domain_dir / "domain_bbox.json", domain_dir / f"{basin_id}_domain.gpkg"
        )
        river_gdf = gpd.read_file(domain_dir / f"{basin_id}_river_network.gpkg")

        river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
        glofas_clip = xr.open_dataset(
            results_dir / "preprocessing_inputs/forcing/glofas_clip.nc"
        )

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

            crossing = f"crossing_{i}_cell_({cell_lat:.3f},{cell_lon:.3f})"
            label = f"basin_{basin_id}_{crossing}"
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
                land_polygons_path=domain_dir / f"{basin_id}_land_polygons.gpkg",
                river_gdf=river_gdf,
                crossings_gdf=crossings_gdf,
                grdc_stations=grdc_stations,
                highlight_crossing_idx=i,
                highlight_station_id=grdc_id,
                diagnostics=diagnostics,
                output_path=analysis_path(basin_id, __file__, f"{crossing}.png"),
                label=label,
            )
            n_written += 1

        print(
            f"basin {basin_id}: {n_written} plot(s) written to {analysis_dir(basin_id)}, "
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


if __name__ == "__main__":
    main()
