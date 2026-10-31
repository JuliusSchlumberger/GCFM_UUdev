"""Draw the preprocessing diagnostic figures from finished outputs; run by rule analysis_preprocess.

Protection levels (03), datum corrections (06), boundary-forcing diagnostics
(07), the cleaned river network (08) and the river water surface (10). Reads
only files rules 03-10 wrote, so drawing them never reruns those rules.
"""

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from src.domain import load_domain
from src.extreme_values import (
    plot_bias_correction,
    plot_cell_diagnostics,
    read_bias_diagnostics,
)
from src.log import setup_logging
from src.plots import (
    plot_clean_network_discharge,
    plot_cleaned_network,
    plot_datum_correction_delta,
    plot_domain_map,
    plot_geoid_offset,
    plot_mdt_ocean,
    plot_protection_levels,
    plot_storm_tide_shapes,
    plot_surge_corrections,
    plot_water_surface_profiles,
    save_figure,
)
from src.protection_levels import load_flopros_table
from src.raster import latlon_dims, read_float_raster, vectorize_land_from_landuse
from src.river_network import normalize_channel_widths

log = setup_logging(snakemake.log[0])
inp, out = snakemake.input, snakemake.output
basin_id = snakemake.wildcards.basin_id
wgs84_bounds, _, domain_poly = load_domain(inp.spec_basins_meta, inp.domain_gpkg)
land = str(inp.land_polygons)


def _placeholder(text: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=13,
        color="grey",
    )
    ax.set_axis_off()
    save_figure(fig, path, dpi=100, bbox_inches="tight")


# 03 protection levels; background from the raw global land use, as the
# protection lookup depends only on the delta polygon. The 0.5 margin matches
# plot_protection_levels' margin_frac default.
delta_polygon = (
    gpd.read_file(inp.delta_polygon).to_crs("EPSG:4326").geometry.union_all()
)
with open(inp.protection_levels) as f:
    summary = json.load(f)
lon_min, lat_min, lon_max, lat_max = delta_polygon.bounds
margin = max(lon_max - lon_min, lat_max - lat_min) * 0.5
land_global = vectorize_land_from_landuse(
    inp.global_landuse,
    (lon_min - margin, lat_min - margin, lon_max + margin, lat_max + margin),
)
with tempfile.TemporaryDirectory() as tmp_dir:
    land_tmp = Path(tmp_dir) / "land_from_landuse.gpkg"
    land_global.to_file(land_tmp, driver="GPKG")
    plot_protection_levels(
        delta_polygon,
        inp.geogunit_raster,
        load_flopros_table(inp.flopros_table),
        summary,
        str(land_tmp),
        out.plot_protection,
    )

# 06 datum corrections
offset, offset_transform, _ = read_float_raster(inp.geoid_offset)
plot_geoid_offset(
    offset_arr=offset,
    offset_transform=offset_transform,
    wgs84_bounds=wgs84_bounds,
    land_polygons_path=land,
    output_path=out.plot_geoid_offset,
)
mdt, mdt_transform, _ = read_float_raster(inp.mdt_ocean)
plot_mdt_ocean(
    mdt_np=mdt,
    mdt_transform=mdt_transform,
    wgs84_bounds=wgs84_bounds,
    land_polygons_path=land,
    output_path=out.plot_mdt_ocean,
)
plot_datum_correction_delta(
    delta=read_float_raster(inp.datum_fathomdem)[0],
    wgs84_bounds=wgs84_bounds,
    land_polygons_path=land,
    output_path=out.plot_datum_fathomdem,
    title="Vertical datum correction — FathomDEM\n(EGM2008 → GOCO06s geoid)",
    colorbar_label="Δ elevation (m)  [GOCO06s − EGM2008]",
    note=(
        "Note: FathomDEM and the underlying land-use map (grey background) are not fully aligned.\n"
        "In areas without FathomDEM coverage, GEBCO data are used (see 06_datum_gebco_correction.png)."
    ),
)
plot_datum_correction_delta(
    delta=read_float_raster(inp.datum_gebco)[0],
    wgs84_bounds=wgs84_bounds,
    land_polygons_path=land,
    output_path=out.plot_datum_gebco,
    title="Vertical datum correction — GEBCO\n(local MSL → GOCO06s: + MDT_CNES-CLS22)",
    colorbar_label="Δ elevation (m)  [GOCO06s − raw GEBCO]",
)

# 07 boundary forcing
if hasattr(out, "plot_storm_tide_shapes"):
    plot_storm_tide_shapes(inp.hydrographs, basin_id, out.plot_storm_tide_shapes)

stations = gpd.read_file(inp.surge_stations)
plot_surge_corrections(stations, output_path=out.plot_surge_correction)

crossings = gpd.read_file(inp.crossings)
has_glofas = (
    crossings["has_glofas"].to_numpy(bool) if len(crossings) else np.zeros(0, bool)
)
river_gdf = normalize_channel_widths(
    gpd.read_file(inp.spec_river_network).to_crs("EPSG:4326"),
    max_width_fill_m=float(snakemake.params.max_width_fill_m),
)
plot_domain_map(
    bbox_poly=domain_poly,
    river_gdf=river_gdf,
    stations=stations,
    crossings=crossings,
    has_glofas=has_glofas,
    land_polygons_path=land,
    output_path=out.plot_map,
)

# EVA of the active crossing with the largest discharge at the diagnostic RP:
# ranked by discharge, not SWORD width, which can diverge from the flow at the
# matched GloFAS cell (distributary vs. main stem).
with xr.open_dataset(inp.glofas_clip) as clip:
    clip = clip.load()
active = np.flatnonzero(has_glofas)
if active.size:
    i = int(active[np.argmax(crossings["q_preview_m3s"].to_numpy(float)[active])])
    time_dim = "valid_time" if "valid_time" in clip.dims else "time"
    with xr.open_dataset(inp.river_forcing, decode_times=False) as rf:
        daily = rf["discharge_daily"].values[i].astype(float)
    plot_cell_diagnostics(
        times=clip[time_dim].values,
        values=daily,
        eva_cfg=dict(snakemake.params.eva),
        output_path=out.plot_eva_diagnostics,
        label=f"crossing{i}_cell({crossings['glofas_cell_lat'].iloc[i]:.3f},"
        f"{crossings['glofas_cell_lon'].iloc[i]:.3f})",
    )
else:
    _placeholder(
        "No active GloFAS crossings\n(EVA could not be fitted)",
        out.plot_eva_diagnostics,
    )

bc_dir = Path(out.plot_bias_correction)
bc_dir.mkdir(parents=True, exist_ok=True)
cells = read_bias_diagnostics(inp.bias_correction)
if cells:
    lat_dim, lon_dim = latlon_dims(clip)
    time_dim = "valid_time" if "valid_time" in clip.dims else "time"
    for cell in cells:
        plot_bias_correction(
            glofas_times=clip[time_dim].values,
            glofas_values=clip["dis24"]
            .isel({lat_dim: cell["i_lat"], lon_dim: cell["i_lon"]})
            .values.astype(float),
            diagnostics=cell["diagnostics"],
            output_path=bc_dir / f"{cell['label']}.png",
            label=cell["label"],
        )
else:
    _placeholder(
        "No GRDC bias correction applied for any crossing\n"
        "(no GRDC station within search radius / insufficient overlap)",
        bc_dir / "none.png",
    )

# 08 river network
rivers_clean = gpd.read_file(inp.clean_river_network).to_crs("EPSG:4326")
plot_cleaned_network(
    rivers_orig=gpd.read_file(inp.spec_river_network).to_crs("EPSG:4326"),
    rivers_clean=rivers_clean,
    bbox_poly=domain_poly,
    land_polygons_path=land,
    output_path=out.plot_clean_network,
)
plot_clean_network_discharge(
    rivers_wgs=rivers_clean,
    bbox_poly=domain_poly,
    land_polygons_path=land,
    output_path=out.plot_discharge_network,
)

# 10 river water surface
plot_water_surface_profiles(
    pd.read_csv(inp.surface_profile_points, dtype={"seed": str}),
    pd.read_csv(inp.surface_profile_paths, dtype={"seed": str}),
    out.plot_water_surface,
    f"Basin {basin_id}: river water surface (rule 10) along each seed's widest arm",
)
log.info("Done")
