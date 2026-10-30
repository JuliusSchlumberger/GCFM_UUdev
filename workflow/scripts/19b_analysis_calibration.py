"""Draw the depth and crest calibration diagnostic figures from rule 11's outputs; run by rule analysis_calibration.

River depth and forward check along the network, crest minus the last round's
water level, dike position with and without ocean-linked wetlands, and per
SFINCS round the water level at one centreline cell per reach and the flood
animation. Reads only files rule 11 wrote, so drawing them never reruns it.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from src.domain import load_domain
from src.log import setup_logging
from src.plots import (
    animate_flood_progression,
    plot_crest_gap_map,
    plot_river_depth_width,
    plot_water_level_timeseries,
    plot_wetland_dike_positions,
)
from src.postprocessing import compute_flood_progression
from src.raster import read_float_raster, read_on_grid
from src.surge import CREST_PRECISION_M

log = setup_logging(snakemake.log[0])
inp, out, params = snakemake.input, snakemake.output, snakemake.params
basin_id = snakemake.wildcards.basin_id
_, _, domain_poly = load_domain(inp.spec_basins_meta, inp.domain_gpkg)
land = str(inp.land_mask_on_grid)
with open(inp.calibration_rounds) as f:
    rounds = json.load(f)
run_hr, last_round = float(rounds["run_hr"]), int(rounds["n_rounds"])

# River depth at the design discharge (colour; dashed where the slope floor
# sets it) and the width used (line width), then the forward check.
segments = gpd.read_file(inp.depth_segments)
region = gpd.read_file(inp.model_region).to_crs(segments.crs).geometry.union_all()
q = params.design_discharge
plot_river_depth_width(
    segments,
    land_mask_path=land,
    domain_outline=region,
    output_path=str(out.plot_river_network_depth),
    depth_label=f"Water depth at {q} discharge (m)",
    title=f"Basin {basin_id}: river depth from the water-surface slope (rule 10), GloFAS {q} Q, "
    f"n = {params.channel_manning_n}",
)
plot_river_depth_width(
    segments,
    land_mask_path=land,
    domain_outline=region,
    output_path=str(out.plot_river_network_forward_check),
    depth_label="Steady GVF level on the bed minus the rule-10 surface (m)",
    title=(
        f"Basin {basin_id}: river bed forward check, GloFAS {q} Q, n = {params.channel_manning_n} "
        "(positive: the bed carries Q above the smoothed FathomDEM level)"
    ),
    value_col="forward_minus_surface_m",
    value_range=(-3.0, 3.0),
    cmap_name="RdBu_r",
)

# Crest minus the last round's period-max water level.
crest, transform, crs = read_float_raster(inp.crest_on_grid)
if not np.isfinite(crest).any():
    log.warning("No weir crest cells -- writing an empty crest gap map sentinel")
    Path(out.plot_crest_gap_map).touch()
else:
    points = gpd.read_file(inp.discharge_points).to_crs(crs)
    plot_crest_gap_map(
        crest,
        transform,
        str(getattr(inp, f"map_r{last_round}")),
        gpd.read_file(inp.coastal_protection_weir).to_crs(crs),
        gpd.read_file(inp.clean_river_network).to_crs(crs),
        points.geometry.x.to_numpy(),
        points.geometry.y.to_numpy(),
        str(out.plot_crest_gap_map),
        basin_id=basin_id,
        run_label=f"calib_r{last_round} -- {rounds['titles'][last_round]}",
        min_overtop_m=CREST_PRECISION_M,
    )

# Dike position with unprotected_ocean_wetlands off vs on.
wetland, grid_transform, _ = read_float_raster(inp.ocean_wetland_mask)
plot_wetland_dike_positions(
    read_on_grid(inp.landuse_on_grid, wetland.shape),
    grid_transform,
    gpd.read_file(inp.dike_wetlands_off),
    gpd.read_file(inp.dike_wetlands_on),
    wetland == 1,
    str(out.plot_dike_positions),
    active_on=bool(params.unprotected_ocean_wetlands),
    basin_id=basin_id,
)

# Per SFINCS round: water level and flood animation.
for r in range(1, last_round + 1):
    run = f"calib_r{r} -- {rounds['titles'][r]}"
    wl = pd.read_csv(getattr(inp, f"water_level_r{r}"))
    reaches = [c for c in wl.columns if c != "time_s"]
    plot_water_level_timeseries(
        wl["time_s"].to_numpy(float) / 86400.0,
        wl[reaches].to_numpy(float),
        str(getattr(out, f"plot_water_level_r{r}")),
        day_markers=[(run_hr / 24.0, f"{run_hr:.0f} h (run end)")],
        station_labels=reaches,
        basin_id=f"Basin {basin_id}",
        run_label=run,
    )
    round_root = Path(getattr(inp, f"map_r{r}")).parent
    da_h = compute_flood_progression(round_root, inp.ocean_mask_on_grid)
    if da_h is None:
        raise RuntimeError(f"[{run}] no zs or zb in the SFINCS output of {round_root}")
    animate_flood_progression(
        da_h,
        domain_poly,
        land,
        str(inp.clean_river_network),
        str(getattr(out, f"animation_r{r}")),
        basin_id=basin_id,
        run_label=run,
        fps=int(params.animation_fps),
    )
log.info("Done")
