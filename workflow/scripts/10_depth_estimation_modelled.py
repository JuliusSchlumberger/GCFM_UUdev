"""
10_depth_estimation_modelled.py -- SFINCS-based river depth and riverbank
weir crest calibration (river_processing.depth_method == "modelled"), an
alternative to the empirical hydraulic-geometry depth estimate (rule
empirical_depth_estimation, 10_depth_estimation_empirical.py).

Three fixed-duration runs of one minimal, disposable SFINCS model (regular
grid only, but otherwise matching the production model's own subgrid setup
-- see the subgrid section below), all with the SAME weir geometry the
production model uses (src.protection_weir.build_coastal_protection_weir --
coast AND river banks merged into one water_like boundary, not a river-only
confinement traced by a separate function) and the same forcing: a steady
discharge equal to the basin's protection-level return period (bankfull as
a fallback) -- linearly RAMPED UP from each crossing's own
bankfull_discharge over river_processing.river_depth_modelling.
discharge_ramp_hours, then held constant for the rest of the run, rather
than forced as an instantaneous step from t=0 (the calibration discharge is
often several times bankfull, and stepping straight to it into a channel
that starts near-dry produces a startup shock wave that can persist/
oscillate for the entire run; mirrors production's own bankfull-lead-in-
then-ramp hydrograph, src.river_forcing.build_design_discharge_matrix/
sinusoidal_wave, just linear here since calibration only needs to reach and
hold the peak) -- plus a steady baseline_m coastal boundary (see below).

Round 0 -- confined, un-excavated. Every weir edge, coastal and riverbank
alike, sits at the artificially high weir_crest_m (e.g. 1000 m): even with
the steady baseline_m boundary, the confined river's water level can exceed
a low real coastal crest right at the coast/river transition near the
mouth, leaking onto the floodplain through the coastal side of the same
merged boundary -- so confinement must be uniform. At every centerline
cell (src.river_depth_calibration.compute_excavation_depth):

    rise   = period_max_water_level - DEM_at_cell
    rivdph = excavation_fraction * max(rise, 0)

then the mouth rules below.

Round 1 -- confined, excavated, with the coastal protection storm tide. The
channel is burned to round 0's rivdph, still behind the same 1000 m walls,
and the sea boundary now carries a storm tide at the coastal protection RP
(FLOPROS; production's own half-cosine wave shape, starting once the river
is at full discharge). With no overbank relief anywhere, this round's water
level is the highest the calibration discharge and the protection-level
storm tide reach, including the coast's own amplification of the boundary
level (shoaling, reflection off the dike) -- so it IS the crest that
contains them, with no iteration needed. Every production weir edge gets its
own crest from the water level on its own water side (the channel or ocean
cell it borders):

    crest(edge) = ceil_0.1( max(zsmax(water-side cell),
                                coastal_protection_crest_m) + freeboard_m )

(zsmax: SFINCS's own maximum over every computational timestep of the
run -- see _read_period_max_zs_field. ceil_0.1: rounded UP to the next
0.1 m, src.surge.ceil_water_level -- sfincs.weir stores crests at 0.1 m
with round-to-nearest, which would otherwise lower up to half the edges.)

One rule for riverbanks, the mouth, discharge-injection cells and the open
coast alike (a coastal edge's crest is the storm-tide level SFINCS reaches
at that stretch of dike, floored at the FLOPROS coastal crest). A never-
wetted water-side cell falls back to coastal_protection_crest_m, so no edge
is ever dropped.

Round 2 -- verification. Same excavated geometry and forcing (discharge +
storm tide), now with the real production weir. Changes nothing: it only
reports every edge whose own water side still exceeds its crest (in theory
none -- round 1 had no water above these crests -- so anything reported
comes from flow the finite crests redistribute) and shows any resulting
flooding in its max-inundation map.

Replaces an iterative scheme (removed 2026-09-10): round 0 used to split
the un-excavated rise between excavation and a crest, then up to
n_correction_iterations coupled rounds raised that crest by
min_crest_increment_per_round_m, snapped it back down and re-verified it,
with separate coastal and river-boundary probe corrections and a hardcoded
0.5 m calibration freeboard. It only needed to iterate because it guessed
the crest from the UN-excavated water level; measuring the crest after
excavation, still confined, gets it in one run.

period_max_water_level (excavation depth, per-cell diagnostics) is each
cell's own MAXIMUM simulated water level over the entire run
(src.river_depth_calibration.compute_period_max_zs, from the hourly zs
field each round writes to its own sfincs_map.nc), not a
windowed-flatness "converged"/stabilized value -- a persistent, non-damping
oscillation at a discharge-injection cell never satisfies a flatness/trend
tolerance no matter how long the run is extended. Sizing channel depth/
crest only needs the worst-case (peak) forced water level, which the period
maximum gives directly, oscillation or not, with no convergence check,
retry, or restart-extension logic at all.

For the excavation depth, each centerline cell's own water level is not
simply the single grid cell the centerline happens to intersect (a noisy,
non-representative sample -- most strikingly a sharp local spike right at
the seed reach's own discharge-injection cell) but the MAX across that
cell's own channel CROSS-SECTION: the contiguous channel_mask run through
the cell along its own grid row vs. its own grid column, whichever is
SHORTER (extended by one cell on each side to also reach the adjacent land
cell the weir itself sits on) -- see _read_cross_section_period_max_zs.
Deliberately a simple axis-aligned comparison, not a true reach-normal
cross-section. The weir crest instead reads SFINCS's own zsmax field
directly, one water-side cell per edge (_read_period_max_zs_field).

Mouth reach(es) (n_rch_dn == 0, topologically the network's own real
coastal outlet(s) -- NOT the same thing as a delta-outline outflow point,
see below) have their own last (most downstream, max along_m) cell
hard-forced to zero excavation (rivdph=0, bed=natural bathymetry) -- that
coastal endpoint's BED is a real physical constraint (the actual seabed),
not a calibration result. The mouth's CREST needs no special case: its
edges get their water level like every other edge.

Every OTHER cell -- in the mouth reach itself, and however far upstream is
needed -- keeps its own completely normal per-cell round-0 bed, UNLESS that
normal bed is shallower than the coastal endpoint, in which case it's
floored down to match it. Rationale: a real channel shouldn't get shallower
right before it meets the sea (the same discharge would then have to
accelerate through a narrower cross-section immediately before the
outlet) -- but this floor must stop being applied the first time, walking
upstream, that the normal per-cell bed is already at least as deep as the
coastal endpoint on its own; otherwise every unrelated, far upstream reach
with a genuinely shallower natural requirement would get needlessly forced
deep too. Implemented as a stateful walk starting at the coast (reach by
reach, via the network's own single-upstream-neighbour adjacency, since
SWORD guarantees a mouth has exactly one upstream neighbour) that stops
permanently -- for that entire mouth's own path -- the first time it finds
a cell that doesn't need clamping, or hits a reach with zero or more than
one upstream neighbour (headwater or confluence).

Separately, a non-seed, non-mouth, non-bifurcation reach that crosses the
delta polygon's own outline (identify_delta_outflow_points, rule
clean_river_network) is a genuine place flow exits the modelled network
WITHOUT reaching a real coastal mouth -- e.g. a distributary clipped by
the domain boundary in a complex, multi-channel delta. This gets a free
outflow boundary (mask=3) instead of any mouth-style bed treatment (there
is no real seabed there, just an arbitrary domain edge) -- see the mask
setup below. Without it, such a reach would be walled off like ordinary
land by the weir/domain-edge closure, so its confined water level -- and
therefore its crest -- would climb chasing water that in production simply
leaves the domain there.

The coastal boundary starts every round at baseline_m (mean sea level in
model coordinates, MDT-corrected, read from surge_forcing.nc -- the level
the sea starts at via zsini, as in production). Round 0 holds it there;
rounds 1 and 2 raise it as a storm tide at the coastal protection RP (see
round 1 above; the calm-sea boundary is kept if no coastal protection RP is
known). The river is a steady calibration discharge throughout.

Rounds 1 and 2 burn using each CELL's own rivdph directly as a
burn_river_channel anchor (rivbed = dem_at_cell - rivdph), rather than
routing a per-reach scalar through src.river_preburn.compute_river_bed_points'
DEM-following constant-offset convention -- every cell already carries its
own calibrated value, so burn_river_channel's existing per-reach interp1d
does the along-reach (and, via its neighbour-anchor-borrowing, cross-reach-
junction) smoothing directly from real calibration data, not DEM shape
alone.

See src.river_depth_calibration's module docstring. This model is
intermediate/disposable -- rule 13 builds the production model separately
from this rule's outputs: the burned river DEM and the traced weir (the
exact weir round 2 verified, imported as-is -- rule 13 never re-traces it).

Sibling alternative to rule empirical_depth_estimation -- exactly one
of the two ever runs (river_processing.depth_method), both writing the
same unified output filename (river_network_depth_estimated.gpkg).

Elevation: uses the SAME two conditioned-DEM layers rule 13's production
build does -- the shared SFINCS-grid-resolution raster (rule
enforce_river_monotonicity's second output, elevation_conditioned_sfincs_grid.tif)
as this model's own coarse sf.elevation.create() background, and the
native-resolution elevation_conditioned.tif for the subgrid table's fine
sub-cell detail (subgrid genuinely needs finer-than-grid-cell resolution;
the coarse layer would just duplicate one value across every sub-cell).
Using the SAME coarse raster rule 13 uses (rather than each independently
letting HydroMT resample from the native file) guarantees calibration and
production sit on the identical background elevation. Rounds 1 and 2
excavate this same background within the shared channel_mask corridor,
using burn_river_channel; the same excavation, burned once more at native
resolution, becomes the production river_burned_dem file rule 13 imports
directly, without re-burning it itself.

Inputs
------
elevation_conditioned              Native-resolution conditioned DEM (rule
                                    enforce_river_monotonicity, 09) --
                                    subgrid's own fine source, and the
                                    native resolution/CRS reference for the
                                    excavation's own native burn.
elevation_conditioned_sfincs_grid  SFINCS-grid-resolution conditioned DEM
                                    (rule enforce_river_monotonicity's
                                    second output) -- this model's own
                                    sf.elevation.create() base layer
                                    (round 0), and the excavation
                                    background for rounds 1 and 2.
surge_forcing                      surge_forcing.nc (rule 07) -- baseline_m
                                    (calm-sea boundary and start level),
                                    the COAST-RP tables for rounds 1-2's
                                    protection-level storm tide, and
                                    coastal_protection_crest_m, the
                                    production weir's crest floor.

Outputs
-------
river_network_depth_estimated  Same reach set, rivdph replaced with the
                          per-reach MEDIAN of its own cells' excavation
                          depth, plus a weir_crest_calibrated column (per-
                          reach median of max(round 1's cross-section water
                          level, coastal crest) + freeboard_m) -- a summary
                          for inspection only: production uses the traced
                          weir file, whose crests are per edge. The same
                          unified filename rule empirical_depth_estimation
                          also writes.
river_burned_dem, river_burned_dem_sfincs_grid
                          Round 0's excavation, burned at native and SFINCS-
                          grid resolution (rule 13 imports both directly).
coastal_protection_weir   The traced production weir, per-edge crests (see
                          round 1 above).
zsini_sea_cells           Real open-sea cells on this rule's own grid,
                          weir-aware (see its own section below).
plot_calibration          Diagnostic: reaches colored by calibrated depth
                          (per-reach median).
plot_water_level_timeseries  Water level over time at one representative
                          cell per reach (nearest that reach's own along_m
                          midpoint) (round 2).
plot_max_inundation       Max inundation depth map (round 2) -- land
                          flooding here means some edge was overtopped.
animation_flood_progress  Instantaneous depth animation (MP4, round 2).
plot_crest_gap_map        Every edge's crest minus round 2's own water level
                          at that edge's water-side cell (negative =
                          overtopped).
plot_dike_positions       Coastal dike position with unprotected_ocean_
                          wetlands off vs on, overlaid on the land-use
                          classes that decide it (the configured one is
                          simulated; the other is traced for the figure only).
calib_root/round{i}/calibration_state.csv  Not a declared Snakemake output
                          (written directly under calib_root, one file per
                          round) -- the per-cell state (reach_id, row, col,
                          along_m, x, y, dem, rivdph, weir_crest, zs) of
                          each round: rivdph is the excavation that round
                          simulated (0 in round 0), weir_crest is NaN in
                          round 0 (confined) and the per-cell summary crest
                          above in rounds 1 and 2. Exists so diagnostics can
                          read ground truth directly instead of re-deriving
                          it from saved SFINCS output.
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import yaml
from affine import Affine
from hydromt_sfincs import SfincsModel
from scipy.ndimage import label as _ndimage_label
from scipy.spatial import cKDTree

from src.protection_weir import LANDUSE_SEA, GridArrays, build_coastal_protection_weir, ocean_linked_wetland_mask
from src.domain import load_domain
from src.geometry import snap_points_into_region
from src.log import setup_logging
from src.plots import (
    animate_flood_progression,
    plot_calibration_round_profiles,
    plot_crest_gap_map,
    plot_max_inundation_map,
    plot_river_depth,
    plot_water_level_timeseries,
    plot_wetland_dike_positions,
)
from src.postprocessing import compute_flood_progression, compute_max_inundation
from src.raster import restrict_waterlevel_boundary_to_sea
from src.river_burn import build_centerline_cells_regular, build_channel_mask_regular, burn_river_channel, constrain_to_coarse_channel_mask, snap_points_to_centerline_cells
from src.river_depth_calibration import (
    build_calibration_seed_discharge,
    compute_excavation_depth,
    compute_period_max_zs,
    gather_calibration_round_profile,
)
from src.river_network import accumulate_discharge, build_downstream_adjacency, compute_hydraulic_depth, normalize_reach_id
from src.sfincs_run import run_sfincs_subprocess
from src.surge import ceil_water_level, read_baseline_m, sinusoidal_wave, storm_tide_at_rp_interpolated

log = setup_logging(snakemake.log[0])

# ── paths & params ────────────────────────────────────────────────────────────
elevation_path       = Path(snakemake.input.elevation_conditioned)
elevation_sfincs_grid_path = Path(snakemake.input.elevation_conditioned_sfincs_grid)
river_elevation_max_path = Path(snakemake.input.river_elevation_max)
river_network_path   = Path(snakemake.input.clean_river_network)
river_forcing_path   = Path(snakemake.input.river_forcing)
protection_levels_path = Path(snakemake.input.protection_levels)
grid_resolution_path = Path(snakemake.input.grid_resolution)
# Grid-aligned land mask (rule grid_align_landuse) -- plot background only.
land_mask_path       = Path(snakemake.input.land_mask_on_grid)
domain_gpkg_path     = Path(snakemake.input.domain_gpkg)
# landuse_on_grid.tif / roughness_on_grid.tif (rule grid_align_landuse, 09b)
# -- already resampled onto the SFINCS grid once, upstream; this rule reads
# them directly with zero further reprojection (see this rule's own zsini
# and roughness sections below).
landuse_on_grid_path = Path(snakemake.input.landuse_on_grid)
roughness_on_grid_path = Path(snakemake.input.roughness_on_grid)
# Native-resolution roughness.tif (rule get_roughness, 05c) -- used ONLY by
# the subgrid table below (sf.subgrid.create), which genuinely needs
# sub-cell detail for both elevation AND roughness together (same DEM
# exception the user asked to keep) -- never for the main regular-grid
# "manning" field, which uses roughness_on_grid_path (coarse) instead.
roughness_path        = Path(snakemake.input.roughness)
surge_forcing_path   = Path(snakemake.input.surge_forcing)
delta_outflow_points_path = Path(snakemake.input.delta_outflow_points)

basin_id    = str(snakemake.wildcards.basin_id)  # plot titles
calib_root  = Path(snakemake.params.calib_root)
sfincs_exe  = Path(snakemake.params.sfincs_exe)
timeout_s   = int(snakemake.params.timeout_s)
flow_accumulation_iterations = int(snakemake.params.flow_accumulation_iterations)
hg_c = float(snakemake.params.hg_c)
hg_f = float(snakemake.params.hg_f)

calibration_days   = float(snakemake.params.calibration_days)
discharge_ramp_hours = float(snakemake.params.discharge_ramp_hours)
weir_crest_m       = float(snakemake.params.weir_crest_m)
weir_par1          = float(snakemake.params.weir_par1)
excavation_fraction = float(snakemake.params.excavation_fraction)
# The ONLY margin on the production weir (added on top of round 1's water
# level, see this module's own docstring) -- there is no separate internal
# calibration freeboard.
weir_freeboard_m = float(snakemake.params.weir_freeboard_m)
# True: wetland/lagoon patches linked to the open sea stay on the seaward
# side of the coastal dike in all three rounds (see
# src.protection_weir.build_coastal_protection_weir).
unprotected_ocean_wetlands = bool(snakemake.params.unprotected_ocean_wetlands)
min_component_cells = int(snakemake.params.min_component_cells)
include_subgrid    = bool(snakemake.params.include_subgrid)
nr_subgrid_pixels  = int(snakemake.params.nr_subgrid_pixels)
nr_levels          = int(snakemake.params.nr_levels)
nrmax              = int(snakemake.params.nrmax)
animation_fps      = int(snakemake.params.animation_fps)
active_mask_enabled = snakemake.params.active_mask_enabled
active_mask_elevation_buffer_m = float(snakemake.params.active_mask_elevation_buffer_m)
outflow_buffer_m = float(snakemake.params.outflow_buffer_m)
waterlevel_buffer_m = float(snakemake.params.waterlevel_buffer_m)

calib_root.mkdir(parents=True, exist_ok=True)

# ── river network + calibration discharge ────────────────────────────────────
rivers = gpd.read_file(river_network_path)
# Power-law comparison column (rivdph_powerlaw) -- computed inline:
# empirical_depth_estimation and modelled_depth_estimation are sibling
# alternatives off the same river_network_clean.gpkg, not a sequential
# chain, so this rule cannot read it from empirical_depth_estimation's output.
rivers["rivdph_powerlaw"] = compute_hydraulic_depth(
    rivers["bankfull_discharge_acc"].values, c=hg_c, f=hg_f,
)

protection_rp_yr = None
with open(protection_levels_path) as f:
    protection = json.load(f)
rp = protection.get("riverine_rp_yr")
if rp is not None and np.isfinite(float(rp)):
    protection_rp_yr = float(rp)
# Coastal protection RP (FLOPROS) -- the storm tide rounds 1 and 2 are forced
# with (see the coastal forcing section before round 1). None: those rounds
# keep round 0's steady calm-sea boundary.
coastal_rp_yr = protection.get("coastal_rp_yr")
coastal_rp_yr = float(coastal_rp_yr) if coastal_rp_yr is not None and np.isfinite(float(coastal_rp_yr)) else None

with xr.open_dataset(river_forcing_path, decode_times=False) as river_ds:
    seed_q = build_calibration_seed_discharge(river_ds, protection_rp_yr)

adjacency = build_downstream_adjacency(rivers)
q_calib = accumulate_discharge(
    rivers, seed_q, adjacency,
    n_iterations=flow_accumulation_iterations,
)
rivers["q_calib"] = q_calib
log.info(
    f"Calibration discharge: {int((q_calib > 0).sum())}/{len(rivers)} reach(es) "
    f"with Q > 0, max Q = {q_calib.max():.1f} m3/s"
)

# Real, steady coastal baseline (mean sea level + SLR/MDT correction) -- same
# value 13_build_sfincs.py/14_run_spinup.py use for their own initial/steady
# conditions -- applied uniformly to every round, rather than isolating the
# river from coastal influence.
#
# coastal_protection_crest_m: the REAL production coastal crest (same
# surge_forcing.nc field rule 13 reads) -- the floor under every production
# weir edge's own water-level-derived crest (see this module's own
# docstring). Rounds 0 and 1 use the uniform weir_crest_m confinement
# instead.
with xr.open_dataset(surge_forcing_path, decode_times=False) as _surge_ds:
    # Rounded UP to the next 0.1 m (read_baseline_m) -- the same calm-sea
    # level production's own skeleton zsini and event lead-in use.
    baseline_m = read_baseline_m(_surge_ds)
    coastal_protection_crest_m = (
        float(_surge_ds["coastal_protection_crest_m"].values)
        if "coastal_protection_crest_m" in _surge_ds else 0.0
    )
log.info(
    f"Coastal boundary baseline read from surge_forcing.nc: {baseline_m:+.4f} m, "
    f"real coastal protection crest: {coastal_protection_crest_m:+.4f} m"
)

# Spatially-varying zsini (sea cells start at baseline_m, not bone-dry) is
# built further below, right after this model's own grid/mask are created
# (needs sf.grid.data["dep"] at THIS model's own exact grid shape -- see
# that section for why landuse_on_grid.tif can't be compared against
# elevation_conditioned_sfincs_grid.tif directly: the latter is
# deliberately EXTENDED beyond the exact SFINCS grid bounds for HydroMT's
# own edge-margin needs, rule enforce_river_monotonicity/09, so its shape
# does not match landuse_on_grid.tif's exact grid shape).

# Native resolution/CRS reference for the exported native-resolution burn
# (river_burned_dem) -- same role elevation_merged plays for rule 11b.
with rasterio.open(elevation_path) as _native_src:
    native_resolution_m = abs(_native_src.transform.a)

# ── minimal SFINCS model (regular grid) ──────────────────────────────────────
with open(grid_resolution_path) as f:
    resolution = float(json.load(f)["resolution"])

# sf.grid.create_from_region(region={"geom": ...}) needs a GeoDataFrame
# (matching 13_build_sfincs.py's delta_domain = gpd.read_file(domain_path)),
# not a raw shapely geometry -- hydromt's parse_region_geom doesn't accept
# a bare Polygon/MultiPolygon here. create_from_region is deterministic
# given the same domain polygon + resolution + "utm" CRS rule, so calling it
# here (as rule 13's production build also does, separately) reproduces the
# identical grid rule 08c (build_sfincs_grid) already built once.
delta_domain = gpd.read_file(domain_gpkg_path)

# Delta-outline outflow points (rule clean_river_network's own
# identify_delta_outflow_points): a non-seed, non-mouth, non-bifurcation
# reach that crosses the delta polygon's own outline -- a genuine place
# flow exits the modelled network without reaching a real coastal mouth
# (e.g. a distributary clipped by the domain boundary in a complex,
# multi-channel delta). Registered as a free outflow boundary below
# (mask=3), matching rule 13's own production build exactly -- without
# this, such a reach has no mouth treatment (it isn't one) AND no outflow
# either, so calibration would just wall it off and its confined water
# level -- and so its crest -- would climb chasing water that in reality/
# production simply leaves the domain there.
delta_outflow_gdf = gpd.read_file(delta_outflow_points_path)
delta_outflow_enabled = not delta_outflow_gdf.empty
log.info(f"Delta-outline outflow points: {len(delta_outflow_gdf)}")

# The excavated channel burned at subgrid resolution (rounds 1 and 2's own
# subgrid source) gets its catalog entry registered upfront -- the FILE
# doesn't exist yet at this point (written after round 0, before round 1's
# own subgrid.create() call reads it), but the catalog only needs the URI
# declared once, at model construction time, matching every other script in
# this codebase (none of them add catalog sources mid-run). The main "dep"
# grid's own excavation is patched directly into sf.grid.data["dep"]'s numpy
# array instead -- no catalog entry needed for that.
river_burned_subgrid_path = calib_root / "river_burned_subgrid.tif"
local_catalog_path = calib_root / "data_catalog_local.yml"
local_catalog = {
    "meta": {"root": str(elevation_path.parent)},
    "local_elevation_conditioned": {
        "data_type": "RasterDataset",
        "uri": str(elevation_path),
        "driver": "rasterio",
    },
    "local_elevation_conditioned_sfincs_grid": {
        "data_type": "RasterDataset",
        "uri": str(elevation_sfincs_grid_path),
        "driver": "rasterio",
    },
    "local_roughness": {
        "data_type": "RasterDataset",
        "uri": str(roughness_on_grid_path),
        "driver": "rasterio",
    },
    # Native resolution -- subgrid table only, see roughness_path's own
    # comment above.
    "local_roughness_native": {
        "data_type": "RasterDataset",
        "uri": str(roughness_path),
        "driver": "rasterio",
    },
    **({
        "local_delta_outflow_points": {
            "data_type": "GeoDataFrame",
            "uri": str(delta_outflow_points_path),
            "driver": "pyogrio",
        },
    } if delta_outflow_enabled else {}),
    "local_river_burned_subgrid": {
        "data_type": "RasterDataset",
        "uri": str(river_burned_subgrid_path),
        "driver": "rasterio",
    },
}
with open(local_catalog_path, "w") as fh:
    yaml.dump(local_catalog, fh, sort_keys=False)

sf = SfincsModel(
    data_libs=[str(local_catalog_path)],
    root=str(calib_root),
    mode="w+",
    write_gis=False,
)
sf.grid.create_from_region(region={"geom": delta_domain}, res=resolution, crs="utm", rotated=False)
log.info(f"Calibration grid created: {resolution} m, auto-UTM")

# tref/tstart/tstop set NOW (Phase 1, matching 13_build_sfincs.py) so the
# forcing sections below (discharge points, water level) build against the
# right time window -- discharge_points.create() internally slices its
# timeseries to the model's configured window and raises NoDataException
# if the window is still at its default (unset) value when it runs.
tref = datetime(2000, 1, 1)
tstop = tref + timedelta(days=calibration_days)
trstout_sec = int(calibration_days * 86400)
sf.config.update({
    "tref": tref, "tstart": tref, "tstop": tstop,
    "dthisout": 3600, "dtmapout": 3600,
    "dtmaxout": trstout_sec,
    "trstout": trstout_sec, "dtrstout": 0,
    "zsini": -9999.0,
})

# ── round-0 elevation (un-excavated), mask, roughness ────────────────────────
# Coarse SFINCS-grid-resolution background -- the SAME shared raster rule
# 13's production build uses (rule enforce_river_monotonicity's second
# output), so calibration and production sit on the identical coarse "dep"
# everywhere the fine channel burn doesn't apply.
sf.elevation.create(elevation_list=[{"elevation": "local_elevation_conditioned_sfincs_grid"}])
# Elevation ceiling on the active mask -- same criterion rule 13's
# production build uses (river_elevation_max.json + active_mask's own
# elevation_buffer_m), so calibration simulates the identical active-cell
# footprint production will actually build.
active_mask_kwargs = {}
if active_mask_enabled:
    with open(river_elevation_max_path) as f:
        river_elevation_max_m = float(json.load(f)["river_elevation_max_m"])
    clip_elevation_m = river_elevation_max_m + active_mask_elevation_buffer_m
    active_mask_kwargs = {"include_zmax": clip_elevation_m}
sf.mask.create_active(include_polygon=delta_domain, **active_mask_kwargs)
log.info(
    "Active mask created: cells within delta polygon"
    + (f" AND below {clip_elevation_m:.1f} m (river max {river_elevation_max_m:.1f} m + "
       f"{active_mask_elevation_buffer_m:.1f} m buffer)" if active_mask_kwargs else "")
)

# Water-level boundary on open sea only, by the model grid's OWN land use
# (landuse_on_grid != 200 -> back to a normal active cell) -- same as rule
# 13's production build; see restrict_waterlevel_boundary_to_sea.
# reset_bounds=False: create_active just built a fresh 0/1 mask, and hydromt's
# create_boundary with reset_bounds=True and no polygon/elevation filter only
# resets and returns -- it would set no boundary cells at all.
sf.mask.create_boundary(btype="waterlevel", reset_bounds=False)
with rasterio.open(landuse_on_grid_path) as _lu_src_bnd:
    _bnd_mask, _n_bnd_on_land = restrict_waterlevel_boundary_to_sea(
        sf.grid.data["mask"].values, _lu_src_bnd.read(1)
    )
if not (_bnd_mask == 2).any():
    raise ValueError(
        "No water-level boundary cell on open sea (landuse_on_grid == 200) at the "
        "active-domain edge -- check the active mask and landuse_on_grid.tif."
    )
sf.grid.data["mask"].values[:] = _bnd_mask
log.info(
    f"Waterlevel boundary set: {int((_bnd_mask == 2).sum())} edge cell(s) on open sea -> mask=2 "
    f"({_n_bnd_on_land} edge cell(s) on land use != 200 left as normal active cells)"
)

# Delta-outline outflow boundary (mask=3, no prescribed water level) --
# reset_bounds=False so this OVERRIDES specific cells the waterlevel step
# above may have already claimed (e.g. a domain-clipped distributary's own
# outline crossing sitting on the active-domain edge would otherwise be
# forced to the coastal baseline water level, which is physically wrong
# for an inland network truncation) rather than replacing the whole
# boundary. include_polygon_buffer turns each point into a small polygon
# so at least one grid cell is captured regardless of exactly where within
# a cell the point falls -- same mechanism and parameter rule 13's
# production build already uses.
if delta_outflow_enabled:
    sf.mask.create_boundary(
        btype="outflow", include_polygon="local_delta_outflow_points",
        include_polygon_buffer=outflow_buffer_m, reset_bounds=False,
    )
    log.info(
        f"Outflow boundary set: {len(delta_outflow_gdf)} delta-outline "
        f"crossing(s), {outflow_buffer_m:.0f} m buffer -> mask=3"
    )

# ── spatially-varying zsini: sea cells start AT baseline_m, not bone-dry ─────
# Built directly from landuse_on_grid.tif (rule grid_align_landuse, 09b),
# read here (not earlier) so the connected-component dry-out check below can
# compare against sf.grid.data["dep"] -- THIS model's own grid, guaranteed
# the same exact shape landuse_on_grid.tif was built on (both "auto-UTM"-fit
# from the same domain_gpkg + grid_resolution.json/sfincs_grid.json).
# elevation_conditioned_sfincs_grid.tif (rule enforce_river_monotonicity/09)
# is NOT usable for this comparison despite being on the same coarse grid's
# transform/phase -- it's deliberately EXTENDED beyond the exact SFINCS grid
# bounds for HydroMT's own edge-margin needs, so its shape doesn't match. A
# coastal/mouth cell starting bone-dry instead would take time to fill in
# from the boundary, producing a sharp transient spike well above its own
# true settled level before declining back down over the following hours.
# Since compute_period_max_zs takes the maximum over the ENTIRE run (no
# windowing), that transient alone would inflate the calibrated crest at
# every affected coastal cell by roughly the spike's own size.
with rasterio.open(landuse_on_grid_path) as _lu_src_zsini:
    _lu_on_grid_zsini = _lu_src_zsini.read(1)
if _lu_on_grid_zsini.shape != sf.grid.data["dep"].shape:
    raise ValueError(
        f"landuse_on_grid.tif shape {_lu_on_grid_zsini.shape} does not match this "
        f"rule's own grid {sf.grid.data['dep'].shape} -- expected pixel-identical grids."
    )
_zsini_arr = np.where(_lu_on_grid_zsini == LANDUSE_SEA, np.float32(baseline_m), np.nan)

# Connected-component dry-out of isolated sea cells (same rationale/
# mechanism as 13_build_sfincs_skeleton.py's identical fix): label connected
# components of sea cells, keep only the main open ocean, and set isolated
# wet cells (dep < baseline_m, so water depth > 0) to dep = dry start.
_dep_zsini = sf.grid.data["dep"].values.astype(np.float32)
_dep_nd_zsini = np.float32(sf.grid.data["dep"].rio.nodata or -9999.0)
_dep_valid_zsini = np.where(np.isclose(_dep_zsini, _dep_nd_zsini), np.float32(1e6), _dep_zsini)
_sea_cells_zsini = np.isclose(_zsini_arr, np.float32(baseline_m))
_labeled_zsini, _n_zsini = _ndimage_label(_sea_cells_zsini)
if _n_zsini > 0:
    _sizes_zsini = np.bincount(_labeled_zsini.ravel())
    _sizes_zsini[0] = 0
    _main_label_zsini = int(np.argmax(_sizes_zsini))
    _connected_zsini = _labeled_zsini == _main_label_zsini
else:
    _connected_zsini = np.zeros_like(_sea_cells_zsini)
_isolated_wet_zsini = _sea_cells_zsini & ~_connected_zsini & (_dep_valid_zsini < np.float32(baseline_m))
_n_fix_zsini = int(_isolated_wet_zsini.sum())
if _n_fix_zsini > 0:
    _zsini_arr = np.where(_isolated_wet_zsini, _dep_valid_zsini, _zsini_arr)
    log.info(f"zsini connectivity fix: {_n_fix_zsini} isolated wet sea-cell(s) set to dep")

# reproj_method="nearest" (not "average"): zsini is a near-binary field
# (baseline_m at sea, NaN/nodata on land) -- and source/destination grids
# are already identical here, so this is a no-op resample regardless (same
# in-memory-DataArray pattern as 13_build_sfincs_skeleton.py's identical
# construction). mask must exist first (create_active/create_boundary
# above), matching production's own ordering.
_da_zsini = xr.DataArray(
    _zsini_arr, dims=sf.grid.data["dep"].dims, coords=sf.grid.data["dep"].coords,
)
_da_zsini = _da_zsini.rio.write_crs(sf.grid.data["dep"].rio.crs)
_da_zsini = _da_zsini.rio.write_transform(sf.grid.data["dep"].rio.transform())
_da_zsini.raster.set_nodata(np.nan)
sf.initial_conditions.create(ini=_da_zsini, reproj_method="nearest")
log.info("Initial conditions: spatially-varying zsini (sea cells start at baseline_m, land dry)")

sf.roughness.create(roughness_list=[{"manning": "local_roughness"}])

# ── shared, round-invariant pieces: grid arrays, channel_mask, landuse ───────
# Same build_coastal_protection_weir rule 13 uses for the production model
# (coast and river channel merged into one water_like boundary, so the mouth
# never gets sealed shut) -- not a separate river-only function, so
# calibration and production build conceptually the same single weir set.
# GridArrays.from_regular copies dep/mask into plain numpy arrays at
# construction time, so it stays valid across every later round even after
# sf.elevation.create() is called again with excavated data -- valid_mask is
# active-mask-based, not elevation-based, so it doesn't need recomputing
# either. channel_mask is likewise a pure function of (rivers, width, grid
# shape/transform) -- used for both the weir corridor and to constrain the
# excavation (see rounds 1-2 below), and identical to what rule 11b/13
# independently compute for production.
rivers_utm = rivers.to_crs(sf.crs)
grid = GridArrays.from_regular(sf.grid.data["dep"], sf.grid.data["mask"], sf.crs)
channel_mask = build_channel_mask_regular(rivers_utm, "width", grid.shape, grid.transform)
# landuse_on_grid.tif (rule grid_align_landuse, 09b) -- already rasterized
# directly onto this rule's own grid (confirmed pixel-identical to
# sfincs_grid.json, both "auto-UTM"-fit from the same domain_gpkg +
# grid_resolution.json), read here with zero further reprojection, rather
# than this rule independently resampling native landuse.tif a second time
# (the previous GridArrays.sample_landuse call -- removed 2026-08-07, see
# 09b_grid_align_landuse.py's own module docstring for why every consumer
# now shares one canonical resample).
with rasterio.open(landuse_on_grid_path) as _lu_src:
    landuse_on_grid = _lu_src.read(1)
if landuse_on_grid.shape != grid.shape:
    raise ValueError(
        f"landuse_on_grid.tif shape {landuse_on_grid.shape} does not match this "
        f"rule's own grid {grid.shape} -- expected pixel-identical grids."
    )
ocean_mask_grid = landuse_on_grid == LANDUSE_SEA

# Coarse-grid ocean mask FILE (compute_max_inundation/compute_flood_
# progression need a path, not an in-memory array -- they call
# data_catalog.get_rasterdataset internally) -- used by every round's own
# diagnostics below instead of native-resolution sea_mask.tif, so every
# sea/land check in this rule shares the same single coarse-grid source (no
# protected_pocket_mask yet at this point -- the production weir isn't
# traced until after round 1 -- but round diagnostics never claimed
# weir-aware precision either; ocean_mask_grid alone is the same "is this
# genuinely open sea" classification the old native sea_mask.tif gave).
_round_ocean_mask_path = calib_root / "ocean_mask_on_grid.tif"
with rasterio.open(
    _round_ocean_mask_path, "w", driver="GTiff", dtype="float32", count=1,
    height=ocean_mask_grid.shape[0], width=ocean_mask_grid.shape[1],
    crs=grid.crs, transform=grid.transform, nodata=-9999.0,
) as _dst:
    _dst.write(np.where(ocean_mask_grid, np.float32(1.0), np.float32(-9999.0)), 1)

# region_wgs84/buf_deg: reused below by both the observation-point and
# discharge-point sections. hydromt's *_points.create() methods clip
# locations against the model's own region using an UNBUFFERED 'intersects'
# check and silently drop anything outside -- snap_points_into_region keeps
# every point by nudging boundary-adjacent ones inward (see its own
# docstring for why a plain buffered filter isn't enough).
region_wgs84 = sf.region.to_crs("EPSG:4326").geometry.union_all()
buf_deg = float(resolution) / 111_000.0

# ── centerline calibration cells (every grid cell each reach's own centerline
# passes through) -- replaces SFINCS "observation points" entirely ──────────
# The real water level varies continuously along a reach -- a single
# sparse observation point can completely miss a sharp local spike right
# at the seed reach's own discharge-injection cell -- so
# calibration tracks depth per CELL instead, sourced directly from
# each round's own sfincs_map.nc (_read_map_zs_at_cells below), not
# sf.observation_points/sfincs_his.nc at all.
cell_gdf = build_centerline_cells_regular(rivers_utm, grid.shape, grid.transform)
if cell_gdf.empty:
    raise RuntimeError("No centerline calibration cells could be identified -- empty river network?")
dem_at_cell = grid.bed_elevation[cell_gdf["row"].to_numpy(), cell_gdf["col"].to_numpy()]
log.info(
    f"Centerline calibration cells: {len(cell_gdf)} across "
    f"{cell_gdf['reach_id'].nunique()} reach(es)"
)

# ── per-cell CROSS-SECTION zs sampling (replaces a single grid cell's own
# reading) ────────────────────────────────────────────────────────────────
# A single grid cell's own water level can be a noisy, non-representative
# sample of "the water level at this point along the river" -- the true
# channel cross-section at a given along-reach position is usually several
# cells wide, and the centerline happens to intersect just one of them.
# Every centerline cell's own "zs" (round 0's excavation depth, and the
# per-cell crest summary/diagnostics of rounds 1-2 -- see
# _read_cross_section_period_max_zs below; the production weir itself reads
# the full field per edge instead) is the MAX water level across the FULL
# cross-section of channel_mask cells
# at that point, found by comparing the contiguous channel_mask run through
# the cell along its own grid ROW vs. its own grid COLUMN, and taking
# whichever run is SHORTER -- the shorter run is the cross-channel
# direction, since a channel is narrow across and long along at any single
# row/column slice (deliberately NOT the union of both: for a reach running
# straight along one grid axis, the OTHER axis's own run is the reach's
# entire length, so unioning would give every cell along that whole reach
# the same single global maximum -- confirmed degenerate, not just a
# theoretical concern). Each run is also extended by one cell beyond
# channel_mask's own True region on either side, to reach the adjacent
# LAND cell where the weir segment itself is actually drawn (the weir sits
# at the land/water_like transition, one cell outside channel_mask's own
# True region -- see _seaward_edges_regular_with_values). Deliberately a
# simple axis-aligned comparison, not a true reach-normal cross-section --
# exact only to the extent the channel isn't running near-diagonally at
# that specific point, the same caveat any grid-aligned raster analysis
# carries.
def _channel_run_through(mask_1d: np.ndarray, idx: int) -> np.ndarray:
    """Contiguous True run in a 1D boolean slice of channel_mask containing
    position idx, extended by one cell on each side beyond the run's own
    boundary (to also reach the adjacent land cell the weir itself sits
    on) -- idx itself is always included even if mask_1d[idx] is somehow
    False (shouldn't happen for a real centerline cell, but never silently
    drop the cell's own position)."""
    if not mask_1d[idx]:
        left = right = idx
    else:
        left = idx
        while left > 0 and mask_1d[left - 1]:
            left -= 1
        right = idx
        while right < len(mask_1d) - 1 and mask_1d[right + 1]:
            right += 1
    left = max(0, left - 1)
    right = min(len(mask_1d) - 1, right + 1)
    return np.arange(left, right + 1)


_cell_rows = cell_gdf["row"].to_numpy()
_cell_cols = cell_gdf["col"].to_numpy()
cell_cross_section_rowcol: list[np.ndarray] = []  # one (n_i, 2) row/col array per cell_gdf row
for _r, _c in zip(_cell_rows, _cell_cols):
    _row_run_cols = _channel_run_through(channel_mask[_r, :], _c)
    _col_run_rows = _channel_run_through(channel_mask[:, _c], _r)
    if len(_row_run_cols) <= len(_col_run_rows):
        _rc = np.column_stack([np.full(len(_row_run_cols), _r), _row_run_cols])
    else:
        _rc = np.column_stack([_col_run_rows, np.full(len(_col_run_rows), _c)])
    cell_cross_section_rowcol.append(_rc)

# Flattened, DEDUPLICATED set of every grid cell any cross-section actually
# needs -- sfincs_map.nc is resolved/read once for this whole set (see
# _read_cross_section_period_max_zs below), not once per cell_gdf row, even
# though a single grid cell can appear in more than one cell_gdf row's own
# cross-section (e.g. two adjacent centerline cells sharing part of the
# same cross-channel run).
_all_cross_section_rc = np.concatenate(cell_cross_section_rowcol, axis=0)
_unique_rc, _cross_section_inverse = np.unique(_all_cross_section_rc, axis=0, return_inverse=True)
cross_section_unique_rows = _unique_rc[:, 0]
cross_section_unique_cols = _unique_rc[:, 1]
cross_section_unique_x, cross_section_unique_y = rasterio.transform.xy(
    grid.transform, cross_section_unique_rows, cross_section_unique_cols
)
cross_section_unique_x = np.asarray(cross_section_unique_x)
cross_section_unique_y = np.asarray(cross_section_unique_y)
# Split _cross_section_inverse back into per-cell_gdf-row groups (same
# lengths/order as cell_cross_section_rowcol) -- cross_section_group_idx[i]
# gives the positional indices into cross_section_unique_rows/cols (and
# therefore into per-unique-cell period-max arrays) that make up cell i's
# own cross-section.
_group_sizes = [len(a) for a in cell_cross_section_rowcol]
_group_bounds = np.cumsum([0] + _group_sizes)
cross_section_group_idx = [
    _cross_section_inverse[_group_bounds[i]:_group_bounds[i + 1]] for i in range(len(cell_gdf))
]
log.info(
    f"Cross-section zs sampling: {len(cell_gdf)} centerline cell(s) map onto "
    f"{len(cross_section_unique_rows)} unique grid cell(s) total "
    f"(median cross-section width: {int(np.median(_group_sizes))} cell(s))"
)

# (n_idx, m_idx): sfincs_map.nc's own internal cell indices matching
# cell_gdf's (row, col) / the cross-section cells / every grid cell --
# resolved lazily, once, from the first sfincs_map.nc read (grid geometry is
# round-invariant, so it's reused unchanged for every later round).
map_cell_idx = None
cross_section_map_cell_idx = None
field_map_cell_idx = None

# One representative cell per reach (nearest that reach's own along_m
# midpoint) -- used only by the water-level-timeseries diagnostic plot,
# which is unreadable/slow at full per-cell density. Positional (cell_gdf
# has a plain 0..n-1 RangeIndex from build_centerline_cells_regular), so
# these double directly as column indices into final_zs (n_time, n_cells).
_reach_along_mid = cell_gdf.groupby("reach_id")["along_m"].transform(lambda s: s.max() / 2.0)
_dist_to_mid = (cell_gdf["along_m"] - _reach_along_mid).abs()
representative_cell_pos = _dist_to_mid.groupby(cell_gdf["reach_id"]).idxmin().to_numpy()
representative_reach_ids = cell_gdf.loc[representative_cell_pos, "reach_id"].to_numpy()

# ── discharge points (constant calibration discharge per crossing) ──────────
n_steps = max(int(calibration_days * 24), 2)
calib_times = pd.date_range(tref, tstop, periods=n_steps)

reach_q = dict(zip(rivers["reach_id"].astype(str), q_calib))
with xr.open_dataset(river_forcing_path, decode_times=False) as river_ds:
    active = river_ds["has_glofas"].values.astype(bool)
    inside_ids = river_ds["inside_reach_id"].values[active] if "inside_reach_id" in river_ds else []
    cross_lons = river_ds["longitude"].values[active]
    cross_lats = river_ds["latitude"].values[active]
    # Per-crossing bankfull discharge -- the ramp's own start value (see
    # below), read at the SAME active crossings, same indexing/order as
    # cross_lons/cross_lats/inside_ids.
    cross_bankfull_q = river_ds["bankfull_discharge"].values[active]

crossing_q = [reach_q.get(str(rid), np.nan) for rid in inside_ids]
crossings_gdf = gpd.GeoDataFrame(
    {"index": range(len(crossing_q))},
    geometry=gpd.points_from_xy(cross_lons, cross_lats),
    crs="EPSG:4326",
)
valid = np.isfinite(crossing_q)
crossings_gdf = crossings_gdf[valid].reset_index(drop=True)
crossing_q = np.asarray(crossing_q)[valid]
crossing_bankfull_q = np.asarray(cross_bankfull_q)[valid]
inside_ids_valid = np.asarray(inside_ids)[valid] if len(inside_ids) else [None] * len(crossing_q)

if crossings_gdf.empty:
    raise RuntimeError("No discharge crossings resolved to a calibration discharge -- cannot calibrate")

# Snap each crossing onto the grid cell its OWN reach's centerline actually
# passes through (cell_gdf), not just wherever its raw domain-entry point
# happens to rasterize to -- an unsnapped point can resolve to a
# neighbouring floodplain cell with no real channel conveyance, which is
# exactly what produced basin 4267691's round-0 369 m water-level pileup.
# reach_ids restricts the nearest-cell search to the point's own reach
# first (matters at confluences/bifurcations), falling back to the nearest
# cell across all reaches when inside_reach_id is missing/unresolved.
crossings_utm = snap_points_to_centerline_cells(
    crossings_gdf.to_crs(sf.crs), cell_gdf, reach_ids=inside_ids_valid, resolution_m=resolution,
)
crossings_gdf = crossings_utm.to_crs("EPSG:4326")

crossings_filt, cross_keep_mask = snap_points_into_region(crossings_gdf, region_wgs84, buf_deg)
crossing_q = crossing_q[cross_keep_mask]
crossing_bankfull_q = crossing_bankfull_q[cross_keep_mask]
if crossings_filt.empty:
    raise RuntimeError("All discharge crossings fall outside the active calibration region")

# Linear ramp from each crossing's own bankfull_discharge up to the full
# calibration discharge over discharge_ramp_hours (from t=0), held flat
# afterward -- NOT a flat step function from t=0. Forcing the full
# calibration discharge (often several times bankfull) as an instantaneous
# step into a channel that starts near-dry produces a startup shock wave
# that can persist/oscillate for the entire run instead of damping out
# (confirmed on basin 2433835: the seed's own first cell showed multi-metre
# swings hour-to-hour under CONSTANT forcing with no ramp). Mirrors
# production's own bankfull-lead-in-then-ramp hydrograph shape
# (src.river_forcing.build_design_discharge_matrix/sinusoidal_wave), linear
# instead of sinusoidal since calibration only needs to reach and hold the
# peak, not simulate a full event recession. A crossing with NaN bankfull
# (shouldn't happen -- same has_glofas-active set river_forcing.nc already
# guarantees bankfull_discharge for) falls back to ramping from 0 rather
# than dropping the ramp entirely.
_ramp_hours = np.arange(n_steps) * (calibration_days * 24.0) / (n_steps - 1)
_ramp_frac = np.clip(_ramp_hours / max(discharge_ramp_hours, 1e-6), 0.0, 1.0)[:, None]
_bankfull_start = np.where(np.isfinite(crossing_bankfull_q), crossing_bankfull_q, 0.0)
dis_values = _bankfull_start[None, :] + (crossing_q[None, :] - _bankfull_start[None, :]) * _ramp_frac
dis_df = pd.DataFrame(dis_values, index=calib_times, columns=range(len(crossings_filt)))
sf.discharge_points.create(timeseries=dis_df, locations=crossings_filt)
log.info(
    f"Discharge forcing: {len(crossings_filt)} source point(s), ramped from bankfull to the full "
    f"calibration discharge over {discharge_ramp_hours:.1f} h, held constant for the remaining "
    f"{max(calibration_days * 24.0 - discharge_ramp_hours, 0.0):.1f} h"
)

# Discharge points in the model CRS -- overlaid on round 2's own crest gap
# map (the same snapped locations discharge is actually injected at).
crossings_utm = crossings_filt.to_crs(grid.crs)

# ── coastal water-level boundary (steady baseline_m) ─────────────────────────
sf.water_level.create_boundary_points_from_mask(bnd_dist=5000.0)
sf.water_level.create_timeseries(shape="constant", offset=baseline_m)
log.info(f"Water-level forcing: flat constant boundary at {baseline_m:+.4f} m (baseline_m)")


def _resolve_map_cell_idx(map_nc_path: Path, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (n_idx, m_idx) into sfincs_map.nc's own zs(time, n, m) matching each
    (x, y) in cell_gdf, via nearest-neighbour match against the file's own
    x(n,m)/y(n,m) cell-center coordinates -- NOT assumed to be n=row,
    m=col directly (SFINCS's own internal index order/orientation relative
    to grid.transform is not guaranteed), so resolved from real coordinates
    instead. The grid is round-invariant, so this only needs to run once
    (see map_cell_idx's own lazy-cache in _run_calibration_round).
    """
    with xr.open_dataset(map_nc_path) as ds:
        grid_x = ds["x"].values.ravel()
        grid_y = ds["y"].values.ravel()
        n_dim, m_dim = ds["x"].shape
    valid = np.isfinite(grid_x) & np.isfinite(grid_y)
    tree = cKDTree(np.column_stack([grid_x[valid], grid_y[valid]]))
    _dist, flat_idx = tree.query(np.column_stack([x, y]))
    orig_flat_idx = np.where(valid)[0][flat_idx]
    n_idx, m_idx = np.unravel_index(orig_flat_idx, (n_dim, m_dim))
    return n_idx, m_idx


def _read_map_zs_at_cells(run_dir: Path, n_idx: np.ndarray, m_idx: np.ndarray):
    # Explicit `with` (not a bare open_dataset()) -- this file gets
    # overwritten by a LATER SFINCS run within the same process (the next
    # round), and xarray's lazy/cached file manager doesn't otherwise
    # release the handle promptly, which fails that later write on Windows
    # ("Permission denied"). Vectorized .isel with dims="cell" pairing
    # (not the outer product a plain .isel(n=n_idx, m=m_idx) would give)
    # pulls out exactly the (n_time, n_cells) slice needed, without ever
    # materializing the full (time, n, m) field in memory.
    with xr.open_dataset(run_dir / "sfincs_map.nc") as ds:
        zs = ds["zs"].isel(
            n=xr.DataArray(n_idx, dims="cell"), m=xr.DataArray(m_idx, dims="cell"),
        ).values
        times_s = (ds["time"].values - ds["time"].values[0]) / np.timedelta64(1, "s")
    return zs, times_s


def _read_zs_and_resolve(round_root: Path):
    """_read_map_zs_at_cells, resolving+caching map_cell_idx on first use
    (round 0's own first sfincs_map.nc) -- see _resolve_map_cell_idx."""
    global map_cell_idx
    if map_cell_idx is None:
        map_cell_idx = _resolve_map_cell_idx(
            round_root / "sfincs_map.nc", cell_gdf["x"].to_numpy(), cell_gdf["y"].to_numpy(),
        )
        log.info(f"Resolved {len(map_cell_idx[0])} centerline cell(s) against sfincs_map.nc's own grid indices")
    return _read_map_zs_at_cells(round_root, *map_cell_idx)


def _read_cross_section_period_max_zs(round_root: Path) -> np.ndarray:
    """Per-centerline-cell period-max water level, one value per cell_gdf
    row, taken as the MAX across that cell's own channel cross-section
    (cross_section_group_idx, precomputed above from channel_mask's row/
    column runs) rather than the single grid cell the centerline happens
    to intersect -- see the precomputation block above cell_gdf's own
    global-cache-var init for the full rationale. This drives round 0's
    excavation depth and the per-cell crest summary/diagnostics (the
    production weir reads _read_period_max_zs_field per edge instead);
    _read_zs_and_resolve (single intersected cell) is kept alongside this, only for
    the water-level-timeseries diagnostic plot's own representative-cell
    indexing, which is a visualization concern, not a calibration one."""
    global cross_section_map_cell_idx
    if cross_section_map_cell_idx is None:
        cross_section_map_cell_idx = _resolve_map_cell_idx(
            round_root / "sfincs_map.nc", cross_section_unique_x, cross_section_unique_y,
        )
        log.info(
            f"Resolved {len(cross_section_map_cell_idx[0])} unique cross-section grid cell(s) "
            f"against sfincs_map.nc's own grid indices"
        )
    zs, _times_s = _read_map_zs_at_cells(round_root, *cross_section_map_cell_idx)
    per_unique_cell_period_max = compute_period_max_zs(zs)
    return np.array(
        [np.nanmax(per_unique_cell_period_max[idx]) for idx in cross_section_group_idx],
        dtype=np.float32,
    )


def _read_period_max_zs_field(round_root: Path) -> np.ndarray:
    """Maximum water level over the whole run at EVERY cell of this rule's
    own grid (grid.shape, row/col order as grid.transform) -- NaN where
    never wetted or inactive. The production weir reads each edge's own
    water-side cell from this (see build_coastal_protection_weir's
    water_side_crest_on_grid), and round 2 checks against it.

    SFINCS's own zsmax (max over EVERY computational timestep; dtmaxout =
    the whole run, so a single frame), not the max over the hourly zs
    output compute_period_max_zs uses for the excavation depth: a weir
    overtops on any timestep, not just at output times -- on basin
    2433835's old round 0 the hourly max sat up to ~0.1 m below zsmax at 1%
    of wet cells. sfincs_map.nc's own (n, m) order is resolved from real
    coordinates (see _resolve_map_cell_idx), once, and cached."""
    global field_map_cell_idx
    map_nc_path = round_root / "sfincs_map.nc"
    if field_map_cell_idx is None:
        _rows, _cols = np.indices(grid.shape)
        _xs, _ys = grid.transform * (_cols.ravel() + 0.5, _rows.ravel() + 0.5)
        field_map_cell_idx = _resolve_map_cell_idx(map_nc_path, np.asarray(_xs), np.asarray(_ys))
        log.info(f"Resolved {grid.shape[0] * grid.shape[1]:,} grid cell(s) against sfincs_map.nc's own grid indices")
    with xr.open_dataset(map_nc_path) as ds:
        zsmax = ds["zsmax"]
        zsmax = zsmax.max(dim=[d for d in zsmax.dims if d not in ("n", "m")]).values
    n_idx, m_idx = field_map_cell_idx
    return zsmax[n_idx, m_idx].reshape(grid.shape).astype(np.float32)


def _run_calibration_round(round_label: str, round_root: Path):
    """Write and run whatever elevation/weir the caller just set up on the
    (shared) `sf` object, into round_root (the caller has already pointed
    sf.root at this via sf.root.set()) -- a single fixed-duration run, no
    restart/extension. Each round starts cold (no restart chained across
    rounds -- the geometry changed between rounds, so resuming a previous
    round's dynamic state would be physically invalid).

    No convergence/stabilization check: the per-cell water level used for
    calibration is simply each cell's own maximum over the full run
    (compute_period_max_zs), not a windowed-flatness "stabilized" value --
    see that function's own docstring for why (a persistent, non-damping
    oscillation at a discharge-injection cell never satisfies a flatness
    tolerance no matter how long the run is extended: the peak is
    exactly what calibration needs, converged or not)."""
    sf.config.set("rstfile", None)
    sf.config.update({
        "tstart": tref, "tstop": tstop, "trstout": trstout_sec, "dtrstout": 0,
        "dtmaxout": trstout_sec, "zsini": -9999.0,
    })
    sf.write()
    log.info(f"Running SFINCS calibration ({round_label}, {calibration_days:.0f} days)")
    run_sfincs_subprocess(
        sfincs_exe, round_root, timeout_s, log,
        label=f"SFINCS calibration ({round_label})", n_threads=snakemake.threads,
    )

    # final_zs/final_times_s (single intersected cell per centerline cell)
    # are kept ONLY for the water-level-timeseries diagnostic plot's own
    # representative_cell_pos indexing -- a visualization concern. The
    # excavation depth instead uses period_max_zs from
    # _read_cross_section_period_max_zs, the MAX water level across each
    # centerline cell's own channel cross-section (see that function's own
    # docstring) -- a single cell's own reading can be a noisy, non-
    # representative sample of the true water level at that point.
    zs, times_s = _read_zs_and_resolve(round_root)
    period_max_zs = _read_cross_section_period_max_zs(round_root)
    log.info(
        f"[{round_label}] Period max water level (cross-section): min={period_max_zs.min():.3f} m, "
        f"max={period_max_zs.max():.3f} m, median={np.median(period_max_zs):.3f} m "
        f"(over the full {calibration_days:.0f}-day run, {zs.shape[0]} output step(s))"
    )
    return period_max_zs, zs, times_s


# wgs84_bounds/domain_poly: round-invariant, needed by every round's own
# diagnostic plots below -- computed once, before the rounds.
wgs84_bounds, domain_crs, domain_poly = load_domain(snakemake.input.spec_basins_meta, domain_gpkg_path)

# Per-round diagnostic files are written directly here (NOT declared
# Snakemake outputs, same convention as calibration_state.csv). Filenames
# put the round number as a SUFFIX (e.g. "max_inundation_round2.png"), not a
# prefix, so a directory listing sorted alphabetically groups by PLOT KIND
# first -- comparing the same diagnostic across rounds means looking at
# consecutive files, not picking them out of an interleaved by-round listing.
round_visuals_dir = Path(snakemake.output.plot_calibration).parent / "rounds"
round_visuals_dir.mkdir(parents=True, exist_ok=True)

N_ROUNDS_AFTER_0 = 2
ROUND_TITLES = [
    "round 0 (confined, un-excavated)",
    "round 1 (confined, excavated)",
    "round 2 (verification, real crests)",
]


def _round_visual_paths(round_idx: int) -> dict[str, Path]:
    return {
        "plot_calibration": round_visuals_dir / f"river_depth_round{round_idx}.png",
        "plot_water_level": round_visuals_dir / f"water_level_timeseries_round{round_idx}.png",
        "plot_max_inundation": round_visuals_dir / f"max_inundation_round{round_idx}.png",
        "animation": round_visuals_dir / f"flood_animation_round{round_idx}.mp4",
        "crest_gap_map": round_visuals_dir / f"crest_gap_map_round{round_idx}.png",
    }


def _start_round(round_idx: int) -> Path:
    """Point sf at this round's OWN output subdirectory (sfincs.inp/
    sfincs_map.nc/sfincs_his.nc/subgrid/...) instead of every round
    overwriting the same filenames in calib_root: reusing the same
    filenames across rounds makes SFINCS itself fail creating its own
    output netCDF files ("Permission denied"/"NetCDF: Not a valid ID") once
    a PREVIOUS round's own diagnostic plotting (compute_max_inundation/
    compute_flood_progression) had read them, since some file handle
    persists past that read on Windows regardless of Python-side
    cache-clearing. The model itself (grid/mask/roughness/weir/discharge
    points/config) is all in-memory on the shared `sf` object -- changing
    root only affects where the NEXT sf.write()/sf.subgrid.create() call
    lands."""
    round_root = calib_root / f"round{round_idx}"
    round_root.mkdir(parents=True, exist_ok=True)
    sf.root.set(round_root)
    return round_root


def _create_subgrid(elevation_list: list[dict], round_label: str) -> None:
    """(Re)build the subgrid table into the current round's own root --
    rebuilt every round, even when the geometry is unchanged (round 2 vs
    round 1), since each round's own diagnostics read the dep_subgrid.tif
    written next to that round's own sfincs_map.nc."""
    if not include_subgrid:
        return
    sf.subgrid.create(
        elevation_list=elevation_list,
        roughness_list=[{"manning": "local_roughness_native"}],
        river_list=[],
        nr_subgrid_pixels=nr_subgrid_pixels,
        nr_levels=nr_levels,
        write_dep_tif=True,
        write_man_tif=True,
        nrmax=nrmax,
    )
    log.info(f"[{round_label}] Subgrid table created: {nr_subgrid_pixels} px/cell")


def _set_weir(weir_gdf: gpd.GeoDataFrame, weir_diagnostics: dict, round_label: str, description: str) -> None:
    if weir_diagnostics.get("applicable", True) and not weir_gdf.empty:
        sf.weirs.set(weir_gdf, merge=False)
        log.info(f"[{round_label}] Weir set: {len(weir_gdf)} segment(s), {description}")
    else:
        log.warning(f"[{round_label}] Coastal protection weir: not applicable -- this round is not confined")


def _update_reach_columns(rivdph: np.ndarray, weir_crest: np.ndarray | None = None) -> None:
    """Per-reach MEDIAN reduction (not mean -- so a mouth reach's hard-forced
    rivdph=0 last cell doesn't skew its own reported value) -- the final
    network output and the river-depth plot need ONE scalar per reach."""
    cell_state = cell_gdf[["reach_id"]].assign(rivdph=rivdph)
    if weir_crest is not None:
        cell_state = cell_state.assign(crest=weir_crest)
    reach_medians = cell_state.groupby("reach_id").median()
    rivers["rivdph"] = rivers["reach_id"].astype(str).map(reach_medians["rivdph"])
    if weir_crest is not None:
        rivers["weir_crest_calibrated"] = rivers["reach_id"].astype(str).map(reach_medians["crest"])
        rivers_utm["weir_crest_calibrated"] = rivers_utm["reach_id"].astype(str).map(reach_medians["crest"])


def _write_calibration_state(round_root: Path, rivdph: np.ndarray, weir_crest: np.ndarray, zs: np.ndarray) -> None:
    cell_gdf.drop(columns="geometry").assign(
        dem=dem_at_cell, rivdph=rivdph, weir_crest=weir_crest, zs=zs,
    ).to_csv(round_root / "calibration_state.csv", index=False)


def _round_diagnostics(
    round_idx: int, round_root: Path, final_zs: np.ndarray, final_times_s: np.ndarray,
    weir_gdf: gpd.GeoDataFrame, crest_gap_crest_on_grid: np.ndarray | None = None,
) -> None:
    """Per-round diagnostic plots. Must run right after each round's own
    SFINCS run, before the next round's own sf.write() -- each round's own
    transient state is only ever read back from its own round_root.
    crest_gap_crest_on_grid (round 2 only): per-cell crest to compare
    against this round's own water level -- a crest-gap map is meaningless
    for the 1000 m confinement rounds."""
    round_label = ROUND_TITLES[round_idx]
    paths = _round_visual_paths(round_idx)

    rivers_wgs = rivers.to_crs("EPSG:4326") if rivers.crs is not None and rivers.crs.to_epsg() != 4326 else rivers
    plot_river_depth(
        rivers_wgs=rivers_wgs,
        bbox_poly=domain_poly,
        land_polygons_path=str(land_mask_path),
        output_path=str(paths["plot_calibration"]),
    )

    plot_water_level_timeseries(
        final_times_s / 86400.0, final_zs[:, representative_cell_pos], str(paths["plot_water_level"]),
        day_markers=[(calibration_days, f"Day {calibration_days:.0f} (run end)")],
        station_labels=[f"reach {rid}" for rid in representative_reach_ids],
        basin_id=f"Basin {basin_id}", run_label=f"Depth calibration -- {round_label}",
    )

    # round_root doubles as both run_dir and sfincs_root for THIS round's
    # own disposable model.
    da_hmax, _da_dep = compute_max_inundation(
        round_root, round_root, _round_ocean_mask_path, hmin=0.0, include_subgrid=include_subgrid,
    )
    if da_hmax is None:
        log.warning(f"[{round_label}] No max inundation data available -- creating empty plot sentinel")
        paths["plot_max_inundation"].touch()
    else:
        plot_max_inundation_map(
            da_hmax, domain_poly, str(land_mask_path), str(river_network_path),
            str(paths["plot_max_inundation"]),
            basin_id=basin_id, run_label=f"calibration -- {round_label}",
        )

    da_h = compute_flood_progression(round_root, _round_ocean_mask_path)
    if da_h is None:
        log.warning(f"[{round_label}] compute_flood_progression returned None -- skipping flood animation")
        paths["animation"].touch()
    else:
        animate_flood_progression(
            da_h, domain_poly, str(land_mask_path), str(river_network_path),
            str(paths["animation"]),
            basin_id=basin_id, run_label=f"calibration -- {round_label}", fps=animation_fps,
        )

    if round_idx == N_ROUNDS_AFTER_0:
        if crest_gap_crest_on_grid is None:
            paths["crest_gap_map"].touch()
        else:
            # Weir line / river network / REAL discharge point(s) overlaid
            # (crossings_utm -- the same snapped locations discharge is
            # actually injected at, not a reach's own line-start coordinate).
            plot_crest_gap_map(
                crest_gap_crest_on_grid, grid.transform, str(round_root / "sfincs_map.nc"),
                weir_gdf, rivers_utm, crossings_utm.geometry.x.to_numpy(), crossings_utm.geometry.y.to_numpy(),
                str(paths["crest_gap_map"]),
                basin_id=basin_id, run_label=f"calibration -- {round_label}",
            )
    log.info(f"[{round_label}] Diagnostic plots written under {round_visuals_dir}")


# Confinement weir (rounds 0 and 1): every edge -- coast and riverbank alike
# -- at weir_crest_m. An all-NaN water_side_crest_on_grid makes
# build_coastal_protection_weir fall back to crest_elevation_m on every edge
# while still tracing the riverbanks. channel_mask_gap_free=True:
# build_channel_mask_regular already uses all_touched=True (no sub-cell
# gaps to patch), so skip the safety-net close_gaps() dilation to keep the
# weir hugging the channel corridor exactly -- the production weir below is
# traced with the same flag, so all three rounds share one weir geometry.
confinement_weir_gdf, confinement_weir_diagnostics = build_coastal_protection_weir(
    grid, landuse_on_grid, channel_mask, weir_crest_m, min_component_cells, weir_par1,
    water_side_crest_on_grid=np.full(grid.shape, np.nan, dtype=np.float32),
    channel_mask_gap_free=True, unprotected_ocean_wetlands=unprotected_ocean_wetlands,
)

# ── dike position with unprotected_ocean_wetlands off vs on (figure only) ────
# The line's POSITION is the same in every round (only crests differ), so the
# confinement trace above stands for the configured setting; the other
# setting is traced once more here purely for the comparison figure -- never
# simulated or exported.
log.info(
    f"Tracing the weir with unprotected_ocean_wetlands={not unprotected_ocean_wetlands} "
    f"(comparison figure only, not simulated)"
)
_alt_weir_gdf, _ = build_coastal_protection_weir(
    grid, landuse_on_grid, channel_mask, weir_crest_m, min_component_cells, weir_par1,
    water_side_crest_on_grid=np.full(grid.shape, np.nan, dtype=np.float32),
    channel_mask_gap_free=True, unprotected_ocean_wetlands=not unprotected_ocean_wetlands,
)
_weir_off, _weir_on = (
    (_alt_weir_gdf, confinement_weir_gdf) if unprotected_ocean_wetlands else (confinement_weir_gdf, _alt_weir_gdf)
)
plot_wetland_dike_positions(
    landuse_on_grid, grid.transform, _weir_off, _weir_on,
    ocean_linked_wetland_mask(landuse_on_grid, ocean_mask_grid, grid),
    str(snakemake.output.plot_dike_positions), active_on=unprotected_ocean_wetlands, basin_id=basin_id,
)

# ── round 0: confined, un-excavated ──────────────────────────────────────────
round_label = ROUND_TITLES[0]
round_root = _start_round(0)
_set_weir(
    confinement_weir_gdf, confinement_weir_diagnostics, round_label,
    f"uniform confinement crest={weir_crest_m:.0f} m",
)
_create_subgrid([{"elevation": "local_elevation_conditioned"}], round_label)
period_max_zs_0, final_zs, final_times_s = _run_calibration_round(round_label, round_root)

rivdph_current = compute_excavation_depth(period_max_zs_0, dem_at_cell, excavation_fraction)
log.info(
    f"[{round_label}] Excavation depth: min={np.nanmin(rivdph_current):.3f} m, "
    f"max={np.nanmax(rivdph_current):.3f} m, median={np.nanmedian(rivdph_current):.3f} m "
    f"(excavation_fraction={excavation_fraction:.3f})"
)

# Mouth reach treatment: a mouth (n_rch_dn == 0, topologically the
# network's own outlet -- is_delta_outflow is NOT used, it's
# unreliable/unpopulated in practice) has its own last (most
# downstream, max along_m) cell hard-forced to natural bathymetry
# (rivdph=0) -- that coastal endpoint is a real physical constraint
# (the actual seabed), not a calibration result.
#
# Every OTHER cell (in the mouth reach itself, and however far
# upstream is needed) keeps the completely normal per-cell round-0
# bed computed above -- UNLESS that normal bed is shallower than the
# coastal endpoint, in which case it's floored down to match it.
# Rationale: a real channel shouldn't get shallower right before it
# meets the sea (the same discharge would then have to accelerate
# through a narrower cross-section just before the outlet) -- but
# this floor must stop being applied the first time, walking
# upstream, that the normal per-cell bed is already at least as deep
# as the coastal endpoint on its own, otherwise every far upstream
# reach with a genuinely shallower natural requirement would get
# needlessly forced deep too. This is a stateful walk (start at the
# coast, stop permanently at the first intersection), NOT a blanket
# elementwise min() over the whole network.
reach_is_mouth = dict(zip(rivers["reach_id"].astype(str), rivers["n_rch_dn"] == 0))
cell_is_mouth_reach = cell_gdf["reach_id"].map(reach_is_mouth).fillna(False).to_numpy(dtype=bool)

if cell_is_mouth_reach.any():
    along_m_arr = cell_gdf["along_m"].to_numpy()
    reach_id_arr = cell_gdf["reach_id"].to_numpy()
    reach_cell_pos: dict[str, np.ndarray] = {
        rid: (pos := np.flatnonzero(reach_id_arr == rid))[np.argsort(along_m_arr[pos])]  # upstream -> downstream
        for rid in cell_gdf["reach_id"].unique()
    }
    downstream_adjacency = build_downstream_adjacency(rivers)
    upstream_of: dict[str, list[str]] = {}
    for rid, dn_list in downstream_adjacency.items():
        for dn in dn_list:
            upstream_of.setdefault(dn, []).append(rid)

    for mouth_rid in cell_gdf.loc[cell_is_mouth_reach, "reach_id"].unique():
        reach_pos = reach_cell_pos[mouth_rid]
        last_pos = reach_pos[-1]
        coastal_endpoint_bed = dem_at_cell[last_pos]
        rivdph_current[last_pos] = 0.0

        current_rid = mouth_rid
        cell_positions = reach_pos[-2::-1]  # this reach's own cells, excl. coastal endpoint, downstream -> upstream
        found_intersection = False
        while not found_intersection:
            for pos in cell_positions:
                bed_normal = dem_at_cell[pos] - rivdph_current[pos]
                if bed_normal > coastal_endpoint_bed:
                    rivdph_current[pos] = dem_at_cell[pos] - coastal_endpoint_bed
                else:
                    found_intersection = True
                    break
            if found_intersection:
                break
            upstream_rids = upstream_of.get(current_rid, [])
            if len(upstream_rids) != 1:
                break  # no further single upstream reach (headwater or confluence) -- stop clamping
            current_rid = upstream_rids[0]
            cell_positions = reach_cell_pos[current_rid][::-1]  # next reach, downstream -> upstream
else:
    log.warning(f"[{round_label}] No mouth (outlet) reach found")

_update_reach_columns(rivdph_current)
_write_calibration_state(
    round_root, rivdph=np.zeros(len(cell_gdf)), weir_crest=np.full(len(cell_gdf), np.nan), zs=period_max_zs_0,
)
_round_diagnostics(0, round_root, final_zs, final_times_s, confinement_weir_gdf)

# ── excavation (shared by rounds 1 and 2) ────────────────────────────────────
# zbed_anchors: one anchor per centerline cell -- rivbed(cell) = DEM_at_cell
# - rivdph_current[cell], a constant DEPTH offset from the real, locally-
# varying terrain at the cell's own location. Mouth reaches need no special
# case here: their own last (coastal) cell was already hard-forced to
# natural bathymetry (rivdph=0) above, so it flows through
# burn_river_channel's own per-reach anchor interpolation like any other
# cell.
zbed_anchors = cell_gdf.assign(rivbed=dem_at_cell - rivdph_current)

# SFINCS-grid resolution: constrained to channel_mask (same shared mask used
# for the weir corridor), so excavated cells == the corridor's own cells,
# exactly -- same invariant production (rule 11b/13) relies on. Patched
# directly into sf.grid.data["dep"]'s own array rather than routed through
# the data catalog/elevation_list mechanism -- sf.write() writes whatever
# ends up in sf.grid.data["dep"] regardless of how it got there.
# natural_dem_path floors the burn against the real conditioned DEM (never
# shallower than it) -- see burn_river_channel's own docstring for why this
# is needed even with terrain-following anchors (interpolation/junction-
# blending between anchors can still locally overshoot). Also written as
# this rule's own river_burned_dem_sfincs_grid output below -- the identical
# burn, not re-run.
burned_arr_sfincs, burned_transform_sfincs, _nd, _stats_sfincs = burn_river_channel(
    rivers=rivers_utm, zbed_anchors=zbed_anchors,
    natural_dem_path=elevation_path,
    utm_crs=sf.crs, resolution_m=grid.cell_size_m,
    out_transform=grid.transform, out_shape=grid.shape,
    channel_mask=channel_mask,
)
valid_burn = np.isfinite(burned_arr_sfincs)
sf.grid.data["dep"].values[valid_burn] = burned_arr_sfincs[valid_burn]
log.info(
    f"Excavated {int(valid_burn.sum()):,} cell(s) into the main grid "
    f"({_stats_sfincs['n_reaches_burned']} reach(es) burned, {_stats_sfincs['n_reaches_skipped']} skipped)"
)

NODATA = np.float32(-9999.0)
if include_subgrid:
    # Burn DIRECTLY onto the subgrid's own phase-locked fine-pixel grid
    # (coarse grid transform subdivided by nr_subgrid_pixels) instead of an
    # independently-bounded native-resolution grid -- otherwise hydromt_
    # sfincs's own subgrid.create() has to reproject/resample this file onto
    # its internal fine grid itself (different resolution AND phase/origin),
    # and THAT resampling step introduces new leakage across the
    # channel_mask boundary even when the source burn file itself has 0
    # pixels outside channel_mask -- hydromt's own dep_subgrid.tif can still
    # show excavated pixels bleeding into a coarse cell channel_mask
    # excludes, and since SFINCS reports a subgrid cell's own "zb" as
    # (effectively) the MINIMUM sub-pixel elevation within it, even a
    # handful of leaked sub-pixels at one corner makes the WHOLE coarse cell
    # read as deeply excavated relative to its own true (unexcavated)
    # surrounding terrain. Burning directly at the subgrid's own
    # resolution/phase eliminates the resampling step (and its leak)
    # entirely, rather than trying to out-guess hydromt's own internal
    # resampling afterward.
    subgrid_transform = grid.transform * Affine.scale(1.0 / nr_subgrid_pixels)
    subgrid_shape = (grid.shape[0] * nr_subgrid_pixels, grid.shape[1] * nr_subgrid_pixels)
    burned_arr_subgrid, transform_subgrid, _nd_subgrid, stats_subgrid = burn_river_channel(
        rivers=rivers_utm, zbed_anchors=zbed_anchors,
        natural_dem_path=elevation_path,
        utm_crs=sf.crs, resolution_m=grid.cell_size_m / nr_subgrid_pixels,
        out_transform=subgrid_transform, out_shape=subgrid_shape,
    )
    # Belt-and-braces: still constrain to channel_mask even though the burn
    # is phase-locked -- a reach's own buf_poly can still rasterize a
    # handful of edge pixels slightly differently than channel_mask's own
    # independent rasterization (same reason the coarse burn always passes
    # channel_mask=... too).
    burned_arr_subgrid = constrain_to_coarse_channel_mask(
        burned_arr_subgrid, transform_subgrid, sf.crs, channel_mask, grid.transform,
    )
    with rasterio.open(
        river_burned_subgrid_path, "w", driver="GTiff", dtype="float32",
        width=burned_arr_subgrid.shape[1], height=burned_arr_subgrid.shape[0],
        count=1, crs=sf.crs, transform=transform_subgrid,
        nodata=float(NODATA), compress="deflate", tiled=True,
    ) as dst:
        dst.write(np.where(np.isnan(burned_arr_subgrid), NODATA, burned_arr_subgrid).astype(np.float32), 1)
    log.info(
        f"Subgrid-resolution burn: {stats_subgrid['n_reaches_burned']} reach(es) burned, "
        f"{stats_subgrid['n_pixels_burned']:,} pixel(s)"
    )
excavated_subgrid_elevation_list = [
    {"elevation": "local_river_burned_subgrid"},
    {"elevation": "local_elevation_conditioned"},
]

# ── coastal forcing for rounds 1 and 2: the protection-level storm tide ─────
# Round 0 keeps its steady calm-sea boundary (the excavation depth is a river
# question). Rounds 1 and 2 are forced at the surge stations production uses
# with a storm tide at the coastal protection RP (FLOPROS), in production's
# own wave shape (half-cosine over surge period_hr): SFINCS's water level at
# the coast can rise well above the level imposed at the boundary (shoaling,
# reflection off the dike, bays), so a crest equal to the offshore COAST-RP
# level does not hold that RP inside the model (basin 2433835: sea levels up
# to ~0.8 m at the dike under a 0.4 m boundary). With the storm tide in round
# 1, every coastal edge's crest comes from the simulated level at the dike
# itself -- the same per-edge rule as the river banks -- and round 2 verifies
# it. The wave starts once the river is at its full calibration discharge
# (after discharge_ramp_hours) and rises from baseline_m, the level the sea
# starts at (zsini), so there is no step. Peaks: per station, linear in RP
# between COAST-RP's tabulated RPs (as the crest's own RP-41 level), MDT-
# corrected and rounded up to 0.1 m like all forcing. River and surge at
# their protection RPs together is conservative near the mouth, where the
# two interact.
if coastal_rp_yr is not None:
    with xr.open_dataset(surge_forcing_path, decode_times=False) as _sds:
        coastal_design_peaks = storm_tide_at_rp_interpolated(_sds, coastal_rp_yr)
        surge_period_hr = float(_sds.attrs["period_hr"])
        _surge_stations = gpd.GeoDataFrame(
            {"index": range(_sds.sizes["station"])},
            geometry=gpd.points_from_xy(_sds["longitude"].values, _sds["latitude"].values),
            crs="EPSG:4326",
        )
    if discharge_ramp_hours + surge_period_hr > calibration_days * 24.0:
        raise ValueError(
            f"calibration run ({calibration_days * 24.0:.0f} h) too short for the storm tide: "
            f"discharge ramp {discharge_ramp_hours:.0f} h + surge period {surge_period_hr:.0f} h -- "
            f"increase river_depth_modelling.calibration_days"
        )
    _surge_hours = np.arange(0.0, calibration_days * 24.0 + 1e-9, 0.25)
    _surge_wl = np.stack([
        sinusoidal_wave(baseline_m, float(peak), _surge_hours, discharge_ramp_hours / 24.0, surge_period_hr)
        for peak in coastal_design_peaks
    ], axis=1)
    sf.water_level.create(
        timeseries=pd.DataFrame(
            _surge_wl, index=pd.DatetimeIndex([tref + timedelta(hours=float(h)) for h in _surge_hours]),
            columns=range(len(_surge_stations)),
        ),
        locations=_surge_stations, buffer=waterlevel_buffer_m, merge=False,
    )
    log.info(
        f"Rounds 1-2 coastal forcing: RP {coastal_rp_yr:.1f} storm tide at {len(_surge_stations)} station(s), "
        f"peak {coastal_design_peaks.min():+.2f}..{coastal_design_peaks.max():+.2f} m (from {baseline_m:+.2f} m), "
        f"{surge_period_hr:.0f} h wave starting at {discharge_ramp_hours:.1f} h"
    )
else:
    log.warning("No coastal protection RP -- rounds 1-2 keep the steady calm-sea boundary")

# ── round 1: confined, excavated ─────────────────────────────────────────────
# sf.weirs still holds round 0's confinement weir -- unchanged here.
round_label = ROUND_TITLES[1]
round_root = _start_round(1)
_create_subgrid(excavated_subgrid_elevation_list, round_label)
period_max_zs_1, final_zs, final_times_s = _run_calibration_round(round_label, round_root)
zs_field_1 = _read_period_max_zs_field(round_root)

# Production weir: every edge's crest = max(round 1's water level on its own
# water side, coastal_protection_crest_m) + weir_freeboard_m (see this
# module's own docstring). Traced once here and exported as-is below --
# round 2 verifies exactly this weir.
weir_gdf, final_weir_diagnostics = build_coastal_protection_weir(
    grid, landuse_on_grid, channel_mask, coastal_protection_crest_m, min_component_cells, weir_par1,
    water_side_crest_on_grid=zs_field_1, freeboard_m=weir_freeboard_m, channel_mask_gap_free=True,
    unprotected_ocean_wetlands=unprotected_ocean_wetlands,
)

# Per-centerline-cell summary of the same rule (cross-section water level in
# place of each edge's own water-side cell) -- reporting only
# (calibration_state.csv, round profiles, weir_crest_calibrated).
weir_crest_cell = ceil_water_level(np.fmax(period_max_zs_1, coastal_protection_crest_m) + weir_freeboard_m)
log.info(
    f"[{round_label}] Crest (centerline summary): min={np.nanmin(weir_crest_cell):.3f} m, "
    f"max={np.nanmax(weir_crest_cell):.3f} m, median={np.nanmedian(weir_crest_cell):.3f} m "
    f"(+{weir_freeboard_m:.2f} m freeboard)"
)
_update_reach_columns(rivdph_current, weir_crest_cell)
_write_calibration_state(round_root, rivdph=rivdph_current, weir_crest=weir_crest_cell, zs=period_max_zs_1)
_round_diagnostics(1, round_root, final_zs, final_times_s, confinement_weir_gdf)

# ── round 2: verification with the real production weir ─────────────────────
round_label = ROUND_TITLES[2]
round_root = _start_round(2)
_set_weir(weir_gdf, final_weir_diagnostics, round_label, "production crests (per edge, from round 1)")
_create_subgrid(excavated_subgrid_elevation_list, round_label)
period_max_zs_2, final_zs, final_times_s = _run_calibration_round(round_label, round_root)

edge_crest_on_grid = None
if final_weir_diagnostics.get("edge_water_side_mask") is not None:
    zs_field_2 = _read_period_max_zs_field(round_root)
    _edge_mask = final_weir_diagnostics["edge_water_side_mask"]
    edge_crest_on_grid = np.where(_edge_mask, final_weir_diagnostics["crest_surface"], np.nan).astype(np.float32)
    # NaN (never-wetted) water side compares False -- nothing to overtop there.
    _exceedance = zs_field_2[_edge_mask] - edge_crest_on_grid[_edge_mask]
    _n_overtopped = int((_exceedance > 0).sum())
    if _n_overtopped:
        log.warning(
            f"[{round_label}] {_n_overtopped} of {int(_edge_mask.sum()):,} edge water-side cell(s) exceed "
            f"their own crest (max exceedance {np.nanmax(_exceedance):.3f} m) -- see the crest gap map; "
            f"crests are NOT adjusted (verification only)"
        )
    else:
        log.info(
            f"[{round_label}] Verified: no edge water-side cell exceeds its own crest "
            f"({int(_edge_mask.sum()):,} cell(s) checked)"
        )
_write_calibration_state(round_root, rivdph=rivdph_current, weir_crest=weir_crest_cell, zs=period_max_zs_2)
_round_diagnostics(2, round_root, final_zs, final_times_s, weir_gdf, crest_gap_crest_on_grid=edge_crest_on_grid)

# ── canonical production outputs: burned river DEM + weir ────────────────────
# Same filenames rule empirical_depth_estimation writes -- see this rule's own
# module docstring. The native-resolution file is burned once more here on
# the native DEM's own grid (not the subgrid-phase-locked grid rounds 1-2
# used); the SFINCS-grid file is the exact burn rounds 1-2 simulated with.
_final_burned_native, _final_transform_native, _nd, _stats = burn_river_channel(
    rivers=rivers_utm, zbed_anchors=zbed_anchors,
    natural_dem_path=elevation_path,
    utm_crs=sf.crs, resolution_m=native_resolution_m,
)
_final_burned_native = constrain_to_coarse_channel_mask(
    _final_burned_native, _final_transform_native, sf.crs, channel_mask, grid.transform,
)
with rasterio.open(
    snakemake.output.river_burned_dem, "w", driver="GTiff", dtype="float32",
    width=_final_burned_native.shape[1], height=_final_burned_native.shape[0],
    count=1, crs=sf.crs, transform=_final_transform_native,
    nodata=float(NODATA), compress="deflate", tiled=True,
) as dst:
    dst.write(np.where(np.isnan(_final_burned_native), NODATA, _final_burned_native).astype(np.float32), 1)
log.info(
    f"Written: {snakemake.output.river_burned_dem} ({_stats['n_reaches_burned']} reach(es) burned, "
    f"{_stats['n_pixels_burned']:,} pixel(s))"
)

with rasterio.open(
    snakemake.output.river_burned_dem_sfincs_grid, "w", driver="GTiff", dtype="float32",
    width=burned_arr_sfincs.shape[1], height=burned_arr_sfincs.shape[0],
    count=1, crs=sf.crs, transform=burned_transform_sfincs,
    nodata=float(NODATA), compress="deflate",
) as dst:
    dst.write(np.where(np.isnan(burned_arr_sfincs), NODATA, burned_arr_sfincs).astype(np.float32), 1)
log.info(
    f"Written: {snakemake.output.river_burned_dem_sfincs_grid} "
    f"({_stats_sfincs['n_reaches_burned']} reach(es) burned, {_stats_sfincs['n_pixels_burned']:,} pixel(s))"
)

Path(snakemake.output.coastal_protection_weir).parent.mkdir(parents=True, exist_ok=True)
weir_gdf.to_file(snakemake.output.coastal_protection_weir, driver="GPKG")
log.info(f"Written: {snakemake.output.coastal_protection_weir} ({len(weir_gdf)} segment(s))")

# ── real open sea, AT THIS RULE'S OWN GRID resolution (no reprojection at
# all -- written using grid.transform/grid.crs exactly as-is) ───────────────
# Mostly a GRID-RESOLUTION MISALIGNMENT fix, not an isolated-pocket one --
# the weir is traced on landuse_on_grid (rule grid_align_landuse, 09b:
# native landuse.tif resampled via nearest-neighbor onto the much coarser
# SFINCS regular grid, ONCE, upstream), so land_mask (the model's own
# "this is the protected side" decision) disagrees with the fine native
# landuse.tif right along the coast -- overlaying the weir on the
# native-resolution raster shows native "sea" (200) pixels sitting on the
# land side, continuously along the coastline, wherever the coarse grid's
# own cell edges cut across the true coastline -- see src.protection_weir.
# build_coastal_protection_weir's own protected_pocket_mask docstring for
# the full mechanism (confirmed via basin 2433835: its landuse.tif/
# sea_mask.tif have exactly ONE connected sea component at native
# resolution, so there was never an isolated "pocket" to find via
# component analysis -- the mismatch is a coastline-wide fringe, not
# discrete blobs).
#
# zsini_sea_cells_on_grid.tif is THE single sea/land classification every
# downstream consumer reads directly, zero further reprojection: THIS
# rule's own zsini (13_build_sfincs_skeleton.py, needs the fix for the
# SIMULATION itself -- the uncorrected classification would start these
# fringe cells wet at baseline_m regardless of the weir, so SFINCS would
# show them as flooded from t=0 purely from the mismatched initial
# condition) AND flood-diagnostic consumers downstream (rules run_spinup/
# sanity_checks/run_event/compute_flood_metrics, via src.postprocessing's
# own sea_mask_path argument). There used to be a SEPARATE, native-
# resolution sea_mask_corrected.tif for the flood-diagnostic consumers --
# removed 2026-08-07b: those consumers reproject whatever sea_mask_path
# they're given onto their own (subgrid- or cell-resolution) output grid
# anyway (src.postprocessing's own da_sea.raster.reproject_like calls), and
# that reprojection is only well-defined/alignment-safe when the SOURCE is
# already grid-aligned (subgrid is an exact integer subdivision of the
# coarse grid, sharing its origin/axes -- unlike the arbitrary-origin
# native FathomDEM/landuse pixel grid). A second, redundant native file
# encoding the exact same boolean was pure duplication once that was
# recognized -- one raster, one resampling pass, one source of truth,
# same principle already applied to zsini/landuse/roughness above.
_protected_pocket_mask = final_weir_diagnostics.get(
    "protected_pocket_mask", np.zeros(landuse_on_grid.shape, dtype=bool)
)
_sea_cells_on_grid = (landuse_on_grid == LANDUSE_SEA) & ~_protected_pocket_mask
_sea_cells_arr = np.where(_sea_cells_on_grid, np.float32(1.0), np.float32(-9999.0))
_sea_cells_profile = {
    "driver": "GTiff", "height": _sea_cells_arr.shape[0], "width": _sea_cells_arr.shape[1],
    "count": 1, "dtype": "float32", "crs": grid.crs, "transform": grid.transform,
    "nodata": -9999.0,
}
Path(snakemake.output.zsini_sea_cells).parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(snakemake.output.zsini_sea_cells, "w", **_sea_cells_profile) as _dst:
    _dst.write(_sea_cells_arr, 1)
log.info(
    f"zsini sea cells (this rule's own grid, no reprojection): "
    f"{int(_sea_cells_on_grid.sum()):,} real open-sea cell(s). "
    f"Written: {snakemake.output.zsini_sea_cells}"
)

# ── seed-to-mouth round-profile diagnostics (bed/crest/water-level per round) ─
# Automatically produced here (not a declared Snakemake output, same
# convention as calibration_state.csv) instead of requiring a separate,
# manually-run script -- and reads the SAME calibration_state.csv every
# round just wrote, so the profile is the real per-cell state this rule
# actually calibrated with, not a reconstruction of it (see
# gather_calibration_round_profile's own docstring). Written under the SAME
# persistent visuals directory as the other canonical diagnostic plots
# (results/{basin_id}/preprocessing_inputs/visuals/10_calibration/) rather
# than calib_root -- calib_root is this rule's own intermediate/disposable
# model directory, not where user-facing output belongs.
visuals_dir = Path(snakemake.output.plot_calibration).parent
visuals_dir.mkdir(parents=True, exist_ok=True)
seed_ids = sorted(
    {normalize_reach_id(rid) for rid, s in zip(rivers["reach_id"], rivers["is_seed"]) if bool(s) is True} - {None}
)
if not seed_ids:
    log.warning("No is_seed reach found -- skipping seed-to-mouth round-profile diagnostics")
for _seed in seed_ids:
    _path_rids, _profiles_by_round = gather_calibration_round_profile(
        calib_root, rivers_utm, _seed, N_ROUNDS_AFTER_0
    )
    plot_calibration_round_profiles(
        basin_id=basin_id, seed=_seed, profiles_by_round=_profiles_by_round,
        n_rounds=N_ROUNDS_AFTER_0,
        output_subplots_path=str(visuals_dir / f"round_profiles_seed{_seed}_subplots.png"),
        output_combined_path=str(visuals_dir / f"round_profiles_seed{_seed}_combined.png"),
        round_titles=ROUND_TITLES,
    )
    pd.concat(
        [df.assign(round=i) for i, df in _profiles_by_round.items()], ignore_index=True
    ).to_csv(visuals_dir / f"round_profiles_seed{_seed}.csv", index=False)
    log.info(f"Seed {_seed}: round-profile diagnostics written (path of {len(_path_rids)} reach(es))")

n_missing = rivers["rivdph"].isna().sum()
if n_missing:
    log.warning(f"{n_missing} reach(es) missing a calibrated depth (no centerline cell?) -- left as NaN")

Path(snakemake.output.depth_estimated_river_network).parent.mkdir(parents=True, exist_ok=True)
rivers.to_file(snakemake.output.depth_estimated_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.depth_estimated_river_network}")

# ── canonical outputs: copies of round 2's own per-round files ───────────────
Path(snakemake.output.plot_calibration).parent.mkdir(parents=True, exist_ok=True)
_last_round_paths = _round_visual_paths(N_ROUNDS_AFTER_0)
for _canonical, _kind in (
    (snakemake.output.plot_calibration, "plot_calibration"),
    (snakemake.output.plot_water_level_timeseries, "plot_water_level"),
    (snakemake.output.plot_max_inundation, "plot_max_inundation"),
    (snakemake.output.animation_flood_progress, "animation"),
    (snakemake.output.plot_crest_gap_map, "crest_gap_map"),
):
    shutil.copy(_last_round_paths[_kind], _canonical)
log.info(f"Canonical diagnostic outputs copied from round {N_ROUNDS_AFTER_0}'s own files")
log.info("Done")

