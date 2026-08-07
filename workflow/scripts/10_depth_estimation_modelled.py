"""
10_depth_estimation_modelled.py -- SFINCS-based river depth calibration
(river_processing.depth_method == "modelled"), an alternative to the
empirical hydraulic-geometry depth estimate (rule empirical_depth_estimation,
10_depth_estimation_empirical.py).

Iterative, backwater-corrected, PER-CELL design: round 0 builds a minimal,
disposable SFINCS model (regular grid only, but otherwise matching the
production model's own subgrid setup -- see the subgrid section below) with
the SAME weir geometry/logic the real production model uses
(src.protection_weir.build_coastal_protection_weir -- coast AND river banks
merged into one water_like boundary, not a river-only confinement traced by
a separate function), but with EVERY segment's crest -- coastal and riverbank
alike -- set to the same artificially high (e.g. 1000 m) confinement value,
not a calibrated or real coastal one: even with the real, steady baseline_m
boundary in place (see below), the confined river's water level can exceed
a low real crest right at the coast/river transition near the mouth,
leaking onto the floodplain through the coastal side of the same merged
boundary -- so round 0 must confine everything uniformly. Forced with a
discharge equal to the basin's protection-level return period (bankfull as
a fallback) -- linearly RAMPED UP from each crossing's own bankfull_discharge
over river_processing.river_depth_modelling.discharge_ramp_hours, then held
constant for the rest of the run, rather than forced as an instantaneous
step from t=0: the calibration discharge is often several times bankfull,
and stepping straight to it into a channel that starts near-dry produces a
startup shock wave that can persist/oscillate for the entire run instead of
damping out (mirrors production's own bankfull-lead-in-then-ramp hydrograph,
src.river_forcing.build_design_discharge_matrix/sinusoidal_wave, just linear
here since calibration only needs to reach and hold the peak). Runs for a
single fixed duration (river_processing.
river_depth_modelling.calibration_days), and splits the simulated rise above
the DEM between channel excavation and an actual production-model weir
crest (river_processing.river_depth_modelling.weir_crest_fraction) --
INDEPENDENTLY AT EVERY SFINCS grid cell along each reach's own centerline,
not once per reach:

    rise = period_max_water_level - DEM_at_cell
    rivdph_calibrated    = (1 - weir_crest_fraction) * rise
    weir_crest_calibrated = DEM_at_cell + weir_crest_fraction * rise

period_max_water_level is each cell's own MAXIMUM simulated water level
over the entire run (src.river_depth_calibration.compute_period_max_zs),
not a windowed-flatness "converged"/stabilized value -- a fixed-duration
steady-discharge run has no reason to converge to a flat window in that
strict sense, and a persistent, non-damping oscillation at a discharge-
injection cell never satisfies a flatness/trend tolerance no matter how
long the run is extended.
Calibration only needs the worst-case (peak) forced water level at each
cell to size channel depth/crest against, which the period maximum gives
directly, oscillation or not, with no convergence check, retry, or
restart-extension logic at all.

Water level is read directly from each round's own sfincs_map.nc (the full
zs(time, n, m) field SFINCS already writes), not from SFINCS "observation
points"/sfincs_his.nc -- a single point per reach misses real, substantial
internal variation within a reach, most strikingly a sharp
local spike right at the seed reach's own discharge-injection cell. Every
cell the centerline actually passes through (src.river_burn.build_centerline_cells_regular)
gets its own independently-tracked depth/crest.

Each centerline cell's own "period_max_water_level" is not simply the
single grid cell the centerline happens to intersect, either -- a single
cell's own reading can be a noisy, non-representative sample of the true
water level at that point along the river. Instead it is the MAX water
level across that cell's own channel CROSS-SECTION: the contiguous
channel_mask run through the cell along its own grid row vs. its own grid
column, whichever is SHORTER (extended by one cell on each side to also
reach the adjacent land cell the weir itself sits on) -- see
_read_cross_section_period_max_zs and the precomputation block above
cell_gdf's own global-cache-var init. Deliberately a simple axis-aligned
comparison, not a true reach-normal cross-section.

The calibrated crest is painted onto the grid with NO along-reach
interpolation and no cross-reach junction blending either
(build_nearest_weir_crest_regular): every raster cell gets its own nearest
centerline anchor's crest value directly, since every anchor is already an
exact per-cell target (zs_max(cross-section) + freeboard/increment, see
below) -- interpolating between exact targets would just reintroduce the
same under/over-shoot a smoothed profile was meant to avoid. Only
burn_river_channel's own bed excavation still uses per-anchor
interpolation (a channel bed is expected to vary gradually, unlike a
target-driven crest).

weir_crest_fraction is applied uniformly at round 0, with no magnitude
threshold that switches some cells to 0% excavation -- a hard switch to
100%-crest/0%-depth below a rise threshold would produce a one-cell cliff
in the bed profile wherever a reach's own rise crosses that threshold.
rivdph_calibrated is set ONCE here, at round 0, and never revisited again --
every correction round below only ever adjusts the crest.

Round 0 alone measures every reach in total isolation (each independently
confined by the 1000 m walls, with nothing downstream to push back against
it) -- it cannot see the backwater effect that couples real, finite-crest
reaches together (a shortfall at one reach raises the head its upstream
neighbour has to overcome, compounding upstream). To correct for this,
river_processing.river_depth_modelling.n_correction_iterations (default 2)
additional ROUNDS run after round 0: each builds a model with round 0's own
(fixed) excavation + the CURRENT calibrated crest (a faithful, coupled
replica of what production will actually build, via the same
burn_river_channel function rules 11b/13 use, plus
build_nearest_weir_crest_regular for the crest -- see above) instead of an
isolated confinement wall, at the SAME calibration discharge, then updates
the crest directly, at every cell, unconditionally (no overtopped/
not-overtopped branching, no smoothing):

    weir_crest_current = period_max_water_level + min_crest_increment_per_round_m

Applied to every cell whether it raises OR lowers that cell's crest
relative to its current value -- repeated every round, this makes the
crest climb by at least min_crest_increment_per_round_m each round on top
of whatever the coupled water level actually settles at (raising the
crest removes overbank relief, forcing more of the same discharge through
the same, round-0-frozen channel depth, which raises the confined water
level in turn -- a flat per-round increment, not a full-gap jump, keeps
this feedback from compounding in one aggressive step). Once a round's own
crest already exceeds that round's own period_max_water_level everywhere
(no overtopping under its own simulation) AND the realized inundated-cell
count has settled back down near round 0's own, a single one-shot
tightening snap runs instead -- see the early-stopping section below.

weir_crest_current is used AS-IS, with no separate freeboard added on top,
both to run every round's own simulated weir and as this rule's own final
output -- so the calibrated crest itself always guarantees at least
freeboard_m of margin above the coupled system's own driven water level
once the one-shot snap has run, rather than relying on a separate, later
top-up (rule 13's own build_coastal_protection_weir freeboard_m parameter,
used this way for the production/coastal crest, would otherwise
double-count it here).
`n_correction_iterations: 0` skips this refinement entirely, using round
0's own isolated-confinement estimate unchanged.

The coastal boundary is a real, STEADY baseline_m level (mean sea level +
SLR/MDT correction, read from surge_forcing.nc -- the same value
14_run_spinup.py/13_build_sfincs.py use for their own initial/steady
conditions), applied uniformly across every round including round 0.
Calibration remains a steady-discharge run throughout -- this is
baseline_m only, not the full dynamic compound event timeseries.

Correction rounds burn using each CELL's own current rivdph_current
directly as a burn_river_channel anchor (rivbed = dem_at_cell -
rivdph_current), rather than routing a per-reach scalar through
src.river_preburn.compute_river_bed_points' DEM-following constant-offset
convention -- every cell already carries its own calibrated value, so
burn_river_channel's existing per-reach interp1d does the along-reach
(and, via its neighbour-anchor-borrowing, cross-reach-junction) smoothing
directly from real calibration data, not DEM shape alone.

Mouth reach(es) (n_rch_dn == 0, topologically the network's own real
coastal outlet(s) -- NOT the same thing as a delta-outline outflow point,
see below) have their own last (most downstream, max along_m) cell
hard-forced to zero excavation (rivdph=0, bed=natural bathymetry) -- that
coastal endpoint's BED is a real physical constraint (the actual seabed),
not a calibration result. The CREST there is a separate concern (see
below) -- it still needs to be an actual protective structure, not the
raw (possibly below-sea-level) seabed elevation.

Separately, a non-seed, non-mouth, non-bifurcation reach that crosses the
delta polygon's own outline (identify_delta_outflow_points, rule
clean_river_network) is a genuine place flow exits the modelled network
WITHOUT reaching a real coastal mouth -- e.g. a distributary clipped by
the domain boundary in a complex, multi-channel delta. This gets a free
outflow boundary (mask=3) instead of any mouth-style bed treatment (there
is no real seabed there, just an arbitrary domain edge) -- see the mask
setup below. Without it, such a reach would have neither treatment: not a
mouth (no downstream-facing bed floor), and walled off like ordinary land
by the weir/domain-edge closure, so its own crest would climb without
bound chasing water that in reality/production simply leaves the domain
there.

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
Bed is untouched by any of this beyond the endpoint itself -- every
non-endpoint cell still gets the normal per-cell round-0 bed formula; only
the coastal endpoint's own bed is hard-forced to natural bathymetry.

The mouth's own CREST, unlike its bed, is not hard-forced to a fixed value
at all -- it is re-targeted every round (round 0, every correction round,
and the final snap) to max(the real simulated water level at the nearest
OCEAN-classified grid cell + freeboard_m, coastal_protection_crest_m)
(_mouth_crest_target). Sampling the ocean side rather than the mouth's own
river-side channel cell matters: that channel cell sits right at the
coast/river transition and its own reading isn't necessarily
representative of the open water the crest there actually has to hold
back. This gives the mouth a real, verified protective structure sized
against actual (backwater/tide-coupled) conditions, rather than reporting
the raw seabed elevation (which can be below sea level) with only a flat
coastal standard as a floor.

weir_crest_current is ALSO floored against coastal_protection_crest_m (the
real production coastal protection standard, read from surge_forcing.nc)
EVERY round, for EVERY cell, right after that round's own update --
build_coastal_protection_weir already applies this max() at actual
simulation/build time; this floor makes the TRACKED weir_crest_current
(calibration_state.csv, the round-profile plots, and the final network's
own weir_crest_calibrated column) agree with what SFINCS actually built,
instead of under-reporting it.

Coastal probe correction (correction rounds only): the river's own
backwater can raise the water level right at the coast above baseline_m,
close enough to the river's own dilated crest coverage to be
masked by it, but just beyond river_crest_dilation_cells' own radius the
flat coastal_protection_crest_m floor can sit below that same locally-
elevated level, letting land there flood even though the river's own crest
a few cells further is correctly raised. Rather than widening the dilation
radius or searching for the nearest bank line (both guesses at geometry),
this probes ocean cells directly (coastal_probe_rows/cols, built once from
ocean_mask -- never river_channel_mask, so it can only ever be driven by
genuinely coastal water, never the river itself) and applies the SAME
two-case update as the river crest to a separate coastal_crest_current
array, rasterized onto the grid by nearest-probe-cell (_rasterize_nearest,
following the real coastline shape, bays included, rather than a uniform
dilation) and combined with river_crest_on_grid via np.maximum inside
build_coastal_protection_weir -- never lowers protection, only raises it
locally where the data says so. Capped at river_crest_dilation_cells: a
probe's own reading only propagates that far before falling back to NaN
(and therefore to the flat coastal_protection_crest_m floor) -- without
this cap, a single genuinely local reading (e.g. a real hydraulic
bottleneck at a discharge injection point, see the river boundary probe's
own docstring below) would otherwise propagate to every cell in the
domain, however far away, since plain nearest-neighbour has no notion of
"too far to be relevant".

See src.river_depth_calibration's module docstring for the full rationale.
This model is intermediate/disposable -- rule 13 builds the production
model separately, using this rule's output network. Rule 13 rebuilds the
SAME weir a second time (necessarily -- it runs on the production grid, not
this rule's disposable one) via the same build_coastal_protection_weir
function, using the FINAL weir_crest_calibrated per reach in place of the
uniform confinement value, maxed against the real coastal crest there.

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
production sit on the identical background elevation. Correction rounds
(>= 1) excavate this same background within the shared channel_mask
corridor, using burn_river_channel; the final round's own burn becomes the
production river_burned_dem file rule 13 imports directly, without
re-burning it itself.

Inputs
------
elevation_conditioned              Native-resolution conditioned DEM (rule
                                    enforce_river_monotonicity, 09) --
                                    subgrid's own fine source, and the
                                    native resolution/CRS reference for
                                    correction rounds' own native burn.
elevation_conditioned_sfincs_grid  SFINCS-grid-resolution conditioned DEM
                                    (rule enforce_river_monotonicity's
                                    second output) -- this model's own
                                    sf.elevation.create() base layer
                                    (round 0), and the excavation
                                    background for correction rounds.
surge_forcing                      surge_forcing.nc (rule 07) -- baseline_m
                                    for the real, steady coastal boundary.

Outputs
-------
river_network_depth_estimated  Same reach set, rivdph replaced with the
                          FINAL (after all correction rounds) per-reach
                          MEDIAN of its own cells' calibrated excavation
                          depth, plus a new weir_crest_calibrated column
                          (absolute elevation, same per-reach median
                          reduction) consumed by rule 13's production weir
                          -- the same unified filename rule
                          empirical_depth_estimation also writes. Production
                          still consumes ONE scalar
                          per reach; the richer per-cell state this rule
                          actually calibrates with is not persisted to this
                          file (see calibration_state.csv per round below).
plot_calibration          Diagnostic: reaches colored by calibrated depth
                          (per-reach median) (last round).
plot_water_level_timeseries  Water level over time at one representative
                          cell per reach (nearest that reach's own along_m
                          midpoint) (last round).
plot_max_inundation       Max inundation depth map (last round) -- should
                          show water confined to the channel plus, in
                          correction rounds, controlled overtopping at most.
animation_flood_progress  Instantaneous depth animation (MP4, last round).
calib_root/round{i}/calibration_state.csv  Not a declared Snakemake output
                          (written directly under calib_root, one file per
                          round) -- the real per-cell state (reach_id, row,
                          col, along_m, x, y, dem, rivdph, weir_crest, zs,
                          frozen) this rule actually calibrated with that
                          round. Exists so future diagnostics can read
                          ground truth directly instead of re-deriving it
                          by replaying these formulas against saved SFINCS
                          output.
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
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as _ndimage_label
from scipy.spatial import cKDTree

from src.protection_weir import LANDUSE_SEA, GridArrays, build_coastal_protection_weir
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
)
from src.postprocessing import compute_flood_progression, compute_max_inundation
from src.river_burn import build_centerline_cells_regular, build_channel_mask_regular, build_nearest_weir_crest_regular, burn_river_channel, constrain_to_coarse_channel_mask, snap_points_to_centerline_cells
from src.river_depth_calibration import (
    build_calibration_seed_discharge,
    compute_calibrated_depth,
    compute_calibrated_weir_crest,
    compute_period_max_zs,
    gather_calibration_round_profile,
)
from src.river_network import accumulate_discharge, build_downstream_adjacency, compute_hydraulic_depth, normalize_reach_id
from src.sfincs_run import run_sfincs_subprocess

log = setup_logging(snakemake.log[0])

# ── paths & params ────────────────────────────────────────────────────────────
elevation_path       = Path(snakemake.input.elevation_conditioned)
elevation_sfincs_grid_path = Path(snakemake.input.elevation_conditioned_sfincs_grid)
river_elevation_max_path = Path(snakemake.input.river_elevation_max)
river_network_path   = Path(snakemake.input.clean_river_network)
river_forcing_path   = Path(snakemake.input.river_forcing)
protection_levels_path = Path(snakemake.input.protection_levels)
grid_resolution_path = Path(snakemake.input.grid_resolution)
land_polygons_path   = Path(snakemake.input.land_polygons)
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
weir_crest_fraction = float(snakemake.params.weir_crest_fraction)
n_correction_iterations = int(snakemake.params.n_correction_iterations)
min_crest_increment_per_round_m = float(snakemake.params.min_crest_increment_per_round_m)
# weir_freeboard_m (config) is reserved for the REAL production top-up,
# added once, uniformly, on the FINAL exported weir only (see the
# "canonical production outputs" section near the end of this script).
# The calibration loop's own internal convergence margin (every
# badly_overtopped/near_zero threshold and every snap target below) uses
# the SEPARATE, fixed _CALIBRATION_FREEBOARD_M instead -- deliberately NOT
# config-driven, since it's an algorithmic convergence tolerance, not a
# production design choice: changing config's freeboard_m should change
# how much margin ends up in the built model, not how aggressively the
# iterative solver converges. 2026-08-05: previously the config value drove
# BOTH roles, which meant the final weir never actually reflected it at all
# (the final trace always passed freeboard_m=0.0, since it assumed the
# margin was already baked into the tracked crest arrays by this same
# config value) -- config's freeboard_m was silently a no-op in production.
weir_freeboard_m = float(snakemake.params.weir_freeboard_m)
_CALIBRATION_FREEBOARD_M = 0.5
# river_crest_dilation_cells_min: FLOOR only -- the actual value used is
# computed per basin (see below, once rivers_utm/grid are available) as
# max(this floor, width / (2 * coarse_resolution)), tying the dilation
# radius to a physically meaningful scale (a wide river's calibrated crest
# needs to reach further onto adjacent land than a narrow one does)
# instead of a single arbitrary constant for every basin.
river_crest_dilation_cells_min = int(snakemake.params.river_crest_dilation_cells)
min_component_cells = int(snakemake.params.min_component_cells)
include_subgrid    = bool(snakemake.params.include_subgrid)
nr_subgrid_pixels  = int(snakemake.params.nr_subgrid_pixels)
nr_levels          = int(snakemake.params.nr_levels)
nrmax              = int(snakemake.params.nrmax)
animation_fps      = int(snakemake.params.animation_fps)
active_mask_enabled = snakemake.params.active_mask_enabled
active_mask_elevation_buffer_m = float(snakemake.params.active_mask_elevation_buffer_m)
outflow_buffer_m = float(snakemake.params.outflow_buffer_m)

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
# conditions. The correction rounds below are meant to reflect the real,
# coupled system, so this is applied uniformly to every round, including
# round 0, rather than isolating the river from coastal influence.
#
# coastal_protection_crest_m: the REAL production coastal crest (same
# surge_forcing.nc field rule 13 reads) -- used as the "elsewhere" (non-
# river) weir floor for CORRECTION rounds only (round_idx >= 1), replacing
# weir_crest_m/1000 there so the whole weir -- not just the river-covered
# segments -- reflects what production will actually build. Round 0 keeps
# the uniform 1000 m confinement everywhere: it's still the deliberately
# isolated, fully-confined baseline measurement, not a production replica.
with xr.open_dataset(surge_forcing_path, decode_times=False) as _surge_ds:
    baseline_m = float(_surge_ds["baseline_m"].values) if "baseline_m" in _surge_ds else 0.0
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

# Native resolution/CRS reference for correction rounds' own native-resolution
# burn (subgrid's source) -- same role elevation_merged plays for rule 11b.
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
# either, so calibration would just wall it off and let its own crest
# climb without bound, chasing water that in reality/production simply
# leaves the domain there.
delta_outflow_gdf = gpd.read_file(delta_outflow_points_path)
delta_outflow_enabled = not delta_outflow_gdf.empty
log.info(f"Delta-outline outflow points: {len(delta_outflow_gdf)}")

# Correction-round burned rasters (native resolution, for subgrid) get their
# own catalog entry per round, registered upfront -- the FILES don't exist
# yet at this point (written inside the loop below, before each round's own
# subgrid.create() call reads them), but the catalog only needs the URI
# declared once, at model construction time, matching every other script in
# this codebase (none of them add catalog sources mid-run). The main "dep"
# grid's own excavation is patched directly into sf.grid.data["dep"]'s numpy
# array instead (see the loop below) -- no catalog entry needed for that.
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
    "local_land_polygons": {
        "data_type": "GeoDataFrame",
        "uri": str(land_polygons_path),
        "driver": "pyogrio",
    },
    **({
        "local_delta_outflow_points": {
            "data_type": "GeoDataFrame",
            "uri": str(delta_outflow_points_path),
            "driver": "pyogrio",
        },
    } if delta_outflow_enabled else {}),
    **({
        f"local_river_burned_round{i}_native": {
            "data_type": "RasterDataset",
            "uri": str(calib_root / f"river_burned_round{i}_native.tif"),
            "driver": "rasterio",
        }
        for i in range(1, n_correction_iterations + 1)
    }),
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

land_polygons_empty = gpd.read_file(land_polygons_path).empty
boundary_kwargs = {} if land_polygons_empty else {"exclude_polygon": "local_land_polygons"}
sf.mask.create_boundary(btype="waterlevel", reset_bounds=True, **boundary_kwargs)
log.info("Waterlevel boundary set: edge cells not on land -> mask=2")

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
# shape/transform) -- the SAME call calibration always made, now also reused
# to constrain correction rounds' own excavation (see the loop below), and
# identical to what rule 11b/13 independently compute for production.
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

# river_crest_dilation_cells: scaled by this basin's own widest reach
# relative to the coarse grid resolution, floored at the configured
# minimum -- see river_crest_dilation_cells_min's own comment above.
_max_river_width_m = float(rivers_utm["width"].max()) if rivers_utm["width"].notna().any() else 0.0
river_crest_dilation_cells = max(
    river_crest_dilation_cells_min, int(_max_river_width_m / (2 * grid.cell_size_m))
)
log.info(
    f"river_crest_dilation_cells = {river_crest_dilation_cells} "
    f"(max reach width={_max_river_width_m:.0f} m, coarse resolution={grid.cell_size_m:.1f} m, "
    f"floor={river_crest_dilation_cells_min})"
)

# ── coastal probe cells (ocean cells near land, driving a separate coastal
# weir correction) ───────────────────────────────────────────────────────────
# A river's own backwater can raise the water level right at the coast
# above baseline_m -- close enough to the river's own dilated crest
# coverage to be masked by it, but
# just beyond that radius the flat coastal_protection_crest_m floor can be
# lower than this locally-elevated level, letting land there flood even
# though the river's own crest, a few cells further, is correctly raised.
# Rather than guessing how far the river's influence reaches (a dilation
# radius or nearest-bank search), this probes the REAL simulated water
# level directly at nearby ocean cells and corrects the coastal "elsewhere"
# floor locally wherever the data says so -- restricted to ocean_mask
# cells specifically (never river_channel_mask), so the correction can
# only ever be driven by genuinely coastal water, never creep up the river
# (that stays entirely the river crest's own job, see the round loop below).
ocean_mask_grid = landuse_on_grid == LANDUSE_SEA

# Coarse-grid ocean mask FILE (compute_max_inundation/compute_flood_
# progression need a path, not an in-memory array -- they call
# data_catalog.get_rasterdataset internally) -- used by the round loop's own
# per-round diagnostics below instead of native-resolution sea_mask.tif, so
# every sea/land check in this rule shares the same single coarse-grid
# source (no protected_pocket_mask yet at this point -- the weir isn't
# final until after the round loop -- but round diagnostics never claimed
# weir-aware precision either; ocean_mask_grid alone is the same "is this
# genuinely open sea" classification the old native sea_mask.tif gave).
_round_ocean_mask_path = calib_root / "ocean_mask_on_grid.tif"
with rasterio.open(
    _round_ocean_mask_path, "w", driver="GTiff", dtype="float32", count=1,
    height=ocean_mask_grid.shape[0], width=ocean_mask_grid.shape[1],
    crs=grid.crs, transform=grid.transform, nodata=-9999.0,
) as _dst:
    _dst.write(np.where(ocean_mask_grid, np.float32(1.0), np.float32(-9999.0)), 1)

_near_land_ocean = (
    ocean_mask_grid & grid.valid_mask & binary_dilation(~ocean_mask_grid, iterations=river_crest_dilation_cells)
)
coastal_probe_rows, coastal_probe_cols = np.where(_near_land_ocean)
coastal_probe_x, coastal_probe_y = rasterio.transform.xy(grid.transform, coastal_probe_rows, coastal_probe_cols)
coastal_probe_x = np.asarray(coastal_probe_x)
coastal_probe_y = np.asarray(coastal_probe_y)
log.info(f"Coastal probe cells (ocean, within {river_crest_dilation_cells} cell(s) of land): {len(coastal_probe_rows)}")

# region_wgs84/buf_deg: reused below by both the observation-point and
# discharge-point sections. hydromt's *_points.create() methods clip
# locations against the model's own region using an UNBUFFERED 'intersects'
# check and silently drop anything outside -- snap_points_into_region keeps
# every point by nudging boundary-adjacent ones inward (see its own
# docstring for why a plain buffered filter isn't enough).
region_wgs84 = sf.region.to_crs("EPSG:4326").geometry.union_all()
region_utm = sf.region.geometry.union_all()
buf_deg = float(resolution) / 111_000.0

# ── centerline calibration cells (every grid cell each reach's own centerline
# passes through) -- replaces SFINCS "observation points" entirely ──────────
# The real water level varies continuously along a reach -- a single
# sparse observation point can completely miss a sharp local spike right
# at the seed reach's own discharge-injection cell -- so
# calibration tracks depth/crest per CELL instead, sourced directly from
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
# Every centerline cell's own "zs" (used everywhere depth/crest calibration
# reads a water level, both round 0's own split and every correction
# round's own update -- see _read_cross_section_period_max_zs below) is now
# the MAX water level across the FULL cross-section of channel_mask cells
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
# cell_gdf's (row, col) -- resolved lazily, once, inside _run_calibration_round
# below, from round 0's own freshly written sfincs_map.nc (grid geometry is
# round-invariant, so it's reused unchanged for every later round).
map_cell_idx = None
cross_section_map_cell_idx = None
coastal_map_cell_idx = None
river_boundary_map_cell_idx = None
mouth_ocean_map_cell_idx = None

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

# ── river boundary probe cells (LAND cells directly bordering the channel,
# driving a separate river-boundary weir correction) ─────────────────────────
# For each such land cell, the water level compared against ITS OWN crest is
# sampled at its own NEAREST channel_mask cell -- "look across the dike,
# what's the water level on the other side" -- not the land cell's own
# reading. A land cell's own period_max_zs is either NaN (never wetted at
# all) or, once it DOES flood, reflects water that arrived by spreading
# from wherever the real breach is, not the local water level THIS
# specific stretch of dike needs to hold back -- neither is the right
# signal to correct THIS segment's own crest with. Sampling the adjacent
# water cell directly mirrors the coastal probe's own convention (which
# already probes the WATER side, ocean cells near land) and needs no
# dilation radius at all: placed at every IMMEDIATE (1-cell) land/channel
# neighbour, the corrected value still reaches every cell further out via
# _rasterize_nearest's own nearest-probe lookup, same as before.
#
# ALSO covers a 5-cell buffer around every discharge injection point
# explicitly, regardless of dike adjacency: a seed's own discharge point
# can land in a single grid cell boxed in by weir segments on both sides
# (e.g. when the reach's true width is truncated to one cell by the domain
# edge), forcing the full discharge through a genuine hydraulic bottleneck
# with a much higher local water level than anywhere else in the reach. The
# immediate-neighbour rule above already catches the segments directly
# bounding that cell, but the buffer adds robustness for cells a little
# further out that are still clearly within the injection's own zone of
# influence.
_dike_adjacent_land = ~channel_mask & grid.valid_mask & binary_dilation(channel_mask, iterations=1)

crossings_utm = crossings_filt.to_crs(grid.crs)
inv_transform = ~grid.transform
_discharge_rc = np.array([inv_transform * (pt.x, pt.y) for pt in crossings_utm.geometry])
_discharge_cols, _discharge_rows = _discharge_rc[:, 0], _discharge_rc[:, 1]
_all_rows, _all_cols = np.indices(grid.shape)
_discharge_buffer_cells = 5.0
_near_discharge = np.zeros(grid.shape, dtype=bool)
for _dr, _dc in zip(_discharge_rows, _discharge_cols):
    _near_discharge |= np.hypot(_all_rows - _dr, _all_cols - _dc) <= _discharge_buffer_cells
_discharge_buffer_land = _near_discharge & ~channel_mask & grid.valid_mask

_river_boundary_probe_mask = _dike_adjacent_land | _discharge_buffer_land
river_boundary_probe_rows, river_boundary_probe_cols = np.where(_river_boundary_probe_mask)
river_boundary_probe_x, river_boundary_probe_y = rasterio.transform.xy(
    grid.transform, river_boundary_probe_rows, river_boundary_probe_cols
)
river_boundary_probe_x = np.asarray(river_boundary_probe_x)
river_boundary_probe_y = np.asarray(river_boundary_probe_y)

# Nearest channel_mask cell for each probe -- the actual water-level
# SAMPLING location for an ORDINARY (non-discharge) probe (see docstring
# above); the probe's own row/col above remains where the CORRECTED crest
# value gets rasterized back onto the grid, since that's the land-side
# position crest_surface is looked up at.
_channel_rows_all, _channel_cols_all = np.where(channel_mask & grid.valid_mask)
_channel_tree = cKDTree(np.column_stack([_channel_rows_all, _channel_cols_all]))
_dist_to_channel, _nearest_channel_idx = _channel_tree.query(
    np.column_stack([river_boundary_probe_rows, river_boundary_probe_cols])
)
channel_cell_x, channel_cell_y = rasterio.transform.xy(grid.transform, _channel_rows_all, _channel_cols_all)
channel_cell_x = np.asarray(channel_cell_x)
channel_cell_y = np.asarray(channel_cell_y)

# Discharge-buffer probes (river_boundary_probe cells inside
# _discharge_buffer_land) sample the WORST (max) water level across every
# channel cell within that SAME discharge-buffer radius of their own
# nearest discharge point, instead of a single nearest-channel-cell
# reading: the discharge injection cell's own water level is not
# necessarily the local peak (a narrow/bottlenecked head can push the true
# peak a cell or two away, see this module's own docstring for the known
# "sharp local spike" caveat) -- sampling the whole neighbourhood's own
# worst case is a safer basis for the crest there. Ordinary dike-adjacent
# probes elsewhere (not near any discharge point) keep the single
# nearest-channel-cell reading above, unchanged.
_is_discharge_probe = _discharge_buffer_land[river_boundary_probe_rows, river_boundary_probe_cols]
_discharge_local_channel_idx: list[np.ndarray] = []
for _dr, _dc in zip(_discharge_rows, _discharge_cols):
    _d = np.hypot(_channel_rows_all - _dr, _channel_cols_all - _dc)
    _discharge_local_channel_idx.append(np.flatnonzero(_d <= _discharge_buffer_cells))
if _is_discharge_probe.any():
    _discharge_tree = cKDTree(np.column_stack([_discharge_rows, _discharge_cols]))
    _, _nearest_discharge_idx = _discharge_tree.query(
        np.column_stack([
            river_boundary_probe_rows[_is_discharge_probe], river_boundary_probe_cols[_is_discharge_probe],
        ])
    )
else:
    _nearest_discharge_idx = np.zeros(0, dtype=int)
log.info(
    f"River boundary probe cells (land, directly bordering channel_mask or within "
    f"{_discharge_buffer_cells:.0f} cell(s) of a discharge point): {len(river_boundary_probe_rows)} "
    f"({int(_is_discharge_probe.sum())} sampling their own discharge-radius max, "
    f"{int((~_is_discharge_probe).sum())} sampling their own nearest channel cell)"
)

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
    global-cache-var init for the full rationale. This is the value that
    drives every depth/crest calibration decision (round 0's split, every
    correction round's own update, the final snap); _read_zs_and_resolve
    (single intersected cell) is kept unchanged alongside this, only for
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


def _read_coastal_probe_period_max_zs(round_root: Path) -> np.ndarray:
    """Same pattern as _read_zs_and_resolve, for the coastal probe cells
    (ocean, near land) instead of the river's own centerline cells --
    empty array if there are no probe cells at all (e.g. a fully inland
    basin with no ocean in the domain)."""
    global coastal_map_cell_idx
    if len(coastal_probe_rows) == 0:
        return np.zeros(0, dtype=np.float32)
    if coastal_map_cell_idx is None:
        coastal_map_cell_idx = _resolve_map_cell_idx(round_root / "sfincs_map.nc", coastal_probe_x, coastal_probe_y)
        log.info(f"Resolved {len(coastal_map_cell_idx[0])} coastal probe cell(s) against sfincs_map.nc's own grid indices")
    zs, _times_s = _read_map_zs_at_cells(round_root, *coastal_map_cell_idx)
    return compute_period_max_zs(zs)


def _read_mouth_ocean_period_max_zs(round_root: Path) -> np.ndarray:
    """Same pattern as _read_coastal_probe_period_max_zs, one probe per
    mouth reach -- the nearest OCEAN-classified cell (mouth_ocean_rows/
    cols/x/y) to that mouth's own last centerline cell, not the river-side
    channel cell itself. Empty array if there is no mouth reach at all."""
    global mouth_ocean_map_cell_idx
    if len(mouth_ocean_rows) == 0:
        return np.zeros(0, dtype=np.float32)
    if mouth_ocean_map_cell_idx is None:
        mouth_ocean_map_cell_idx = _resolve_map_cell_idx(round_root / "sfincs_map.nc", mouth_ocean_x, mouth_ocean_y)
        log.info(f"Resolved {len(mouth_ocean_map_cell_idx[0])} mouth ocean probe cell(s) against sfincs_map.nc's own grid indices")
    zs, _times_s = _read_map_zs_at_cells(round_root, *mouth_ocean_map_cell_idx)
    return compute_period_max_zs(zs)


def _mouth_crest_target(round_root: Path) -> np.ndarray:
    """The mouth's own last centerline cell's CREST target, every round
    including round 0 and the final snap -- max(the real, seaward water
    level there + _CALIBRATION_FREEBOARD_M, the real coastal protection standard).
    Deliberately NOT the mouth cell's own bed elevation (natural
    bathymetry stays a separate, unrelated BED/depth concern, hard-forced
    elsewhere) -- a coastal outlet still needs an actual protective crest
    sized against how high the water there actually gets, sampled on the
    OPEN-WATER side of the coast/river transition (mouth_ocean_rows/cols),
    not the river-side channel cell's own reading, which sits right at
    that transition and isn't necessarily representative of the water it's
    actually confining against."""
    if len(mouth_ocean_rows) == 0:
        return np.zeros(0, dtype=np.float32)
    mouth_ocean_period_max_zs = _read_mouth_ocean_period_max_zs(round_root)
    return np.maximum(mouth_ocean_period_max_zs + _CALIBRATION_FREEBOARD_M, coastal_protection_crest_m)


def _read_river_boundary_probe_period_max_zs(round_root: Path) -> np.ndarray:
    """Per-probe period-max water level, resolved once against the FULL
    channel-cell set (channel_cell_x/y) rather than one point per probe.
    An ORDINARY (non-discharge) probe reads its own single nearest channel
    cell (_nearest_channel_idx, "look across the dike, what's the water
    level on the other side"). A DISCHARGE-buffer probe instead reads the
    MAX water level across every channel cell within its own nearest
    discharge point's own buffer radius (_discharge_local_channel_idx) --
    see the probe setup's own docstring for why. Falls back to the
    ordinary nearest-cell reading if that local neighbourhood is somehow
    empty (a discharge point with no channel cell within its own radius,
    not expected in practice but not a reason to crash). Empty array if
    there are no probe cells at all."""
    global river_boundary_map_cell_idx
    if len(river_boundary_probe_rows) == 0:
        return np.zeros(0, dtype=np.float32)
    if river_boundary_map_cell_idx is None:
        river_boundary_map_cell_idx = _resolve_map_cell_idx(round_root / "sfincs_map.nc", channel_cell_x, channel_cell_y)
        log.info(
            f"Resolved {len(river_boundary_map_cell_idx[0])} channel cell(s) (river boundary probe "
            f"sampling) against sfincs_map.nc's own grid indices"
        )
    zs, _times_s = _read_map_zs_at_cells(round_root, *river_boundary_map_cell_idx)
    channel_period_max_zs = compute_period_max_zs(zs)
    result = np.empty(len(river_boundary_probe_rows), dtype=np.float32)
    result[~_is_discharge_probe] = channel_period_max_zs[_nearest_channel_idx[~_is_discharge_probe]]
    for _i, _pos in enumerate(np.flatnonzero(_is_discharge_probe)):
        _di = _nearest_discharge_idx[_i]
        _local_idx = _discharge_local_channel_idx[_di]
        result[_pos] = (
            np.nanmax(channel_period_max_zs[_local_idx]) if len(_local_idx)
            else channel_period_max_zs[_nearest_channel_idx[_pos]]
        )
    return result


def _rasterize_nearest(
    rows: np.ndarray, cols: np.ndarray, values: np.ndarray, out_shape: tuple[int, int],
    max_distance_cells: float,
) -> np.ndarray:
    """Full-grid raster where every cell within max_distance_cells of its
    nearest (rows, cols) point takes that point's own value -- turns the
    probes' own discrete, per-cell crest values into a continuous surface
    covering surrounding land, the same role build_smoothed_weir_crest_regular's
    own dilation plays for the river crest, but following the actual
    ocean-cell/channel layout (and therefore the real coastline shape, bays
    included) instead of a fixed search radius.

    max_distance_cells is a hard requirement, not a tuning knob to loosen:
    without it, a single probe's own genuinely local reading (e.g. a real
    hydraulic bottleneck right at a discharge injection point) propagates
    to EVERY cell in the entire domain, however far away, since plain
    nearest-neighbour has no notion of "too far to be relevant" -- cells
    beyond the cutoff get NaN here instead, which
    build_coastal_protection_weir's own crest_surface already falls back to
    the flat crest_elevation_m floor for (see its own docstring), exactly
    matching how river_crest_on_grid's bounded grid.dilate_values(...)
    behaves for the river crest."""
    all_rows, all_cols = np.indices(out_shape)
    tree = cKDTree(np.column_stack([rows, cols]))
    dist, idx = tree.query(np.column_stack([all_rows.ravel(), all_cols.ravel()]))
    out = values[idx].astype(np.float32)
    out[dist > max_distance_cells] = np.nan
    return out.reshape(out_shape)


def _gap_contained(gap: np.ndarray | None) -> bool:
    """True if every entry of `gap` (crest minus period-max water level) is
    comfortably contained (> 0). None (no probe cells in this set at all,
    e.g. no ocean in the domain) is trivially contained.

    Deliberately NOT `bool(np.all(gap > 0))`: a probe cell never wetted at
    all during this round's own run has period_max_zs == NaN (see
    compute_period_max_zs's own docstring), making gap NaN too --
    `NaN > 0` is False, so a plain np.all(gap > 0) would report that cell
    as "not contained" forever (blocking convergence indefinitely), even
    though the round's own update formula (badly_overtopped/near_zero, both
    of which also evaluate False for NaN) already correctly finds nothing
    to correct there -- a never-wetted cell (too high to ever flood) is not
    a problem. `np.any(gap <= 0)` also evaluates False for NaN, so a NaN
    entry is correctly treated as "nothing to worry about" instead of a
    permanent failure."""
    return gap is None or not bool(np.any(gap <= 0))


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
    # representative_cell_pos indexing -- a visualization concern. Every
    # calibration decision (round 0's split, every correction round's own
    # update, the final snap) instead uses period_max_zs from
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
# diagnostic plots below -- computed once, before the loop.
wgs84_bounds, domain_crs, domain_poly = load_domain(snakemake.input.spec_basins_meta, domain_gpkg_path)

# Per-round diagnostic files are written directly here (NOT declared
# Snakemake outputs, same convention as calibration_state.csv) -- a round
# skipped by early-stopping gets NO file at all rather than an empty
# placeholder, since Snakemake would otherwise require every one of them
# to exist. Filenames put the round number as a SUFFIX (e.g.
# "max_inundation_round4.png"), not a prefix, so a directory listing
# sorted alphabetically groups by PLOT KIND first -- comparing the same
# diagnostic across rounds means looking at consecutive files, not
# picking them out of an interleaved by-round listing.
round_visuals_dir = Path(snakemake.output.plot_calibration).parent / "rounds"
round_visuals_dir.mkdir(parents=True, exist_ok=True)


def _round_visual_paths(round_idx: int) -> dict[str, Path]:
    return {
        "plot_calibration": round_visuals_dir / f"river_depth_round{round_idx}.png",
        "plot_water_level": round_visuals_dir / f"water_level_timeseries_round{round_idx}.png",
        "plot_max_inundation": round_visuals_dir / f"max_inundation_round{round_idx}.png",
        "animation": round_visuals_dir / f"flood_animation_round{round_idx}.mp4",
        "crest_gap_map": round_visuals_dir / f"crest_gap_map_round{round_idx}.png",
    }

# ── round 0 + N correction rounds ─────────────────────────────────────────────
# Round 0: isolated confinement (unchanged from the original single-pass
# design). Rounds 1..n_correction_iterations: current-best excavation + real
# per-reach crest, correcting for the backwater coupling round 0 alone can't
# see (see this module's own docstring).
rivdph_current = None
weir_crest_current = None
weir_crest_simulated = None
mouth_natural_bathymetry = None
is_last_mouth_cell = None
mouth_ocean_rows = np.zeros(0, dtype=int)
mouth_ocean_cols = np.zeros(0, dtype=int)
mouth_ocean_x = np.zeros(0, dtype=float)
mouth_ocean_y = np.zeros(0, dtype=float)
gap = None
converged_round_idx = None
crest_before_snap = None
snap_attempted = False
snap_reverted = False
coastal_crest_current = None
coastal_gap = None
coastal_crest_before_snap = None
river_boundary_crest_current = None
river_boundary_gap = None
river_boundary_crest_before_snap = None

for round_idx in range(n_correction_iterations + 1):
    round_label = "round 0 (isolated)" if round_idx == 0 else f"correction round {round_idx}/{n_correction_iterations}"

    # Each round gets its OWN output subdirectory (sfincs.inp/sfincs_map.nc/
    # sfincs_his.nc/subgrid/...) instead of all rounds overwriting the same
    # filenames in calib_root: reusing the same filenames across rounds
    # makes SFINCS itself fail creating its own output netCDF files
    # ("Permission denied"/"NetCDF: Not a valid ID") once a PREVIOUS round's
    # own diagnostic plotting (compute_max_inundation/
    # compute_flood_progression) had read them, since some file handle
    # persists past that read on Windows regardless of Python-side
    # cache-clearing. Redirecting root sidesteps the issue entirely instead
    # of chasing exactly which cache/library was holding it open. The model
    # itself (grid/mask/roughness/weir/obs points/discharge points/config)
    # is all in-memory on the shared `sf` object -- changing root only
    # affects where the NEXT sf.write()/sf.subgrid.create() call lands.
    round_root = calib_root / f"round{round_idx}"
    round_root.mkdir(parents=True, exist_ok=True)
    sf.root.set(round_root)

    if round_idx == 0:
        river_crest_on_grid = np.where(channel_mask, np.float32(weir_crest_m), np.float32(np.nan))
    else:
        # Snapshot the crest that is ABOUT TO BE painted/simulated THIS
        # round -- i.e. whatever the PREVIOUS round's own update left it at
        # -- before it gets reassigned further down by THIS round's own
        # freeboard update. calibration_state.csv reports this (not the
        # post-update value) as "weir_crest", so the per-cell diagnostics
        # show the crest that actually produced this round's own period-max
        # water level/flooding, not next round's not-yet-simulated target --
        # the post-update value would make it look like the crest already
        # exceeded zs everywhere while the SAME round's own animation/
        # max-inundation map (built from the actually-simulated, pre-update
        # crest) still shows real flooding.
        weir_crest_simulated = weir_crest_current.copy()

        # zbed_anchors: one anchor per centerline cell, built directly from
        # this round's own per-cell calibration state -- rivbed(cell) =
        # DEM_at_cell - rivdph_current[cell], a constant DEPTH offset from
        # the real, locally-varying terrain at the cell's own location, no
        # DEM re-sampling or reach-level filtering needed. Mouth reaches
        # need no special case here: their own last (coastal) cell was
        # already hard-forced to natural bathymetry (rivdph=0) in the
        # round-0 mouth block below and re-forced at the end of this same
        # branch, so it flows through burn_river_channel's own per-reach
        # anchor interpolation like any other cell, giving the mouth's
        # downstream end a real anchor at natural bathymetry automatically.
        zbed_anchors = cell_gdf.assign(rivbed=dem_at_cell - rivdph_current)

        # SFINCS-grid resolution: constrained to channel_mask (same shared
        # mask used for the weir corridor), so excavated cells == the
        # corridor's own cells, exactly -- same invariant established this
        # session for production (rule 11b/13). Patched directly into
        # sf.grid.data["dep"]'s own array rather than routed through the
        # data catalog/elevation_list mechanism -- simpler than
        # pre-registering per-round catalog entries for the main grid, and
        # sf.write() below writes whatever ends up in sf.grid.data["dep"]
        # regardless of how it got there. natural_dem_path floors the burn
        # against the real conditioned DEM (never shallower than it) -- see
        # burn_river_channel's own docstring for why this is needed even
        # with terrain-following anchors (interpolation/junction-blending
        # between anchors can still locally overshoot).
        burned_arr_sfincs, _t, _nd, _stats = burn_river_channel(
            rivers=rivers_utm, zbed_anchors=zbed_anchors,
            natural_dem_path=elevation_path,
            utm_crs=sf.crs, resolution_m=grid.cell_size_m,
            out_transform=grid.transform, out_shape=grid.shape,
            channel_mask=channel_mask,
        )
        valid_burn = np.isfinite(burned_arr_sfincs)
        sf.grid.data["dep"].values[valid_burn] = burned_arr_sfincs[valid_burn]
        log.info(
            f"[{round_label}] Excavated {int(valid_burn.sum()):,} cell(s) into the main grid "
            f"({_stats['n_reaches_burned']} reach(es) burned, {_stats['n_reaches_skipped']} skipped)"
        )

        if include_subgrid:
            # Burn DIRECTLY onto the subgrid's own phase-locked fine-pixel
            # grid (coarse grid transform subdivided by nr_subgrid_pixels)
            # instead of an independently-bounded native-resolution grid --
            # otherwise hydromt_sfincs's own subgrid.create() has to
            # reproject/resample this file onto its internal fine grid
            # itself (different resolution AND phase/origin), and THAT
            # resampling step introduces new leakage across the
            # channel_mask boundary even when the source burn file itself
            # has 0 pixels outside channel_mask -- hydromt's own
            # dep_subgrid.tif can still show excavated pixels bleeding into
            # a coarse cell channel_mask excludes, and since SFINCS reports
            # a subgrid cell's own "zb" as (effectively) the MINIMUM
            # sub-pixel elevation within it, even a handful of leaked
            # sub-pixels at one corner makes the WHOLE coarse cell read as
            # deeply excavated relative to its own true (unexcavated)
            # surrounding terrain. Burning directly at the subgrid's own
            # resolution/phase eliminates the resampling step (and its
            # leak) entirely, rather than trying to out-guess hydromt's own
            # internal resampling afterward.
            subgrid_transform = grid.transform * Affine.scale(1.0 / nr_subgrid_pixels)
            subgrid_shape = (grid.shape[0] * nr_subgrid_pixels, grid.shape[1] * nr_subgrid_pixels)
            burned_arr_native, transform_native, _nd_native, stats_native = burn_river_channel(
                rivers=rivers_utm, zbed_anchors=zbed_anchors,
                natural_dem_path=elevation_path,
                utm_crs=sf.crs, resolution_m=grid.cell_size_m / nr_subgrid_pixels,
                out_transform=subgrid_transform, out_shape=subgrid_shape,
            )
            # Belt-and-braces: still constrain to channel_mask even though
            # the burn is now phase-locked -- a reach's own buf_poly can
            # still rasterize a handful of edge pixels slightly differently
            # than channel_mask's own independent rasterization (same
            # reason the coarse burn always passes channel_mask=... too).
            burned_arr_native = constrain_to_coarse_channel_mask(
                burned_arr_native, transform_native, sf.crs, channel_mask, grid.transform,
            )
            NODATA = np.float32(-9999.0)
            native_out_path = calib_root / f"river_burned_round{round_idx}_native.tif"
            with rasterio.open(
                native_out_path, "w", driver="GTiff", dtype="float32",
                width=burned_arr_native.shape[1], height=burned_arr_native.shape[0],
                count=1, crs=sf.crs, transform=transform_native,
                nodata=float(NODATA), compress="deflate", tiled=True,
            ) as dst:
                dst.write(np.where(np.isnan(burned_arr_native), NODATA, burned_arr_native).astype(np.float32), 1)
            log.info(
                f"[{round_label}] Native-resolution burn (subgrid source): "
                f"{stats_native['n_reaches_burned']} reach(es) burned, "
                f"{stats_native['n_pixels_burned']:,} pixel(s)"
            )

        # Direct per-cell nearest-anchor painting (build_nearest_weir_crest_regular)
        # instead of the interpolated/smoothed profile (build_smoothed_weir_crest_regular)
        # -- the calibration loop's own per-cell targets (weir_crest_current)
        # are already exact requirements (zs_max(cross-section) + freeboard/
        # increment), so interpolating BETWEEN them would reintroduce the
        # same under/over-shoot smoothing was meant to avoid. See that
        # function's own docstring for why it's a separate function rather
        # than a mode switch on build_smoothed_weir_crest_regular.
        river_crest_on_grid = build_nearest_weir_crest_regular(
            rivers_utm, "width", grid.shape, grid.transform,
            cell_gdf=cell_gdf, crest_values=weir_crest_current,
        )

    # Round 0: uniform 1000 m confinement everywhere (the deliberately
    # isolated baseline). Correction rounds: the REAL production coastal
    # crest as the "elsewhere" floor too, not just for river-covered cells --
    # otherwise the weir stays a hybrid of real (river) and artificial
    # (coastal) crests instead of a faithful production replica, which is
    # the whole point of these rounds.
    round_crest_elevation_m = weir_crest_m if round_idx == 0 else coastal_protection_crest_m
    # freeboard_m=0.0 always: this is the CALIBRATION's own per-round
    # simulation weir, not the final production output (see the
    # "canonical production outputs" section near the end of this script,
    # which always re-traces the actually-exported weir separately with
    # the real, config-driven weir_freeboard_m). weir_crest_current is used
    # AS-IS to run the model here, with no separate freeboard added on top
    # of it during simulation -- correction rounds' own ensure-minimum-
    # freeboard update (see the round-0/correction-round branch below)
    # already guarantees _CALIBRATION_FREEBOARD_M of margin directly in the
    # tracked crest value itself, so adding it again here would double-
    # count it for purposes of the iterative convergence check.
    #
    # coastal_crest_on_grid: round 0 has no per-probe tracking at all (its
    # own uniform weir_crest_m confinement already covers near-coast land
    # too). First correction round starts it flat at coastal_protection_crest_m
    # (identical to the scalar fallback, so behaviourally unchanged from
    # today) -- from the round after that, it carries whatever the coastal
    # probe correction below computed from the PREVIOUS round's own
    # simulated water level.
    coastal_crest_on_grid = None
    if round_idx > 0 and len(coastal_probe_rows) > 0:
        if coastal_crest_current is None:
            coastal_crest_current = np.full(len(coastal_probe_rows), coastal_protection_crest_m, dtype=np.float32)
        coastal_crest_on_grid = _rasterize_nearest(
            coastal_probe_rows, coastal_probe_cols, coastal_crest_current, landuse_on_grid.shape,
            max_distance_cells=river_crest_dilation_cells,
        )
    # river_boundary_crest_on_grid: same role as coastal_crest_on_grid, but
    # for LAND cells near the channel (see river boundary probe cells'
    # own setup comment above) -- folded into the SAME "elsewhere" floor via
    # np.maximum, since build_coastal_protection_weir's own
    # coastal_crest_on_grid parameter already means exactly this ("optional
    # per-cell elsewhere-crest override, combined via max, never lowers
    # protection"), so no changes are needed there at all.
    if round_idx > 0 and len(river_boundary_probe_rows) > 0:
        if river_boundary_crest_current is None:
            river_boundary_crest_current = np.full(
                len(river_boundary_probe_rows), coastal_protection_crest_m, dtype=np.float32
            )
        river_boundary_crest_on_grid = _rasterize_nearest(
            river_boundary_probe_rows, river_boundary_probe_cols, river_boundary_crest_current, landuse_on_grid.shape,
            max_distance_cells=river_crest_dilation_cells,
        )
        coastal_crest_on_grid = (
            river_boundary_crest_on_grid if coastal_crest_on_grid is None
            else np.maximum(coastal_crest_on_grid, river_boundary_crest_on_grid)
        )
    weir_gdf, weir_diagnostics = build_coastal_protection_weir(
        grid, landuse_on_grid, channel_mask, round_crest_elevation_m,
        min_component_cells, weir_par1, river_crest_on_grid=river_crest_on_grid,
        coastal_crest_on_grid=coastal_crest_on_grid,
        freeboard_m=0.0,
        # build_channel_mask_regular already uses all_touched=True (no
        # sub-cell gaps to patch), so skip the safety-net close_gaps()
        # dilation to keep the weir hugging the channel corridor exactly --
        # rule 13's production build passes the same flag for the same
        # reason (see build_coastal_protection_weir's channel_mask_gap_free
        # docstring), so calibration and production trace the weir
        # identically.
        channel_mask_gap_free=True,
        # Wider than the 1-cell default to close a "salt-and-
        # pepper" gap at reach junctions (e.g. near a river mouth) -- see
        # build_coastal_protection_weir's own docstring. Harmless for round
        # 0's own uniform confinement value (dilation radius doesn't matter
        # for a spatially-constant value), so applied identically every
        # round rather than only for corrections.
        river_crest_dilation_cells=river_crest_dilation_cells,
    )

    if weir_diagnostics.get("applicable", True) and not weir_gdf.empty:
        sf.weirs.set(weir_gdf, merge=False)
        log.info(
            f"[{round_label}] Coastal + channel weir set: {len(weir_gdf)} segment(s), "
            f"elsewhere crest={round_crest_elevation_m:+.2f} m (no separate freeboard added -- "
            f"already baked into weir_crest_current, see round-loop comment)"
        )
    else:
        log.warning(f"[{round_label}] Coastal protection weir: not applicable -- calibration will not be confined")

    if include_subgrid:
        if round_idx == 0:
            subgrid_elevation_list = [{"elevation": "local_elevation_conditioned"}]
        else:
            subgrid_elevation_list = [
                {"elevation": f"local_river_burned_round{round_idx}_native"},
                {"elevation": "local_elevation_conditioned"},
            ]
        sf.subgrid.create(
            elevation_list=subgrid_elevation_list,
            roughness_list=[{"manning": "local_roughness_native"}],
            river_list=[],
            nr_subgrid_pixels=nr_subgrid_pixels,
            nr_levels=nr_levels,
            write_dep_tif=True,
            write_man_tif=True,
            nrmax=nrmax,
        )
        log.info(f"[{round_label}] Subgrid table created: {nr_subgrid_pixels} px/cell")

    period_max_zs, final_zs, final_times_s = _run_calibration_round(round_label, round_root)

    if round_idx == 0:
        rivdph_current = compute_calibrated_depth(period_max_zs, dem_at_cell, weir_crest_fraction)
        weir_crest_current = compute_calibrated_weir_crest(period_max_zs, dem_at_cell, weir_crest_fraction)
        log.info(
            f"[{round_label}] Calibrated depth: min={np.nanmin(rivdph_current):.3f} m, "
            f"max={np.nanmax(rivdph_current):.3f} m, median={np.nanmedian(rivdph_current):.3f} m "
            f"(weir_crest_fraction={weir_crest_fraction:.2f})"
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
        #
        # Crest is untouched by any of this -- still the normal per-cell
        # round-0 formula for every interior cell; only the coastal
        # endpoint's own crest is hard-forced (and floored against
        # coastal_protection_crest_m, see below).
        reach_is_mouth = dict(zip(rivers["reach_id"].astype(str), rivers["n_rch_dn"] == 0))
        cell_is_mouth_reach = cell_gdf["reach_id"].map(reach_is_mouth).fillna(False).to_numpy(dtype=bool)

        is_last_mouth_cell = np.zeros(len(cell_gdf), dtype=bool)
        mouth_natural_bathymetry = {}
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
                is_last_mouth_cell[last_pos] = True
                coastal_endpoint_bed = dem_at_cell[last_pos]
                rivdph_current[last_pos] = 0.0
                mouth_natural_bathymetry[mouth_rid] = coastal_endpoint_bed

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

        # Mouth ocean probe cell(s): one per mouth, the nearest OCEAN-
        # classified grid cell to that mouth's own last centerline cell --
        # crest there (via _mouth_crest_target below) is sized against the
        # real open-water level, not the river-side channel cell's own
        # reading, which sits right at the coast/river transition and
        # isn't necessarily representative of the water it actually
        # confines against.
        if is_last_mouth_cell.any():
            _ocean_rows_all, _ocean_cols_all = np.where(ocean_mask_grid)
            _ocean_tree = cKDTree(np.column_stack([_ocean_rows_all, _ocean_cols_all]))
            _mouth_positions = np.flatnonzero(is_last_mouth_cell)
            _mouth_rowcol = cell_gdf.loc[_mouth_positions, ["row", "col"]].to_numpy()
            _, _nearest_ocean_idx = _ocean_tree.query(_mouth_rowcol)
            mouth_ocean_rows = _ocean_rows_all[_nearest_ocean_idx]
            mouth_ocean_cols = _ocean_cols_all[_nearest_ocean_idx]
            mouth_ocean_x, mouth_ocean_y = rasterio.transform.xy(grid.transform, mouth_ocean_rows, mouth_ocean_cols)
            mouth_ocean_x = np.asarray(mouth_ocean_x)
            mouth_ocean_y = np.asarray(mouth_ocean_y)
            log.info(
                f"[{round_label}] Mouth ocean probe cell(s): {len(mouth_ocean_rows)} "
                f"(nearest ocean cell to each mouth's own coastal endpoint)"
            )

        # Depth (rivdph_current) is set HERE, once, and never revisited again
        # in any correction round -- see this rule's own module docstring for
        # the full rationale. Only the crest keeps adjusting below.
        log.info(f"[{round_label}] Excavation depth fixed for all subsequent rounds")
        weir_crest_current[is_last_mouth_cell] = _mouth_crest_target(round_root)
    else:
        # Direct per-cell crest update -- no two-case overtopped/near-zero
        # branching, no smoothing algorithm applied to the centerline's own
        # target: every centerline cell's crest is set directly from THIS
        # round's own simulated cross-section water level
        # (period_max_zs, see _read_cross_section_period_max_zs) plus a flat
        # per-round increment, applied to EVERY cell unconditionally --
        # raising OR lowering it relative to whatever it currently is, not
        # just where it was overtopped. Repeated every round, this makes
        # the crest climb by min_crest_increment_per_round_m each round on
        # top of whatever the coupled water level actually settles at. Once
        # a round's own crest already exceeds that round's own
        # period_max_zs everywhere (no overtopping under its own
        # simulation), the one-shot snap-and-verify step below (unchanged)
        # tightens the crest down to the freeboard-exact target and spends
        # one more round confirming it still holds.
        gap = weir_crest_current - period_max_zs  # margin: negative = overtopped by this much
        n_overtopped = int((gap < 0).sum())
        log.info(
            f"[{round_label}] {n_overtopped}/{len(gap)} cell(s) overtopped by their own "
            f"period-max water level (min gap={np.min(gap):.3f} m) -- crest set directly to "
            f"zs_max(centerline) + {min_crest_increment_per_round_m:.2f} m at every cell"
        )
        weir_crest_current = period_max_zs + min_crest_increment_per_round_m
        # Mouth outlet cell(s): crest re-targeted every round, independent
        # of the formula above -- see _mouth_crest_target's own docstring
        # (bed/depth there is the separate, unrelated natural-bathymetry
        # floor set once in round 0, untouched here).
        weir_crest_current[is_last_mouth_cell] = _mouth_crest_target(round_root)
        log.info(
            f"[{round_label}] Updated crest: min={np.nanmin(weir_crest_current):.3f} m, "
            f"max={np.nanmax(weir_crest_current):.3f} m, median={np.nanmedian(weir_crest_current):.3f} m "
            f"(depth unchanged from round 0)"
        )

        # Coastal probe correction: SAME two-case update as the river crest
        # above, applied to coastal_crest_current instead of weir_crest_current
        # -- entirely separate state, driven only by ocean-cell water levels
        # (coastal_probe_rows/cols, restricted to ocean_mask at setup time),
        # so this can never be driven by, or leak onto, the river itself.
        # coastal_crest_on_grid (built before this round's own weir, from
        # whatever coastal_crest_current held at that point) only ever
        # combines with river_crest_on_grid via np.maximum inside
        # build_coastal_protection_weir -- never lowers protection, only
        # raises it locally where the data says so.
        if len(coastal_probe_rows) > 0:
            coastal_period_max_zs = _read_coastal_probe_period_max_zs(round_root)
            coastal_gap = coastal_crest_current - coastal_period_max_zs
            coastal_badly_overtopped = coastal_gap < -_CALIBRATION_FREEBOARD_M
            coastal_near_zero = np.abs(coastal_gap) <= _CALIBRATION_FREEBOARD_M
            coastal_excess = -coastal_gap
            coastal_crest_current = np.where(
                coastal_badly_overtopped,
                coastal_crest_current + np.maximum(coastal_excess, min_crest_increment_per_round_m),
                coastal_crest_current,
            )
            coastal_crest_current = np.where(
                coastal_near_zero, coastal_crest_current + min_crest_increment_per_round_m, coastal_crest_current
            )
            coastal_crest_current = np.maximum(coastal_crest_current, coastal_protection_crest_m)
            log.info(
                f"[{round_label}] Coastal probe crest: min={coastal_crest_current.min():.3f} m, "
                f"max={coastal_crest_current.max():.3f} m, median={float(np.median(coastal_crest_current)):.3f} m "
                f"({int(coastal_badly_overtopped.sum())} badly overtopped, {int(coastal_near_zero.sum())} near "
                f"target, of {len(coastal_crest_current)} probe cell(s))"
            )

        # River boundary probe correction: SAME two-case update, applied to
        # river_boundary_crest_current instead -- entirely separate state,
        # driven only by LAND-cell water levels near the channel (see river
        # boundary probe cells' own setup comment above). This directly
        # verifies whatever river_crest_on_grid's own dilation claims to
        # cover, instead of trusting it -- a land cell the centerline never
        # reaches (e.g. an array-edge corner near a seed) gets its own
        # tracked, corrected crest here even though it has no cell_gdf row
        # at all.
        if len(river_boundary_probe_rows) > 0:
            river_boundary_period_max_zs = _read_river_boundary_probe_period_max_zs(round_root)
            river_boundary_gap = river_boundary_crest_current - river_boundary_period_max_zs
            river_boundary_badly_overtopped = river_boundary_gap < -_CALIBRATION_FREEBOARD_M
            river_boundary_near_zero = np.abs(river_boundary_gap) <= _CALIBRATION_FREEBOARD_M
            river_boundary_excess = -river_boundary_gap
            river_boundary_crest_current = np.where(
                river_boundary_badly_overtopped,
                river_boundary_crest_current + np.maximum(river_boundary_excess, min_crest_increment_per_round_m),
                river_boundary_crest_current,
            )
            river_boundary_crest_current = np.where(
                river_boundary_near_zero,
                river_boundary_crest_current + min_crest_increment_per_round_m,
                river_boundary_crest_current,
            )
            river_boundary_crest_current = np.maximum(river_boundary_crest_current, coastal_protection_crest_m)
            log.info(
                f"[{round_label}] River boundary probe crest: min={river_boundary_crest_current.min():.3f} m, "
                f"max={river_boundary_crest_current.max():.3f} m, "
                f"median={float(np.median(river_boundary_crest_current)):.3f} m "
                f"({int(river_boundary_badly_overtopped.sum())} badly overtopped, "
                f"{int(river_boundary_near_zero.sum())} near target, of {len(river_boundary_crest_current)} probe cell(s))"
            )

    # Coastal protection floor, EVERY round, EVERY cell: the tracked crest
    # must never read below the real coastal protection standard -- the
    # mouth's own hard-forced last cell would otherwise report raw natural
    # bathymetry (which can be below sea level) with nothing enforcing the
    # production requirement that crest = max(river-derived crest, coastal
    # protection crest). build_coastal_protection_weir already applies this floor at
    # actual simulation/build time (round_crest_elevation_m), but the
    # TRACKED weir_crest_current did not, creating a misleading gap between
    # what calibration_state.csv/plots/the final gpkg report and what SFINCS
    # actually builds.
    weir_crest_current = np.maximum(weir_crest_current, coastal_protection_crest_m)

    # Per-reach MEDIAN reduction (not mean -- so a mouth reach's hard-forced
    # rivdph=0 last cell doesn't skew its own reported value) -- the final
    # network output and per-round diagnostics still need ONE scalar per
    # reach; median, and this reduction, only ever have a per-cell view.
    cell_state = cell_gdf[["reach_id"]].assign(rivdph=rivdph_current, crest=weir_crest_current)
    reach_medians = cell_state.groupby("reach_id")[["rivdph", "crest"]].median()
    rivers["rivdph"] = rivers["reach_id"].astype(str).map(reach_medians["rivdph"])
    rivers["weir_crest_calibrated"] = rivers["reach_id"].astype(str).map(reach_medians["crest"])
    rivers_utm["weir_crest_calibrated"] = rivers_utm["reach_id"].astype(str).map(reach_medians["crest"])

    # ── per-round calibration_state.csv (real per-cell ground truth) ─────────
    # Written directly under calib_root (not a declared Snakemake output) --
    # guards against drift between this script's actual formulas and any
    # independent replay of them (e.g.
    # tests/plot_calibration_round_profiles.py's own reconstruct_crest_by_round),
    # which is not guaranteed to match exactly for junction-adjacent
    # reaches. Writing the real state directly means future diagnostics can
    # read ground truth instead of re-deriving it.
    # "weir_crest" here is the crest that ACTUALLY produced this round's own
    # period_max_zs/flooding -- round 0's own freshly-split estimate (no
    # earlier simulated crest exists yet), or the PRE-update snapshot
    # (weir_crest_simulated) for every correction round, NOT the post-update
    # weir_crest_current (that value is this round's own target for the
    # NEXT round's simulation, still untested at this point -- see the
    # weir_crest_simulated snapshot's own comment above).
    calibration_state_path = round_root / "calibration_state.csv"
    cell_gdf.drop(columns="geometry").assign(
        dem=dem_at_cell, rivdph=rivdph_current,
        weir_crest=(weir_crest_current if round_idx == 0 else weir_crest_simulated),
        zs=period_max_zs,
    ).to_csv(calibration_state_path, index=False)

    # ── per-round diagnostic plots ────────────────────────────────────────────
    # Must happen HERE, inside the loop, before the next round overwrites
    # this disposable calibration model's own sfincs_map.nc/sfincs_his.nc/
    # subgrid files -- each round's own transient state is only ever
    # available during its own iteration.
    _round_paths = _round_visual_paths(round_idx)
    plot_calibration_path = _round_paths["plot_calibration"]
    plot_water_level_path = _round_paths["plot_water_level"]
    plot_max_inundation_path = _round_paths["plot_max_inundation"]
    animation_path = _round_paths["animation"]
    crest_gap_map_path = _round_paths["crest_gap_map"]

    rivers_wgs = rivers.to_crs("EPSG:4326") if rivers.crs is not None and rivers.crs.to_epsg() != 4326 else rivers
    plot_river_depth(
        rivers_wgs=rivers_wgs,
        bbox_poly=domain_poly,
        osm_land_path=str(land_polygons_path),
        output_path=str(plot_calibration_path),
    )

    round_day_markers = [(calibration_days, f"Day {calibration_days:.0f} (run end)")]
    plot_water_level_timeseries(
        final_times_s / 86400.0, final_zs[:, representative_cell_pos], str(plot_water_level_path),
        day_markers=round_day_markers,
        station_labels=[f"reach {rid}" for rid in representative_reach_ids],
        basin_id=f"Basin {calib_root.parent.name}", run_label=f"Depth calibration -- {round_label}",
    )

    # round_root doubles as both run_dir and sfincs_root for THIS round's
    # own disposable model.
    da_hmax, _da_dep = compute_max_inundation(
        round_root, round_root, _round_ocean_mask_path, hmin=0.0, include_subgrid=include_subgrid,
    )
    if da_hmax is None:
        log.warning(f"[{round_label}] No max inundation data available -- creating empty plot sentinel")
        plot_max_inundation_path.touch()
    else:
        plot_max_inundation_map(
            da_hmax, domain_poly, str(land_polygons_path), str(river_network_path),
            str(plot_max_inundation_path),
            basin_id=calib_root.parent.name, run_label=f"calibration -- {round_label}",
        )

    da_h = compute_flood_progression(round_root, _round_ocean_mask_path)
    if da_h is None:
        log.warning(f"[{round_label}] compute_flood_progression returned None -- skipping flood animation")
        animation_path.touch()
    else:
        animate_flood_progression(
            da_h, domain_poly, str(land_polygons_path), str(river_network_path),
            str(animation_path),
            basin_id=calib_root.parent.name, run_label=f"calibration -- {round_label}", fps=animation_fps,
        )

    # Crest-vs-actual-water-level gap map -- the painted crest (river_crest_on_grid,
    # THIS round's own actual weir) minus THIS round's own actual simulated
    # period-max water level, with the weir line / river network / REAL
    # discharge point(s) overlaid (crossings_utm -- the same snapped
    # locations discharge is actually injected at, not a reach's own
    # line-start coordinate).
    plot_crest_gap_map(
        river_crest_on_grid, grid.transform, str(round_root / "sfincs_map.nc"),
        weir_gdf, rivers_utm, crossings_utm.geometry.x.to_numpy(), crossings_utm.geometry.y.to_numpy(),
        str(crest_gap_map_path),
        basin_id=calib_root.parent.name, run_label=f"calibration -- {round_label}",
    )
    log.info(f"[{round_label}] Diagnostic plots written under {plot_calibration_path.parent}")

    # ── early-stopping check (correction rounds only) ────────────────────────
    # n_correction_iterations is a MAXIMUM, not a fixed target: once the crest
    # already exceeds the period-max water level everywhere (no raw
    # overtopping under this round's own simulation, centerline + coastal
    # probes + river boundary probes), the raise-only correction process has
    # nothing left to correct -- but it can leave real excess margin behind
    # (it only ever raises crest, never lowers it, by design -- see the
    # correction-round update's own comment for why). So instead of stopping
    # immediately, this first convergence tightens every cell's crest down to
    # exactly period_max_zs + freeboard_m in one shot (removing that excess)
    # and spends exactly one more round verifying the tightened crest still
    # holds. If it does, THAT round is the final, converged one. If not (the
    # network's own coupling pushed some other cell back over), the whole
    # snap is abandoned and reverted -- not patched cell-by-cell -- and the
    # round before the snap becomes the converged one instead. Both outcomes
    # stop the loop and backfill whatever round slots remain (see just after
    # the loop) rather than spending further SFINCS runtime on rounds that
    # can't change the outcome.
    #
    # 2026-08-06: previously ALSO required the realized inundated-land-cell
    # count (da_hmax, already computed above) to settle within 10% of round
    # 0's own count before convergence would fire -- round 0 is confined by
    # an effectively un-overtoppable 1000 m wall, so its own flooded footprint
    # is artificially tiny, and a real (correctly contained) crest can easily
    # have a much larger, but perfectly genuine and stable, flooded footprint
    # than that -- removed per explicit direction: containment (no overtopping
    # anywhere) is the real, physically meaningful criterion; comparing total
    # flooded area against an artificially walled-off baseline was overly
    # protective and could block convergence indefinitely even with zero
    # overtopping observed for many rounds running.
    if round_idx == 0:
        pass  # round 0's own 1000 m confinement is never eligible for convergence
    elif snap_attempted:
        # This round is the verification of the PREVIOUS round's own snap
        # (see the snap applied in the branch below) -- checked regardless
        # of n_correction_iterations, since the snap already committed this
        # round's slot to being the verification, whether or not that
        # leaves further budget. `gap` here was computed from the SNAPPED
        # crest that actually built/simulated this round (weir_crest_simulated),
        # BEFORE this round's own correction-round update ran -- so it still
        # correctly reflects whether the snap itself held, independent of
        # whatever that update did on top of it afterward.
        contained = _gap_contained(gap) and _gap_contained(coastal_gap) and _gap_contained(river_boundary_gap)
        if contained:
            log.info(
                f"[{round_label}] Snap verified: crest still exceeds the period-max water level "
                f"everywhere (centerline, coastal probes, river boundary probes) after tightening "
                f"every cell to exactly freeboard_m margin -- converged"
            )
            converged_round_idx = round_idx
        else:
            log.info(
                f"[{round_label}] Snap verification failed: tightening the crest to exactly "
                f"freeboard_m margin caused overtopping elsewhere -- abandoning the snap entirely "
                f"and reverting to round {round_idx - 1}'s own (pre-snap) crest instead"
            )
            weir_crest_current = crest_before_snap
            if coastal_crest_before_snap is not None:
                coastal_crest_current = coastal_crest_before_snap
            if river_boundary_crest_before_snap is not None:
                river_boundary_crest_current = river_boundary_crest_before_snap
            cell_state = cell_gdf[["reach_id"]].assign(rivdph=rivdph_current, crest=weir_crest_current)
            reach_medians = cell_state.groupby("reach_id")[["rivdph", "crest"]].median()
            rivers["weir_crest_calibrated"] = rivers["reach_id"].astype(str).map(reach_medians["crest"])
            rivers_utm["weir_crest_calibrated"] = rivers_utm["reach_id"].astype(str).map(reach_medians["crest"])
            converged_round_idx = round_idx - 1
            # weir_gdf (this loop iteration's own traced weir) was built from
            # the FAILED, over-tightened snap crest -- it no longer matches
            # weir_crest_current now that it's been reverted above, so the
            # exported gpkg would otherwise report a crest that was never
            # actually accepted. Flagged here, rebuilt once after the loop
            # (see snap_reverted's own use below) rather than re-traced
            # again right here, since coastal_crest_on_grid/
            # river_boundary_crest_on_grid at this point in the loop still
            # reflect the (also reverted) pre-snap state consistently only
            # once every relevant variable has actually settled post-revert.
            snap_reverted = True
        break
    elif round_idx < n_correction_iterations:
        contained = _gap_contained(gap) and _gap_contained(coastal_gap) and _gap_contained(river_boundary_gap)
        if contained:
            log.info(
                f"[{round_label}] Converged: crest exceeds the period-max water level everywhere "
                f"(centerline, coastal probes, river boundary probes) -- tightening every cell's "
                f"crest down to exactly period_max_zs + freeboard_m (removing whatever excess "
                f"margin the raise-only correction process left behind) and spending one more "
                f"round to verify it still holds"
            )
            crest_before_snap = weir_crest_current.copy()
            weir_crest_current = period_max_zs + _CALIBRATION_FREEBOARD_M
            weir_crest_current[is_last_mouth_cell] = _mouth_crest_target(round_root)
            weir_crest_current = np.maximum(weir_crest_current, coastal_protection_crest_m)
            # np.where(isfinite(...), target, current) here, NOT a plain
            # np.maximum(...) -- unlike centerline cells (always wet, always
            # finite), a probe cell can be genuinely dry in the specific
            # round the snap fires. Overwriting unconditionally would
            # permanently poison that probe to NaN with no later round ever
            # able to recover it (the regular per-round update above already
            # preserves-on-NaN via np.where; the snap must match it).
            if coastal_crest_current is not None:
                coastal_crest_before_snap = coastal_crest_current.copy()
                _coastal_snap_target = np.maximum(coastal_period_max_zs + _CALIBRATION_FREEBOARD_M, coastal_protection_crest_m)
                coastal_crest_current = np.where(
                    np.isfinite(coastal_period_max_zs), _coastal_snap_target, coastal_crest_current
                )
            if river_boundary_crest_current is not None:
                river_boundary_crest_before_snap = river_boundary_crest_current.copy()
                _river_boundary_snap_target = np.maximum(
                    river_boundary_period_max_zs + _CALIBRATION_FREEBOARD_M, coastal_protection_crest_m
                )
                river_boundary_crest_current = np.where(
                    np.isfinite(river_boundary_period_max_zs), _river_boundary_snap_target, river_boundary_crest_current
                )
            snap_attempted = True
            # deliberately no break -- next round runs as the verification
        else:
            _still_over = []
            if not _gap_contained(gap):
                _still_over.append("centerline")
            if not _gap_contained(coastal_gap):
                _still_over.append("coastal probes")
            if not _gap_contained(river_boundary_gap):
                _still_over.append("river boundary probes")
            log.info(
                f"[{round_label}] Not yet converged: crest does not exceed the period-max water "
                f"level everywhere ({', '.join(_still_over)})"
            )

# ── backfill round slot(s) skipped by early stopping ──────────────────────────
# `round_idx` (the for loop's own variable, still holding its last value
# here whether the loop broke early or ran to completion) is the last
# round that ACTUALLY simulated something -- its own output files are
# genuine and must NEVER be overwritten, even when that round turned out
# to be a REJECTED snap attempt (converged_round_idx == round_idx - 1 in
# that case): destroying a round's own real diagnostics the moment it's
# superseded is exactly what made an earlier failed-snap investigation on
# basin 2433835 have to reconstruct round 6's own state from raw
# sfincs_map.nc instead of just reading its already-corrupted
# max_inundation_round6.png.
#
# Rounds strictly BETWEEN the last simulated one and the final slot never
# ran at all -- these are genuinely SKIPPED: no visual output files at
# all (not even an empty placeholder), since these are no longer declared
# Snakemake outputs (see this rule's own .smk comment) and a round that
# never ran has nothing worth a file for. calibration_state.csv is the
# one exception: it still gets a copy of the converged (accepted) round's
# own file, since gather_calibration_round_profile (below) reads every
# round index's own CSV unconditionally -- harmless here, since no
# genuine state ever existed for these rounds to misrepresent in the
# first place.
#
# The FINAL slot (round n_correction_iterations) always gets a full copy
# of the converged round's own genuine, ACCEPTED files -- "jump directly
# to round n_correction_iterations's own plot & diagnostics" -- since the
# canonical-output copy section further down reads from exactly that slot.
if converged_round_idx is not None and round_idx < n_correction_iterations:
    log.info(
        f"Calibration converged at round {converged_round_idx}/{n_correction_iterations} "
        f"(last actually-simulated round: {round_idx}) -- rounds {round_idx + 1}.."
        f"{n_correction_iterations - 1} never ran and are left with no visual output at all; "
        f"round {n_correction_iterations}'s own slot gets a direct copy of round "
        f"{converged_round_idx}'s own accepted diagnostics"
    )
    for _skip_idx in range(round_idx + 1, n_correction_iterations):
        _skip_round_root = calib_root / f"round{_skip_idx}"
        _skip_round_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            calib_root / f"round{converged_round_idx}" / "calibration_state.csv",
            _skip_round_root / "calibration_state.csv",
        )

    _final_round_root = calib_root / f"round{n_correction_iterations}"
    _final_round_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        calib_root / f"round{converged_round_idx}" / "calibration_state.csv",
        _final_round_root / "calibration_state.csv",
    )
    _converged_paths = _round_visual_paths(converged_round_idx)
    _final_paths = _round_visual_paths(n_correction_iterations)
    for _kind in _converged_paths:
        shutil.copy(_converged_paths[_kind], _final_paths[_kind])

# ── canonical production outputs: burned river DEM + weir, from the FINAL
# converged state ──────────────────────────────────────────────────────────
# Same filenames rule empirical_depth_estimation writes -- see this rule's own module
# docstring. Deliberately re-burnt here from rivdph_current/channel_mask
# directly (not reused from whichever round happened to run last inside
# the loop above), since round 0 itself never burns anything into its own
# grid (excavation is only ever APPLIED starting the round after it's
# computed) -- rebuilding once here from the final, converged state is
# correct regardless of which round the loop actually stopped at,
# including the n_correction_iterations=0 edge case.
_final_zbed_anchors = cell_gdf.assign(rivbed=dem_at_cell - rivdph_current)
NODATA = np.float32(-9999.0)

_final_burned_native, _final_transform_native, _nd, _stats = burn_river_channel(
    rivers=rivers_utm, zbed_anchors=_final_zbed_anchors,
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

_final_burned_coarse, _final_transform_coarse, _nd2, _stats2 = burn_river_channel(
    rivers=rivers_utm, zbed_anchors=_final_zbed_anchors,
    natural_dem_path=elevation_path,
    utm_crs=sf.crs, resolution_m=grid.cell_size_m,
    out_transform=grid.transform, out_shape=grid.shape,
    channel_mask=channel_mask,
)
with rasterio.open(
    snakemake.output.river_burned_dem_sfincs_grid, "w", driver="GTiff", dtype="float32",
    width=_final_burned_coarse.shape[1], height=_final_burned_coarse.shape[0],
    count=1, crs=sf.crs, transform=_final_transform_coarse,
    nodata=float(NODATA), compress="deflate",
) as dst:
    dst.write(np.where(np.isnan(_final_burned_coarse), NODATA, _final_burned_coarse).astype(np.float32), 1)
log.info(
    f"Written: {snakemake.output.river_burned_dem_sfincs_grid} "
    f"({_stats2['n_reaches_burned']} reach(es) burned, {_stats2['n_pixels_burned']:,} pixel(s))"
)

# weir_gdf: ALWAYS re-traced here from the final, accepted crest arrays
# (weir_crest_current/coastal_crest_current/river_boundary_crest_current),
# never reused directly from the loop's own last per-round weir_gdf --
# that per-round weir was built with freeboard_m=0.0 for the calibration's
# own internal simulation purposes (see its own comment above), so reusing
# it as-is would export a weir carrying only _CALIBRATION_FREEBOARD_M of
# margin and silently ignore config's real weir_freeboard_m. This is also
# needed for a failed snap verification specifically: there,
# weir_crest_current/coastal_crest_current/river_boundary_crest_current
# were reverted back to their PRE-snap (accepted) values, but the loop's
# own weir_gdf still holds the FAILED, over-tightened snap's own traced
# geometry -- re-tracing from the now-reverted arrays instead ensures the
# exported gpkg always matches the crest that was actually accepted, never
# a rejected one.
# 2026-08-05: previously this re-trace only ran `if snap_reverted`, and
# even then passed freeboard_m=0.0 -- meaning config's weir_freeboard_m was
# NEVER actually applied to the exported weir in either code path. Now
# unconditional, and uses the real config value as the one place freeboard
# actually reaches the final crest, on top of _CALIBRATION_FREEBOARD_M
# already baked into the tracked arrays above.
_final_river_crest_on_grid = build_nearest_weir_crest_regular(
    rivers_utm, "width", grid.shape, grid.transform,
    cell_gdf=cell_gdf, crest_values=weir_crest_current,
)
_final_coastal_crest_on_grid = None
if coastal_crest_current is not None:
    _final_coastal_crest_on_grid = _rasterize_nearest(
        coastal_probe_rows, coastal_probe_cols, coastal_crest_current, landuse_on_grid.shape,
        max_distance_cells=river_crest_dilation_cells,
    )
if river_boundary_crest_current is not None:
    _final_river_boundary_crest_on_grid = _rasterize_nearest(
        river_boundary_probe_rows, river_boundary_probe_cols, river_boundary_crest_current, landuse_on_grid.shape,
        max_distance_cells=river_crest_dilation_cells,
    )
    _final_coastal_crest_on_grid = (
        _final_river_boundary_crest_on_grid if _final_coastal_crest_on_grid is None
        else np.maximum(_final_coastal_crest_on_grid, _final_river_boundary_crest_on_grid)
    )
weir_gdf, _final_weir_diagnostics = build_coastal_protection_weir(
    grid, landuse_on_grid, channel_mask, coastal_protection_crest_m,
    min_component_cells, weir_par1, river_crest_on_grid=_final_river_crest_on_grid,
    coastal_crest_on_grid=_final_coastal_crest_on_grid,
    freeboard_m=weir_freeboard_m, channel_mask_gap_free=True,
    river_crest_dilation_cells=river_crest_dilation_cells,
)
log.info(
    f"Re-traced weir for export: {len(weir_gdf)} segment(s), "
    f"+{weir_freeboard_m:.2f} m production freeboard applied uniformly"
    + (" (snap was reverted -- traced from the accepted, pre-snap crest)" if snap_reverted else "")
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
_protected_pocket_mask = _final_weir_diagnostics.get(
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
        calib_root, rivers_utm, _seed, n_correction_iterations
    )
    plot_calibration_round_profiles(
        basin_id=calib_root.parent.name, seed=_seed, profiles_by_round=_profiles_by_round,
        n_rounds=n_correction_iterations,
        output_subplots_path=str(visuals_dir / f"round_profiles_seed{_seed}_subplots.png"),
        output_combined_path=str(visuals_dir / f"round_profiles_seed{_seed}_combined.png"),
    )
    pd.concat(
        [df.assign(round=i) for i, df in _profiles_by_round.items()], ignore_index=True
    ).to_csv(visuals_dir / f"round_profiles_seed{_seed}.csv", index=False)
    log.info(
        f"Seed {_seed}: round-profile diagnostics written "
        f"({n_correction_iterations} correction round(s), path of {len(_path_rids)} reach(es))"
    )

n_missing = rivers["rivdph"].isna().sum()
if n_missing:
    log.warning(f"{n_missing} reach(es) missing a calibrated depth (no centerline cell?) -- left as NaN")
log.info(
    f"Final calibrated weir crest: min={np.nanmin(weir_crest_current):.3f} m, "
    f"max={np.nanmax(weir_crest_current):.3f} m, median={np.nanmedian(weir_crest_current):.3f} m"
)

Path(snakemake.output.depth_estimated_river_network).parent.mkdir(parents=True, exist_ok=True)
rivers.to_file(snakemake.output.depth_estimated_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.depth_estimated_river_network}")

# ── canonical outputs: copies of the LAST round's own per-round files ────────
# The per-round loop above already produced a full diagnostic set for every
# round, including the final one (round_idx == n_correction_iterations) --
# copy those straight to the fixed canonical filenames downstream consumers/
# convention expect, rather than recomputing the same plots a second time.
Path(snakemake.output.plot_calibration).parent.mkdir(parents=True, exist_ok=True)
_last_round = n_correction_iterations
_last_round_paths = _round_visual_paths(_last_round)
for _canonical, _kind in (
    (snakemake.output.plot_calibration, "plot_calibration"),
    (snakemake.output.plot_water_level_timeseries, "plot_water_level"),
    (snakemake.output.plot_max_inundation, "plot_max_inundation"),
    (snakemake.output.animation_flood_progress, "animation"),
    (snakemake.output.plot_crest_gap_map, "crest_gap_map"),
):
    shutil.copy(_last_round_paths[_kind], _canonical)
log.info(f"Canonical diagnostic outputs copied from round {_last_round}'s own files")
log.info("Done")
