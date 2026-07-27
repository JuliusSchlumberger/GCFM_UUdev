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
boundary -- so round 0 must confine everything uniformly. Forced with a steady
discharge equal to the basin's protection-level return period (bankfull as
a fallback), runs for a single fixed duration (river_processing.
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
gets its own independently-tracked depth/crest, so the calibrated crest
varies smoothly along a reach and across reach junctions
(build_smoothed_weir_crest_regular's own per-reach interpolation + junction
blend, generalized from one scalar per reach to many anchors per reach),
matching how the burned bed already varies via burn_river_channel's own
per-anchor interpolation.

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
burn_river_channel/build_smoothed_weir_crest_regular functions rules 11b/13
use) instead of an isolated confinement wall, at the SAME calibration
discharge, then updates the crest, both cases expressed as plain
differences/sums only (no abs(), no ratios, so they stay correct for
below-datum/negative elevations, e.g. a mouth's own natural bathymetry):

    gap = weir_crest_current - period_max_water_level  (negative = overtopped)

    BADLY OVERTOPPED (gap < -freeboard_m): close the full gap, floored at a
    minimum step size so the crest keeps making real progress every round
    instead of stalling on a residual that never grows enough to matter:
        weir_crest_current += max(-gap, min_crest_increment_per_round_m)

    NEAR ZERO (|gap| <= freeboard_m -- barely overtopped, or contained with
    less than freeboard_m of margin): raised by the SAME flat
    min_crest_increment_per_round_m step, unconditionally -- NOT targeted at
    period_max_water_level + freeboard_m. Treating the whole
    [-freeboard_m, +freeboard_m] band identically avoids a sharp
    discontinuity between two physically-adjacent cells that land on
    opposite sides of gap == 0 by a few mm of simulated-zs noise (one side
    would otherwise jump by min_crest_increment_per_round_m, the other by
    the near-zero freeboard-exact nudge).

    Comfortably contained (gap > freeboard_m): left unchanged every round
    while raise-only correction is still ongoing -- the crest is never
    lowered round-to-round while cells elsewhere are still overtopped. Once
    EVERY cell is simultaneously contained with the inundated-cell count
    stable (the early-stopping condition below), a single one-shot
    tightening snap runs instead, see that section.

weir_crest_current is used AS-IS, with no separate freeboard added on top,
both to run every round's own simulated weir and as this rule's own final
output -- so the calibrated crest itself always guarantees at least
freeboard_m of margin above the coupled system's own driven water level,
rather than relying on a separate, later top-up (rule 13's own
build_coastal_protection_weir freeboard_m parameter, used this way for the
production/coastal crest, would otherwise double-count it here).
`n_correction_iterations: 0` skips this refinement entirely, using round
0's own isolated-confinement estimate unchanged.

Jumping straight to period_max_water_level + freeboard_m for badly
overtopped cells (the same rule as the near-zero case) would risk a
self-reinforcing loop for inland reaches, since depth is frozen after
round 0 and the crest is the only lever left: raising the crest removes
overbank relief, forcing more of the same discharge through the same
(unchanged) channel depth, which raises the confined water level, which
then needs an even higher crest, without converging. The floored-full-gap
closure above is a more conservative, incremental correction for large
gaps specifically, to avoid compounding that loop in one aggressive jump.

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
hard-forced to zero excavation (rivdph=0, crest=natural bed) -- that
coastal endpoint is a real physical constraint (the actual seabed), not a
calibration result.

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
Crest is untouched by any of this -- every non-endpoint cell still gets the
normal per-cell round-0 crest formula; only the coastal endpoint's own
crest is hard-forced (and then floored against the real coastal protection
crest, see below).

The forced natural-bathymetry bed/crest at each mouth's own last cell is
re-asserted at the end of every correction round too, after the
ensure-minimum-freeboard crest update -- its bed and crest are externally
fixed for the entire calibration, never excess/freeboard-driven at all.

weir_crest_current is floored against coastal_protection_crest_m (the real
production coastal protection standard, read from surge_forcing.nc) EVERY
round, for EVERY cell, right after that round's own update: the mouth's
hard-forced last cell would otherwise report raw natural bathymetry (which
can be below sea level) with nothing enforcing the production requirement
that crest = max(river-derived crest, coastal protection crest).
build_coastal_protection_weir already applies this max()
at actual simulation/build time; this floor makes the TRACKED
weir_crest_current (calibration_state.csv, the round-profile plots, and the
final network's own weir_crest_calibrated column) agree with what SFINCS
actually built, instead of under-reporting it.

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
following the real coastline shape, bays included, instead of an assumed
search radius) and combined with river_crest_on_grid via np.maximum inside
build_coastal_protection_weir -- never lowers protection, only raises it
locally where the data says so.

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
    plot_max_inundation_map,
    plot_river_depth,
    plot_water_level_timeseries,
)
from src.postprocessing import compute_flood_progression, compute_max_inundation
from src.river_burn import build_centerline_cells_regular, build_channel_mask_regular, build_smoothed_weir_crest_regular, burn_river_channel, constrain_to_coarse_channel_mask
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

# ── scope guard: regular grid only ───────────────────────────────────────────
if snakemake.params.quadtree_enabled:
    raise NotImplementedError(
        "River depth calibration (river_processing.depth_method='modelled') "
        "only supports regular grids in this version -- quadtree calibration "
        "is explicitly out of scope for now. Disable sfincs.grid.quadtree.enabled "
        "or use river_processing.depth_method='empirical' instead."
    )

# ── paths & params ────────────────────────────────────────────────────────────
elevation_path       = Path(snakemake.input.elevation_conditioned)
elevation_sfincs_grid_path = Path(snakemake.input.elevation_conditioned_sfincs_grid)
river_elevation_max_path = Path(snakemake.input.river_elevation_max)
river_network_path   = Path(snakemake.input.clean_river_network)
river_forcing_path   = Path(snakemake.input.river_forcing)
protection_levels_path = Path(snakemake.input.protection_levels)
grid_resolution_path = Path(snakemake.input.grid_resolution)
roughness_path       = Path(snakemake.input.roughness)
land_polygons_path   = Path(snakemake.input.land_polygons)
domain_gpkg_path     = Path(snakemake.input.domain_gpkg)
landuse_path         = Path(snakemake.input.landuse)
surge_forcing_path   = Path(snakemake.input.surge_forcing)
sea_mask_path        = Path(snakemake.input.sea_mask)
delta_outflow_points_path = Path(snakemake.input.delta_outflow_points)

calib_root  = Path(snakemake.params.calib_root)
sfincs_exe  = Path(snakemake.params.sfincs_exe)
timeout_s   = int(snakemake.params.timeout_s)
flow_accumulation_iterations = int(snakemake.params.flow_accumulation_iterations)
hg_c = float(snakemake.params.hg_c)
hg_f = float(snakemake.params.hg_f)

calibration_days   = float(snakemake.params.calibration_days)
weir_crest_m       = float(snakemake.params.weir_crest_m)
weir_par1          = float(snakemake.params.weir_par1)
weir_crest_fraction = float(snakemake.params.weir_crest_fraction)
n_correction_iterations = int(snakemake.params.n_correction_iterations)
min_crest_increment_per_round_m = float(snakemake.params.min_crest_increment_per_round_m)
weir_crest_junction_blend_m = float(snakemake.params.weir_crest_junction_blend_m)
weir_freeboard_m = float(snakemake.params.weir_freeboard_m)
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

# ── spatially-varying zsini: sea cells start AT baseline_m, not bone-dry ─────
# Replicates 13_build_sfincs.py's own zsini.tif construction verbatim (same
# sea_mask.tif + baseline_m + connected-component isolated-pocket fix) --
# a coastal/mouth cell starting bone-dry instead would take time to fill in
# from the boundary, producing a sharp transient spike well above its own
# true settled level before declining back down over the following hours.
# Since compute_period_max_zs takes the maximum over the ENTIRE run (no
# windowing), that transient alone would inflate the calibrated crest at
# every affected coastal cell by roughly the spike's own size.
with rasterio.open(sea_mask_path) as _sea_src:
    _sea_mask_arr = _sea_src.read(1).astype(np.float32)
    _sea_meta = _sea_src.meta.copy()
_zsini_arr = np.where(_sea_mask_arr == np.float32(1.0), np.float32(baseline_m), np.float32(-9999.0))

with rasterio.open(elevation_path) as _dep_src_zsini:
    _dep_zsini = _dep_src_zsini.read(1).astype(np.float32)
    _dep_nd_zsini = np.float32(_dep_src_zsini.nodata if _dep_src_zsini.nodata is not None else -9999.0)
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

_zsini_path = calib_root / "zsini.tif"
with rasterio.open(_zsini_path, "w", **_sea_meta) as _zsini_dst:
    _zsini_dst.write(_zsini_arr, 1)
log.info(f"zsini written: {_zsini_path} (sea cells = {baseline_m:+.4f} m)")

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
    "local_zsini": {
        "data_type": "RasterDataset",
        "uri": str(_zsini_path),
        "driver": "rasterio",
    },
    "local_roughness": {
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

# reproj_method="nearest" (not "average"): zsini is a near-binary field
# (baseline_m at sea, NaN/nodata on land) -- see 13_build_sfincs.py's own
# identical choice for why "average" would dilute/NaN-out genuine sea
# cells at the native/SFINCS-grid resolution boundary. mask must exist
# first (create_active/create_boundary above), matching production's own
# ordering. quadtree is out of scope for this rule (guarded above), so no
# quadtree_initial_conditions/ncinifile workaround is needed here.
sf.initial_conditions.create(ini="local_zsini", reproj_method="nearest")
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
landuse_on_grid = grid.sample_landuse(landuse_path)

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
_near_land_ocean = ocean_mask_grid & binary_dilation(~ocean_mask_grid, iterations=river_crest_dilation_cells)
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

# (n_idx, m_idx): sfincs_map.nc's own internal cell indices matching
# cell_gdf's (row, col) -- resolved lazily, once, inside _run_calibration_round
# below, from round 0's own freshly written sfincs_map.nc (grid geometry is
# round-invariant, so it's reused unchanged for every later round).
map_cell_idx = None
coastal_map_cell_idx = None
river_boundary_map_cell_idx = None

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

crossing_q = [reach_q.get(str(rid), np.nan) for rid in inside_ids]
crossings_gdf = gpd.GeoDataFrame(
    {"index": range(len(crossing_q))},
    geometry=gpd.points_from_xy(cross_lons, cross_lats),
    crs="EPSG:4326",
)
valid = np.isfinite(crossing_q)
crossings_gdf = crossings_gdf[valid].reset_index(drop=True)
crossing_q = np.asarray(crossing_q)[valid]

if crossings_gdf.empty:
    raise RuntimeError("No discharge crossings resolved to a calibration discharge -- cannot calibrate")

crossings_filt, cross_keep_mask = snap_points_into_region(crossings_gdf, region_wgs84, buf_deg)
crossing_q = crossing_q[cross_keep_mask]
if crossings_filt.empty:
    raise RuntimeError("All discharge crossings fall outside the active calibration region")

dis_df = pd.DataFrame(
    data=np.tile(crossing_q, (len(calib_times), 1)),
    index=calib_times,
    columns=range(len(crossings_filt)),
)
sf.discharge_points.create(timeseries=dis_df, locations=crossings_filt)
log.info(f"Discharge forcing: {len(crossings_filt)} constant source point(s)")

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
# SAMPLING location (see docstring above); the probe's own row/col above
# remains where the CORRECTED crest value gets rasterized back onto the
# grid, since that's the land-side position crest_surface is looked up at.
_channel_rows_all, _channel_cols_all = np.where(channel_mask & grid.valid_mask)
_channel_tree = cKDTree(np.column_stack([_channel_rows_all, _channel_cols_all]))
_dist_to_channel, _nearest_channel_idx = _channel_tree.query(
    np.column_stack([river_boundary_probe_rows, river_boundary_probe_cols])
)
_river_boundary_sample_rows = _channel_rows_all[_nearest_channel_idx]
_river_boundary_sample_cols = _channel_cols_all[_nearest_channel_idx]
river_boundary_sample_x, river_boundary_sample_y = rasterio.transform.xy(
    grid.transform, _river_boundary_sample_rows, _river_boundary_sample_cols
)
river_boundary_sample_x = np.asarray(river_boundary_sample_x)
river_boundary_sample_y = np.asarray(river_boundary_sample_y)
log.info(
    f"River boundary probe cells (land, directly bordering channel_mask or within "
    f"{_discharge_buffer_cells:.0f} cell(s) of a discharge point): {len(river_boundary_probe_rows)}"
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


def _read_river_boundary_probe_period_max_zs(round_root: Path) -> np.ndarray:
    """Same pattern as _read_coastal_probe_period_max_zs, for the river
    boundary probe cells -- but sampled at each probe's own NEAREST
    channel_mask cell (river_boundary_sample_x/y), not the land probe's own
    position (river_boundary_probe_x/y, used only for rasterizing the
    corrected value back onto the grid afterward) -- see the probe setup's
    own docstring for why. Empty array if there are no probe cells at all."""
    global river_boundary_map_cell_idx
    if len(river_boundary_probe_rows) == 0:
        return np.zeros(0, dtype=np.float32)
    if river_boundary_map_cell_idx is None:
        river_boundary_map_cell_idx = _resolve_map_cell_idx(
            round_root / "sfincs_map.nc", river_boundary_sample_x, river_boundary_sample_y
        )
        log.info(
            f"Resolved {len(river_boundary_map_cell_idx[0])} river boundary probe cell(s) "
            f"against sfincs_map.nc's own grid indices"
        )
    zs, _times_s = _read_map_zs_at_cells(round_root, *river_boundary_map_cell_idx)
    return compute_period_max_zs(zs)


def _rasterize_nearest(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Full-grid raster where every cell takes the value of its nearest
    (rows, cols) point -- turns the coastal probes' own discrete,
    per-cell crest values into a continuous surface covering all
    surrounding land, the same role build_smoothed_weir_crest_regular's
    own dilation plays for the river crest, but following the actual
    ocean-cell layout (and therefore the real coastline shape, bays
    included) instead of a fixed search radius."""
    all_rows, all_cols = np.indices(out_shape)
    tree = cKDTree(np.column_stack([rows, cols]))
    _dist, idx = tree.query(np.column_stack([all_rows.ravel(), all_cols.ravel()]))
    return values[idx].reshape(out_shape).astype(np.float32)


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

    zs, times_s = _read_zs_and_resolve(round_root)
    period_max_zs = compute_period_max_zs(zs)
    log.info(
        f"[{round_label}] Period max water level: min={period_max_zs.min():.3f} m, "
        f"max={period_max_zs.max():.3f} m, median={np.median(period_max_zs):.3f} m "
        f"(over the full {calibration_days:.0f}-day run, {zs.shape[0]} output step(s))"
    )
    return period_max_zs, zs, times_s


# wgs84_bounds/domain_poly: round-invariant, needed by every round's own
# diagnostic plots below -- computed once, before the loop.
wgs84_bounds, domain_crs, domain_poly = load_domain(snakemake.input.spec_basins_meta, domain_gpkg_path)

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
gap = None
round0_inundated_count = None
converged_round_idx = None
crest_before_snap = None
snap_attempted = False
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

        river_crest_on_grid = build_smoothed_weir_crest_regular(
            rivers_utm, "width", "weir_crest_calibrated", grid.shape, grid.transform,
            blend_distance_m=weir_crest_junction_blend_m,
            crest_anchors=cell_gdf.assign(crest=weir_crest_current),
        )

    # Round 0: uniform 1000 m confinement everywhere (the deliberately
    # isolated baseline). Correction rounds: the REAL production coastal
    # crest as the "elsewhere" floor too, not just for river-covered cells --
    # otherwise the weir stays a hybrid of real (river) and artificial
    # (coastal) crests instead of a faithful production replica, which is
    # the whole point of these rounds.
    round_crest_elevation_m = weir_crest_m if round_idx == 0 else coastal_protection_crest_m
    # freeboard_m=0.0 always: weir_crest_current is used AS-IS to run the
    # model, with no separate freeboard added on top of it during
    # simulation -- correction rounds' own ensure-minimum-freeboard update
    # (see the round-0/correction-round branch below) already guarantees
    # weir_freeboard_m of margin directly in the tracked/output crest value
    # itself, so adding it again here would double-count it.
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
            roughness_list=[{"manning": "local_roughness"}],
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
                weir_crest_current[last_pos] = dem_at_cell[last_pos]
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

        # Depth (rivdph_current) is set HERE, once, and never revisited again
        # in any correction round -- see this rule's own module docstring for
        # the full rationale. Only the crest keeps adjusting below.
        log.info(f"[{round_label}] Excavation depth fixed for all subsequent rounds")
    else:
        # Two-case crest update, both cases expressed as plain differences/
        # sums only (no abs(), no ratios) so they stay correct regardless of
        # sign -- elevations here are relative to a vertical datum and are
        # routinely negative (e.g. the mouth's own natural bathymetry, -1.85
        # m in this basin), not distances, so an accidental abs() would
        # silently corrupt the update for any below-datum cell.
        #
        # Case 1 -- BADLY OVERTOPPED (gap < -weir_freeboard_m): close the
        # full gap, floored at a minimum step size so the crest keeps making
        # real, non-vanishing progress every round instead of stalling on a
        # residual that never grows enough to matter:
        #     delta = max(period_max_zs - weir_crest_current, min_crest_increment_per_round_m)
        # Case 2 -- NEAR ZERO (|gap| <= weir_freeboard_m), i.e. either
        # barely overtopped or contained but with less than freeboard_m of
        # margin: raised by the SAME flat min_crest_increment_per_round_m
        # step, unconditionally -- NOT targeted at zs + freeboard_m. A cell
        # already comfortably contained (gap > weir_freeboard_m) is left
        # untouched by either case -- the crest never gets lowered here
        # (only the separate one-shot snap-and-verify step, once every cell
        # is contained, ever tightens it -- see the early-stopping check
        # below).
        #
        # Case 2 deliberately targets a flat min_crest_increment_per_round_m
        # step rather than the freeboard-exact zs + weir_freeboard_m: gap == 0
        # is where Case 1's own min-increment floor and Case 2 meet, and
        # neighbouring cells can straddle that boundary by a few mm of
        # simulated-zs noise -- a freeboard-exact Case 2 would jump one side
        # by min_crest_increment_per_round_m and the other by a near-zero
        # nudge, producing a one-cell notch in an otherwise smooth crest
        # profile. Treating the whole [-freeboard_m, +freeboard_m] band
        # identically keeps adjacent cells moving together instead of
        # diverging. This is deliberately less precise than a freeboard-exact
        # target -- the one-shot snap-and-verify step (once every cell is
        # contained) is what achieves the tight final margin, not these
        # intermediate rounds.
        gap = weir_crest_current - period_max_zs  # margin: negative = overtopped by this much
        badly_overtopped = gap < -weir_freeboard_m
        near_zero = np.abs(gap) <= weir_freeboard_m
        n_badly_overtopped = int(badly_overtopped.sum())
        n_near_zero = int(near_zero.sum())
        log.info(
            f"[{round_label}] {n_badly_overtopped}/{len(gap)} cell(s) badly overtopped by their "
            f"own period-max water level (min gap={np.min(gap):.3f} m, more than "
            f"{weir_freeboard_m:.2f} m below crest); {n_near_zero} more within "
            f"{weir_freeboard_m:.2f} m of the target either side -- crest raised at "
            f"{n_badly_overtopped + n_near_zero}/{len(gap)} cell(s) total"
        )
        excess = -gap  # = period_max_zs - weir_crest_current, positive where overtopped
        weir_crest_current = np.where(
            badly_overtopped,
            weir_crest_current + np.maximum(excess, min_crest_increment_per_round_m),
            weir_crest_current,
        )
        weir_crest_current = np.where(
            near_zero, weir_crest_current + min_crest_increment_per_round_m, weir_crest_current
        )
        # Mouth outlet cell(s) are permanently fixed to natural bathymetry,
        # independent of either case above.
        weir_crest_current[is_last_mouth_cell] = dem_at_cell[is_last_mouth_cell]
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
            coastal_badly_overtopped = coastal_gap < -weir_freeboard_m
            coastal_near_zero = np.abs(coastal_gap) <= weir_freeboard_m
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
            river_boundary_badly_overtopped = river_boundary_gap < -weir_freeboard_m
            river_boundary_near_zero = np.abs(river_boundary_gap) <= weir_freeboard_m
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
    plot_calibration_path = Path(getattr(snakemake.output, f"plot_calibration_round{round_idx}"))
    plot_water_level_path = Path(getattr(snakemake.output, f"plot_water_level_round{round_idx}"))
    plot_max_inundation_path = Path(getattr(snakemake.output, f"plot_max_inundation_round{round_idx}"))
    animation_path = Path(getattr(snakemake.output, f"animation_round{round_idx}"))
    plot_calibration_path.parent.mkdir(parents=True, exist_ok=True)

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
        round_root, round_root, landuse_path, hmin=0.0, include_subgrid=include_subgrid,
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

    da_h = compute_flood_progression(round_root, landuse_path)
    if da_h is None:
        log.warning(f"[{round_label}] compute_flood_progression returned None -- skipping flood animation")
        animation_path.touch()
    else:
        animate_flood_progression(
            da_h, domain_poly, str(land_polygons_path), str(river_network_path),
            str(animation_path),
            basin_id=calib_root.parent.name, run_label=f"calibration -- {round_label}", fps=animation_fps,
        )
    log.info(f"[{round_label}] Diagnostic plots written under {plot_calibration_path.parent}")

    # ── early-stopping check (correction rounds only) ────────────────────────
    # n_correction_iterations is a MAXIMUM, not a fixed target: once the crest
    # already exceeds the period-max water level everywhere (no raw
    # overtopping under this round's own simulation) AND the realized
    # inundated-land-cell count (same da_hmax already computed above) has
    # settled back down to within 10% of round 0's own (near-zero, since
    # round 0 is confined by an effectively un-overtoppable 1000 m wall)
    # count, the raise-only correction process has nothing left to correct --
    # but it can leave real excess margin behind (it only ever raises crest,
    # never lowers it, by design -- see the correction-round update's own
    # comment for why). So instead of stopping immediately, this first
    # convergence tightens every cell's crest down to exactly
    # period_max_zs + freeboard_m in one shot (removing that excess) and
    # spends exactly one more round verifying the tightened crest still
    # holds. If it does, THAT round is the final, converged one. If not
    # (the network's own coupling pushed some other cell back over), the
    # whole snap is abandoned and reverted -- not patched cell-by-cell --
    # and the round before the snap becomes the converged one instead. Both
    # outcomes stop the loop and backfill whatever round slots remain (see
    # just after the loop) rather than spending further SFINCS runtime on
    # rounds that can't change the outcome.
    current_inundated_count = int((da_hmax.values > 0).sum()) if da_hmax is not None else None
    if round_idx == 0:
        round0_inundated_count = current_inundated_count
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
        flood_stable = (
            current_inundated_count is not None and round0_inundated_count is not None
            and abs(current_inundated_count - round0_inundated_count) <= 0.10 * max(round0_inundated_count, 1)
        )
        if contained and flood_stable:
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
        break
    elif round_idx < n_correction_iterations:
        contained = _gap_contained(gap) and _gap_contained(coastal_gap) and _gap_contained(river_boundary_gap)
        flood_stable = (
            current_inundated_count is not None and round0_inundated_count is not None
            and abs(current_inundated_count - round0_inundated_count) <= 0.10 * max(round0_inundated_count, 1)
        )
        if contained and flood_stable:
            log.info(
                f"[{round_label}] Converged: crest exceeds the period-max water level everywhere "
                f"(centerline, coastal probes, river boundary probes), and inundated-cell count "
                f"({current_inundated_count}) is within 10% of round 0's own ({round0_inundated_count}) "
                f"-- tightening every cell's crest down to exactly period_max_zs + freeboard_m "
                f"(removing whatever excess margin the raise-only correction process left behind) "
                f"and spending one more round to verify it still holds"
            )
            crest_before_snap = weir_crest_current.copy()
            weir_crest_current = period_max_zs + weir_freeboard_m
            weir_crest_current[is_last_mouth_cell] = dem_at_cell[is_last_mouth_cell]
            weir_crest_current = np.maximum(weir_crest_current, coastal_protection_crest_m)
            if coastal_crest_current is not None:
                coastal_crest_before_snap = coastal_crest_current.copy()
                coastal_crest_current = np.maximum(coastal_period_max_zs + weir_freeboard_m, coastal_protection_crest_m)
            if river_boundary_crest_current is not None:
                river_boundary_crest_before_snap = river_boundary_crest_current.copy()
                river_boundary_crest_current = np.maximum(
                    river_boundary_period_max_zs + weir_freeboard_m, coastal_protection_crest_m
                )
            snap_attempted = True
            # deliberately no break -- next round runs as the verification
        elif not contained:
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
        else:
            log.info(
                f"[{round_label}] Not yet converged: inundated-cell count ({current_inundated_count}) "
                f"still more than 10% away from round 0's own ({round0_inundated_count})"
            )

# ── backfill any round slot(s) skipped by early stopping ─────────────────────
# n_correction_iterations still fixes how many round-indexed output files
# Snakemake expects (declared at DAG-build time, before this script runs) --
# stopping the actual simulation loop early doesn't reduce that. Every
# remaining, never-simulated round slot gets a plain copy of the last round
# that actually ran, both for the four declared per-round Snakemake outputs
# and for calibration_state.csv (which gather_calibration_round_profile
# below expects to exist for every round index) -- honestly representing
# "nothing changed after this round" rather than leaving a gap.
if converged_round_idx is not None and converged_round_idx < n_correction_iterations:
    log.info(
        f"Calibration converged at round {converged_round_idx}/{n_correction_iterations} -- "
        f"backfilling round {converged_round_idx + 1}..{n_correction_iterations}'s own output "
        f"slot(s) from round {converged_round_idx}'s own files (no further simulation needed)"
    )
    for _skip_idx in range(converged_round_idx + 1, n_correction_iterations + 1):
        _skip_round_root = calib_root / f"round{_skip_idx}"
        _skip_round_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            calib_root / f"round{converged_round_idx}" / "calibration_state.csv",
            _skip_round_root / "calibration_state.csv",
        )
        for _kind in ("plot_calibration", "plot_water_level", "plot_max_inundation", "animation"):
            shutil.copy(
                getattr(snakemake.output, f"{_kind}_round{converged_round_idx}"),
                getattr(snakemake.output, f"{_kind}_round{_skip_idx}"),
            )

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

# weir_gdf: already the final round's own actual traced weir (loop-
# persistent, correctly holds whichever round the loop last executed --
# the converged round if early-stopping fired, round n_correction_iterations
# otherwise) -- written as-is, no re-derivation.
Path(snakemake.output.coastal_protection_weir).parent.mkdir(parents=True, exist_ok=True)
weir_gdf.to_file(snakemake.output.coastal_protection_weir, driver="GPKG")
log.info(f"Written: {snakemake.output.coastal_protection_weir} ({len(weir_gdf)} segment(s))")

# ── seed-to-mouth round-profile diagnostics (bed/crest/water-level per round) ─
# Automatically produced here (not a declared Snakemake output, same
# convention as calibration_state.csv) instead of requiring a separate,
# manually-run script -- and reads the SAME calibration_state.csv every
# round just wrote, so the profile is the real per-cell state this rule
# actually calibrated with, not a reconstruction of it (see
# gather_calibration_round_profile's own docstring). Written under the SAME
# persistent visuals directory as the other canonical diagnostic plots
# (results/{basin_id}/visuals/input_data/10_calibration/) rather than
# calib_root -- calib_root is this rule's own intermediate/disposable model
# directory, not where user-facing output belongs.
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
for _canonical, _kind in (
    (snakemake.output.plot_calibration, "plot_calibration"),
    (snakemake.output.plot_water_level_timeseries, "plot_water_level"),
    (snakemake.output.plot_max_inundation, "plot_max_inundation"),
    (snakemake.output.animation_flood_progress, "animation"),
):
    shutil.copy(getattr(snakemake.output, f"{_kind}_round{_last_round}"), _canonical)
log.info(f"Canonical diagnostic outputs copied from round {_last_round}'s own files")
log.info("Done")
