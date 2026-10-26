# Methods

How the pipeline builds, runs and adapts a SFINCS compound flood model for one
delta. One `basin_id` (the `BasinID2` of the delta polygons of Edmonds et al.,
2020) is one delta and one model. Rule numbers match the files in
`workflow/rules/` and `workflow/scripts/`; config keys are in
`config/config.yml`. Dataset sources and versions: [data.md](data.md).

## Pipeline stages

Rules 01-11 are basin-level and shared by all scenarios. Rules 13-17 split into a
basin-level part (skeleton, spin-up) and a per-scenario part (forcing, event
run, metrics). Rules 18-19 run per scenario and strategy.

**01 `split_delta_polygons`.** Extracts the basin's delta polygon from the
delta-polygon dataset.

**02 `determine_model_domain`.** Picks the UTM CRS, buffers the delta polygon by
`domain.delta_buffer_m` for the clipping box, and fits the model grid frame: the
minimum rotated rectangle around the polygon when `sfincs.grid.rotated` (default
`true`, per-basin `rotated_overrides`), else an axis-aligned box
(`src/grid.py:fit_grid_frame`).

**02b `prepare_landuse`.** Writes the basin land-use raster in pipeline codes
(source classes, 200 = open sea, 255 = nodata), WGS84 at the source resolution.
`landuse.source` selects Copernicus LC100 (100 m) or ESA WorldCover (10 m). See
[Land use and the sea](#land-use-and-the-sea).

**03 `get_land_polygons`.** Vectorises land (land use ≠ 200) as the background of
figures made before the model grid exists. Nothing computational reads it.

**04 `get_protection_levels`.** Finds the delta's FLOPROS protection standards;
see [Protection levels](#protection-levels).

**05a `get_elevation`.** Merges FathomDEM and GEBCO into
`elevation_merged.tif` in the GOCO06s frame; see [Datums](#datums-and-vertical-references)
and [Terrain merge](#terrain-merge).

**05b `get_landuse`.** Warps land use onto the elevation grid by mode and writes
the sea mask (land use 200).

**05c `get_roughness`.** Writes Manning's n on the elevation grid subdivided to
about the land-use pixel size, area-averaged from the land-use source rather
than taken from the dominant class (`landuse.roughness_aggregation`:
`area_mean` or `conveyance`, the harmonic mean). Inside the river channel (each
cleaned reach buffered by `width/2`, `src.river_burn.channel_polygons`) every
source pixel takes `river_processing.channel_manning_n` before the averaging,
the n the river bed is fitted with (rule 10), so it runs after rule 08.

**06 `get_river_network` / `plot_river_network`.** Clips SWORD to the domain.
The clip runs before 02b because the network is the barrier of the open-sea
fill; its figure needs rule 03 and is a separate rule.

**07a-c `select_surge_stations`, `extract_gtsm_series`, `storm_tide_hydrographs`.**
Select the COAST-RP stations that force the sea boundary, extract their hourly
GTSM water level and surge from the station subset (built once for all basins by
`tools/subset_gtsm_stations.py` after `select_surge_stations` has run), and derive per-station average tide and per-RP surge
shapes (enabled by `boundary_forcings.surge.hydrograph.enabled`).

**07 `get_boundary_forcings`.** Writes `surge_forcing.nc` and `river_forcing.nc`:
storm-tide return levels, calm-sea levels, the coastal protection crest and SLR
fingerprints per station; return-period tables, bankfull and mean discharge per
river crossing. See [Forcing](#forcing).

**08 `clean_river_network`.** Keeps the network reachable downstream of the rule
07 crossings, fixes widths and routes bankfull discharge through it; see
[River network](#river-network).

**08b `channel_landuse`.** Writes `landuse_channels.tif`: land use with every
non-sea pixel of the river channel (the same polygons) set to water
(`landuse.water_code`, 80), so built-up or crop pixels under the river count
neither as exposed nor as dry land. Every rule after 09 reads it; rule 09
keeps the original, whose islands it must see to skip them.

**08c `build_sfincs_grid`.** Subdivides the grid frame at the basin resolution
and writes `{basin_id}_sfincs_grid.json`, the one grid every later rule uses; see
[Grid](#grid).

**09 `enforce_river_monotonicity`.** Builds the river water surface from
FathomDEM and conditions the DEM to it (`src.water_surface`, settings under
`river_processing.water_surface`). FathomDEM's channels carry a flattened water
surface: flat per water body, dropping in steps, with stray peaks (bridges,
bars, bank pixels) and dips. On the node network (see *Nodes* under method
`gvf`): the median of the water pixels (land use = `landuse.water_code`, 80)
along the centreline within `median_half_window_m` (1000) of each node, the
window running on into the neighbouring reaches; islands, bars and banks the
centreline crosses are skipped, so their nodes take the level of the water
around them (gaps are filled from neighbouring nodes) (so a short lock or bridge
reach is outvoted by the river around it); structures detected where the reach
level drops (see *Structures*) but, with `split_at_structures: false`
(default), smoothed across: weirs and locks open in floods, and a mean-flow
level drop kept as a step would become a step in the flood bed; a downstream
rise is cut (`cut_rising_edges`); made non-increasing
downstream by a length-weighted isotonic regression on the whole node graph
(Hildreth's dual coordinate ascent, in node-disjoint edge batches, so loops
and junctions are solved together); smoothed over `smoothing_m` (5000), never
across a cut, ends mirrored so a slope runs through; non-increasing
again. Nodes of a network part without any water pixel take `calm_sea_level_m`. Every land pixel
of each channel buffer (`width/2`) is set to that surface (lower wins where
buffers overlap); sea pixels keep their GEBCO bed. Writes the surface
(`river_water_surface.gpkg`, layers `nodes` and `structures`: the GVF target),
the conditioned DEM at native resolution and on the model grid, and
`river_elevation_max.json` (highest surface level), which caps the active
cells at `river_elevation_max_m + sfincs.grid.active_mask.elevation_buffer_m`. The figure `visuals/09_water_surface.png` shows, per seed along
its widest arm, the FathomDEM water pixels, node medians, smoothed surface and
structure junctions by SWORD obstruction type.

**09b `grid_align_landuse`.** Resamples `landuse_channels.tif` onto the model grid
once (mode) and area-averages roughness onto it from the native source, with the
channel at `channel_manning_n` as in rule 05c. These are the only
grid-level land-use and roughness fields downstream.

**10 `modelled_depth_estimation`.** Sets river depth, burns the channel and
calibrates the coastal and riverbank weir crests with a disposable SFINCS model;
see [River depth and crest calibration](#river-depth-and-crest-calibration).
Also writes `zsini_sea_cells_on_grid.tif`: land-use sea minus pockets protected
by the weir, the sea mask of the initial condition, flood metrics and
adaptation.

**11 `test_upstream_boundary`, `test_bifurcation_calibration_options`.**
Diagnostics only; nothing depends on them. The first checks that river forcing
points lie beyond the reach of the surge wave from every mouth: distance =
min(kinematic, attenuation), kinematic `T_eff (c − v)` with
`c = sqrt(g (h + A0))`, attenuation `A(x) = A0 exp(−μx)`, `μ = r / 2c`,
`r = (8/3π) g n² U0 / h^(4/3)`, with `U0` fixed at the mouth and `A0` the
`testing.upstream_boundary_check.surge_rp_yr` level of the mouth's nearest
station. The second compares bifurcation discharge splits of the modified and
original SWORD files.

**13 `build_sfincs_skeleton`.** Builds the scenario-independent model once per
basin: grid, elevation, masks, weir, roughness, subgrid table, observation
points and the default initial water level. Nothing is reprojected here: every
input is already on the model grid. Not runnable on its own (no forcing, no
times).

**13 `build_sfincs`.** Per scenario: loads the skeleton, writes only the
water-level and discharge forcing, and hand-writes `sfincs.inp` with relative
references to the skeleton files (HydroMT's config writer makes references
outside the model root absolute).

**13c `land_bed_level`.** Once per basin: the skeleton's subgrid bed level on
the grid flood depths are computed on, masked to NaN on open sea (the corrected
`zsini_sea_cells_on_grid.tif`). Spin-up, sanity checks, event runs, flood
metrics and attribution read it, so its non-null pixels are the land domain of
every flood statistic; `adapt_apply_pre` writes the same file for each adapted
skeleton.

**14 `run_spinup`, 15 `sanity_checks`.** One spin-up per basin; see
[Spin-up](#spin-up). Rule 15 reports the share of land pixels wetted during
spin-up and animates it.

**16 `run_event`.** Runs the scenario's `sfincs.inp` and writes the
inundation-ratio map, an animation and `flood_timeseries.csv`. The flood volume
there is the volume present at each map output (every `dtmapout`), so its maximum
is the peak volume at that sampling; summing each pixel's maximum depth would
overstate it.

**17 `compute_flood_metrics`.** Downscales the maximum water level to the subgrid
bed (`downscale_floodmap`) as `max_flood_depth.tif`, masks the sea, and writes
`flood_metrics.csv`: domain, land and urban areas, flooded area and share
(depth ≥ `metrics.hmin`), urban exposure (land use `metrics.urban_landuse_code`),
mean and maximum depth, volume.

**13b, 18a-19** adaptation; see [Adaptation](#adaptation).

## Datums and vertical references

Every elevation and water level is in the GOCO06s geoid frame.

| quantity | source frame | to GOCO06s |
|---|---|---|
| FathomDEM | EGM2008 | add `N_EGM2008 − N_GOCO06s`, synthesised with pyshtools on the WGS84 ellipsoid (boule), EGM2008 truncated to the GOCO06s degree (about 300) |
| GEBCO | local mean sea level | add the MDT: `H_GOCO06s = H_MSL + MDT` |
| COAST-RP storm-tide levels | local mean sea level | `storm_tide_at_rp_m = storm_tide_at_rp_raw_m + mdt` |

The MDT is CNES-CLS22 (HYBRID-CNES-CLS2022), the mean sea surface above the
geoid. It is used twice: on GEBCO in rule 05a and on the COAST-RP levels in rule
07. It is NaN over land: rule 05a loads it with a `datum_correction.mdt_load_margin_deg`
(5°) margin and extrapolates it over land by inverse-distance fill before
resampling, so nearshore GEBCO pixels are not lost; rule 07 takes the nearest
valid cell within ±`datum_correction.fallback_search_deg` (3°) of each station.
The MDT is never applied to the DEM.

`calm_sea_level_m` is the mean MDT over a basin's selected stations: mean sea
level in the model frame, without SLR. It is the sea level of the initial
condition, of the round-0 calibration boundary and of the start of the spin-up;
each station's own calm level (`station_calm_sea_level_m`, its MDT) is where the
event lead-in starts. Coastal water levels are rounded to the nearest centimetre
(`src.surge.round_level`).

## Terrain merge

FathomDEM (30 m) stores int32 centimetres in lowercase 1° tiles
`{ns}{lat:02d}{ew}{lon:03d}.tif` (e.g. `n00e010.tif`); rule 05a divides by 100.
Steps (`05a_get_elevation.py`):

1. FathomDEM to the UTM working grid, coverage-weighted.
2. FathomDEM set to NaN on sea cells (land use 200): a terrestrial DEM reports
   spurious shallow elevation over water, so GEBCO fills sea, creeks, estuaries
   and lagoons.
3. Datum corrections as above.
4. GEBCO clamped to at most `terrain.gebco_max_depth_m` (10 m) below local MSL,
   because the deepest active cell sets SFINCS's CFL time step for the whole
   domain; and, on sea cells only, to at least `terrain.gebco_min_depth_m` (2 m)
   below it, which is also the depth of every sea-connected creek GEBCO cannot
   resolve. Local MSL is the MDT surface.
5. FathomDEM where valid, GEBCO elsewhere, blended linearly over
   `terrain.coast_blend_m` (300 m) seaward of the coast in open water only
   (water at least twice the blend width); FathomDEM pixels never change. GEBCO
   tiles merge with target-aligned pixels, so the clipping box does not shift
   the data.
6. Remaining interior gaps filled from data within 100 px; outside the domain is
   NaN.

## Land use and the sea

Land use 200 is the single sea/land criterion: sea mask, land masks, the weir's
ocean side, the initial condition and the water-level boundary all derive from
it. ESA WorldCover has no sea class (sea, lagoons, rivers and lakes are all 80),
so rule 02b derives 200 (`src/landuse.py`): WorldCover water that reaches the
domain border, with the basin's own river network as a barrier (width
`max(SWORD width / 2 × river_barrier_width_factor, river_barrier_min_width_m)`,
minus LC100 sea) so the sea stops at the mouths; only water bodies holding an
LC100 open-sea pixel qualify; WorldCover nodata where LC100 says sea is added.
LC100 is therefore needed in both modes. The sea morphology runs on a coarsened
working grid (at most 60 M pixels) and is applied back to the fine water mask,
streaming rows in blocks of 20 M pixels.

## Grid

| setting | value |
|---|---|
| cell size `dx` | `sfincs.grid.resolution_m` (100 m) for every basin; `resolution_overrides_m` per basin (`4267691: 150`) to fit memory |
| subgrid pixel | `dx / sfincs.subgrid.nr_subgrid_pixels` (10 m at the defaults); channels narrower than a cell live in the subgrid tables |
| orientation | rotated to the delta by default; preprocessing rasters (02b-09) stay north-up, clipped to the box around the rotated rectangle |

`{basin_id}_sfincs_grid.json` (rule 08c) holds `x0, y0, dx, dy, mmax, nmax,
rotation, epsg` and the affine transform in SfincsModel orientation:
`translation(x0, y0) × rotation(rotation) × scale(dx, dy)` with `dy` positive, so
row 0 is on the origin side (the south side of an unrotated grid), row = SFINCS
`n`, column = `m`; rotation is counter-clockwise in degrees, 0-360. Unrotated
frames are flipped to y-ascending when the JSON is written, so it matches the
grid HydroMT builds. Models are created with `src.grid.create_model_grid`, which
raises if the SfincsModel transform differs from the JSON.

On a rotated raster the pixel size is `hypot(a, d)` and the pixel area
`|a·e − b·d|` (`src.grid.cell_size_m`, `cell_area_m2`), not `transform.a`.
rioxarray's `rio.resolution`, `rio.bounds`, `rio.transform` and `rio.reproject`
are wrong there; use HydroMT's `.raster.*`.

SfincsModels are created with `mode="w+"`, never `r+`; an existing model is
opened read-only (`"r"`) and redirected to a new root with
`sf.root.set(<root>, mode="w+")`.

## River network

The network is SWORD v17c (`river_network`: one GeoPackage per region, layer
`reaches`), read by `src.sword.read_sword_reaches`, which maps the published
topology (`rch_id_dn_1..4`, `rch_id_up_1..4`) to the `rch_id_dn`/`rch_id_up`
list strings used downstream. Rule 06 then overwrites `width` on the reaches in
`config/sword_width_corrections.csv` (from satellite-image inspection; the rule
stops if a reach's SWORD width no longer matches the expected value) and marks
them `width_source = manual`. Rule 11's bifurcation comparison runs the network
without and with these widths. Reach ids are normalised with `str(int(float(x)))`
(`src.river_network.normalize_reach_id`) before any dict or set operation;
`rch_id_dn` is a list string split on commas and whitespace.

Rule 08, in order: fix T-junction tails; walk downstream from the rule 07
crossings; turn seeds fed from inside the network into outflows; trim seeds to
the domain; write the delta-outline outflow points (reaches crossing the delta
outline that are no seed, mouth or bifurcation, later the SFINCS outflow
boundary, mask 3); drop reaches without any width; swap `width`/`max_width`
where `max_width < width` (the 30 m `max_width` fill value is never swapped in);
raise mouth widths to their widest upstream neighbour; accumulate
`bankfull_discharge_m3s` into `bankfull_discharge_acc_m3s`. At a bifurcation
discharge splits in proportion to arm width only (equally if all widths are
zero); canal arms (SWORD lakeflag 2 on the arm's chain to the next junction,
plus `river_processing.canal_routing.force_exclude`) get no share unless every
arm is a canal.

## River depth and crest calibration

Rule 10 is the only source of river depth and weir crests. Round 0 sets the
river bed by GVF (no SFINCS run); one disposable SFINCS model (production grid
and subgrid set-up, production weir geometry) then runs twice:

| round | weir | channel | sea boundary | purpose |
|---|---|---|---|---|
| 0 | -- | GVF bed | -- | river bed (no SFINCS run) |
| 1 | every edge at `weir_crest_m` (1000 m) | burned GVF bed | storm tide at the FLOPROS coastal RP | crest per weir edge |
| 2 | production crests | as round 1 | as round 1 | verification: reports overtopped edges, changes nothing |

River forcing in both SFINCS rounds is a steady discharge at the FLOPROS riverine
return period (`river_protection_rp_yr`, log-RP interpolated from the discharge
table; bankfull when absent), ramped linearly from each crossing's bankfull over
`discharge_ramp_hr` to avoid a startup shock in a near-dry channel.

**Depth (GVF).** River bed from a steady gradually-varied-flow solution
fitted to rule 09's water surface at the mean discharge (Neal et al.,
2021, WRR 57, e2020WR028301; `src/gvf*.py`, settings under
`river_depth_modelling.gvf`). The channel the bed is burned into carries the
solver's roughness (`river_processing.channel_manning_n`, 0.03) in
`roughness.tif` and `roughness_on_grid.tif` (rules 05c and 09b), so the subgrid
tables of rules 10 and 13 and the main grid see the n the bed was fitted for.

- *Discharge.* GloFAS `mean_discharge_m3s` at the seed crossings, accumulated
  through the network like the calibration discharge (summed at confluences,
  width split at bifurcations, canal arms excluded), paired with the FathomDEM
  surface taken as the mean water level.
- *Nodes* (`src.gvf_target`). Every `water_surface.node_spacing_m` (500) along each reach,
  ends included. Reach lines are reversed where they run upstream
  (`orient_downstream`). Reach ends meeting at a junction share one node; ends
  either side of a structure stay separate, so the level can drop across it.
- *Target level.* Rule 09's water surface, interpolated by chainage onto the
  same nodes (same oriented lines, spacing and structure edges).
- *Structures* (rule 09; the surface and node network are split there only with
  `split_at_structures`), at junctions only: the reach level drops by more than
  `step_threshold_m` (0.5), or either reach is a SWORD obstruction
  (`obstr_type` 1-4, `use_sword_obstructions`) with a drop of at least
  `structure_min_drop_m` (0.2), or the upstream reach is listed in
  `structures`. A downstream rise above `step_threshold_m` is inconsistent data
  and is cut likewise (`cut_rising_edges`).
- *Node classes.* `water_body`: wider than `water_body.min_width_m` (1000) and
  mostly sea class (fraction > `sea_class_fraction`, 0.5) or with the DEM at the
  GEBCO floor; bed from the DEM, level from the target. `fixed_bed`: other
  mostly-sea nodes; bed from the DEM, forward solve only. `invert`: bed
  estimated. In the channel rule 10 burns, every `water_body` and `fixed_bed`
  node then takes the lower of its DEM bed and the bed of the nearest estimated
  node upstream (deepest one at a confluence; `src.gvf.carry_bed_to_sea`), so
  GEBCO's coarse coastal bathymetry leaves no sill where the river enters the
  sea; the solve itself is unchanged.
- *Forward model* (`src.gvf_solver`), rectangular channel of width w,
  discharge Q, Manning n:
  `dh/dx = (S0 − Sf) / (1 − Fr²)`, `Sf = n² Q² / (A² R^(4/3))`, `A = w h`,
  `R = w h / (w + 2h)`, `Fr² = Q² / (g w² h³)`. Subcritical, integrated
  upstream with RK4 (substeps ≤ 50 m), bed and width linear between nodes;
  where Fr would reach `froude_cap` (0.95) the depth is held at that Froude
  number and the node is flagged. Reaches are solved downstream first. The
  downstream level of a reach is the simulated level at its end node (mean over
  the downstream reaches at a bifurcation), the target above a structure,
  `calm_sea_level_m` at a sea mouth, and the target at any other network end. A
  network end is a sea mouth unless it is a free outflow (`is_delta_outflow`),
  and only if its last node is `water_body`/`fixed_bed` or its target is within
  `mouth_level_tol_m` (0.5) of sea level.
- *Inversion.* Start from the Manning normal depth on the target slope (slope
  at least `slope_floor`, 1e-5); nudge the bed `nudging.passes` (3) times by
  `alpha` (1.0) × (target − simulated level); then, per reach downstream first,
  bounded least squares on
  `[η_sim − η_target ; ρ (S_bed / s_scale − 1) ; w_p (h − h_p) / h_p]` with
  `least_squares.rho_m` (0.05) and `prior_weight_m` (0.05). The slope term pulls
  the bed slope towards the target slope, not flat, and acts only between two
  inverted beds; the prior term keeps the depth near normal depth h_p where a
  near-flat target (tidal delta) carries no depth information.
  `least_squares.apply`: `all`, `flagged` (reach level RMSE above `rmse_tol_m`,
  0.10, after nudging) or `none`. Depth stays within `h_min_m`-`h_max_m`
  (0.5-30) below the target.
- *No-flow reaches* (Q ≤ `q_min_m3s`, 1.0: tidal side channels, canal arms)
  cannot be inverted; their depth comes from a width-depth fit h = c w^b of the
  inverted reaches, and their level is ignored at a junction with a flowing
  reach.
- The bed at each model centreline cell (`src.gvf.values_at_points`) is the
  absolute bed elevation (GOCO06s) used as burn anchor.

**Burn.** Each reach is burned independently from its own bed anchors
(`src.river_burn.burn_river_channel`); junction anchors take the mean of the
neighbouring reaches, extrapolation is clamped, and seed reaches get a flat end
cap. The burn is done once, on the subgrid pixel grid, clipped to the coarse
channel mask, as `river_burned_subgrid.tif`; the main-grid `dep` of channel cells
is the per-cell mean of those pixels (`river_burned_dem_sfincs_grid.tif`).
Rounds 1-2 and the production skeleton read the same file, and the skeleton
raises if it is not on its own subgrid pixel grid. Discharge points are snapped
onto centreline cells of the channel.

**Crest.** Per weir edge:
`crest = ceil_cm(max(zsmax_round1(water side), coastal_protection_crest_m) + freeboard_m)`,
with `freeboard_m` 0. Round 1 has no overbank relief, so its maximum level holds
the calibration discharge and storm tide including coastal amplification. An
edge whose water side never wets takes the coastal crest. Crests are rounded up
to the next centimetre (`src.surge.ceil_crest`), and `sfincs.weir` is rewritten at
1 cm (`src.sfincs_run.write_weir_file`) because hydromt-sfincs writes 0.1 m.
Crests are calibrated at present-day sea level.

**Weir geometry** (`src.protection_weir.build_coastal_protection_weir`). One
continuous line along the coast and both river banks, built on the model grid
edges between land and water-like cells (sea plus channel). Land or water
patches smaller than `min_component_cells` (30) are found with 8-connectivity
and dissolved into the other class, so no ring is drawn around them and no gap
is left in the larger feature. With `unprotected_ocean_wetlands: true`, wetland
classes 80/90/95 touching the open sea stay seaward of the dike. Weir discharge
uses `weir_par1` (Cd 0.6).

## Forcing

**River (rule 07).** Crossings are the network reaches cutting the domain
outline; each is moved to where it enters the domain and must flow into the
network. Each takes the GloFAS v4.0 cell (`dis24`, daily, 0.05°) with the
highest mean discharge within `glofas_search_radius_km` (5 km), at least
`glofas_min_mean_discharge_m3s` (10). Extreme-value analysis
(`src/extreme_values.py`) needs at least `min_days` (9131, about 25 years):

- bankfull discharge: annual maxima, GEV, at `bankfull_rp_yr` (1.5);
- return-period table: peaks over threshold with a GPD fit (L-moments),
  threshold searched for 1-5 peaks per year, evaluated at 212 standard return
  periods (1, 1.5, 2, every 5 years to 1000, every 1000 years to 10000);
- `preview_rp_yr` (100) is a diagnostic; it and the bankfull value must be finite
  for the crossing to be forced (`has_glofas`). Any other fit failure stops the
  rule.

GRDC bias correction (`bias_correction.enabled`, off by default) maps GloFAS onto
the nearest GRDC gauge within 15 km by empirical quantile mapping (200
quantiles, linear tail extrapolation) when the overlap is at least 730 days; the
reported correlation is restricted to days at or above the 90th GRDC percentile.
The GRDC extract has 83 stations covering only the test basins; other basins get
no correction.

**River (rule 13).** A scenario's return period is looked up in the table by
log-RP interpolation. `event_hydrograph.source: glofas_event` uses the real
GloFAS flood of the main inflow whose peak is nearest that level, scaled to it
(above the record: the mean of the `n_largest` events, each scaled); other
inflows carry their own discharge on the same dates times the same factor.
`cosine` uses a synthetic wave of `period_hr` on the mean discharge. The event
blends from the spin-up mean into the flood over
`max(min_ramp_hr, gap / (max_rise_per_hr × peak))`. `compound.lag_hr` shifts discharge relative to surge
(positive = later); in `coastal_only` the discharge points stay with zero flow.

**Surge.** Rule 07a picks 6-10 COAST-RP stations (deduplicated within 1 km,
search radii 10-1000 km). Return levels are exact table lookups at the
tabulated return periods; each station forces the nearest water-level boundary
cell within `water_level_buffer_m`, and SFINCS interpolates between them. With
hydrographs enabled the event is the average tide plus a surge shape peaking on
tidal high water over ±`window_hr` (60 h; 168 h for `2433835`), shapes taken from
historic GTSM events near each return level (HGRAPHER, Dullaart et al., 2023;
GTSMv3 hourly reanalysis 1950-2024). The river peak is aligned with the surge
peak. In `river_only` the sea boundary is flat at
`-(gebco_max_depth_m + 0.5)` m.

**Sea-level rise.** Rule 07 stores only a dimensionless fingerprint per station:
local over global-mean AR6 regional sea-level change (Garner et al., 2021, Zenodo
DOI 10.5281/zenodo.5914710; values in mm; `ssp245`, medium confidence, 2100,
median; 1.0 when no AR6 location lies within 3°). Rule 13 adds
`fingerprint × slr_m`, ramped in from 0 over `sfincs.boundary_setup.slr_ramp_hr`
(24 h) after the event start because the shared spin-up has no SLR.
`surge_forcing.nc` never contains SLR, so changing `slr_m` reruns only the
scenario.

## Spin-up

Rule 14 runs once per basin for `spinup_days` with each crossing's constant mean
discharge and the sea ramped over `boundary_ramp_hr` from `calm_sea_level_m` to
each station's calm level, the levels every event starts from, so there is no
jump at the restart. It writes `sfincs.YYYYMMDD.HHMMSS.rst` at
`tref + spinup_days`, used by every scenario. `storevel` is 0 in the spin-up.

The initial water level (`zsini`) is `calm_sea_level_m` on the sea cells of
`zsini_sea_cells_on_grid.tif` and dry on land; sea cells outside the largest
connected sea body that lie below that level start dry at bed level
(`src.raster.dry_isolated_sea_cells`).

## Protection levels

Rule 04 (`src/protection_levels.py`) takes the dominant WRI Aqueduct
`geogunit_107` unit (largest pixel count) inside the delta polygon, read in a
window, and its FLOPROS (Scussolini et al., 2016) design return periods. About
74% of units have a riverine value and about 25% a coastal one; each hazard falls
back to the other hazard, then to `flopros_range.default_rp_yr` (2), and is capped
at `max_rp_yr` (1000). Outputs: `river_protection_rp_yr` and
`surge_protection_rp_yr` (with `*_raw_yr` and `*_source`).

The coastal value sets `coastal_protection_crest_m` (station-mean COAST-RP level
at that return period plus the mean MDT), the floor of every weir crest, and the
round 1-2 storm tide of the calibration. The riverine value sets the calibration
discharge. Protection is represented only by the calibrated weirs; forcing is
never reduced by it.

## Adaptation

Strategies (`config/adaptation_strategies.yml`) combine measures
(`config/measures.yml`: `offshore_barrier`, `pumps`, `nbs_land_reclamation`,
`water_retention`, `river_levee`, `coastal_levee`, `urban_raising`, `retreat`).
Families (`advance`, `grey_protect_open`, `NbS_protect_open`, `protect_closed`,
`accommodate`, `retreat`) are written at scale 1 and expanded over `scales`
(0.4, 0.9, 1.0) into `<family>_04`, `_09`, `_1`; only parameters flagged
`scale: true` are multiplied, then checked against their bounds. A parameter is a
number, a `{coastal, river, compound}` map picked by the scenario's `event`, a
run-time value (`max_coastal_water_level`, `max_river_water_level`,
`max_discharge`), `current_protection` (never scaled) or `{fixed: x}`. Every
strategy and event combination is validated when the DAG is built.

A measure's `locations` is a geometry under `<raw_data_root>`, or `auto/<name>`:
a line that rule 13b `build_adaptation_lines` derives per basin from the
skeleton's baseline weir (`src.adaptation_lines`), so it follows the grid's cell
edges. Each weir segment is labelled by the water it faces: coast (open sea,
plus sea-linked wetlands when `unprotected_ocean_wetlands`), river (channel), or
for a segment between two land cells the water around the small merged island
it faces. Every SWORD outlet
reach gets a closure: the shortest straight crossing within 60° of perpendicular
to the river at the outlet reach's upstream end that reaches protected land on
both sides, snapped to cell edges. Sea cells the closures cut off from the open
sea form the estuary; coast segments facing it become estuary banks. The pump
line runs along the river centreline from 200 m upstream of the closure to the
outlet reach's seaward end (inland to sea). Outputs: `coastal_protection_weir_PC`
(coast + closures, `protect_closed`), `coastal_protection_weir_PO` (coast,
`grey_protect_open`), `river_protection_weir_PO` (river and estuary banks) and
`pump_river_mouth`. The rule fails when the river behind a closure is still
connected to the open sea.

| method | rule | approach |
|---|---|---|
| `pre` | 18a | edits a copy of the skeleton (weirs, drainage structures, subgrid), reuses the basin spin-up restart and reruns SFINCS (rules `adapt_build_forcing_pre`, `adapt_run_event_pre`, `adapt_flood_metrics_pre`) |
| `post` | 18b | chains each measure on the baseline `max_flood_depth.tif` without SFINCS |

Rule 18c classifies the baseline flood map by dominant source: 1 river,
2 coastal, 3 compound, 4 permanent water (wet in the spin-up, overriding the
others), from the river-only and coastal-only sibling scenarios.
`baseline_excess_volume.json` is the excess volume of classes 1 and 3, which
sizes water-retention measures. Rule 19 gathers every baseline, `pre` and `post`
`flood_metrics.csv` of a basin into `runs/metrics_comparison.csv`.

The adaptation rules have not been run on a rotated grid.

## References

Full citations and versions of the datasets: [data.md](data.md).

- Dullaart et al. (2023): HGRAPHER storm-tide hydrographs.
- Edmonds, D.A., Caldwell, R.L., Brondizio, E.S. et al. Coastal flooding will
  disproportionately impact people on river deltas. Nat Commun 11, 4741 (2020).
  https://doi.org/10.1038/s41467-020-18531-4
- Garner et al. (2021), IPCC AR6 Sea Level Projections, Zenodo DOI
  10.5281/zenodo.5914710.
- Kvas et al. (2021), GOCO06s — A satellite-only gravity field model. Earth
  System Science Data 13(1), 99–118.
- Neal et al. (2021), WRR 57, e2020WR028301.
- Pavlis et al. (2012), The development and evaluation of the EGM2008. J.
  Geophys. Res. 117, B04406.
- Scussolini, P., Aerts, J.C.J.H., Jongman, B., Bouwer, L.M., Winsemius, H.C., de
  Moel, H., Ward, P.J. (2016). FLOPROS: an evolving global database of flood
  protection standards. Natural Hazards and Earth System Sciences, 16, 1049-1061.
