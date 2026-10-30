# Methods

How the pipeline builds, runs and adapts a SFINCS compound flood model for one
delta. One `basin_id` (the `BasinID2` of the delta polygons of Edmonds et al.,
2020) is one delta and one model. Rule numbers match the files in
`workflow/rules/` and `workflow/scripts/`; config keys are in
`config/config.yml`. Dataset sources and versions: [data.md](data.md).

## Pipeline stages

Rules are numbered in the order they run. Rules 01-11 are basin-level and shared
by all scenarios. Rules 12-16 split into a basin-level part (skeleton, spin-up)
and a per-scenario part (forcing, event run, metrics). Rules 17-18 run per
scenario and strategy. Diagnostics that feed no rule are stand-alone scripts in
`analysis/` (river-network map, upstream-boundary check, bifurcation options).

**01 `determine_model_domain`.** Extracts the basin's delta polygon from the
delta-polygon dataset, picks the UTM CRS, buffers the polygon by
`domain.delta_buffer_m` for the clipping box. Rule 04 fits the axis-aligned
model grid to the polygon (`src/grid.py:build_grid_def`).

**02 `get_river_network`.** Clips SWORD to the domain and applies the manual
widths. It runs before the land use because the network is the barrier of the
open-sea fill.

**03 `get_protection_levels`.** Finds the delta's FLOPROS protection standards;
see [Protection levels](#protection-levels).

**04 `build_sfincs_grid`.** Subdivides the grid frame at the basin resolution
and writes `{basin_id}_sfincs_grid.json`, the one grid every later rule uses; see
[Grid](#grid).

**05 `prepare_landuse`.** Writes the basin land-use raster in pipeline codes
(source classes, 200 = open sea, 255 = nodata), WGS84 at the source resolution,
and its land (land use ≠ 200) as polygons, the background of figures made before
the model grid exists. `landuse.source` selects Copernicus LC100 (100 m) or ESA
WorldCover (10 m). See [Land use and the sea](#land-use-and-the-sea).

**06 `get_elevation`.** Merges FathomDEM and GEBCO into `elevation_merged.tif` in
the GOCO06s frame, and writes the land use on the same grid (`landuse.tif`, mode),
whose sea class blanks FathomDEM over water; see
[Datums](#datums-and-vertical-references) and [Terrain merge](#terrain-merge).

**07a-c `select_surge_stations`, `extract_gtsm_series`, `storm_tide_hydrographs`.**
Select the COAST-RP stations that force the sea boundary, extract their hourly
GTSM water level and surge from the station subset (built once for all basins by
`tools/subset_gtsm_stations.py` after `select_surge_stations` has run), and derive per-station average tide and per-RP surge
shapes (enabled by `boundary_forcings.surge.hydrograph.enabled`).

**07d `get_boundary_forcings`.** Writes `surge_forcing.nc` and `river_forcing.nc`:
storm-tide return levels, calm-sea levels, the coastal protection crest and SLR
fingerprints per station; return-period tables, bankfull and mean discharge per
river crossing. See [Forcing](#forcing).

**08 `clean_river_network`.** Keeps the network reachable downstream of the rule
07d crossings, fixes widths and routes bankfull discharge through it; see
[River network](#river-network).

**09 `grid_align_landuse`.** The land surface with the river channel (each
cleaned reach buffered by `width/2`, `src.river_burn.channel_polygons`):
`landuse_channels.tif` on the elevation grid, every non-sea channel pixel set to
water (`landuse.water_code`, 80), so built-up or crop pixels under the river
count neither as exposed nor as dry land; `roughness.tif` on the elevation grid
subdivided to about the land-use pixel size, area-averaged from the land-use
source rather than taken from the dominant class
(`landuse.roughness_aggregation`: `area_mean` or `conveyance`, the harmonic
mean), every channel pixel at `river_processing.channel_manning_n`, the n the
river bed is fitted with (rule 11); and land use (mode) and Manning's n
(area-averaged, same channel n) on the model grid, the only grid-level land-use
and roughness fields downstream, with the grid-aligned land mask.

**10 `river_surface`.** Builds the river water surface and its slope from
FathomDEM and conditions the DEM to it (`src.water_surface`, settings under
`river_processing.water_surface`). FathomDEM's channels carry a flattened water
surface: flat per water body, dropping in steps, with stray peaks (bridges,
bars, bank pixels) and dips. On the node network (see *Network* under
*Depth*), which rule 11 rebuilds with the same node ids:

- *Node median* of the water pixels (land use = `landuse.water_code`, 80)
  along the centreline within `median_half_window_m` (1000) of each node, the
  window running on into the neighbouring reaches; islands, bars and banks the
  centreline crosses are skipped, so their nodes take the level of the water
  around them (gaps are filled from neighbouring nodes), and a short lock or
  bridge reach is outvoted by the river around it.
- *Node class* from the DEM and `landuse_channels.tif` over a disc of the
  channel width: `water_body` (wider than `water_body.min_width_m`, 1000 m,
  and mostly sea class or at the GEBCO floor), `fixed_bed` (mostly sea class),
  else `invert` (river).
- *Sea pin.* A network end is a sea mouth unless it is a free outflow, and
  only if its last node is not a river node or its FathomDEM level is within
  `mouth_level_tol_m` (0.5 m) of `calm_sea_level_m`; no mouth fails the rule.
  The mouths are set to `calm_sea_level_m` and every node median below it is
  raised to it: FathomDEM flattens open water to 0 m on the EGM2008 geoid and
  the lower rivers join that level, while the model's sea is the geoid plus
  the MDT, so the flattened lower river sits up to ~1.3 m below it (620947,
  1475442, 3279946), and no steady river level lies below its sea.
- *Fit*: non-increasing downstream by a length-weighted isotonic regression on
  the whole node graph (Hildreth's dual coordinate ascent, in node-disjoint
  edge batches, so loops and junctions are solved together); smoothed over
  `smoothing_m` (5000), ends mirrored so a slope runs through; non-increasing
  again. The mouths are held fixed throughout, so the mouth level survives.
  Structures detected where the reach level drops (see *Structures*) and
  rises of more than `step_threshold_m` are reported and smoothed across:
  weirs and locks open in floods, and a mean-flow level drop kept as a step
  would become a step in the flood bed.
- *Slope*: the same fit smoothed over `slope_smoothing_m` (40000) instead;
  its fall per metre at each node along its largest-Q path (routed bankfull
  discharge of rule 08, `bankfull_discharge_acc_m3s`): the central difference
  inside a reach, and at a junction the edges of the largest-Q up- and
  downstream reaches, so a small arm does not set the slope of the main
  channel. Not floored. FathomDEM's surface on low-gradient rivers is a
  staircase of flat treads; over a few km its slope is near zero on the treads
  and steep at the risers, which puts the depth on the slope floor and the bed
  in steps. The level keeps the short smoothing: a long one rides metres above
  a concave profile (2209166: p90 4.5 m at 20 km, 11 m at 40 km), and the bed
  would rise with it.
- *SWOT check*: per reach with a reliable positive SWOT `slope_obs_p50`, the
  reach-mean slope of the slope surface over SWOT's, logged per reach and
  summarised (median, share within a factor 2), written to the node table and
  the figure title. It is not used in the surface.

*Smoothing lengths and the SWOT check, nine test basins*
(`analysis/check_river_bed.py`; level at 5 km throughout; slope at 5 km vs 40 km):

| basin | SWOT reaches | slope / SWOT median | within ×2 | nodes on the slope floor | bed steps > 1 m |
|---|---|---|---|---|---|
| 2433835 Ebro | 3 | 1.23 → 1.44 | 67 → 67 % | 54 → 29 % | 3 → 0 |
| 1248635 Rhine-Meuse | 24 | 0.01 → 0.25 | 4 → 21 % | 77 → 66 % | 244 → 147 |
| 1416812 | 12 | 0.68 → 0.92 | 50 → 75 % | 49 → 25 % | 63 → 20 |
| 1475442 | 5 | 0.85 → 0.93 | 60 → 80 % | 11 → 0 % | 6 → 0 |
| 2209166 | 3 | 0.83 → 1.15 | 100 → 100 % | 0 → 0 % | 36 → 34 |
| 2444235 | 2 | 2.55 → 1.69 | 0 → 0 % | 25 → 0 % | 4 → 0 |
| 3279946 | 24 | 0.41 → 1.59 | 25 → 58 % | 48 → 25 % | 441 → 59 |
| 4267691 Mississippi | 49 | 0.19 → 0.88 | 20 → 41 % | 77 → 52 % | 694 → 358 |
| 620947 | 1 | 0 → 0 | 0 % | 100 → 100 % | 0 → 0 |

At 5 km the slope on the low-gradient deltas is mostly too small against SWOT
(Rhine-Meuse, Mississippi, 3279946: median ratio 0.01–0.4), because it is near
zero on FathomDEM's flat treads; 40 km brings the median ratio to 0.9–1.6 and
halves the floor share and the bed steps. Where FathomDEM is flat at sea level
over the whole delta (620947, most of Rhine-Meuse), no smoothing length gives
a slope, and the depth stays on the floor. Smoothing the level itself over
20 or 40 km gives the same slopes, but the level then departs from FathomDEM
(p90 |level − FathomDEM|: 2209166 2.5 → 4.5 → 11 m, Ebro 0.5 → 1.5 → 2.1 m at
5 / 20 / 40 km), hence the two lengths. A 60 km slope lowers the floor share
further on some basins (Ebro 29 → 11 %) with no fewer bed steps, and is as
long as the whole modelled river on small deltas.

Every land pixel of each channel buffer (`width/2`) is set to the surface
(lower wins where buffers overlap); sea pixels keep their GEBCO bed. Writes the
node table (`river_water_surface.gpkg`, layer `nodes`: level, node median,
slope, node class, DEM, width, mouth flag, SWOT slope and ratio per reach;
layer `structures`: the detected structures), the conditioned DEM at native
resolution and on the model grid, and `river_elevation_max.json` (highest
surface level), which caps the active cells at
`river_elevation_max_m + sfincs.grid.active_mask.elevation_buffer_m`. The
figure `figures/analysis/10_river_surface.png` shows, per seed along its widest arm, the
FathomDEM water pixels, node medians, the surface, structure junctions by SWORD
obstruction type, and the SWOT check of the path.

**11 `modelled_depth_estimation`.** Sets river depth, burns the channel and
calibrates the coastal and riverbank weir crests with a disposable SFINCS model;
see [River depth and crest calibration](#river-depth-and-crest-calibration).
Also writes `zsini_sea_cells_on_grid.tif`: land-use sea minus pockets protected
by the weir, the sea mask of the initial condition, flood metrics and
adaptation.

**12a `build_sfincs_skeleton`.** Builds the scenario-independent model once per
basin: grid, elevation, masks, weir, roughness, subgrid table, observation
points and the default initial water level. Nothing is reprojected here: every
input is already on the model grid. Not runnable on its own (no forcing, no
times).

**12b `land_bed_level`.** Once per basin: the skeleton's subgrid bed level on
the grid flood depths are computed on, masked to NaN on open sea (the corrected
`zsini_sea_cells_on_grid.tif`). Spin-up, sanity checks, event runs, flood
metrics and attribution read it, so its non-null pixels are the land domain of
every flood statistic; `adapt_apply_pre` writes the same file for each adapted
skeleton.

**13a `run_spinup`, 13b `sanity_checks`.** One spin-up per basin; see
[Spin-up](#spin-up). Rule 13b reports the share of land pixels wetted during
spin-up and animates it.

**14 `build_sfincs`.** Per scenario: loads the skeleton, writes only the
water-level and discharge forcing, and hand-writes `sfincs.inp` with relative
references to the skeleton files (HydroMT's config writer makes references
outside the model root absolute). It runs after the spin-up, whose restart file
the event starts from.

**15 `run_event`.** Runs the scenario's `sfincs.inp` and writes the
inundation-ratio map; with `analysis: true` also an animation and
`flood_timeseries.csv` (about 12 min per run). The flood volume
there is the volume present at each map output (every `dtmapout`), so its maximum
is the peak volume at that sampling; summing each pixel's maximum depth would
overstate it.

**16 `compute_flood_metrics`.** Downscales the maximum water level to the subgrid
bed (`downscale_floodmap`) as `max_flood_depth.tif`, masks the sea, and writes
`flood_metrics.csv`: domain, land and urban areas, flooded area and share
(depth ≥ `metrics.hmin`), urban exposure (land use `metrics.urban_landuse_code`),
mean and maximum depth, volume.

**12c, 17a-18** adaptation; see [Adaptation](#adaptation).

**Diagnostics (`analysis/`).** `plot_river_network.py` maps the clipped network.
`check_upstream_boundary.py` checks that river forcing points lie beyond the
sea's influence on the river level: steady GVF levels on the round-0 bed over
the whole network (`src.gvf_solver.forward_network`) at mean and bankfull
discharge, with the sea mouths at the nearest station's calm level and at
SWL_ext: per station the highest storm tide plus SLR × fingerprint over every
scenario in `scenarios.yml` with a surge RP and the stress case
(`testing.upstream_boundary_check.stress_surge_rp_yr`, `stress_slr_m`; RP100 +
1 m). Along each forcing point's main stem, the sea's influence ends
where the rise of the river level falls below `max(level_tol_m,
amplitude_threshold_fraction × A0)`; at mean discharge the forcing point
passes if its rise is below `level_tol_m` and it lies `margin_km` beyond that
end. `benchmark_upstream_boundary.py <basin> ...` reruns built models steady
at mean discharge with the sea calm and at SWL_ext and compares SFINCS with
GVF along the main arms (`figs/upstream_boundary_benchmark/`, with a
generated README of method, results and parameter choice); in three basins
(low gradient with small and with 4 m surge, one steep) SFINCS put the
influence end within 2.9 km of the GVF estimate, with no amplification along
the channels, which sets `margin_km` 5. Bankfull discharge moves the end
seaward on the steep river (−40 %) and by 1–2 % on the low-gradient ones.
The 10 m above MSL rule (`high_gradient_elevation_m`) is reported for
comparison only: on a low-gradient river with a 4 m surge the 10 m contour
lay only 0.5 km beyond the influence end. Method: `HANDOFF_upstream_boundary.md`.
`check_bifurcation_options.py` compares
bifurcation discharge splits with and without the manual SWORD widths.

## Datums and vertical references

Every elevation and water level is in the GOCO06s geoid frame.

| quantity | source frame | to GOCO06s |
|---|---|---|
| FathomDEM | EGM2008 | add `N_EGM2008 − N_GOCO06s`, synthesised with pyshtools on the WGS84 ellipsoid (boule), EGM2008 truncated to the GOCO06s degree (about 300) |
| GEBCO | local mean sea level | add the MDT: `H_GOCO06s = H_MSL + MDT` |
| COAST-RP storm-tide levels | local mean sea level | `storm_tide_at_rp_m = storm_tide_at_rp_raw_m + mdt` |

The MDT is CNES-CLS22 (HYBRID-CNES-CLS2022), the mean sea surface above the
geoid. It is used twice: on GEBCO in rule 06 and on the COAST-RP levels in rule
07. It is NaN over land: rule 06 loads it with a `datum_correction.mdt_load_margin_deg`
(5°) margin and extrapolates it over land by inverse-distance fill before
resampling, so nearshore GEBCO pixels are not lost; rule 07d takes the nearest
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
`{ns}{lat:02d}{ew}{lon:03d}.tif` (e.g. `n00e010.tif`); rule 06 divides by 100.
Steps (`06_get_elevation.py`):

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
so rule 05 derives 200 (`src/landuse.py`): WorldCover water that reaches the
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
| orientation | axis-aligned (north-up), `rotation` 0 |

`{basin_id}_sfincs_grid.json` (rule 04) holds `x0, y0, dx, dy, mmax, nmax,
rotation, epsg` and the affine transform in SfincsModel orientation:
`translation(x0, y0) × scale(dx, dy)` with `dy` positive, so row 0 is on the
south side, row = SFINCS `n`, column = `m`. The grid is flipped to y-ascending
when the JSON is written, so it matches the grid HydroMT builds. Models are created with `src.grid.create_model_grid`, which
raises if the SfincsModel transform differs from the JSON.

On the south-up model grid rioxarray's `rio.bounds` and `rio.reproject` can
crop or raise; use HydroMT's `.raster.*`.

SfincsModels are created with `mode="w+"`, never `r+`; an existing model is
opened read-only (`"r"`) and redirected to a new root with
`sf.root.set(<root>, mode="w+")`.

## River network

The network is SWORD v17c (`river_network`: one GeoPackage per region, layer
`reaches`), read by `src.sword.read_sword_reaches`, which maps the published
topology (`rch_id_dn_1..4`, `rch_id_up_1..4`) to the `rch_id_dn`/`rch_id_up`
list strings used downstream, and reverses every reach line that runs upstream
(SWORD v17c draws all reaches of the current basins from mouth to source), so
every rule reads lines that run downstream. Rule 02 then overwrites `width` on the reaches in
`config/sword_width_corrections.csv` (from satellite-image inspection; the rule
stops if a reach's SWORD width no longer matches the expected value) and marks
them `width_source = manual`. Rule 11's bifurcation comparison runs the network
without and with these widths. Reach ids are normalised with `str(int(float(x)))`
(`src.river_network.normalize_reach_id`) before any dict or set operation;
`rch_id_dn` is a list string split on commas and whitespace.

Rule 08, in order: fix T-junction tails; walk downstream from the rule 07d
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

Rule 11 is the only source of river depth and weir crests. Round 0 sets the
river bed from the water-surface slope (no SFINCS run); one disposable SFINCS model (production grid
and subgrid set-up, production weir geometry) then runs twice:

| round | weir | channel | sea boundary | purpose |
|---|---|---|---|---|
| 0 | -- | estimated bed | -- | river bed (no SFINCS run) |
| 1 | every edge at `weir_crest_m` (1000 m) | burned bed | storm tide at the FLOPROS coastal RP, held | crest per weir edge |
| 2 | production crests | as round 1 | as round 1 | verification: reports overtopped edges, changes nothing; runs only with `verify_crests: true` |

Both SFINCS rounds hold the sea and ramp only the rivers:

- sea: every station at its storm tide at the FLOPROS coastal return period from
  the start; sea and river channel cells start no lower than the highest
  station's storm tide, so no rising sea runs up the channels and the level at
  the mouth responds to the river alone (`analysis/benchmark_peak_travel.py`
  times it);
- rivers: each crossing from its reach's mean discharge (the flow that holds the
  initial river level, below) to the discharge at the FLOPROS riverine return
  period (`river_protection_rp_yr`, log-RP interpolated from the discharge
  table; bankfull when absent), over `ramp_hr` (6 h) with the half-cosine weight
  `0.5 (1 − cos πs)` used for every river forcing (`src.river_forcing.cosine_ramp`),
  then held to the run end.

The run lasts `ramp_hr + T_walled + calibration_margin_hr` (12 h), so the full
discharge reaches the coast and every level settles before the run ends. `T_walled`
is the slowest travel time from a crossing to its network end with the whole
calibration discharge in the channel (no overtopping), along the crossing's mainstem
(the larger-discharge arm at each bifurcation) on round 0's bed (steady GVF level, sea
held at the highest station level; `src.river_profile.walled_travel_time`). Per node
the wave speed follows the regime, from the linearised friction rate
`r = g n² u / h^(4/3)` against the ramp `T = ramp_hr`: where `r·T` is below
`wave_regime_rT[0]` (0.3) friction acts slower than the forcing changes, and the
change travels as a gravity wave, `u + √(gh)` (wide, deep, slow reaches: 1248635's
Meuse water bodies, 10 m deep at 0.015 m/s, `r·T` ≈ 0.15); elsewhere it takes the
kinematic speed `5/3·u`, the in-bank flood-wave speed of a wide Manning channel,
which is also the slower one in the transition up to `wave_regime_rT[1]` (3)
(1248635 Waal, `r·T` ≈ 12). The kinematic speed alone gave 516 h on 1248635's
80 km Meuse path, the regime speed 51 h. All-kinematic and all-gravity times are
logged for comparison. In the earlier 24 h runs, round 1 settled within
about 4 h of `ramp_hr + T_walled` on 2433835 (64 km path, `T_walled` 6 h), and on
3279946 (288 km, `T_walled` 34 h) the upper 150 km still rose 0.2–0.8 m in the
last 3 h. Runs longer than `calibration_max_days` fail. After each round,
`steady_check` warns when centreline cells still rose more than `tol_m` over the
last `window_hr`.

Round 2 reports some overtopped edges even though its inputs equal round 1's
except the crests: lowering the crests from 1000 m changes SFINCS's solution by
a few cm even where no water reaches them, and a few unstable cells (levels
swinging metres per hour) amplify it. 2433835: with every crest 1 m above any
water, levels still differ by up to 0.09 m; round 2 overtops 200 of 2,020 edges,
mostly by a few cm. Accepted as minor; crests are not raised for it.

**Depth** (`src/river_bed.py`, settings under `river_depth_modelling.bed`). At
the low Froude numbers of these rivers the water-surface slope equals the
friction slope, also in backwater, so the local depth follows from Manning with
the surface slope; rule 10's surface places the bed. The channel the bed is
burned into carries `river_processing.channel_manning_n` (0.03) in
`roughness.tif` and `roughness_on_grid.tif` (rule 09), the n the depth is
computed with.

- *Network.* Nodes every `water_surface.node_spacing_m` (500) along each reach,
  reach lines oriented downstream, junction ends shared (`src.gvf_target`);
  rules 10 and 11 build it identically, and rule 11 checks rule 10's node ids
  and positions against it. Structures are open: no level step at weirs or
  locks, which open in floods.
- *Discharge.* GloFAS discharge at the seed crossings, accumulated through the
  network like the calibration discharge (summed at confluences, width split at
  bifurcations, canal arms excluded): `bankfull_discharge_m3s` with
  `design_discharge: bankfull` (default), `mean_discharge_m3s` with `mean`. The
  depth hangs below the mean-flow level, so `bankfull` treats that level as the
  bankfull stage: the bed sits lower by `h_bankfull − h_mean` than with `mean`.
- *Level and slope*: rule 10's surface and slope at every node.
- *Depth*, at every node: Manning normal depth of a rectangular channel
  (hydraulic radius `W h / (W + 2h)`, the section of the GVF solver) at
  `max(slope, slope_floor)` (1e-5 m/m). Normal depth diverges on a flat
  surface, so the floor bounds the depth there while the surface keeps its own
  slope. W is the nearest SWORD node width, scaled so each reach's median
  matches the pipeline reach width (manual corrections included), as a running
  median and then mean over `width_smoothing_m` (5 km) of network, across reach
  boundaries; raw node widths give ±5 m bed jumps (4267691: 172 steps > 3 m),
  and the per-reach scaling alone steps the width at reach ends. Nodes with
  Q ≤ `q_min_m3s` (1.0) take `h = c W^b` fitted to the others.
- *Bed* = surface − depth. Where the DEM is GEBCO bathymetry rather than a
  FathomDEM water surface, i.e. at `water_body` nodes whose DEM is below
  GEBCO's shallow clamp (`calm_sea_level_m − gebco_min_depth_m`; a bed at the
  clamp means GEBCO was too shallow and would form a sill), the bed is the
  DEM; river beds within `bathymetry_handover_m` (5 km) of such a node blend
  linearly into it.
- *Forward check*: a steady GVF solve on the bed over the whole network at the
  design discharge, from the surface at the network ends
  (`src.gvf_solver.forward_network`), minus the surface. The depth uses the
  40 km slope and the floor while the surface keeps the 5 km smoothing, so they
  differ by design: positive where the bed carries the design discharge above
  the smoothed FathomDEM level (FathomDEM's flat, sea-level lower rivers; the
  floor), negative where below. Logged for all river nodes and the largest-Q
  paths (p5/median/p95, share within 0.5 and 1 m) and mapped in
  `figures/analysis/11_calib_river_forward_check.png` (fixed −3 to 3 m).
- The bed at each model centreline cell (`src.river_bed.values_at_points`) is
  the absolute bed elevation (GOCO06s) used as burn anchor. Per-node results,
  with the forward level, are written to
  `visuals/11_calibration/river_bed/<basin>_river_bed_nodes.*`.

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
Crests are calibrated at present-day sea level. SFINCS keeps a weir face only if
its crest is above the face's subgrid minimum level (`uv_zmin`); a lower face
opens as plain subgrid flow, so later runs would not have round 1's dike
(2433835: 372 of 3,266 faces, round 2 up to 0.19 m off). Every crest is
therefore at least `uv_zmin` + 1 cm of its face (round 1's subgrid table, which
round 2 and rule 12a share); the bank holds water to that level anyway
(`src.protection_weir.floor_crests_above_face_zmin`). Round 2 then reproduces
round 1. Overtopping is reported above 1 cm, the crest precision.

**Weir geometry** (`src.protection_weir.build_coastal_protection_weir`). One
continuous line along the coast and both river banks, built on the model grid
edges between land and water-like cells (sea plus channel). Land or water
patches smaller than `min_component_cells` (30) are found with 8-connectivity
and dissolved into the other class, so no ring is drawn around them and no gap
is left in the larger feature. With `unprotected_ocean_wetlands: true`, wetland
classes 80/90/95 touching the open sea stay seaward of the dike. Weir discharge
uses `weir_par1` (Cd 0.6).

## Forcing

**River (rule 07d).** Crossings are the network reaches cutting the domain
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

*GloFAS is used uncorrected; GRDC gauges were evaluated and are not used.* The
evaluation matched the 15 inflows of the 8 basins with river forcing against
6,503 GRDC stations (four GRDC exports, 2026). A gauge counted when it lay
outside the domain, its drainage area was within ±30 % of the inflow's SWORD
`facc`, and its record held at least 10 complete years in 1979–2025. Reasons
for not using it:

- Few inflows have a gauge. One inflow has one at the inflow (Ebro, Tortosa,
  4.7 km, 39 overlap years); two more have one on the same river 21–41 km away
  (Rhine at Rees, Meuse at Venlo); 12 have no usable gauge within 50 km (the
  Red River gauge at Alexandria lies 1 km away but its record ends in 1983),
  five of them none with a plausible drainage area within 150 km. A correction would apply
  to a few inflows of a few basins, and correcting one inflow but not the
  others changes their ratio in every basin event (rule 14 scales all inflows
  by the main inflow's factor).
- Where a gauge matches, GloFAS shows no mean bias. GloFAS/gauge mean
  discharge on their common days (90 % year-block bootstrap interval): Ebro
  0.97 (0.93–1.02, 1979–2021), Rhine 1.02 at Rees and 1.04 at Lobith, Meuse
  1.02 at Lith and 1.08 at Venlo. GloFAS v4 is itself calibrated against about
  2,000 gauges (KGE'). Comparing record means from different periods suggests
  a bias that is not there: Ebro 289 m³/s (GloFAS, 1979–2026) against 421 m³/s
  (Tortosa, 1913–2021, largely before the reservoirs).
- Matching by distance picks the wrong river: the nearest gauge to the Meuse
  inflow (27,647 km²) is the Niers at Goch (1,203 km², 8 m³/s), 14 km away. Any
  use needs an area check, which in turn rejects most nearby gauges.
- Daily quantile mapping is unstable in the tail: beyond the overlap's range it
  extrapolates with the slope of the two largest overlap values, so the
  100-year discharge depends on a few days of record.

The evaluation stopped at this inventory; flood quantiles of GloFAS and the
gauges were not compared. `analysis/inspect_grdc_correlation.py` plots GloFAS
against the nearest GRDC gauge of each inflow (`figs/grdc_overview/`). The quantile-mapping code in rule 07d
(`boundary_forcings.river.bias_correction`) remains and stays off
(`enabled: false`).

**River (rule 14).** A scenario's return period is looked up in the table by
log-RP interpolation. `event_hydrograph.source: glofas_event` uses the real
GloFAS flood of the main inflow whose peak is nearest that level, scaled to it
(above the record: the mean of the `n_largest` events, each scaled); other
inflows carry their own discharge on the same dates times the same factor.
`cosine` uses a synthetic wave of `period_hr` on the mean discharge. The daily
means are joined by a monotone cubic (PCHIP), so the slope is continuous and the
peak keeps its value and hour. The rising limb starts at the main inflow's lowest
daily flow within `max_rising_days` before the peak; in storm-tide mode rule 07d
moves both peaks later to make room for it. Up to there the discharge holds the
spin-up mean and blends into the flood with a half-cosine weight over
`max(min_ramp_hr, π/2 × gap / (max_rise_per_hr × peak))`, ending at the rising-limb
start, so the blend's steepest rise is `max_rise_per_hr × peak` per hour.
Large rivers' floods rise over weeks, so their rising-limb start can still be well
above the mean. `compound.lag_hr` shifts discharge relative to surge
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

**No tide-gauge bias correction.** The surge forcing is not corrected against
GESLA v4.1 tide-gauge records. That option was tested on the nine basins and
rejected for these reasons:

- **Coverage.** Within 50 km of the domain, outside it, only three basins
  (Rhine–Meuse, Mississippi, Ebro) have an open-coast gauge with ≥ 10 complete
  years. Most nearby gauges are marsh, river or estuary stations inside the
  domain. A correction could therefore only be a per-basin option.
- **Screening.** After deduplication, step detection on the monthly
  gauge − GTSM difference and a hourly cross-correlation lag test, only the
  Rhine–Meuse kept usable gauges (Scheveningen, Cadzand, Brouwersdam,
  Noordwijk; 21-54 yr). The Gulf of Mexico gauges lag GTSM by 41-90 min (they
  sit in passes and bays), Tarragona by 47 min (42 km from its GTSM station,
  r = 0.77), and two Sabine-area records share a 9-11 cm datum step.
- **Datums.** Gauge datums (chart, station, harbour) are not tied to the
  GOCO06s geoid used here, so only variability, not absolute level, can be
  compared.
- **Circularity.** COAST-RP is itself derived from GTSM, so correcting GTSM
  against a few gauges would not make the return levels consistent across
  basins.

`analysis/inspect_gesla_gauges.py` reproduces the coverage counts and the
screening per gauge (`figs/gesla_overview/`).

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
The forcing's calm lead-in (`boundary_forcings.lead_days`) must equal
`spinup_days`, so the event run starts where the lead-in ends; the workflow
refuses to start otherwise.

The initial water level (`zsini`) is `calm_sea_level_m` on the sea cells of
`zsini_sea_cells_on_grid.tif`, the steady GVF level of the routed mean discharge
on rule 11's bed on the river channel cells (`mean_level_m` of the river-bed node
table, network ends at rule 10's surface), and dry on land; sea cells outside the
largest connected sea body that lie below that level start dry at bed level
(`src.raster.dry_isolated_sea_cells`). Rule 11's calibration rounds start from
the same river level, raised with the sea to at least the coastal protection
storm tide. Rule 10's surface itself is the bankfull stage when
`bed.design_discharge` is `bankfull`, 1-5 m above the mean-flow level on the
2433835, 2209166 and 3279946 rivers, which the spin-up then had to drain.

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
a line that rule 12c `build_adaptation_lines` derives per basin from the
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
| `pre` | 17b | edits a copy of the skeleton (weirs, drainage structures, subgrid), reuses the basin spin-up restart and reruns SFINCS (rules `adapt_build_forcing_pre`, `adapt_run_event_pre`, `adapt_flood_metrics_pre`) |
| `post` | 17c | chains each measure on the baseline `max_flood_depth.tif` without SFINCS |

Rule 17a classifies the baseline flood map by dominant source: 1 river,
2 coastal, 3 compound, from the river and coastal counterpart scenarios, and
4 embanked water, overriding the others: every pixel wet in rule 11's last
crest calibration round, i.e. the channel and foreshore up to the dikes at the
design discharge and coastal storm tide (present-day sea level).
`baseline_excess_volume.json` is the flood volume of classes 1 and 3, the water
beyond what the dikes hold, which sizes water-retention measures. Rule 18 gathers every baseline, `pre` and `post`
`flood_metrics.csv` of a basin into `runs/metrics_comparison.csv`.

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
- Pavlis et al. (2012), The development and evaluation of the EGM2008. J.
  Geophys. Res. 117, B04406.
- Scussolini, P., Aerts, J.C.J.H., Jongman, B., Bouwer, L.M., Winsemius, H.C., de
  Moel, H., Ward, P.J. (2016). FLOPROS: an evolving global database of flood
  protection standards. Natural Hazards and Earth System Sciences, 16, 1049-1061.
