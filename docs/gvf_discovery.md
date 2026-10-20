# GVF river-bed estimation: discovery

Read-only investigation of the GCFM_UU pipeline, done 2026-10-12, as input
for planning a gradually-varied-flow (GVF) estimate of river bed levels. No
code was changed.

**Test delta: 1248635 (Rhine–Meuse).** Five of the eight deltas are single
channels with 3–6 reaches and no junctions; 1248635 is the only delta with
a complete run that has confluences, bifurcations and bifurcate-then-rejoin
loops. Its results are from the 2026-10-10 all-basin run on the current
defaults (rotated grid, 100 m cells, 10 m subgrid pixels).

Figures are in `docs/figures/gvf/`, together with the per-reach tables
behind them (`reaches_1248635.csv`, `profiles_1248635.csv`,
`widths_1248635.csv`, `stage4_per_reach_1248635.csv`).

> **Bug found: the calibration discharge of 1248635 is inflated about
> fivefold.** `src.river_network.accumulate_discharge` keeps a seed reach's
> value fixed each iteration (`next_q[seed] = q[seed]`) and then adds that
> iteration's upstream inflow on top, so a seed reach that has upstream
> reaches inside the network gains its inflow once per iteration (100
> iterations). In 1248635, seed `23261000015` (Drecht) has a GloFAS RP-1000
> discharge of 245 m³/s but ends up with 103,693 m³/s (inflow ≈1,034 m³/s
> × 100 + 245); its `bankfull_discharge_acc` is likewise 33,993 m³/s
> instead of ≈400. With unit seed discharges the result grows with the
> iteration count (9.0 after 100 iterations, 20.6 after 200). The total
> calibration inflow is 127,935 m³/s; the other four seeds sum to 24,242
> m³/s (Rhine 17,421, Meuse 4,689, two Scheldt-side inflows 1,397 and 735),
> which is plausible for RP 1000. No other delta has a seed with upstream
> reaches, so only 1248635 is affected. Every 1248635 number below that
> depends on discharge (rise, rivdph, steady state) is distorted by this.

---

## Network

### 1. Columns kept; format of `rch_id_dn`

**Answer.** The cleaned network keeps all 66 SWORD v17c reach attributes
and adds four: `linked_to_source`, `is_seed`, `is_delta_outflow`,
`bankfull_discharge_acc`. The depth-estimated network adds `q_calib`,
`rivdph`, `weir_crest_calibrated`.

- Present: `wse`, `wse_var`, `slope`, `width`, `width_var`, `max_width`,
  `n_chan_max`, `n_chan_mod`, `dist_out`, `reach_len`, `facc`,
  `main_side`, `n_rch_up`, `n_rch_dn`, `rch_id_up`, `rch_id_dn`, plus
  SWORD's path attributes (`path_freq`, `path_order`, `main_path_id`,
  `hydro_dist_out`, ...).
- `wse` is stored in 0.1 m steps (values such as 0.7, 8.1, 8.3), relative
  to EGM2008. `slope` is in m/km.
- `rch_id_dn` and `rch_id_up` are **strings** holding a bracketed list,
  e.g. `'[23250700195, 23250700445]'`; `n_rch_dn` / `n_rch_up` are
  integers. They are parsed by `src.river_network.build_downstream_adjacency`.
- Geometry: LineStrings in EPSG:4326. Their coordinates run downstream: at
  all 198 junctions of 1248635, a reach's last vertex is the one closest to
  its downstream neighbour's first vertex. The profile sampling
  (`src.river_network._sample_line_cells`) relies on this without checking
  it.

**Source.** `workflow/rules/06_river_network.smk`,
`workflow/scripts/08_clean_river_network.py`; files
`{basin}_river_network_clean.gpkg`, `{basin}_river_network_depth_estimated.gpkg`.

### 2. Network topology per delta

Counted on the in-network graph (edges to reaches outside the cleaned
network are ignored). Loops = independent cycles of the undirected graph
(edges − nodes + components); there are no directed cycles.

| Basin | Reaches | Length | Seeds | Confluences | Bifurcations | Loops | Network ends | Delta-outline outflows | Run complete |
|---|---|---|---|---|---|---|---|---|---|
| 2444235 | 4 | 27 km | 1 | 0 | 0 | 0 | 1 | 0 | yes |
| 2433835 | 5 | 39 km | 1 | 0 | 0 | 0 | 1 | 0 | yes |
| 620947 | 3 | 28 km | 1 | 0 | 0 | 0 | 1 | 0 | yes |
| 2209166 | 6 | 42 km | 1 | 0 | 0 | 0 | 1 | 0 | yes |
| 1475442 | 6 | 69 km | 1 | 0 | 0 | 0 | 1 | 0 | yes |
| **1248635** | **182** | **1,080 km** | **5** | **20** | **24** | **17** | **9** | **2** | yes |
| 1416812 | 29 | 255 km | 1 | 0 | 4 | 0 | 5 | 0 | no (memory, fixed since) |
| 4267691 | 155 | 1,587 km | 4 | 12 | 12 | 9 | 5 | 0 | no (memory, fixed since) |

"Network ends" are reaches with no downstream reach in the network: real
mouths plus reaches that stop at the domain edge. In 1248635, 2 of the 9
are flagged `is_delta_outflow` (free outflow, see item 9); 5 of the 9 are
real mouths with calibration cells.

**Source.** `build_downstream_adjacency`; `is_delta_outflow` from
`identify_delta_outflow_points` (rule 08).

### 3. Discharge: splits, merges, and what round 0 uses

**Answer.**

- **Seed values** (`src.river_depth_calibration.build_calibration_seed_discharge`):
  per GloFAS crossing, the discharge at the basin's riverine protection
  return period, log-RP interpolated from the crossing's
  `discharge_rp_table` (RPs 1–10,000 yr) in `river_forcing.nc`. The RP is
  `riverine_rp_yr` from `protection_levels.json` (FLOPROS). Fallback when
  no RP is available: the crossing's `bankfull_discharge`. Crossings are
  snapped to seed reaches by `snap_crossings_to_reaches` (largest value
  kept per reach).
- **Propagation** (`src.river_network.accumulate_discharge`, rule 10
  around line 370): 100 iterations, one hop downstream per iteration. At a
  bifurcation the upstream discharge is split in proportion to the
  receiving reaches' SWORD `width` only. At a confluence contributions are
  summed. Seed values are held fixed (see the bug above).
- **1248635:** riverine RP = 1000 yr (FLOPROS, NLD dominant geounit).
  Seeds: Rhine `23261000201` 17,421; Meuse `23250801131` 4,689;
  `23250500175` 1,397; `23250700205` 735; Drecht `23261000015` 103,693
  (should be ≈1,280 = 245 + its inflow). Per-reach `q_calib` quantiles
  (p5/p25/p50/p75/p95): 138 / 1,121 / 1,965 / 5,451 / 12,580 m³/s; all
  182 reaches in `reaches_1248635.csv`.
- `bankfull_discharge_acc` (rule 08) uses the same function with bankfull
  seeds and has the same fault.

---

## Target profile

### 4. Centreline profiles: raw, stage 1, smooth monotonic fit

Method: every reach sampled along its centreline at the DEM pixel size
(30.9 m) from `elevation_merged.tif` ("raw": FathomDEM, datum-corrected to
GOCO06s, GEBCO where land use is sea) and `elevation_conditioned.tif`
(after rule 09's running minimum and full-width flattening). The smooth
monotonic fit is an isotonic (non-increasing) regression, a 1 km running
mean, and a second isotonic pass. Reaches shorter than three samples are
skipped (163 of 182 kept).

**All reaches, 1248635:**

| Measure | Value |
|---|---|
| Noise std of raw around the fit | median 0.11 m, p90 1.57 m |
| Mean slope of the fit | median 0.008 m/km |
| Mean slope of the stage-1 profile | median 0.000 m/km |
| Fraction of length flat (zero slope) in stage 1 | median 100%, length-weighted 98% |
| Fraction flat in the smooth fit | median 47% |
| Raw minus stage 1 along the centreline | median +0.55 m |

Representative reaches (figure `profiles_representative_1248635.png`):

| Reach | Position | Length | Width | Q (m³/s) | Noise std | Fit slope (m/km) | SWORD slope (m/km) | Raw median | rivdph |
|---|---|---|---|---|---|---|---|---|---|
| 23250500175 | seed | 3.0 km | 202 m | 1,397 | 0.31 | 0.000 | 0.064 | −0.21 | 4.95 |
| 23250700205 | seed | 1.9 km | 780 m | 735 | 0.00 | 0.002 | 0.004 | −0.16 | 0.84 |
| 23250801131 | seed (Meuse) | 8.8 km | 120 m | 4,689 | 0.08 | 0.008 | 0.108 | 7.14 | 12.05 |
| 23261000201 | seed (Rhine) | 6.2 km | 392 m | 17,421 | 0.00 | 0.000 | 0.292 | 8.13 | 13.73 |
| 23250700135 | mid-delta | 2.0 km | 1,025 m | 1,060 | 0.00 | 0.002 | 0.000 | 0.05 | 6.23 |
| 23261001411 | mid-delta | 0.4 km | 259 m | 7,928 | 0.16 | 0.046 | 0.000 | 0.08 | 15.33 |
| 23250900036 | mid-delta/mouth | 0.2 km | 2,413 m | 3,477 | 0.00 | 0.000 | 0.000 | 0.20 | 4.35 |
| 23250700016 | mouth | 0.2 km | 1,606 m | 1,965 | 0.00 | 0.000 | 0.000 | −0.58 | 8.13 |
| 23250700276 | mouth | 0.2 km | 4,604 m | 154 | 1.91 | 0.000 | 0.000 | −5.00 | 8.06 |
| 23250700306 | mouth | 0.1 km | 5,633 m | 1,456 | 0.00 | 0.000 | 0.000 | −0.85 | 8.38 |

What the profiles show:

- **FathomDEM is hydro-flattened over these rivers.** Most reaches have
  zero noise and a flat raw profile; FathomDEM carries a water surface, not
  a bed, and its along-river slope (≈0.008 m/km) is far below SWORD's
  (Rhine 0.29, Meuse 0.11 m/km).
- **The stage-1 running minimum is "sticky".** One low pixel sets the
  minimum for the rest of the reach and, through the inherited minimum,
  for every reach downstream. Example: `23250500175` drops from −0.2 to
  −5.6 m at 0.9 km and stays there; the mouth reaches sit at −9.9 m
  (GEBCO clamped to MSL − 10 m) while raw is −0.6 to −5 m. Stage 2's
  "lower wins" rule at overlapping buffers spreads such values sideways
  into neighbouring reaches too.
- Mouth reaches in SWORD are very short (0.1–0.2 km), so they say little
  about the profile on their own.

**Source.** `src.river_conditioning.enforce_river_monotonicity`
(rule 09); `src.river_network._sample_line_cells`.

### 5. SWORD `wse` / `slope` against FathomDEM

Available for 163 reaches with profiles (figure `sword_vs_dem_1248635.png`).

| Measure | Value |
|---|---|
| SWORD `wse` − raw DEM median, mean | +0.68 m (SWORD higher) |
| Same, median absolute difference | 0.60 m |
| Correlation of the two per reach | r = 0.88 |
| SWORD `wse` ≤ 0 | 8% of reaches |
| SWORD slope, median | < 0.0001 m/km (most delta reaches report ≈0) |
| Monotone-fit slope, median | 0.008 m/km |
| Correlation of the slopes | r = 0.25 |

SWORD `wse` is relative to EGM2008 and the DEM to GOCO06s; the geoid
difference here is only about +0.05 m, so most of the 0.68 m is a real
difference (SWORD water levels are SWOT/altimetry-derived means, the DEM is
FathomDEM's flattened water surface). SWORD's slope is informative on the
upstream main stems (Rhine 0.29, Meuse 0.11 m/km) and ≈0 in the tidal
delta, where its 0.1 m `wse` steps cannot resolve a slope.

---

## SFINCS representation

### 6. Effective burned width

**Answer.** The burn is applied at **subgrid resolution (10 m)** over each
reach's own `buffer(width/2)` (flat end cap at seed reaches,
`_flush_capped_buffer`), rasterised with `all_touched=True`, then limited
to subgrid pixels whose parent model cell is in the 100 m `all_touched`
channel mask (`constrain_to_coarse_channel_mask`). Rule 10 does **not**
pass `channel_mask` into `burn_river_channel`, so the 100 m mask only
clips, it does not define the burn.

Measured as area / reach length (subgrid pixels or mask cells assigned to
the nearest reach centreline; reaches ≥ 300 m; figure
`burned_width_1248635.png`):

| | Burned (10 m subgrid) / SWORD width | 100 m corridor mask / SWORD width |
|---|---|---|
| All reaches (n = 169), median | 1.06 | 1.57 |
| p10 – p90 | 0.23 – 2.09 | 0.85 – 5.13 |
| SWORD width < 100 m | 1.30 | 3.51 |
| Reach within 15° of a grid axis | 1.06 | 1.49 |
| Reach more than 30° off-axis (diagonal) | 1.07 | 1.59 |

The subgrid burn matches SWORD width well and is insensitive to
orientation. The low p10 comes partly from the nearest-centreline
assignment at junctions and overlapping wide reaches (pixels counted for a
neighbour), so treat individual low values with care. The 100 m corridor,
which confines the calibration and carries the weirs, is 1.5× wider than
the channel in the median and 3.5× for narrow channels.

**Source.** `src.river_burn.burn_river_channel`,
`constrain_to_coarse_channel_mask`, `build_channel_mask_regular`;
`workflow/scripts/10_depth_estimation_modelled.py` around line 1390.

### 7. Manning n on channel subgrid pixels

**Answer.** Not channel-specific. The subgrid tables take Manning's n from
`{basin}_roughness.tif` (rule 05c: land use converted with
`lu_to_roughness_lookup.csv` and area-averaged onto the ~10 m grid), via
`sf.subgrid.create(roughness_list=[{"manning": "local_roughness_native"}])`.
Under the burned channel of 1248635:

- Land-use classes: 200 sea 47%, 80 water 38%, 30 grassland 6%, 40
  cropland 3%, 10 trees 3%, 50 built-up 2%.
- n values: 0.020 for 83% of pixels (water and sea), then 0.034, 0.037,
  0.035; maximum 0.12 where the buffer covers trees.

So channel n is mostly uniform 0.02, but wherever the SWORD buffer extends
over land the burned bed gets the land's roughness. The coarse `manning`
field (`roughness_on_grid.tif`) is only used for regular, non-subgrid
cells.

**Source.** `src.landuse.write_roughness_raster`, `read_roughness_lookup`;
round-0 `subgrid/manning_subgrid.tif`.

### 8. Round-0 SFINCS settings and steady state

**Settings** (from round 0's `sfincs.inp` and `sfincs.log`, SFINCS v2.3.0):

| Setting | Value |
|---|---|
| Advection | on, first-order upwind (`advection = 1`) |
| theta | not set in `sfincs.inp`, so SFINCS's default (1.0 per the SFINCS documentation; not printed in the log) |
| alpha | 0.5 |
| Viscosity | on, `nuvisc = 0.01` in the inp; the log reports `nuvisc = 1.0` |
| Coriolis | off (no latitude given) |
| huthresh | 0.01 m |
| Subgrid | 10 × 10 pixels per cell, 20 levels, `nrmax` 2000; table variables `z_zmin, z_zmax, z_volmax, z_level, uv_zmin, uv_zmax, uv_ffit, uv_navg, uv_havg, uv_nrep, uv_pwet` (hydromt_sfincs v1.1+ format with the fitted uv method) |
| Log warning | "some subgrid uv point(s) have a difference (z_zmin − uv_zmin) > 0.1 m" |
| Run | 24 h (`calibration_days: 1`), discharge ramped linearly from bankfull to Q over the first 6 h, then held; map output hourly, `dtmaxout` 24 h |
| Boundary | sea held at `baseline_m`; 1000 m weirs on all river banks and coast |
| Average time step | 2.07 s (1248635) |

**Steady state.** Change in water level at the centreline cells over the
last hours of round 0 (figure `round0_zs_timeseries_1248635.png`):

| Basin | Centreline cells | Median \|Δzs\| last 3 h | p95 | Median \|Δzs\| last 1 h | Cells still rising > 1 cm in last 3 h |
|---|---|---|---|---|---|
| 620947 | 366 | 0.007 m | 0.009 m | 0.002 m | 0% |
| 2433835 | 502 | 0.008 m | 0.008 m | 0.002 m | 0% |
| 1475442 | 896 | 0.000 m | 0.000 m | 0.000 m | 0% |
| 1248635 | 12,850 | 0.39 m | 0.80 m | 0.11 m | 75% |

The small single-channel deltas are steady. **1248635 is far from steady
after 24 h**: a rough storage-change estimate over the last 3 h (wet cells
× 100 m² × Δzs) is about 24,000 m³/s against a total inflow of 127,935
m³/s, i.e. a large share of the inflow is still filling the confined
network. The inflated Drecht discharge is part of this, but a 1,080 km
network with 17 loops would likely need longer than 24 h even with correct
discharge. The excavation uses the period maximum, so it was taken before
the water level had settled. No outflow is recorded per mouth (the map
file has no cross-sections), so inflow/outflow could only be estimated
from storage change.

**Source.** `workflow/scripts/10_depth_estimation_modelled.py`;
`config.yml` `river_processing.river_depth_modelling` and `sfincs.subgrid`.

### 9. Mouth handling

Mean sea level in the model datum: `baseline_m` = **+0.112 m** (GOCO06s),
from `surge_forcing.nc`; it is the mean of the selected COAST-RP stations'
MDT correction (CNES-CLS22 mean dynamic topography), SLR excluded, rounded
to the centimetre by `src.surge.read_baseline_m`. The same value is the
sea's starting level (`zsini`) and the calibration's coastal boundary.

Mouth cells of 1248635 (most downstream calibration cell of each network
end that has cells):

| Mouth reach | DEM at cell (m) | Land use at cell | rivdph | Round-0 zs (m) |
|---|---|---|---|---|
| 23270100646 | −9.89 | 200 sea | 0 | 5.11 |
| 23250900036 | −0.90 | 50 built-up | 0 | 4.48 |
| 23250700016 | −9.88 | 80 water | 0 | 0.28 |
| 23250700306 | −9.88 | 80 water | 0 | 0.59 |
| 23250700276 | −9.89 | 200 sea | 0 | 0.18 |

- Four of five mouth cells sit at −9.89 m: GEBCO clamped at MSL − 10 m
  (`terrain.gebco_max_depth_m`), so the "natural bathymetry" the mouth rule
  fixes the bed to is the clamp floor. One mouth cell is built-up land at
  −0.9 m.
- Round-0 water levels at two mouths are 4.5–5.1 m above a sea held at
  +0.11 m: the confined river is backing up strongly at those outlets.
- The mouth rule (rule 10): the last cell of each mouth reach gets
  rivdph = 0; walking upstream, any cell whose bed would be shallower than
  that endpoint is lowered to it, until the first cell that is already deep
  enough or a confluence/headwater.
- **Free-outflow reaches** (`is_delta_outflow`, 2 in 1248635): a point
  where a non-seed, non-mouth, non-bifurcation reach crosses the delta
  outline (`identify_delta_outflow_points`, rule 08). In the model those
  points are buffered by `sfincs.boundary_setup.outflow_buffer_m` (500 m)
  and set to mask = 3 (`sf.mask.create_boundary(btype="outflow",
  reset_bounds=False)`), overriding any water-level boundary there. No bed
  rule is applied to them.

### 10. Interface of `burn_river_channel`

```python
burn_river_channel(rivers, zbed_anchors, natural_dem_path, utm_crs,
                   resolution_m, width_column="width", margin_m=500.0,
                   out_transform=None, out_shape=None, channel_mask=None)
    -> (burned_arr, transform, terrain, stats)
```

- **Anchors** (`zbed_anchors`): a GeoDataFrame with one point per
  model-grid centreline cell, columns `reach_id`, `rivbed`, geometry. Built
  in rule 10 as `cell_gdf.assign(rivbed=dem_at_cell - rivdph)`.
  `rivbed` is an **absolute** bed elevation (m, GOCO06s), not a depth;
  `dem_at_cell` is the model-grid (100 m) conditioned elevation at that
  cell.
- **Along each reach**: anchors are placed by `line_locate_point` and
  interpolated linearly (`interp1d`), clamped to the end values beyond the
  first/last anchor (no extrapolation).
- **Reach ends**: if a reach's first/last anchor is more than 1 m
  (`_BOUNDARY_BLEND_EPS_M`) from its end, a synthetic anchor is added at
  the end. Its value comes from `src.river_network._junction_value`: the
  mean of this reach's anchor nearest the junction and the neighbour's
  anchor nearest the junction. With several neighbours (confluence or
  bifurcation), the mean over all neighbours' junction values is used.
- **Per pixel**: burned bed = `min(natural DEM, interpolated rivbed)`, so
  the burn never raises the bed; pixels selected by the reach's buffer
  (or by `channel_mask` if given; rule 10 does not give it).
- **Downstream use**: rule 10 calls it on the subgrid pixel grid
  (`out_transform` = model transform / 10), applies
  `constrain_to_coarse_channel_mask`, and writes
  `river_burned_subgrid.tif`; the model-grid `dep` is the per-cell mean
  (`mean_elevation_on_coarse_grid`). Rule 13 imports both unchanged.

A GVF method that produces `rivbed` per centreline cell (or per reach,
expanded to the cells) can feed this function without changes.

---

## Baseline

### 11. Current stage-4 results

Per-reach values in `stage4_per_reach_1248635.csv` (172 reaches with
calibration cells, 12,850 cells).

| Measure | p10 | p50 | p90 |
|---|---|---|---|
| Rise (round-0 max zs − DEM), per-reach median | 7.2 m | 14.6 m | 19.0 m |
| rivdph (per-reach median, `river_network_depth_estimated`) | 5.8 m | 11.6 m | 15.2 m |

No centreline cell has a rise ≤ 0. These depths (median 11.6 m) are much
larger than the single-channel deltas' and are inflated by the discharge
bug and by the unsteady, still-rising round 0; they should not be used as
a reference for 1248635 until both are addressed.

`excavation_fraction: 0.8` is set in `config.yml`
(`river_processing.river_depth_modelling`, line 80). It was introduced on
2026-09-10 (CHANGELOG entry "river depth/crest calibration reduced to
three fixed runs"; commit `556b8a4`, 2026-09-14) as "the same excavation
share the old `weir_crest_fraction: 0.2` implied" — it was carried over
from the previous iterative scheme, not calibrated against observed
depths. No other note on how 0.8 was chosen was found.

### 12. Runtime and test setup

Round-0 SFINCS runtime (`Total time` in round 0's `sfincs.log`, 10
threads):

| Basin | Active cells | Round-0 runtime |
|---|---|---|
| 2444235 | 28,963 | 7 s |
| 620947 | 92,217 | 23 s |
| 2433835 | 68,110 | 29 s |
| 1475442 | 230,859 | 60 s |
| 2209166 | 123,780 | 88 s |
| 1416812 | 1,988,399 | 525 s (rotated, 2026-10-10 run that later failed) |
| 1248635 | 1,102,258 | 796 s |
| 4267691 | 3,833,580 | 3,492 s (unrotated run of 2026-09-16) |

Rule 10 runs three such SFINCS rounds plus subgrid builds; its log has no
timestamps, so the wall time of the whole rule was not measured.

Tests and CI:

- There is **no pytest suite**: no `conftest.py`, no `[tool.pytest]`
  section in `pyproject.toml`, no fixtures. `tests/` holds 17 stand-alone
  analysis and plotting scripts (e.g. `check_crest_calibration.py`,
  `plot_calibration_round_profiles.py`) plus `check_code_health.py`, an
  import check to run by hand.
- CI (`.github/workflows/ci.yml`) runs only the pre-commit hooks on pull
  requests to `main`: ruff format, ruff check, pyright, pydocstyle and the
  YAML/whitespace checks. No Snakemake dry run (raw data is not available
  on the runner) and no tests.
- **Candidate regression deltas:** 2444235 (4 reaches, round 0 in 7 s) or
  620947 (3 reaches, 23 s) for a fast single-channel case. No small delta
  has bifurcations or loops; for those the smallest available case is
  1416812 (29 reaches, 4 bifurcations, no loops), then 1248635 and
  4267691. A synthetic network would be needed for a fast junction test.
