# River depth, GVF and calibration — test and implementation plan

Oct 30, 2026 · @J

## How to read this plan

Scope is two basins: Chao Phraya (3279946) and Ebro (2433835). Ebro is the sweep basin (a confined round-1 run costs 0.26 min), Chao Phraya the confirmation basin (37.5 min for 59 h, not yet steady). The other seven deltas come back only when a default has flipped on these two.

The Phase 0 read-out (30 Oct, commit `edaf0be`) is done and folded in below. What it changed:

- No SWOT RiverSP product is on disk, but SWORD v17c already carries node-level SWOT water-surface percentiles and reach slopes. Phase 3 starts from those; a RiverSP download is a fallback.
- No surveyed bathymetry, levee inventory or flood map exists for either basin. Phase 8 leans on depth sensitivity and GESLA gauges; truth-data sourcing is a separate errand.
- No main run has ever completed on either basin (`runs/default` holds inputs only). A baseline main run is now a prerequisite (Phase 1.0).
- Round 2 is already off by default (`verify_crests: false`); Phase 6.3 reduces to a cleanup.
- GTSM tide + surge hydrographs exist (`storm_tide_hydrographs.nc`) and drive the main runs; Phase 6.2 reuses the same builder for the crest run.
- Bank-top elevations are not extracted anywhere; `dem_m` in `calibration_state.csv` is the channel-flattened grid elevation. Phase 1.1 needs its own extraction from the natural DEM.

Principles: every method change sits behind a config switch whose default reproduces current behaviour until its gate passes; every phase ends with the same two-row table (Chao Phraya, Ebro); a gate is a number with a threshold; no plotting or model-physics change rides along; diagnostics that survive become `analysis: true` outputs. Each phase is one handoff to Claude Code; the first one is drafted in the last section.

## Phase 0 — Read-out findings (done)

The four open points are settled; the table maps each answer to the phase it changes.

| Point | Answer from the read-out | Consequence |
| --- | --- | --- |
| SWOT | No RiverSP on disk. SWORD v17c nodes carry `wse_obs_p10..p90`, `_range`, `_mad`, `n_obs`; reaches carry `slope_obs_p50` and `slope_obs_reliable`. Datum of `wse_obs` not stated; static SWORD `wse` is EGM2008. Pipeline reads only `slope_obs_p50` as a check. | Phase 3 uses the SWORD statistics first; a datum test comes before anything else; RiverSP becomes a fallback. |
| Bifurcation split | Width shares live in `river_network.accumulate_discharge` (l.448–458); the solver receives routed Q only; the junction level in `gvf_solver.forward_network` (l.188–191) is the unweighted mean of the flowing arms. | Phase 5 adds a share override to the routing and an outer iteration in `river_bed`; the junction condition changes in the solver. |
| Calibration sea | Held at the COAST-RP level at the FLOPROS RP plus MDT from t = 0, no ramp (rule 11 l.1039–1045). Tide + surge hydrographs exist (`storm_tide_hydrograph.py`, `surge.build_design_surge_matrix`) but drive rule 14 only. Crest = `ceil_cm(max(zs_max, coastal crest) + freeboard)` plus a face `uv_zmin` floor. | Phase 6.2 calls the same hydrograph builder inside rule 11; the `freeboard` value is reported with the crests. |
| SFINCS | v2.3.0, regular mesh; quadtree supported by the binary and hydromt\_sfincs, unused. Subgrid 10 px, 20 levels; `advection` 1, `viscosity` 1 (`nuvisc` 0.01), `huthresh` 0.01. No `crsfile` in any run. Walls are land / water-like edges, so no wall sits between two channel cells. | Phase 2 matrix is fixed; Phase 1.5 adds `crsfile` lines; the blocked-face census becomes a channel-connectivity census. |
| Flags | `integrate_reach` returns `capped`; `forward_network` drops it. | Phase 1.4 is a small API change. |
| State on disk | Both basins: round-1 map, node table (`forward_level_m`, `level_m`), boundary check. Chao Phraya also has a spin-up restart; neither has a main run. Chao Phraya not steady at 59 h (9172 of 12462 cells rose > 5 cm, max +0.77 m); Ebro steady at 20 h. | Phase 1.0 baseline main run; Phase 6.1 sizes Chao Phraya's run on L²/D (open item B34). |
| Open items | B32 (no river-only run for the lag), B33 (zsini vs spin-up end), B34 (Chao Phraya not steady), B35 (1D–2D gap). | B32 → 6.2 and 7; B33 → 1.6; B34 → 6.1; B35 → 1.7 and 2. |

Still unknown and left open: the time span and cycle count behind the SWORD SWOT statistics, and whether the Mississippi results can be rebuilt (out of scope now).

## Phase 1 — Diagnostics on Chao Phraya and Ebro (done, commit `5e4db0b`)

Goal: the baseline every later gate is measured against. Mostly analysis scripts under `analysis/` on outputs already on disk; three items need small code additions (marked). Expected effort: small to medium; the main cost is one Chao Phraya round-1 rerun with cross-sections (about 40 min) and the baseline main runs.

| # | Diagnostic | Method | Output | Decision it feeds |
| --- | --- | --- | --- | --- |
| 1.0 | Baseline main run (prerequisite) | Rebuild stale skeletons (B30) and run the default scenario end to end on Ebro, then Chao Phraya. Record wall-clock time. | Flood map, run time | Everything in 1.9 and Phase 8; the first flood result the pipeline has produced on these basins |
| 1.1 | Bankfull freeboard (new extraction) | Bank top per node from the rule-6 FathomDEM (GOCO06s, before rule-10 conditioning): 90th percentile of the DEM in a band from W/2 to W/2 + 200 m on each side, lower bank taken; `water_body` nodes skipped. Freeboard = bank top − GVF level at Q\_bankfull, for the stored bankfull bed and for a mean-Q bed re-estimated in the script via `estimate_river_bed_from_files`. | Median, 5/95 % freeboard; share < 0 and > 2 m, per option | `design_discharge` default (Phase 4) |
| 1.2 | Slope-floor census | `on_floor` share from the node table; bed re-estimated at floors 5e-6, 1e-5, 2e-5 in the script. | Floor share; median depth change | Scope of Phase 4 |
| 1.3 | Forward-check gate | `forward_level_m − level_m` from the node table. Pass: median within ±0.25 m, 95th within ±0.75 m. Write the summary to a CSV; today it is only logged. | Pass/fail plus the two numbers | Baseline for Phases 3–5 |
| 1.4 | Froude-cap flags (code) | Make `forward_network` return the per-node `capped` flag already produced by `integrate_reach`; extend `test_gvf.py`. Count flags in the stored bed solve. | Count, reaches | Input-quality check; expect zero |
| 1.5 | SFINCS bifurcation split (code, Chao Phraya) | Add `crsfile` lines across each arm of every bifurcation on the two forcing paths; rerun round 1; compare end-of-run `crs` discharge with the width shares from `accumulate_discharge`. Ebro: only the mouth split, if inside the domain. | Fraction per arm: SFINCS vs rule | Phase 5 |
| 1.6 | Spin-up convergence | Chao Phraya: dζ/dt over the last 6 h of the existing spin-up map, and spin-up end minus `mean_level_m` (B33). Ebro: run the spin-up (about 1 min), same numbers. | mm/h; median offset | Run-length rule (6.1); zsini check |
| 1.7 | 1D–2D gap by angle and width | Round-1 end state minus `river_profile.forward_levels` at the calibration Q, per reach, with channel angle to the grid, width in cells, and a connectivity census (channel cells whose only channel neighbours are diagonal). | Table and scatter per basin | Hypotheses for Phase 2 |
| 1.8 | SWOT coverage and datum | From SWORD v17c nodes: share of nodes with `n_obs` ≥ 10 and finite `wse_obs_p50`; reaches with `slope_obs_reliable`. Datum: `wse_obs_p50` minus static `wse`, and minus the rule-10 level after the geoid shift, on nodes upstream of the boundary-check influence end. | Coverage share; median offsets | Whether Phase 3 can run on SWORD statistics alone, and which datum they sit on |
| 1.9 | Depth sensitivity (after 1.0) | Ebro: beds ±30 % through the rule-11 burn, then the default main run; flood volume and extent against the baseline. Chao Phraya if the 1.0 run time allows. | Volume and extent change | How much Phases 3–5 are worth |

Gate to proceed: none; this phase measures. Deliverable: one table with the two basins as rows and one figure per diagnostic in the analysis figure folder.

## Phase 1 — Results and decisions

Four things are settled: river depth barely moves the Ebro flood (±30 % depth is ±1–3 % of volume), the GloFAS "bankfull" discharge is not a channel-forming flow on Chao Phraya, SWOT levels cover 92–100 % of the nodes, and the width split holds at three of four bifurcations. Two things are not: the 1D–2D gap test was inconclusive as designed, and the one number that decides the bed work, Chao Phraya's river-driven depth sensitivity, is still missing.

| Item | Ebro | Chao Phraya | Reading |
| --- | --- | --- | --- |
| 1.0 main run | 15.8 min end to end; 64 % of land flooded, 0.41 km³ | Stopped after the spin-up (rule 11 33 min, spin-up 22 min) | Chao Phraya's event run is still needed for 1.9 |
| 1.1 freeboard at Q\_bf | Bankfull bed: median 3.5 m, 7 % negative. Mean-Q bed: 1.7 m, 16 % | Bankfull bed: 1.2 m, 9 %. Mean-Q bed: −2.9 m, 88 % negative | With the bankfull bed the GVF reproduces the surface by construction, so that option measures bank height above the FathomDEM surface, not the bed. The mean-Q option says Q\_bf needs about 4 m more stage than Chao Phraya's banks allow: Q\_bf (1.5-yr GEV from GloFAS) is not a channel-forming flow there |
| 1.2 slope floor | 29 % of nodes; floor ×2 → −19 % depth, ÷2 → +24 % | 25 %; −20 % / +24 % | The floor sets a quarter of the depths within ±25 % |
| 1.3 forward check | Median +0.20 m, p95 0.88 m: fail | +0.44 m, 1.12 m: fail | Baseline for Phases 3–4 |
| 1.4 Froude cap | 0 nodes | 1 node | Closed |
| 1.5 split | No bifurcation on the forcing path | 3 of 4 within 0.03; one off by 0.20 with dominance reversed | The width rule mostly holds; Phase 5 targets one junction |
| 1.6 spin-up | Converged; end minus GVF mean level +0.04 m | p95 rate 0.01 m/h; end minus GVF mean +0.28 m (p95 +2.1 m) | The 2D mean-flow level sits above the 1D one on Chao Phraya: the 1.7 gap at mean flow. B33 closes as documented |
| 1.7 1D–2D gap | +0.76 m median over 8 reaches; Spearman with width −0.55, with chord angle +0.19 | +0.29 m (p5 −0.9, p95 +1.4); no correlation; run not steady | Inconclusive: a reach chord angle does not measure a meander's staircase, and Chao Phraya had not converged. Redo as slope excess against path-length ratio on steady runs |
| 1.8 SWOT | 99.7 % of nodes observed; SWOT median 1.0 m below the rule-10 surface | 91.5 %; 0.2–0.3 m above | Coverage settles Phase 3 without a RiverSP download. The 1 m Ebro offset is a finding in itself: the FathomDEM-based surface is uncertain at the metre level. No reach has a reliable SWOT slope |
| 1.9 depth sensitivity | ±30 % depth: area −1.3 % / −0.3 %, volume −0.9 % / +2.6 % | Not run | On a sea-dominated delta the bed hardly matters; Chao Phraya is the river-driven case and decides |

Decisions:

1. `design_discharge` stays `bankfull` for now. Neither option is consistent with the data: the bankfull option over-deepens the channel wherever the FathomDEM surface is a mean-flow surface, the mean option cannot carry Q\_bf. The fix is to match the surface to its own discharge (Phase 3: SWOT median level with the GloFAS median over the SWOT period) and to compute channel capacity Q\_cap, the discharge at which the GVF reaches the bank top, as the diagnostic Q\_bf is compared with.
2. Phase 6.1 (run length, hard convergence gate) moves ahead of Phase 2: every Chao Phraya number in 1.6 and 1.7 is contaminated by non-convergence.
3. Phase 2 starts with a no-run re-analysis and the Ebro sweep, extended with two roughness runs; the synthetic channel comes only if the sweep leaves the Ebro gap unexplained.
4. Phase 4 is conditional on Chao Phraya 1.9: if ±30 % depth moves its flood volume by under 5 %, the sea inversion is deferred and the floor stays, documented.
5. Phase 5 is downgraded to a targeted fix of the one reversed bifurcation.
6. Phase 3 goes ahead on the SWORD statistics alone; the datum is taken as EGM2008 (SWOT's reference) with the 0.07 m offset applied; the slope-from-SWOT option is dropped.
7. Outside the plan: the Ebro default scenario floods 64 % of the delta's land. The scenario definition is worth a check before it serves as a baseline for anything else.

## Phase 2 — 1D–2D gap: re-analysis, Ebro sweep, synthetic test if needed

Goal: find whether SFINCS's excess over the 1D GVF (Ebro +0.76 m, Chao Phraya at least +0.3 m) comes from the grid, from a SFINCS setting, from roughness on mixed faces, or from the real geometry. Three parts, in cost order. Chao Phraya is only re-examined on the steady runs Phase 6.1 produces. Expected effort: small for 2.0 and 2a, medium for 2b.

**2.0 Re-analysis, no runs.** The 1.7 test used the reach chord angle, which says nothing about a meandering reach, and the level gap, which accumulates along the path. Replace both: per reach, slope excess = d/dx of (SFINCS level − 1D level) along the centreline in m/km, and path ratio = Manhattan length of the consecutive centreline cells over the Euclidean centreline length. Spearman of slope excess against path ratio and against width in cells; plus the level gap against distance to the mouth along each main path. A staircase effect shows as slope excess rising with path ratio and a gap that grows upstream.

**2a Ebro sweep** (real geometry, round 1 in a scratch root, 0.26 min per SFINCS run):

| Run | Current | Sweep | Hypothesis |
| --- | --- | --- | --- |
| `nr_levels` | 20 | 50 | Subgrid table discretisation |
| `advection` | 1 | 0 | Upwind advection in a channel that turns every few cells |
| `nuvisc` | 0.01 | 0 | Viscosity term |
| `huthresh` | 0.01 | 0.05 | Wet-face threshold |
| Calibration Q | Q\_prot | 0.5 Q\_prot | Does the gap scale with u² (advection) or u (friction)? |
| Manning n, whole grid | by land class | 0.03 everywhere | Roughness averaging on faces that straddle channel and bank |
| Manning n, corridor | by land class | 0.03 on the channel mask plus one cell either side | Same, locally |

The two roughness runs test the hypothesis that fits Phase 1 best: at Q\_prot the stage exceeds the bank inside the walled corridor on most reaches (freeboard 1.2 m median on Chao Phraya), so bank pixels with land roughness are wet on every mixed face, and narrow channels have the most mixed faces, which is Ebro's −0.55 correlation of gap with width.

**2b Synthetic channel**, only if 2.0 and 2a leave the Ebro gap unexplained: straight rectangular channel, 100 km, bed slope 1e-5, 8 m deep at the mouth, n 0.03, banks 15 m above the bed, 2 km floodplain, Fr ≈ 0.1, downstream level fixed; built with the pipeline's burn and subgrid code; factors angle (0°, 22.5°, 45°), width (1, 3, 8 cells), banks (DEM vs weir lines), corridor quadtree at 50 m. Reference: analytic normal depth and `gvf_solver`.

Decision rule:

- A roughness run closes most of the Ebro gap: change how the subgrid averages roughness on mixed faces, or set the corridor roughness in the pipeline (config), rerun 1.7 on the steady Chao Phraya run, done.
- Another 2a setting closes it: change it pipeline-wide in config, same confirmation.
- 2.0 shows slope excess rising with path ratio: the grid; remedies in order of cost are quadtree refinement of the corridor (hydromt\_sfincs has the components, the binary supports it), a burn that widens channels under 3 cells with conserved area, last an n correction per reach.
- Neither: the cause is in the real geometry (meander resolution, width smoothing, the split); return to 1.5 and 2.0 before changing anything.

Gate to close: the 1.7 gap rerun after the remedy gives a median within ±0.2 m on both basins, on steady runs.

## Phase 3 — Rule 10: SWOT levels from SWORD as surface source, one fit, period-matched discharge

Goal: use `wse_obs_p50` from SWORD v17c (observed on 92–100 % of the nodes, Phase 1.8) as the primary water surface, derive level and slope from one fit, and pair the surface with the discharge it actually corresponds to. No download: coverage is sufficient and no reach has a reliable SWOT slope, so the slope comes from the fit. Expected effort: medium.

Steps:

1. Datum and the Ebro metre (3.1). Treat `wse_obs` as EGM2008, SWOT's published reference, and apply `raster.compute_geoid_offset_arr` (0.07 m here). Then find where Ebro's 1.0 m offset between SWOT and the rule-10 surface comes from: compare `wse_obs_p50` with the raw FathomDEM node medians, with the surface after the sea pin, after the isotonic fit, and after smoothing, per reach. If the metre is in the raw medians, FathomDEM's water pixels on the incised upper Ebro include bank pixels; if it appears at a fit stage, that stage lifts the surface.
2. Read the columns (3.2). Extend `sword.read_sword_nodes` with `wse_obs_p50`, `wse_obs_mad`, `wse_obs_range`, `n_obs`, carried through `08_clean_river_network.py` into the clean network.
3. Source switch (3.3). In `water_surface.build_water_surface`: `surface_source: fathom | sword_swot` (default `fathom`). `sword_swot` sets the node target to the shifted `wse_obs_p50` where `n_obs ≥ 10` and `wse_obs_mad ≤ 0.5 m`, else to the FathomDEM median; then the sea pin and the isotonic fit as now. Report the share of nodes falling back to FathomDEM.
4. One fit (3.4). `surface_fit: two_scale | single` (default `two_scale`). `single` replaces the 5 km level fit plus 40 km slope fit with one penalised monotone fit on the node graph; slope is its derivative along the main path. SWOT node levels are a per-node median over passes, so their noise is a few decimetres and a 20–40 km fit resolves 1e-5.
5. Period-matched discharge (3.5). New option `design_discharge: period_median`: the GloFAS median at each crossing over the SWOT observation window (from July 2023 to the end of the series), routed as now. This is the discharge the SWOT median surface corresponds to; Phase 4 hangs the depth below the surface with it. The `bankfull` and `mean` options stay for comparison.
6. Validation (3.6). On both basins, for each source and fit combination: forward check (1.3), floor share (1.2), freeboard (1.1), and the per-reach difference between the SWOT and FathomDEM surfaces.

Gate to flip `surface_source` and `surface_fit`: forward check passes on both basins; floor share halves; the Ebro metre is explained.

Risk: `wse_obs_mad` and `_range` are the spread over passes; on tidal reaches that is the tidal range, so the 0.5 m MAD threshold may reject exactly the lower-river nodes. Report how many it rejects per basin before choosing the threshold; Phase 4 uses the same columns as its tidal flag.

## Phase 4 — Rule 11: channel capacity, and the sea inversion if the bed matters

Goal: make the bed consistent with the surface it is hung from, report what the channel can carry, and only then decide whether the floored nodes are worth a sea inversion. Conditional: the sea inversion (steps 3–5) runs only if Chao Phraya's depth sensitivity (Phase 1.9, delivered with the next handoff) moves flood volume by 5 % or more for ±30 % depth. Ebro's answer was 1–3 %. Expected effort: small for 1–2, medium for 3–5.

Steps:

1. Consistent bed (4.1). With `surface_source: sword_swot` and `design_discharge: period_median` from Phase 3, the bed is the SWOT median surface minus the normal depth at the period-median discharge. This replaces both current options as the default candidate; the bankfull option hangs a 1.5-yr flood depth below a median-stage surface and over-deepens the channel by metres on Chao Phraya (Phase 1.1).
2. Channel capacity (4.2, new diagnostic). Per node, Q\_cap = the discharge at which the GVF level reaches the bank top from the Phase 1.1 extraction (solve the network at a set of discharges, interpolate). Report Q\_cap / Q\_bf per reach and per basin. Where Q\_cap is below Q\_bf the channel overflows at its 1.5-yr flood; on Chao Phraya that is expected and is information, not an error. Where Q\_cap is below Q\_mean something is wrong with the width or the bank top, and the node is flagged.
3. Classify (4.3). `resolved` if the fitted slope exceeds `slope_resolved_min` (propose 3e-5) and `on_floor` is false; `unresolved` otherwise. Phase 1.2 gives the baseline (25–29 % on the floor).
4. Sea inversion (4.4). Per main stem, one unknown: the coefficient c in h = c · W^b, b fitted on the resolved nodes with `gvf_solver.fit_width_depth`, fallback 0.4. `forward_network` at the period-median discharge from the mouth at the calm sea level; iterate c until the level at the first resolved node matches the surface within 2 cm; downstream of it the bed is the GVF surface minus c · W^b, upstream the Manning inversion. Blend over `bathymetry_handover_m`.
5. Tidal reaches (4.5, optional). Where `wse_obs_mad` exceeds 0.5 m, a tidal-prism lower bound on depth behind `tidal_prism_bound: false`.

Config: `design_discharge: bankfull | mean | period_median`, `bed_method: manning | manning_sea_inversion` (default `manning`), `slope_resolved_min`, `tidal_prism_bound`; move `froude_cap` into config (Phase 1.4 showed it binds on one node, so it is a reporting setting now).

Gate to flip `design_discharge` to `period_median`: forward check passes on both basins; Q\_cap / Q\_bf has no node below Q\_mean. Gate to flip `bed_method`: the same plus `on_floor` share below 10 %. If Phase 1.9 on Chao Phraya comes back under 5 %, step 1–2 still run (they cost nothing), steps 3–5 are deferred and the floor is documented as a known ±25 % depth uncertainty on a quarter of the nodes.

## Phase 5 — Bifurcation split: a targeted fix

Goal: fix the one bifurcation where the width rule reverses the dominant arm (Chao Phraya 44306300275: SFINCS 0.57 / 0.43 against width 0.37 / 0.63) without disturbing the three that are within 0.03 (Phase 1.5). Downgraded from a solver rewrite to a bounded change; last in the sequence. Expected effort: small.

Method:

1. Routing hook. `river_network.accumulate_discharge` takes an optional `share_override` keyed by (reach, arm); default stays the width share. Canal exclusions unchanged.
2. Hydraulic shares. `gvf_solver.solve_network_splits`: route with current shares, call `forward_network`, update the shares by Newton on the arm-level mismatch at each bifurcation, to 2 cm or 5 iterations. The junction level is then the common arm level; the unweighted mean (l.188–191) stays the fallback for width-share runs.
3. Rule 11 iterates split → depth → split twice under `split_method: hydraulic` (default `width`).
4. Report per bifurcation: width share, hydraulic share, SFINCS share from 1.5 (rerun on the steady run after Phase 2's remedy), and the arm-level mismatch the mean was hiding.

Gate to flip `split_method`: on Chao Phraya the reversed bifurcation comes within 0.05 of the SFINCS share and the other three stay within 0.05; the 1D–2D gap on the arms does not grow. If the hydraulic split cannot meet that, a `split_override` entry in config for that one bifurcation is the fallback, with the SFINCS share as its value and a note that it is basin-specific.

## Phase 6 — Calibration redesign

Goal: crests from a converged run forced by the event a levee is designed against, and a run length derived from the reach. 6.1 is now the first implementation step of the whole plan: Chao Phraya's round 1 at 59 h is not steady (max rise +0.77 m in 3 h, Phase 0; spin-up end still 2 m above the GVF mean level at the 95th percentile, Phase 1.6), so its gap, its crests and its split shares are all provisional until it converges. Expected effort: medium; Chao Phraya compute is about 0.6 min per simulated hour, so a 240 h run is 2–3 h.

**6.1 Run length and a hard convergence gate.** In rule 11 l.1131–1171: for each inflow's main stem, D = Q\_prot / (2 W S) with the reach-median width and the fitted slope (floored at 1e-5 for this purpose only), t\_settle = L² / D over the path to the boundary-check influence end (276 km on Chao Phraya, 56 km on Ebro). `run_hr = ceil(ramp_hr + max(T_walled, 3 · t_settle) + calibration_margin_hr)`. Raise `calibration_max_days` to 10; the ValueError stays. `_steady_check` (l.1235–1250) gets `steady_check.fail` in config and writes its cell count and max rise into `calibration_state.csv`; today it only warns. `flood_wave_travel_time_methods.md` l.205–210 already proposes this rule; implement it there and in the script. Ebro is the regression check (it converges at 20 h and should not change). The r·T regime switch stays only as the T\_river lower bound for Phase 7.

**6.2 Two runs instead of one.** Split round 1 into:

- a river-timing run: sea held at the calm level, rivers ramped to Q\_prot and held, length from 6.1; it gives T\_river from the SFINCS levels themselves (half-rise time per centreline cell) and the convergence check, and closes B32;
- a crest run: restart from the timing run's end, rivers held at Q\_prot, sea forced with `surge.build_design_surge_matrix` at the FLOPROS coastal RP (average tide plus the surge shape scaled to the COAST-RP level, the same call rule 14 makes), over the full ±60 h window; crest = water-side `zs_max` as now, through `build_coastal_protection_weir` with the same `freeboard` and `uv_zmin` floor.

Config: `calibration_sea: held | hydrograph` (default `held`). Report along the lower 50 km of each basin the crest difference hydrograph minus held; that difference is the estuarine amplification the held sea cannot produce. Before trusting it, one sanity check: GTSM storm-tide levels at the basin's stations against the GESLA v4.1 gauges already on disk (`raw_data/GESLA_v41/`, not catalogued).

**6.3 Round-2 cleanup.** Round 2 is already off (`verify_crests: false`) and is bit-identical to round 1 wherever nothing overtops. Remaining: `analysis/check_crest_calibration.py` hard-codes `round2/`; retire it with the plotting task, and replace its purpose with a scenario run one return period above the protection level.

Gate to flip `calibration_sea`: both basins converge within the cap under 6.1; the GESLA check shows no systematic storm-tide bias above 0.3 m at the stations used; the amplification numbers are recorded. With no levee inventory for either basin, this gate is on convergence and boundary quality, not on crest truth.

## Phase 7 — Compound lag from reanalysis statistics

Goal: make river–surge timing an explicit scenario choice backed by observed co-occurrence, with the modelled travel time used only to translate the inflow boundary to the coast. Independent of Phases 2–6. The data is on disk: `gtsm_hourly.nc` per basin (surge residual, 8 stations for Chao Phraya, 1950–2024) and `river_forcing.nc` `discharge_daily` per crossing (1979–2026); overlap 1979–2024. Expected effort: small.

Method (one `analysis/` script, then one config key in rule 14):

1. Event pairing. Surge residual at the station nearest the mouth; surge peaks above the 50th percentile of annual maxima, 3-day independence. For each, the discharge peak at the forcing crossing within ±7 days, with the 14-day independence already used by `river_forcing.real_flood_event_matrix`. Lag = t\_Q − t\_surge in hours. Also the reverse pairing from discharge peaks.
2. Report per basin: lag median, interquartile range, share with |lag| < 1 day, share of surge peaks with no discharge peak in the window, for each pairing direction.
3. Translation to the coast: lag at the coast = lag at the boundary − T\_river, with T\_river from the 6.2 timing run (until then, the walled estimate as a lower bound).
4. Materiality: whether |lag at the coast| exceeds τ95 of the inflow hydrograph (`river_forcing.py` event shapes). Where it does not, the mode below makes no difference and the report says so.
5. Scenario option in rule 14: `lag_mode: aligned | observed_median | none` (default `aligned`, current behaviour). `observed_median` shifts the river hydrograph by the Phase 7 median; `none` keeps source-series timing for `glofas_event` scenarios.

Gate: none to flip; this is a scenario definition. Deliverable: the table and a methods paragraph stating which mode each scenario uses. Recommended: `aligned` for design-standard scenarios (worst case), `observed_median` for representative-event scenarios.

## Phase 8 — Standing validation suite and documentation

Goal: turn the checks into `analysis: true` rules that run on every basin and write one summary table per basin. Neither basin has surveyed bathymetry, a levee inventory or a national flood map, so the suite leans on internal consistency, SWOT, GESLA and sensitivity; truth-data sourcing is a separate errand listed last. Expected effort: medium.

| # | Check | Data | Pass criterion (proposal) | Built in |
| --- | --- | --- | --- | --- |
| 1 | Bankfull freeboard | FathomDEM bank tops (new extraction) | Median 0–1.5 m on resolved reaches | Phase 1.1 |
| 2 | Surface against SWOT node levels | SWORD v17c `wse_obs_p50` | Median within ±0.3 m, 95 % within ±1 m, on covered nodes | Phase 3.5 |
| 3 | Forward check | Rule 11 node table | Median ±0.25 m, 95 % ±0.75 m | Phase 1.3 |
| 4 | Storm-tide boundary against gauges | GESLA v4.1 (on disk, to catalogue) | No systematic bias above 0.3 m at the stations used | Phase 6.2 |
| 5 | Lag against reanalysis co-occurrence | GloFAS, GTSM hourly | Modelled T\_river within the observed lag interquartile range | Phase 7 |
| 6 | Calibration convergence | `calibration_state.csv` max rise | Below 5 cm over the last 3 h | Phase 6.1 |
| 7 | Depth sensitivity | Default scenario with beds ±30 % | Reported, no pass/fail: flood volume and extent change | Phase 1.9 |

Item 7 decides how much Phases 3–5 are worth: if ±30 % depth moves flood volume by less than 10 %, bed work drops in priority and the effort goes to crests and run length.

Truth data to source (separate errand, outside Claude Code): lower-Ebro channel surveys and delta protection heights from the Ebro basin authority; Chao Phraya channel surveys and Bangkok flood-wall heights from the Thai river authorities. Until something arrives, the bathymetry and levee gates in Phases 4 and 6 stay off.

Documentation: one methods paragraph per flipped default, written at the time of the flip with the gate numbers. The hard-coded constants the read-out listed (`froude_cap` 0.95, `max_step_m` 50, `h_dry_m` 0.01, forward-check bands, bathymetry margin, burn `margin_m` 500) move into config under `river_processing` with a one-line comment each, in one commit, with no value change.

## Dependencies and order of handoffs

&#91;embedded content: phase dependencies · 9 phases, 5 gates\]

Rows top to bottom: diagnostics, calibration, bed chain, scenario timing. Phases 0 and 1 are done. Phase 6.1 now comes first, because every Chao Phraya number depends on a converged run; Phase 2's re-analysis and Ebro sweep go with it; Phase 5 waits for Phase 2's remedy because the SFINCS shares may change with it; Phase 6.2 waits for 6.1 and that remedy; Phase 4's sea inversion waits for Chao Phraya's depth sensitivity.

Suggested handoff sequence, one Claude Code task each: P6.1 + P2.0 + P2a + Chao Phraya 1.0/1.9 (next, drafted below) → P3 → P4.1–4.2 (and 4.3–4.5 only if Chao Phraya's sensitivity says so) → P2b if still needed → P6.2 → P7 → P5 → P8. Ebro proves each step in minutes; the Chao Phraya reruns in 6.1, 1.9 and 6.2 are the only steps above an hour each.

## Next handoff: Phase 6.1 + Phase 2.0/2a + Chao Phraya sensitivity

One task, four parts: the run-length rule with a hard convergence gate (6.1), the no-run gap re-analysis (2.0), the Ebro settings and roughness sweep (2a), and the Chao Phraya main run with the depth sensitivity that Phase 1 left open (1.0 and 1.9). Paste as is.

```markdown
# Task: Phase 6.1, 2.0, 2a and Chao Phraya sensitivity

Reference: `analysis/phase1_report.md` (commit 5e4db0b), the plan, and
`docs/readout_river_depth_gvf.md` for code locations. Basins: Ebro
(2433835) and Chao Phraya (3279946) only. Deliverable:
`analysis/phase2_report.md` with the Phase 1 conventions: the script named
next to every table, figures under the analysis figure folder, a "not
computed and why" list, a code-change table, and `snakemake -n` output
showing the default DAG is unchanged except where part C changes rule 11.

Rules: new code only where marked (code), additive and behind config;
sweeps run in `<results_dir>/../scratch_phase2/`; production outputs
change only through part C. Order: C on Ebro, then A and B, then C on
Chao Phraya, then D.

## C. 6.1 Run length and hard convergence gate (code, rule 11)
At rule 11 l.1131-1171 (edaf0be numbering): for each inflow's main stem,
D = Q_prot / (2 * W * S) with the reach-median width and the fitted slope
floored at 1e-5 for this purpose only; t_settle = L^2 / D over the path
from the crossing to the boundary-check influence end (`x_inf_km`);
run_hr = ceil(ramp_hr + max(T_walled, 3 * t_settle) +
calibration_margin_hr). Config: `calibration_max_days: 10` (the
ValueError stays); `steady_check.fail: false` (true makes an unconverged
round a hard failure). `_steady_check` writes its cell count, share and
max rise into `calibration_state.csv`. Update
`flood_wave_travel_time_methods.md` l.205-210 to the implemented rule.
Run rule 11 on Ebro first: run_hr and crests must be unchanged. Then on
Chao Phraya: report old and new run_hr, T_walled and t_settle per
crossing, SFINCS wall-clock, the steady-check numbers, and the Phase 1.7
gap table recomputed on this steady run.

## A. 2.0 Gap re-analysis (no runs)
On the Ebro Phase 1 round-1 output and the steady Chao Phraya run from C.
Per reach: slope excess = d/dx of (SFINCS end level - 1D level) along
the centreline cells, in m/km; path ratio = Manhattan length of the
consecutive centreline cells (sum of |dx| + |dy|) over the Euclidean
centreline length; width in cells. Report Spearman of slope excess
against path ratio and against width, per basin, with scatter figures;
and the level gap against distance to the mouth along each main path.
1D level as in Phase 1.7 (`forward_network` at the calibration Q).

## B. 2a Ebro sweep (scratch, one factor at a time)
Baseline = the Ebro round-1 setup after C. Runs:
1. `nr_levels` 50
2. `advection` 0
3. `nuvisc` 0
4. `huthresh` 0.05
5. calibration Q = 0.5 * Q_prot
6. Manning n = 0.03 on the whole grid
7. Manning n = 0.03 on the channel mask plus one cell either side, land
   values elsewhere
For each: the 1.7 gap (median, p5, p95) and the slope excess from A.
One table, baseline first; flag any run whose gap median falls below
0.2 m. For runs 6 and 7 also report the share of wet faces inside the
walled corridor that straddle channel and bank pixels at the end of the
baseline run.

## D. Chao Phraya main run and depth sensitivity
After C on Chao Phraya: run the default scenario end to end (rule 12a
onward, from the new rule-11 outputs); report wall-clock per rule,
flooded area and volume. Then `depth_scale` 0.7 and 1.3 in scratch roots
as in Phase 1.9, each with rule 11 and the default build; report area
and volume ratios to the baseline. If one event run takes more than 4 h,
do the baseline and 0.7 only and say so.

## E. Scenario note
State in the report header what the `default` scenario is for each
basin: surge RP, river RP, protection applied, SLR, event timing. Ebro's
64 % flooded land share from Phase 1 needs that context.
```
