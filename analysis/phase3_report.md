# Phase 3: SWOT levels as the rule-10 surface: Ebro (2433835) and Chao Phraya (3279946)

Question: should rule 10's surface come from SWOT node levels (SWORD v17c `wse_obs_p50`) instead of FathomDEM, and from one fit instead of two? Outcome: the gate fails on the floor share, so the defaults stay `surface_source: fathom` and `surface_fit: two_scale`. The options are in code and config for Phase 4.

Scripts:

- `phase3_surface_stages.py`: SWOT against each rule-10 stage, writing `90_phase3_surface_stages_{reaches.csv,along.png}`.
- `phase3_surface_options.py`: rebuilds the surface and round-0 bed per option, writing `90_phase3_surface_options_{summary.csv,profile.png}`. The surfaces themselves go to `figures/analysis/phase3_surfaces/`.

## 3.1 Where the surface departs from SWOT

SWOT levels are shifted from EGM2008 to GOCO06s with rule 06's geoid offset (0.07 m), and each is matched to the nearest node of its own reach.

| | Ebro | Chao Phraya |
|---|---|---|
| raw FathomDEM median − SWOT, all observed nodes | +0.18 m | −0.05 m |
| after the sea pin | +0.33 m | +0.02 m |
| after the isotonic fit / final level | +0.33 / +0.34 m | +0.04 / +0.04 m |

**The Ebro metre** is in the raw FathomDEM medians on the incised upper reaches: reach 071 (the gorge, TODO B32b) is +1.7 m and reach 051 is +0.9 m. The fit stages add nothing there.

**The sea pin** adds about 0.4 m on Ebro's lower 35 km. SWOT sees the river 0.3–0.6 m below the model's calm sea level.

**Chao Phraya main stem:** the FathomDEM staircase stands 2–5 m above SWOT between 20 and 150 km below the upstream boundary. The basin median is near zero only because the tidal lower 120 km and the branches agree.

## SWOT pass-to-pass spread (`wse_obs_mad`)

The spread includes the seasonal and tidal range, not just noise:

| | Ebro | Chao Phraya |
|---|---|---|
| median spread | 1.08 m | 1.41 m |
| rejected at 0.5 / 1 / 2 / 3 m | 65 / 53 / 29 / 14 % | 99 / 77 / 37 / 22 % |
| median spread of nodes > 1 m off their neighbours, against the rest | 2.84 against 1.05 m | 3.64 against 1.31 m |

- The plan's 0.5 m threshold would reject almost every node.
- `swot_max_mad_m` is therefore 3.0. A median over each node's 1 km window drops the remaining isolated spikes.

## 3.2–3.5 Code (defaults unchanged)

- `sword.read_swot_node_levels`: SWOT node level in the model datum and CRS.
- `water_surface.build_water_surface(swot_nodes=...)` takes three new options:
  - `surface_source: sword_swot` makes the fit start from the SWOT levels. Gaps are filled from SWOT neighbours, and reaches with no SWOT node use FathomDEM. Mouths and structures stay FathomDEM-based.
  - `surface_fit: single` takes level and slope from one fit over `single_smoothing_m`.
  - The node table gains `level_source_m` and `level_source`.
- `river_bed.period_median_discharge`, used by `design_discharge: period_median`: the GloFAS daily median from `period_start` (2023-07-01) to the series end.

## 3.6 Options (round-0 bed per surface and design discharge)

Columns:
- **forward:** GVF level at the design discharge minus the surface, median / p95.
- **floor:** share of flowing nodes whose depth uses the slope floor.
- **freeboard:** bank top minus the GVF level at Q_bankfull, as the median and the share of nodes below 0.

| basin | surface | design Q | surface − SWOT p50 | forward p50 / p95 (m) | floor | depth p50 (m) | freeboard p50 (m) / < 0 |
|---|---|---|---|---|---|---|---|
| Ebro | FathomDEM two-scale | bankfull | — | 0.20 / 0.87 | 0.29 | 5.1 | 3.46 / 7 % |
| Ebro | SWOT two-scale | bankfull | 0.30 | 0.18 / 0.79 | 0.26 | 5.3 | 3.44 / 7 % |
| Ebro | SWOT single 20 km | period median | 0.39 | 0.06 / 0.12 | 0.43 | 2.6 | 1.22 / 31 % |
| Ebro | SWOT single 40 km | period median | 0.49 | 0.02 / 0.09 | 0.26 | 1.9 | 0.77 / 32 % |
| Chao Phraya | FathomDEM two-scale | bankfull | — | 0.44 / 1.12 | 0.25 | 11.2 | 1.18 / 9 % |
| Chao Phraya | SWOT two-scale | bankfull | 0.02 | 0.62 / 1.54 | 0.36 | 11.0 | 1.29 / 8 % |
| Chao Phraya | SWOT single 20 km | period median | 0.03 | 0.22 / 0.72 | 0.40 | 4.6 | −3.92 / 91 % |
| Chao Phraya | SWOT single 40 km | period median | 0.04 | 0.17 / 0.66 | 0.36 | 4.3 | −3.93 / 91 % |

Every surface × design combination is in the CSV.

- **Isotonic treads:** the downstream-monotone fit turns noisy SWOT levels into flat treads, just as it does with the FathomDEM staircase. On them the bankfull bed drops to −39 m (Chao Phraya, 145 km).
- **Floor share:** neither source halves it. It reflects flat, mostly tidal lower reaches that SWOT confirms are flat, so it is a matter for Phase 4's sea inversion, not for the surface source.
- **Freeboard with the period-median bed:** the bed is shallower, so the channel overflows at Q_bankfull on 91 % of Chao Phraya nodes and 31 % of Ebro nodes. This is Phase 4.2's capacity question.

## Gate

| condition | result |
|---|---|
| forward check passes on both basins | only the SWOT single fits with the period-median discharge |
| floor share halves | fails: Ebro 0.29 → 0.26–0.43, Chao Phraya 0.25 → 0.36–0.40 |
| Ebro metre explained | yes: FathomDEM's raw medians on the incised upper reaches |

The gate fails, so the defaults stay. SWOT single 40 km with `period_median` is the strongest candidate. It goes to Phase 4 as the "consistent bed" (4.1), and its channel capacity (4.2) decides the flip together with Phase 4's own gate.
