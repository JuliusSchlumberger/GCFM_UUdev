# Phase 2: 1D-2D gap: Ebro (2433835) and Chao Phraya (3279946)

Question: why do steady SFINCS channel levels stand above the 1D GVF profile the bed is hung from (Ebro +0.76 m, Chao Phraya +3.4 m median after Phase 6.1)? Outcome: SFINCS momentum advection on a staircase cell path. Rules 11 and 12a now set `sfincs.advection: 0` and `sfincs.alpha: 0.25`.

Scripts:

- `phase2_gap_slope.py`: gap per reach, writing `<results_dir>/{basin_id}/figures/analysis/90_phase2_gap_slope_{along,scatter}.png`.
- `phase2_ebro_sweep.py`: one-factor sweep of the Ebro round-1 calibration, writing `phase2_ebro_sweep_{summary.csv,gap.png}`.
- `phase2_advection_channel.py`: synthetic trench at normal depth, writing `phase2_advection_channel_{summary.csv,profiles.png,map.png}`. Its SFINCS test is `tests/unit/test_advection_channel.py`, marker `needs_sfincs`.

## 2.0 Gap per reach (advection on)

- **Staircase in the slope:** no. The slope excess does not follow the path ratio (Spearman −0.05 on Chao Phraya).
- **Chao Phraya branches:** the gap follows how SFINCS splits the discharge, not the grid.
  - A constriction on the central branch (222-255 km) raised the junction at 280 km by 7 m.
  - The narrow western and middle arms carried 1.9-2.0 times their 1D discharge and stood 6-7 m high.
  - The separate eastern inflow stood +1 m, like Ebro.

## 2a Ebro sweep (round-1 calibration, one factor at a time)

| setting | gap median (m) |
|---|---|
| baseline (advection 1) | 0.76 |
| `nr_levels` 50 | 0.76 |
| `huthresh` 0.05 | 0.76 |
| `nuvisc` 0 | 0.50 |
| uniform Manning | 0.45 |
| Manning on the corridor | 0.43 |
| discharge halved | 0.43 |
| **advection 0** | **0.19** |

Only advection off closes the gap. On the steady Chao Phraya round 1:

- the median falls from 3.43 to 0.25 m;
- the central-branch constriction disappears (slope excess 0.63 → −0.02 m/km);
- branches: middle 6.07 → 0.11 m, west 5.78 → 0.35 m, east 1.06 → 0.71 m.

## 2b Synthetic trench (exact answer known)

A 100 m wide trench (n 0.03, slope 1e-4, walls 25 m) is held at normal depth by a downstream level boundary. It is laid three ways across a 100 m grid: aligned with it, at 45°, and as a meander.

Median level error against normal depth at a stable `alpha`:

| layout | Q (m³/s) | advection 0 | advection 1 |
|---|---|---|---|
| aligned | 3000 | +0.38 m | +0.39 m |
| 45° | 3000 | +0.76 m | +2.33 m |
| meander | 3000 | +1.11 m | +8.70 m |
| 45° | 500 | +0.51 m | +0.80 m |
| meander | 500 | +0.68 m | +2.23 m |

- **Aligned:** advection makes no difference.
- **Diagonal and meander:** advection adds head that grows upstream and with discharge. Momentum is lost at each corner of the staircase cell path, so this is a grid artefact.
- **Physics:** the delta rivers run at Froude 0.1-0.2, where the local-inertia equations (advection off) are the appropriate SFINCS formulation.
- **Time step:** `alpha` 0.5 with advection off went unstable in 2 of 6 synthetic runs; 0.25 was stable in all of them.

## Event stability: `alpha`

The first full rerun with advection off kept SFINCS's `alpha` 0.5.

- **Ebro `river_100_03SLR` went unstable** above about 2500 m³/s.
- **At the inflow:** the cell is a closed nook at the domain tip, draining through a 3-cell trench. Its level swung 5-9 m per hour up to 54 m.
- **Upstream gorge:** the backwater stood it at a flat 24 m.

Four scratch reruns of that event put the cause on `alpha`, not on where the inflow sits:

| inflow | `alpha` | largest hourly step at inflow | peak at inflow | peak at km 2.3 / 5.1 / 12 |
|---|---|---|---|---|
| domain tip | 0.5 | 9.04 m | 53.8 m | 24.1 / 23.1 / 11.2 m |
| 1.6 km inside | 0.5 | 6.74 m | 41.3 m | 30.1 / 21.8 / 11.5 m |
| domain tip | 0.25 | 0.19 m | 16.7 m | 15.1 / 12.8 / 10.1 m |
| 1.6 km inside | 0.25 | 0.17 m | 15.6 m | 15.1 / 12.8 / 10.1 m |

The pipeline run at `alpha` 0.25 reproduces the third row. Fast level changes left in the four events:

- **Chao Phraya coast:** 5478 cells step more than 2 m in one hour. These are wetting fronts: low-lying cells (median bed 0.4 m), during the surge (hours 42-74), and none reverses.
- **Ebro river:** 102 cells, 2 of which reverse once.
- **Ebro coast:** 34 cells. **Chao Phraya river:** 19 cells.

Steady calibrations and stable events give identical results at 0.25 and 0.5:

- crests are unchanged;
- Chao Phraya flood metrics are unchanged.

## Gap and flood effect (advection 0, `alpha` 0.25)

Gap per reach, median of reach medians:

| | Ebro | Chao Phraya |
|---|---|---|
| advection 1 | 0.76 m | 3.43 m |
| advection 0 | 0.19 m (reaches 0.06 to 0.29) | 0.25 m (reaches −0.55 to 1.28) |
| main path, upstream 20 % | 0.97 → 0.24 m | 6.59 → 0.07 m |

Calibrated crests, median: Ebro 4.27 → 3.73 m, Chao Phraya 11.15 → 9.58 m.

100-year scenarios with 0.3 m SLR, change from advection 1 / `alpha` 0.5:

| | flooded area | volume | mean depth |
|---|---|---|---|
| Ebro coast | 260.4 → 260.4 km² (0 %) | −0.1 % | 0 % |
| Ebro river | 173.1 → 178.6 km² (+3.2 %) | −10.1 % | −12.3 % |
| Chao Phraya coast | 1866 → 1937 km² (+3.8 %) | +1.0 % | −2.8 % |
| Chao Phraya river | 1553 → 1099 km² (−29.3 %) | −28.6 % | +4.2 % |

- **Chao Phraya river:** the large drop is the backwater from the advection artefact leaving the upper delta. The 1553 km² baseline was inflated by it. That also explains Phase 1.9's non-monotonic depth sensitivity: 1082 km² at depth × 0.7 and 1154 km² at × 1.3, both under the old baseline.
- **Ebro river:** the max depth falls from 26.8 to 11.4 m. The old maximum was the inflow pile-up.

## Decisions and follow-ups

- `sfincs.advection: 0` and `sfincs.alpha: 0.25` are the defaults for calibration and events.
- **Phase 1.5:** the bifurcation shares were measured under advection 1, whose backwater drove the reversed splits. Rerun on the new round 1 before Phase 5.
- **Phase 1.9:** the Chao Phraya depth sensitivity was measured against the inflated baseline. Rerun it with the Phase 4 bundle.
- **Phase 8:** make the hourly level change in event runs a standing check.
- **TODO B32b:** in the Ebro upper gorge the conditioned DEM on the centreline is 13.7-16.4 m (6-12 m either side), so rule 11 hangs a 9-11 m deep channel there.
