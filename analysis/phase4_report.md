# Phase 4: channel capacity and the sea inversion: Ebro (2433835) and Chao Phraya (3279946)

Question: does a bed hung consistently from the surface carry the discharge it should, and does a sea inversion fix the nodes whose depth sits on the slope floor? Outcome: no option passes the gates, so the defaults stay (`design_discharge: bankfull`, `bed_method: manning`). The inversion is in code as an option.

Script: `phase4_capacity.py`, writing `90_phase4_capacity_{nodes.csv,profile.png}` and `phase4_capacity_summary.csv`.

## 4.2 Channel capacity

Q_cap is the discharge at which the steady GVF level on the bed reaches the bank top (the Phase 1.1 extraction).
- **Method:** the network is solved at 0.05–5 × the routed Q_bankfull, on a log grid of 25 steps.
- **Excluded:** nodes whose bank top is not above the calm sea level (6 on Ebro, 11 on Chao Phraya). These are mouths with no banks, so any level counts as overflowing.

## 4.3–4.4 Sea inversion (`bed_method: manning_sea_inversion`)

- **Which nodes:** connected nodes with a surface slope at or below `slope_resolved_min` (3e-5) that reach a mouth or GEBCO bathymetry.
- **Their depth:** h = c·W^b, with b fitted on the resolved nodes.
- **How c is set:** by bisection, so that the GVF level from the sea meets the surface at the resolved nodes next to the set, to within 2 cm. Every run in the table converged to within 2 cm.

## Results

Columns are Ebro / Chao Phraya. Forward is the GVF level at the design discharge minus the surface.

| option | Q_cap/Q_bf p50 | nodes with Q_cap < Q_mean | forward p50 (m) | forward p95 (m) | floor share | inverted share | depth of the inverted set |
|---|---|---|---|---|---|---|---|
| default: FathomDEM, bankfull, Manning | 3.91 / 1.28 | 2 / 2 | 0.20 / 0.44 | 0.87 / 1.12 | 0.29 / 0.25 | — | — |
| FathomDEM, period median | 1.49 / 0.39 | 7 / 629 | 0.16 / 0.26 | 0.61 / 0.85 | 0.29 / 0.25 | — | — |
| SWOT 40 km, period median | 1.38 / 0.41 | 7 / 523 | 0.02 / 0.17 | 0.09 / 0.66 | 0.26 / 0.36 | — | — |
| default + inversion | 5.00 / 1.35 | 0 / 0 | 0.00 / 0.10 | 0.70 / 0.92 | 0 / 0.15 | 0.38 / 0.12 | 4.65·W^0.38 (≈ 35 m) / 39.6 m |
| SWOT 40 km, period median + inversion | 1.41 / 0.45 | 5 / 428 | 0.00 / 0.01 | 0.11 / 0.53 | 0 / 0.20 | 0.39 / 0.21 | 0.51·W^0.33 (≈ 3 m) / 8.5 m |

**Bankfull discharge with the inversion.** Holding the flat lower reaches at the observed mean stage takes a channel 35–40 m deep. This is the Phase 1.1 inconsistency in another form: a 1.5-year flood cannot sit below a mean-stage surface in a channel of realistic depth.

**Period median discharge.** The surface-and-discharge pair is consistent and the depths are plausible: 3 m on Ebro, 8.5 m in Chao Phraya's tidal reach. But on Chao Phraya, 21–32 % of the nodes cannot carry the mean discharge within their banks.
- The plan reads Q_cap below Q_mean as a width or bank-top error.
- Bangkok's flood walls are not in FathomDEM's P90 bank top, which is a likely part of it.
- n = 0.03 and the SWORD widths are the other candidates.

## Gates

| gate | conditions | result |
|---|---|---|
| `design_discharge: period_median` | forward check passes on both basins; no node with Q_cap below Q_mean | **fails**: 7 / 523 nodes with SWOT, 7 / 629 with FathomDEM |
| `bed_method: manning_sea_inversion` | the same, plus a floor share below 10 % | **fails**: the bankfull variant fails Chao Phraya's p95 (0.92 m) and floor share (0.15), and its depths are not physical; the period-median variant has 5 / 428 capacity nodes and a 0.20 floor share on Chao Phraya |

Defaults unchanged: `surface_source: fathom`, `surface_fit: two_scale`, `design_discharge: bankfull`, `bed_method: manning`.

## Follow-ups

- **Phase 1.9 rerun:** the depth sensitivity is rerun against the advection-0 baseline with these defaults (the "Phase 4 bundle", unchanged).
- **TODO B38:** Chao Phraya's bank tops and widths where Q_cap is below Q_mean, under the period-median bed. Flood walls are missing from the DEM bank tops.
