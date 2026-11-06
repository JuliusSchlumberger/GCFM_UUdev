# River bed, GVF and timing: method choices and their evidence

How the river bed is set, why SFINCS runs with advection off, how long the river
peak takes to reach the coast, where the sea's influence ends upstream, and how
sensitive flooding is to the depth. Numbers are for the Ebro (2433835) and the
Chao Phraya (3279946), scenarios `coast_100_03SLR` and `river_100_03SLR`.

## Reproduce

```
snakemake analysis_suite --cores 14 --rerun-triggers mtime
```

Basins, events and switches: `config/analysis_suite.yml`. Figures go to
`<results_dir>/figures/analysis/`:

| folder | script | figures |
|---|---|---|
| `gvf_setup/` | `analysis/gvf_setup.py` | `concept`, `froude_1d`, `froude_sfincs_<scenario>`, `advection`, `long_profile`, `bed_options` |
| `travel_time/` | `analysis/travel_time.py` | `travel_time` |
| `upstream_boundary/` | `analysis/upstream_boundary.py` | `upstream_boundary` |
| `depth_sensitivity/` | `analysis/depth_sensitivity.py` | `depth_sensitivity` (opt-in: `depth_sensitivity.enabled`) |

The depth-sensitivity runs are a full rule 11 + build per basin and scale, about
12 h per scale for the Chao Phraya on 14 cores. `--rerun-triggers mtime` keeps
Snakemake from rebuilding production outputs whose code or params metadata
changed without changing their content.

## 1. How the bed is set (`gvf_setup/concept.png`)

1. **Water surface (rule 10):** FathomDEM river-centre medians per 500 m node,
   mouths pinned at the calm sea level, a monotone downstream fit, two-scale
   smoothing.
2. **Slope S:** the surface slope along each node's largest-Q path, floored at
   `slope_floor` (1e-5) for the depth only.
3. **Depth:** Manning normal depth of a rectangular channel,
   h = (n Q / (W √S))^(3/5), with Q the routed GloFAS bankfull discharge, W the
   SWORD width and n `channel_manning_n` (0.03).
4. **Bed:** surface − h (rule 11, round 0).
5. **Forward check:** a steady GVF solve on that bed at Q_bankfull should
   return the surface. The bed is hung from the surface, so the GVF solve
   checks the bed but does not set it.
6. **SFINCS:** the bed is burned into the subgrid. Rule 11 runs the walled model
   to set the coastal crests.

## 2. Advection off, and when that is valid (`gvf_setup/advection.png`, `froude_1d.png`)

**Why advection is off.** On a 100 m grid a diagonal or meandering river is a
staircase of cells. SFINCS's advection term loses momentum at every corner and
adds head that grows upstream. In a synthetic trench (exact answer known), the
median level error is:
- aligned: the same with and without advection;
- diagonal: +2.33 m with advection, +0.76 m without;
- meander: +8.70 m with advection, +1.11 m without.

On the deltas, the steady 2D level stood 0.76 m (Ebro) and 3.43 m (Chao Phraya)
above the 1D profile with advection. Without it the gap is 0.19 and 0.24 m
(median). Rules 11 and 12a set `sfincs.advection: 0`.

**Validity.** The local-inertia equations (advection off) hold where inertia is
small: errors are negligible below Froude 0.3 and grow above 0.5 (de Almeida &
Bates 2013).
- **Chao Phraya:** under 0.3 on every node at Q_bankfull and the event peak,
  and on 99.6 % at Q_mean. The p95 at the peak is 0.17.
- **Ebro:** 8–12 % of nodes are between 0.3 and 0.5, all in the upper gorge
  (the first 10 km below the inflow); none is above 0.5. The p95 at the peak is
  0.37.
- **1D solver:** it keeps the (1 − Fr²) term of the GVF equation. At p95 this
  steepens the 1D profile by a factor of 1.03 (Chao Phraya) and 1.15 (Ebro)
  against SFINCS's form, which is small next to the bed uncertainty.
- **SFINCS Froude numbers:** the event maps store `vmax`
  (`sfincs.simulation.storevelmax: 1`). `froude_sfincs_<scenario>.png` maps the
  largest vmax / √(g h) per wet cell. Runs from before that setting have no
  vmax; the figure says so.

**Time step.** With advection off, `alpha` 0.5 went unstable on the Ebro river
event, with the inflow level swinging 5–9 m per hour. `sfincs.alpha` is 0.25.

A synthetic search (`analysis/alpha_stability.py`) tested whether anything else
lets the time step grow:
- no ramp, bed step, inflow position, cell size, `theta`,
  `wiggle_suppression` or `viscosity` setting replaces the small `alpha`;
- `alpha` 0.35 is stable in every 100 m case but untested on a real event.

## 3. The bed and its inputs (`gvf_setup/long_profile.png`, `bed_options.png`)

| choice | options compared | kept | evidence |
|---|---|---|---|
| surface source | FathomDEM; SWOT node levels (SWORD v17c) | FathomDEM | SWOT does not reduce the share of nodes on the slope floor |
| surface fit | two-scale; single 20 km; single 40 km | two-scale | single fits pass the forward check only with the period-median discharge |
| design discharge | bankfull; mean; GloFAS median since 2023-07 | bankfull | beds for lower discharges overflow at Q_bankfull on 31 % (Ebro) and 91 % (Chao Phraya) of nodes, and 7 / 523–629 nodes cannot carry Q_mean |
| bed method | Manning; Manning + sea inversion | Manning | the inversion needs 35–40 m deep channels at bankfull |
| slope floor | 5e-6; 1e-5; 2e-5 | 1e-5 | depth on the floor nodes (25–29 %) changes +24 % / −20 % |

- **Forward check:** the default bed returns the surface within a median of
  +0.20 m (Ebro) and +0.44 m (Chao Phraya); the p95 is 0.87 and 1.12 m.
- **Floor nodes:** these are the flat, mostly tidal lower reaches (Ebro: the
  lower 18 km), where the depth is set by the floor rather than the slope.
- **Chao Phraya surface between 150 and 270 km:** SWOT levels and the mean-flow
  levels (GVF and SFINCS spin-up) lie 4–6 m below the fitted FathomDEM surface.
  The bed is hung for bankfull flow from that surface.
- **SWOT vertical spread:** the pass-to-pass spread includes the seasonal and
  tidal range (median 1.1–1.4 m), so `swot_max_mad_m` is 3.0.

**Bifurcations.** The Chao Phraya's discharge splits by SWORD width
(`river_network.accumulate_discharge`). With advection off, SFINCS matches the
width split within 0.03 at three of four bifurcations. At 44306300275 the SWORD
arms are misaligned and SFINCS sends 0.55 instead of 0.37 down the narrow arm.

Rule 11 checks every split:
- it writes `round1/bifurcation_split.csv`;
- it stops when an arm's share is more than 0.05 off;
- a split can be set with `river_processing.split_override`;
- 44306300275 is accepted until the network is fixed (TODO B37).

## 4. Calibration run and crests

- **Run length.** The walled travel time gives when the wave front reaches the
  coast, not when the levels behind it stop rising. The run lasts
  ramp + max(T_walled, 1.75 × L²/D) + margin, with D = Q / (2 W S) (Hayami).
  - Ebro: 65 h, settled by 21 h.
  - Chao Phraya: 219 h, settled by 171 h.
  - No centreline cell still rises by more than 5 cm (`round1/steady_check.csv`).
- **Calibration sea.** The calibration holds the sea at the storm tide of the
  coastal protection RP. GTSM matches the Tarragona gauge (13 years) within
  0.03 m on the median annual maximum and the 10-year level. The Chao Phraya
  has no usable gauge.

## 5. River peak travel to the mouth (`travel_time/travel_time.png`)

Hours after the inflow peak, main mouth, river-only event:

| reference point | Ebro | Chao Phraya |
|---|---|---|
| first maximum of the tide-filtered mouth discharge | 16.3 | 24.5 |
| half rise | −3.5 | 56.2 |
| plateau (within 2 % of the maximum): start / midpoint / end | 1.0 / 17.8 / 34.7 | −4.8 / 54.6 / 114.0 (open) |
| configured lag | 21.2 | 41.2 |
| walled estimate at the calibration discharge (rule 11) | 6.2 | 40.8 |
| walled estimate at bankfull discharge | 15.8 | 65.5 |

- **Reference point.** The plateau midpoint is the one used; it sets
  `compound.lag_overrides_hr`.
  - The half rise and the plateau start are distorted by attenuation.
  - The first maximum is a single point on a plateau that is nearly flat.
  - The Chao Phraya plateau is still open when the run ends, so its midpoint is
    a lower bound (TODO B32).
- **Parametric estimate.** The walled estimate at bankfull discharge matches the
  model in both deltas.
  - Above bankfull, the extra flow spills onto the floodplain, and the peak
    travels at the bankfull channel speed.
  - At the calibration discharge the estimate is 2.6 times too fast for the
    Ebro.
  - It only sizes the calibration run, which the settling time already sets
    longer.
- **Compound lag.** Both configured lags put the surge peak on the river's
  plateau: the aligned worst case.
  - Reanalysis (`analysis/lag_cooccurrence.py`, GTSM and GloFAS 1979–2024):
    Chao Phraya surge and river peaks are seasonally disjoint (NE vs SW
    monsoon).
  - On the Ebro, one surge peak in five is followed by a river peak 3–4 days
    later.

## 6. Upstream boundary (`upstream_boundary/upstream_boundary.png`)

The rise of the river level at mean discharge over its calm-sea level along each
forcing mainstem, and where it ends (below max(0.05 m, 5 % of the rise at the
mouth)):

| forcing reach | forcing point | SFINCS event | GVF, event sea level | GVF, extreme sea level | margin at extreme (≥ 5 km) |
|---|---|---|---|---|---|
| Ebro 21604100071 | 65.5 km | 53.3 km | 53.8 km | 56.2 km | 9.3 km |
| Chao Phraya 44306200155 | 264.5 km | 198.3 km | 197.3 km | 202.7 km | 61.8 km |
| Chao Phraya 44306300031 | 293.5 km | 275.4 km | 275.4 km | 276.4 km | 17.1 km |

- **Agreement.** The steady GVF estimate and the SFINCS event agree within
  1 km.
- **Why.** The reach ends where the river surface at mean flow rises above the
  sea level, which topography sets. So neither the event's finite duration nor
  the size of the surge matters much: the extreme sea (RP100 + 1 m SLR) moves
  the end 1–5 km.
- **Timing.** The surge peak reaches the end of its reach 2 h (Ebro) and
  9–14 h (Chao Phraya) after its peak at the sea boundary.

## 7. Depth sensitivity (`depth_sensitivity/depth_sensitivity.png`)

Burned depth × 0.7 and × 1.3, crests recalibrated on each bed, relative to the
production bed:

| | flooded area | volume | urban exposed |
|---|---|---|---|
| Ebro coast | 1.00 / 1.00 | 0.94 / 1.06 | 1.00 / 1.00 (2.2 km²) |
| Ebro river | 1.04 / **0.36** | 1.08 / 0.82 | 1.43 / **0.15** (3.4 km²) |
| Chao Phraya coast | 0.95 / 1.06 | 0.87 / 1.13 | 0.93 / 1.11 (650 km²) |
| Chao Phraya river | 1.01 / 0.97 | 0.98 / 1.04 | 1.02 / 0.95 (48 km²) |

- **Direction.** A deeper bed carries more of a river flood in the channel but
  lets the surge further inland. Depth errors therefore shift the balance
  between river and coastal flooding rather than cancelling.
- **Size.** The Ebro river flood is the sensitive case: a 30 % deeper bed
  removes 64 % of the flooded area.

## Open

- SFINCS Froude numbers need event runs with `storevelmax: 1`.
- Steady SFINCS boundary benchmark on the current bed, including the stress
  case (TODO B31).
- Chao Phraya river run ends on the mouth plateau (TODO B32).
- Held vs hydrograph calibration sea for the crests (TODO B36).
- Bifurcation 44306300275 network fix (TODO B37); Chao Phraya bank tops and
  flood walls (TODO B38).
- Shorter coast-only event runs (TODO B39).
- `alpha` 0.35 on a real event (`analysis/alpha_stability.py event`).
