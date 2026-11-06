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
| `gvf_setup/` | `analysis/gvf_setup.py` | `concept`, `froude_1d`, `froude_sfincs_<river event>`, `long_profile_<basin>`, `bed_alternatives` |
| `travel_time/` | `analysis/travel_time.py` | `travel_time` |
| `upstream_boundary/` | `analysis/upstream_boundary.py` | `upstream_boundary` |
| `depth_sensitivity/` | `analysis/depth_sensitivity.py` | `depth_sensitivity` (opt-in: `depth_sensitivity.enabled`) |

Each figure has a description `<figure>.txt` next to it: what the panels show
and what each colour, line and marker stands for.

The depth-sensitivity runs are a full rule 11 + build per basin and scale, about
12 h per scale for the Chao Phraya on 14 cores. `--rerun-triggers mtime` keeps
Snakemake from rebuilding production outputs whose code or params metadata
changed without changing their content.

## 1. How the bed is set (`gvf_setup/concept.png`)

1. **Observed water surface:** FathomDEM river-centre medians per 500 m node.
2. **Fitted surface and slope (rule 10):** mouths pinned at the calm sea level,
   a monotone downstream fit, two-scale smoothing. The slope is taken along each
   node's largest-Q path and floored at `slope_floor` (1e-5) for the depth only.
   The pin is the calm level, not a surge level, because the terrain model
   records the river at ordinary water levels; the surge enters later, as the
   event's sea boundary. On the flat reach near the sea the depth follows from
   Q, W and the minimum slope, not from the pin, so a higher pin would lift
   surface and bed together and leave the channel shallower relative to the sea.
3. **Depth and bed (rule 11, round 0):** Manning normal depth of a rectangular
   channel, h = (n Q / (W √S))^(3/5), with Q the routed GloFAS bankfull
   discharge, W the SWORD width and n `channel_manning_n` (0.03); bed =
   surface − h.
4. **Forward check:** a steady GVF solve on that bed at Q_bankfull should
   return the surface. The bed is hung from the surface, so the GVF solve
   checks the bed but does not set it. On floored reaches the channel is
   shallower than the flat surface needs, so the computed level climbs above
   it. Upstream the excess decays as a backwater curve rather than adding up.
5. **Flood model:** the bed is burned into the SFINCS subgrid. Rule 11 runs the
   model with the river walled in to set the dikes along the river to the
   protection standard.

## 2. Advection off, and when that is valid (`gvf_setup/froude_1d.png`, `froude_sfincs_<river event>.png`)

**Why advection is off.** The rivers are narrower than a 100 m cell and run
diagonally or meander across the grid, so SFINCS routes them through a staircase
of cells. With the advection term on, the flow loses momentum at every corner of
that staircase. The model makes up for the loss with extra water-surface slope,
so the level rises and the error grows upstream. It is a grid artefact, not
physics.

**Effect with advection on**, steady calibration runs against the 1D profile on
the same bed:
- the 2D channel level stood a median 0.76 m (Ebro) and 3.43 m (Chao Phraya)
  above the 1D level, up to 6-7 m on the narrow Chao Phraya branches;
- the backwater reversed the discharge split at two Chao Phraya bifurcations
  (0.31 and 0.52 of the flow down arms whose width gives 0.16 and 0.20);
- the calibrated crests were too high (median Ebro 4.27 m, Chao Phraya
  11.15 m);
- the Chao Phraya 100-year river flood covered 1553 km2.

**With advection off** (rules 11 and 12a, `sfincs.advection: 0`), the median
gap is 0.19 m (Ebro) and 0.24 m (Chao Phraya), and the splits match the widths
within 0.03. The crest medians drop to 3.73 and 9.58 m, and the Chao Phraya
river flood to 1099 km2 (-29 %). The cost is the time step: with advection off,
`alpha` 0.5 went unstable on the Ebro river event (inflow level swinging 5-9 m
per hour), so `sfincs.alpha` is 0.25, which roughly doubles run times.
`alpha` 0.35 may also be stable but is untested on a real event
(`analysis/alpha_stability.py`).

**Validity.** Without advection SFINCS solves the local-inertia equations, which
hold where inertia is small compared with gravity and friction, that is at low
Froude numbers: errors are negligible below about 0.3 and grow above 0.5
(de Almeida & Bates 2013). The check is on the river channels at high
discharge, where the flow is fastest. Coastal inundation is slow, deep-water
flow and is not the case the choice depends on.
- **Chao Phraya:** under 0.3 on every node at Q_bankfull and the event peak,
  and on 99.6 % at Q_mean. The p95 at the peak is 0.17.
- **Ebro:** 8-12 % of nodes are between 0.3 and 0.5, all in the upper gorge
  below the inflow; none is above 0.5. The p95 at the peak is 0.37.
- **1D solver:** it keeps the (1 - Fr^2) term of the GVF equation. At p95 this
  steepens the 1D profile by a factor of 1.03 (Chao Phraya) and 1.15 (Ebro)
  against SFINCS's form, which is small next to the bed uncertainty.
- **SFINCS Froude numbers:** the river event map stores `vmax`
  (`sfincs.simulation.storevelmax: 1`), and `froude_sfincs_<river event>.png`
  maps the largest vmax / sqrt(g h) per wet cell. Runs from before that setting
  have no vmax (TODO B40).

## 3. The bed and its alternatives (`gvf_setup/long_profile_<basin>.png`, `bed_alternatives.png`)

`bed_alternatives.png` compares ten ways to set the bed (A-J, described in
its caption) with the production bed A, delta by delta, on what matters for
the flood model:
- **consistency:** computed level at bankfull minus the fitted surface, |median|
  and p95 (equal within 0.02 m);
- **crest demand:** the dike crest rule 11 would set minus the terrain at the
  river centre, median and p95 (equal within 0.05 m). Overtopping in itself is
  not a criterion, since rule 11 builds a dike wherever the river would
  overtop; what a bed costs is how high those dikes must be;
- **advection-off validity:** 1D Froude number at the event peak, median and
  maximum, scored by band (up to 0.3, 0.3-0.5, above 0.5);
- **depth plausibility:** depth median and p95, worse when more than 50 % deeper
  than A;
- **reliance on the minimum slope:** share of nodes on the slope floor, shown for
  information.

**Decision rule.** An alternative replaces A only if it is no worse than A on
every scored criterion in every delta and better on at least one. None is:

| alternative | better than A | worse than A |
|---|---|---|
| B, D, F, H, J (recent median discharge) | consistency | crest demand about doubled (Chao Phraya median 10 m vs 4.4 m); Froude maximum above 0.5 |
| C (SWOT, two-scale, bankfull) | Ebro consistency and crest demand | Chao Phraya consistency, crest p95, Froude maximum 0.75 |
| E (SWOT, single 20 km, bankfull) | consistency, crest demand in both deltas | Chao Phraya Froude maximum 0.46 vs 0.22 |
| G (SWOT, single 40 km, bankfull) | consistency in both deltas | Ebro crest p95 (5.9 vs 5.3 m); Chao Phraya Froude maximum 0.41 vs 0.22 |
| I (sea inversion) | consistency, crest demand | channels 39-40 m deep (p95), four times A's on the Ebro |

A is kept. E comes closest: it would need lower dikes and reproduces its
surface better in both deltas, but it raises the peak Froude number on the
Chao Phraya into the range where dropping advection becomes questionable, and
it relies more on the minimum slope (40-43 % of nodes). A misses the
consistency check by design: the depth uses the floored, long-smoothed slope,
which the surface does not share (concept.png, step 4).

The slope floor was varied separately: 5e-6 and 2e-5 change the depth on the
25-29 % of nodes on the floor by +24 % and -20 %.

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
