# River bed, momentum and timing in a global delta flood model: summary for review

## Context

The pipeline builds compound (storm surge + river) flood models of river
deltas from global data only, with SFINCS (reduced-physics 2D shallow-water
solver, v2.3.0) on a 100 m grid with a 10 x 10 subgrid. Rivers are mostly
narrower than a cell and enter through the subgrid. There is no river
bathymetry, so the channel bed must be estimated. Inputs:
- FathomDEM (30 m terrain; its river-centre values are water-surface levels);
- GloFAS discharge (daily reanalysis; bankfull taken as about the 1.5-year flood);
- SWORD v17c river network and widths, with SWOT node water levels;
- GTSM storm tides;
- FLOPROS protection standards.

The two test deltas are the Ebro (one channel, about 65 km modelled, Q_bankfull
about 970 m3/s) and the Chao Phraya (a branching network over about 290 km,
two inflows, Q_bankfull about 4000 m3/s on the main inflow). Each delta has one
coastal-only and one river-only event, both 100-year with 0.3 m sea-level rise.

## Method as implemented

1. **Water surface.** FathomDEM river-centre medians every 500 m along the
   network, pinned to the calm (tide-averaged) sea level at the mouths, made
   monotone towards the sea and smoothed at two length scales.
2. **Depth.** Manning normal depth of a rectangular channel at bankfull
   discharge, h = (n Q / (W sqrt(S)))^(3/5), with n = 0.03, W the SWORD width
   and S the surface slope. S is floored at 1e-5 because normal depth diverges
   on a flat surface.
3. **Bed.** The bed is the surface minus h.
4. **Check.** A steady gradually-varied-flow (GVF) solve on that bed at bankfull
   should reproduce the surface ("forward check"). The GVF equation includes
   the (1 - Fr^2) term.
5. **Crests.** The bed is burned into the subgrid. Dikes along the rivers and
   coast are calibrated in a steady SFINCS run, with the rivers walled in at
   the protection-standard discharge and the sea held at the protection
   standard's storm tide: crest = calibrated water level (+ freeboard, 0).
6. **Events.** Each event restarts from a spin-up at mean discharge with a
   calm sea.

## Findings and decisions

**Momentum advection is off.**
- **Why:** with advection on, the steady 2D channel level stood a median 0.76 m
  (Ebro) and 3.43 m (Chao Phraya) above the 1D GVF level on the same bed, and
  up to 6-7 m on narrow branches.
- **Diagnosis:** a grid artefact. On a 100 m grid a sub-cell channel that runs
  diagonally or meanders becomes a staircase of cells, momentum is lost at
  each corner, and the head grows upstream. The cause was isolated with
  synthetic trenches with an exact answer.
- **Other effects of advection on:** it reversed two bifurcation splits and
  inflated the Chao Phraya 100-year river flood (1553 vs 1099 km2).
- **With advection off:** the gap is 0.19 and 0.24 m.
- **Cost:** the time-step factor alpha had to drop from 0.5 to 0.25 (about 2x
  run time); at 0.5 a deep one-cell inflow trench oscillated by 5-9 m per
  hour.

**The local-inertia approximation is justified by low Froude numbers**, from
the 1D solve on the bed at the river event peak with all flow kept in the
channel:
- Chao Phraya: p95 0.17, with every node below 0.3;
- Ebro: p95 0.37, with 8-12 % of nodes (the upper gorge) between 0.3 and 0.5
  and none above 0.5.

The SFINCS-side Froude check (from vmax) awaits reruns.

**The production bed is kept against nine alternatives.** The alternatives
vary:
- the surface: SWOT instead of FathomDEM;
- the fit: one fit over 20 or 40 km;
- the design discharge: the GloFAS median since 2023 instead of bankfull;
- the bed method: depth from a width power law calibrated so levels from the
  sea meet the surface (the "sea inversion").

They are scored against production per delta on:
- consistency: forward check |median| and p95;
- crest demand: crest minus the river-centre DEM, median and p95;
- Froude band at the event peak (up to 0.3 / 0.3-0.5 / above 0.5);
- depth plausibility: worse when more than 50 % deeper than production.

An alternative replaces production only if it is no worse on every criterion
in every delta and better on one. None is:
- **median-discharge beds:** too shallow; crests roughly double (Chao Phraya
  median 10 m vs 4.4 m) and the maximum Froude number exceeds 0.5;
- **sea inversion:** needs 39-40 m deep channels;
- **closest, SWOT surface with a 20 km single fit:** lower crests and better
  consistency in both deltas, but the Chao Phraya peak Froude number rises
  from 0.22 to 0.46 and 40-43 % of nodes rely on the slope floor.

**Production fails its own consistency check**: forward-check median 0.20 /
0.44 m, p95 0.87 / 1.12 m.
- **Cause:** the depth uses a slope smoothed over a longer distance than the
  surface keeps, so the two differ by design.
- **Floored nodes:** the miss is not concentrated where the slope is floored
  (25-29 % of nodes, the flat tidal reaches). Floored nodes: Ebro median
  0.09 m, against 0.34 m elsewhere; Chao Phraya 0.50 m, against 0.41 m.
- **Floor sensitivity:** moving the floor to 5e-6 or 2e-5 changes the depth on
  floored nodes by +24 % / -20 %.

**Calibration run length must cover diffusive settling, not wave arrival.**
- **The rule:** run = ramp + max(T_walled, 1.75 L^2/D), with Hayami diffusivity
  D = Q / (2 W S).
- **Why:** sizing the run on the walled wave-arrival time left the Chao Phraya
  still filling, with crests up to 5 m too low.
- **Now:** the Ebro settles by 21 h of a 65 h run, the Chao Phraya by 171 h of
  219 h.

**Bifurcations split by SWORD width.** SFINCS, with advection off, matches the
width split within 0.03 at three of four Chao Phraya bifurcations. The fourth
is a misaligned network junction.

**River peak travel to the mouth.** Hours after the inflow peak, at the main
mouth:

| | first maximum | plateau midpoint | walled estimate at calibration Q | at bankfull Q |
|---|---|---|---|---|
| Ebro | 16.3 h | 17.8 h | 6.2 h | 15.8 h |
| Chao Phraya | 24.5 h | at least 54.6 h | 40.8 h | 65.5 h |

- **Mouth signal:** the discharge plateau (within 2 % of its maximum) is used,
  because attenuation distorts the half-rise and the first maximum.
- **Estimate:** a walled kinematic estimate at bankfull discharge matches;
  water above bankfull is stored on the floodplain.
- **Compound lag:** the configured lag (river peak relative to surge peak)
  centres the surge on the river plateau, a design worst case.
- **Reanalysis (GTSM and GloFAS, 1979-2024):**
  - Chao Phraya surge and river peaks are seasonally disjoint (NE vs SW
    monsoon);
  - on the Ebro, one surge peak in five is followed by a river peak, median
    +89 h later.

**Upstream boundary.** Where the sea's influence on the river level ends, at
mean discharge:
- the steady GVF estimate and the SFINCS coastal event agree within 1 km in
  both deltas;
- the reach ends abruptly where the river surface rises above the storm-tide
  level, so it is set by topography;
- the extreme case (100-year plus 1 m sea-level rise) moves the end only
  1-5 km;
- the forcing points lie 9-62 km beyond the end (at least 5 km required).

**Depth sensitivity.** Burned depth x 0.7 / x 1.3, crests recalibrated:

| | flooded area | volume | urban exposed |
|---|---|---|---|
| Ebro coast | 1.00 / 1.00 | 0.94 / 1.06 | 1.00 / 1.00 |
| Ebro river | 1.04 / 0.36 | 1.08 / 0.82 | 1.43 / 0.15 |
| Chao Phraya coast | 0.95 / 1.06 | 0.87 / 1.13 | 0.93 / 1.11 |
| Chao Phraya river | 1.01 / 0.97 | 0.98 / 1.04 | 1.02 / 0.95 |

A deeper bed contains more of a river flood but carries the surge further
inland. Depth errors therefore shift the balance between river and coastal
flooding rather than cancelling.

## Known weaknesses

- **No bathymetry:** there is no observed depth to validate against; the bed is
  judged only by internal consistency and plausibility.
- **Tidal reaches:** the depth there comes from an assumed minimum slope.
  Tidal flow, not river discharge, probably shapes those channels, so the
  bankfull Manning depth is likely too shallow.
- **Chao Phraya surface:** between 150 and 270 km, SWOT and mean-flow levels lie
  4-6 m below the FathomDEM surface the bed hangs from. FathomDEM may capture
  high stages there.
- **SWOT:** the node levels scatter by 1-1.4 m between passes (tide and season
  included), so it is weak as a surface source.
- **Single global roughness:** n = 0.03, and rectangular sections.
- **Bankfull:** taken from GloFAS, uncorrected.
- **Coverage:** two deltas and one event per scenario; the Chao Phraya
  river-event plateau has not ended when the run does.

## Questions for the reviewer

1. Is the local-inertia approximation adequately justified by a 1D Froude
   number below about 0.3-0.5 at peak flow? Is the threshold right (de Almeida
   & Bates 2013 and later work), and should the 2D check use other indicators?
2. Is hanging the bed from a DEM water surface with Manning normal depth at
   bankfull defensible against alternatives in the literature? Candidates:
   hydraulic geometry (Leopold & Maddock; Andreadis et al. 2013), GVF
   inversion from water surfaces, and global depth products.
3. How should depth be set in tidal reaches where the surface is flat? What
   does tidal channel geometry (tidal prism, exponential widening) imply, and
   is a slope floor of 1e-5 reasonable?
4. Is "replace production only if it is no worse on every criterion in every
   delta" a sound decision rule for two deltas? Are consistency, crest demand,
   Froude band and depth the right criteria, or is one missing?
5. Does matching flood-wave travel with a bankfull channel celerity agree with
   the literature on overbank storage slowing flood waves?
6. Is the steady backwater reach a sufficient upstream-boundary criterion for
   compound events, given that it matched the dynamic run here?
7. What does the depth sensitivity imply for reporting: should flood metrics
   carry a depth-uncertainty band, and is x 0.7 / x 1.3 a defensible range?
