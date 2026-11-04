# Phase 5: bifurcation split, Chao Phraya (3279946)

Question: does SFINCS split the discharge at the bifurcations the way the width rule routes it (`river_network.accumulate_discharge`)? Outcome: three of the four bifurcations agree once advection is off. The fourth is a river-network problem, so rule 11 now checks every split and stops on a mismatch.

Scripts:

- `phase1_bifurcation_split.py run|compare 3279946`: round 1 rerun with cross-sections, written to `<scratch>/p15_adv0`.
- Figure: `<results_dir>/3279946/figures/analysis/90_phase5_bifurcation_44306300275.png`.

## SFINCS share against width share (steady round 1)

| bifurcation | arm (width) | advection 1: SFINCS / width | advection 0: SFINCS / width | \|error\|, advection 0 |
|---|---|---|---|---|
| 44306300031 | 025 (54 m) | 0.31 / 0.16 | 0.15 / 0.16 | 0.01 |
| 44306300125 | 565 (63 m) | 0.52 / 0.20 | 0.19 / 0.20 | 0.01 |
| 44306300265 | 425 (78 m) | 0.30 / 0.42 | 0.45 / 0.42 | 0.02 |
| 44306300275 | 200025 (84 m) | 0.62 / 0.37 | 0.55 / 0.37 | 0.18 |

The two earlier reversals at 031 and 125 were the advection backwater (Phase 2).

## Bifurcation 44306300275

- **Geometry:** the two arms form a loop that rejoins at reach 44306100055, 60.4 km down the narrow arm and 56.0 km down the wide one.
- **Bed:** SFINCS's subgrid bed follows the 1D bed on both arms. Over the first 35 km the wide arm (145 m) is about 10 m deeper, around −13 m against −3.5 m.
- **1D expectation:** Manning conveyance at the shared level would send about 0.8 of the flow down the wide arm.
- **SFINCS:** it sends 0.45 down the wide arm. Its level there stands 0.2–0.4 m above the narrow arm over km 10–35.
- **Cause:** the SWORD arms are misaligned at the junction. The flow runs straight on into the narrow arm instead of turning into the wide one.

## Why the planned hydraulic split was dropped

The planned split tuned the shares until the 1D levels matched at each junction. It cannot set a split here:

- round 0 hangs each arm's bed below the observed surface, at the normal depth for whatever share the arm gets;
- the forward GVF solve on that bed therefore reproduces the surface on every arm, whatever the share;
- so the 1D model accepts any split.

## Decisions

- Routing keeps the width rule. `accumulate_discharge` takes `share_override` (`river_processing.split_override`, empty by default), which rules 08 and 11 and the river bed use.
- Rule 11's round 1 always sets bifurcation cross-sections. It writes `round1/bifurcation_split.csv` and stops when an arm's share differs by more than `bifurcation_check.max_share_error` (0.05) from the routed share. The fix is either to correct the network or to set `split_override`.
- 44306300275 is listed in `bifurcation_check.accepted` until the river network is revised there (TODO B37).
- Results are unchanged: rule 11's round-0 node table is byte-identical on 2433835 and 3279946.
