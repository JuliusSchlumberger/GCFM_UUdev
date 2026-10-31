"""Phase 1.4: nodes held at the Froude-cap depth in the stored bed solve.

Re-runs rule 11's round 0 (src.river_bed.estimate_river_bed_from_files), then
the forward solve at the design discharge with
src.gvf_solver.forward_network(..., return_capped=True), which is the solve
behind forward_level_m. Counts flagged nodes per basin and per reach.

Writes per basin 90_phase1_froude_flags_reaches.csv to
<results_dir>/<basin>/figures/analysis/ and phase1_froude_flags_summary.csv to
<results_dir>/figures/analysis/.

Usage: python analysis/phase1_froude_flags.py [basin_id ...]
"""

import logging

import numpy as np
import pandas as pd
from _common import analysis_path, script_parser
from _phase1 import (
    PHASE1_BASINS,
    load_config,
    network_end_levels,
    reach_q_dict,
    round0,
    write_summary,
)


def flags_basin(basin_id: str, cfg: dict) -> dict:
    from src.gvf_solver import forward_network

    bed, rivers, q_design, _ = round0(basin_id, cfg)
    net = bed.net
    q = reach_q_dict(rivers, q_design)
    level, capped = forward_network(
        net,
        bed.bed_m,
        bed.width_m,
        {r: q.get(r, 0.0) for r in net.reach_ids},
        float(cfg["river_processing"]["channel_manning_n"]),
        network_end_levels(bed),
        return_capped=True,
    )
    owner = np.asarray(net.owner)
    per_reach = (
        pd.DataFrame({"reach_id": owner, "capped": capped})
        .groupby("reach_id")["capped"]
        .agg(["sum", "size"])
        .rename(columns={"sum": "n_capped", "size": "n_nodes"})
        .reset_index()
    )
    per_reach = per_reach[per_reach["n_capped"] > 0].sort_values(
        "n_capped", ascending=False
    )
    per_reach.to_csv(analysis_path(basin_id, __file__, "reaches.csv"), index=False)
    return {
        "basin_id": basin_id,
        "n_nodes": int(net.n_nodes),
        "n_capped": int(capped.sum()),
        "share_capped": float(capped.mean()),
        "max_abs_level_diff_vs_stored_solve_m": float(
            np.nanmax(np.abs(level - bed.forward_level_m))
        ),
        "capped_reaches": " ".join(
            f"{r}({n})" for r, n in zip(per_reach["reach_id"], per_reach["n_capped"])
        ),
    }


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    rows = [flags_basin(str(b), cfg) for b in args.basin_ids]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
