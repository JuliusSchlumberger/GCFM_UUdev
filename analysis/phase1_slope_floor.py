"""Phase 1.2: slope-floor census and the depth change for other slope floors.

Share of nodes with on_floor in the stored node table (rule 11), then the bed
re-estimated (rule 11 round 0, design_discharge as configured) at slope_floor
5e-6, 1e-5 and 2e-5: floor share and the median depth change relative to 1e-5
over flowing nodes (q > q_min_m3s), and over the flowing nodes on the
floor at either value (the only ones whose depth changes).

Writes phase1_slope_floor_summary.csv to <results_dir>/figures/analysis/.

Usage: python analysis/phase1_slope_floor.py [basin_id ...]
"""

import logging

import pandas as pd
from _common import analysis_path, script_parser
from _phase1 import PHASE1_BASINS, basin_paths, load_config, pct, round0, write_summary

FLOORS = (5e-6, 1e-5, 2e-5)
REFERENCE_FLOOR = 1e-5


def floor_basin(basin_id: str, cfg: dict) -> list[dict]:
    stored = pd.read_csv(basin_paths(basin_id)["node_table"])
    q_min = float(cfg["river_processing"]["river_depth_modelling"]["bed"]["q_min_m3s"])
    beds = {f: round0(basin_id, cfg, slope_floor=f)[0] for f in FLOORS}
    ref = beds[REFERENCE_FLOOR]
    flowing = ref.q_m3s > q_min
    rows = []
    for f, bed in beds.items():
        d = bed.depth_m - ref.depth_m
        affected = flowing & (bed.on_floor | ref.on_floor)
        rows.append(
            {
                "basin_id": basin_id,
                "slope_floor": f,
                "n_nodes": int(bed.net.n_nodes),
                "stored_on_floor_share": float(stored["on_floor"].astype(bool).mean()),
                "on_floor_share": float(bed.on_floor.mean()),
                "on_floor_share_flowing": float(bed.on_floor[flowing].mean()),
                "median_depth_m": pct(bed.depth_m[flowing], 50),
                "median_depth_change_m": pct(d[flowing], 50),
                "median_depth_change_pct": pct(
                    100.0 * d[flowing] / ref.depth_m[flowing], 50
                ),
                # nodes on the floor at either slope floor (the only ones that change)
                "n_affected": int(affected.sum()),
                "median_depth_change_affected_m": pct(d[affected], 50),
                "median_depth_change_affected_pct": pct(
                    100.0 * d[affected] / ref.depth_m[affected], 50
                ),
            }
        )
    return rows


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config()
    rows = [r for b in args.basin_ids for r in floor_basin(str(b), cfg)]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
