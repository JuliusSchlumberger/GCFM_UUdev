"""Phase 1.3: forward-check gate on the stored rule-11 node table.

Residual = forward_level_m - level_m on flowing (q_m3s > q_min_m3s),
non-bathymetry nodes (the set src.river_bed.forward_deviation_summary uses).
Pass when the median residual is within +-0.25 m and the 95th percentile of
|residual| within 0.75 m.

Writes <results_dir>/<basin>/preprocessing_inputs/diagnostics/<basin>_forward_check.csv
(one row) and phase1_forward_check_summary.csv to <results_dir>/figures/analysis/.

Usage: python analysis/phase1_forward_check.py [basin_id ...]
"""

import logging

import numpy as np
import pandas as pd
from _common import analysis_path, script_parser
from _phase1 import PHASE1_BASINS, basin_paths, load_config, pct, write_summary

MEDIAN_TOL_M = 0.25
P95_TOL_M = 0.75


def check_basin(basin_id: str, cfg: dict) -> dict:
    p = basin_paths(basin_id)
    t = pd.read_csv(p["node_table"])
    q_min = float(cfg["river_processing"]["river_depth_modelling"]["bed"]["q_min_m3s"])
    use = (t["q_m3s"] > q_min) & ~t["bathymetry"].astype(bool)
    res = (t["forward_level_m"] - t["level_m"])[use].to_numpy(float)
    med, p95 = pct(res, 50), pct(np.abs(res), 95)
    row = {
        "basin_id": basin_id,
        "n_nodes": int(np.isfinite(res).sum()),
        "median_residual_m": med,
        "p95_abs_residual_m": p95,
        "median_tol_m": MEDIAN_TOL_M,
        "p95_tol_m": P95_TOL_M,
        "pass": bool(abs(med) <= MEDIAN_TOL_M and p95 <= P95_TOL_M),
    }
    out = p["diagnostics"] / f"{basin_id}_forward_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False, float_format="%.4f")
    return row


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    rows = [check_basin(str(b), cfg) for b in args.basin_ids]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
