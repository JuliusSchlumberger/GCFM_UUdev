"""Phase 1.6: spin-up convergence on the river channel (TODO B33).

From <results_dir>/<basin>/spin_up/sfincs_map.nc: per channel cell (finite in
rule 11's burned river DEM on the SFINCS grid) the rate dzs/dt over the last
6 h of the spin-up, (zs_end - zs_end-6h) / 6 h; reported as max |rate| and
median rate. And the median of (spin-up end level - mean-discharge level) over
channel cells, the mean-discharge level being the skeleton's zsini.tif on
those cells (rule 12a sets channel cells to rule 11's mean_level_m there).

Writes phase1_spinup_convergence_summary.csv to <results_dir>/figures/analysis/.

Usage: python analysis/phase1_spinup_convergence.py [basin_id ...]
"""

import logging

import numpy as np
from _common import analysis_path, script_parser
from _phase1 import PHASE1_BASINS, basin_paths, map_cells, pct, raster_at, write_summary

WINDOW_HR = 6.0


def spinup_basin(basin_id: str) -> dict:
    from src.grid import load_grid_def

    p = basin_paths(basin_id)
    map_path = p["spin_up"] / "sfincs_map.nc"
    if not map_path.exists():
        raise FileNotFoundError(f"{basin_id}: no spin-up map at {map_path}")
    crs = load_grid_def(p["sfincs_grid"])["crs"]
    x, y, zs, t = map_cells(map_path)
    channel = np.isfinite(raster_at(p["burned_dem_grid"], x, y, crs))
    k0 = int(np.argmin(np.abs(t - (t[-1] - WINDOW_HR * 3600.0))))
    dt_hr = (t[-1] - t[k0]) / 3600.0
    end = zs[-1]
    wet = channel & np.isfinite(end) & np.isfinite(zs[k0])
    rate = (end - zs[k0]) / dt_hr
    zsini = raster_at(p["skeleton"] / "zsini.tif", x, y, crs)
    both = wet & np.isfinite(zsini)
    return {
        "basin_id": basin_id,
        "spinup_hr": float((t[-1] - t[0]) / 3600.0),
        "window_hr": float(dt_hr),
        "n_channel_cells": int(wet.sum()),
        "max_abs_rate_m_per_hr": float(np.nanmax(np.abs(rate[wet])))
        if wet.any()
        else np.nan,
        "median_rate_m_per_hr": pct(rate[wet], 50),
        "p95_abs_rate_m_per_hr": pct(np.abs(rate[wet]), 95),
        "n_cells_with_zsini": int(both.sum()),
        "median_end_minus_mean_level_m": pct((end - zsini)[both], 50),
        "p5_end_minus_mean_level_m": pct((end - zsini)[both], 5),
        "p95_end_minus_mean_level_m": pct((end - zsini)[both], 95),
    }


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rows = [spinup_basin(str(b)) for b in args.basin_ids]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
