"""Phase 1.7: 1D-2D gap per reach against channel angle, width and diagonal-only connectivity.

Gap = round-1 end-of-run SFINCS level minus the steady 1D level, median over
the reach's centreline cells (round1/calibration_state.csv). The 1D level is
src.gvf_solver.forward_network on rule 11's round-0 bed at the calibration
discharge (q_calib of the depth-estimated network), with the calibration sea
level (highest station storm tide at the FLOPROS coastal RP) at the mouths and
rule 10's surface at the other network ends. The network solve stands in for
river_profile.forward_levels, which covers only each crossing's main stem;
both integrate the same reach equations. Angle: reach chord direction against
the grid axes, folded to 0-45 deg. Width in cells: SWORD width / grid dx.
Connectivity: channel cells (finite in rule 11's burned DEM on the grid)
whose only channel neighbours are diagonal, assigned to the nearest
centreline cell's reach.

Writes per basin 90_phase1_gvf_gap_reaches.csv and _scatter.png to
<results_dir>/<basin>/figures/analysis/, and phase1_gvf_gap_summary.csv to
<results_dir>/figures/analysis/.

Usage: python analysis/phase1_gvf_gap.py [basin_id ...]
"""

import logging

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from _common import analysis_path, script_parser
from _phase1 import (
    PHASE1_BASINS,
    basin_paths,
    calibration_sea_level_m,
    git_hash,
    load_config,
    map_cells,
    pct,
    reach_q_dict,
    round0,
    write_summary,
)
from scipy.spatial import cKDTree

WIDTH_CLASSES = [
    (0, 3, "< 3 cells"),
    (3, 10, "3-10 cells"),
    (10, np.inf, ">= 10 cells"),
]


def diagonal_only_cells(channel: np.ndarray) -> np.ndarray:
    """Return channel cells with no orthogonal but at least one diagonal channel neighbour."""
    c = np.pad(channel, 1)
    orth = c[:-2, 1:-1] | c[2:, 1:-1] | c[1:-1, :-2] | c[1:-1, 2:]
    diag = c[:-2, :-2] | c[:-2, 2:] | c[2:, :-2] | c[2:, 2:]
    return channel & ~orth & diag


def gap_basin(basin_id: str, cfg: dict, tag: str) -> dict:
    from src.grid import load_grid_def
    from src.gvf_solver import forward_network
    from src.river_bed import values_at_points
    from src.river_network import normalize_reach_id

    p = basin_paths(basin_id)
    grid = load_grid_def(p["sfincs_grid"])
    dx = float(np.hypot(grid["transform"].a, grid["transform"].d))
    rotation = float(grid.get("rotation", 0.0))
    bed, rivers, _, _ = round0(basin_id, cfg)
    net = bed.net
    depth_net = gpd.read_file(p["rivers_depth"])
    q_calib = reach_q_dict(depth_net, depth_net["q_calib"].to_numpy(float))
    sea = calibration_sea_level_m(basin_id)
    mouths = set(int(m) for m in np.asarray(bed.mouths))
    ends = {
        int(net.reach_nodes[r][-1]): (
            sea
            if int(net.reach_nodes[r][-1]) in mouths
            else float(bed.level_m[net.reach_nodes[r][-1]])
        )
        for r in net.reach_ids
        if not net.downstream[r]
    }
    level_1d = forward_network(
        net,
        bed.bed_m,
        bed.width_m,
        {r: q_calib.get(r, 0.0) for r in net.reach_ids},
        float(cfg["river_processing"]["channel_manning_n"]),
        ends,
    )

    cells = pd.read_csv(p["round1"] / "calibration_state.csv")
    cells["reach_id"] = cells["reach_id"].map(normalize_reach_id)
    mx, my, zs, _t = map_cells(p["round1"] / "sfincs_map.nc")
    _, k = cKDTree(np.column_stack([mx, my])).query(cells[["x", "y"]].to_numpy())
    cells["zs_end_m"] = zs[-1][k]
    cells["level_1d_m"] = values_at_points(
        net, level_1d, cells["reach_id"], cells["x"].to_numpy(), cells["y"].to_numpy()
    )
    cells["gap_m"] = cells["zs_end_m"] - cells["level_1d_m"]

    with rasterio.open(p["burned_dem_grid"]) as src:
        burned = src.read(1).astype(float)
        nodata, transform = src.nodata, src.transform
    channel = np.isfinite(burned) & (burned != nodata if nodata is not None else True)
    rr, cc = np.nonzero(diagonal_only_cells(channel))
    dx_c, dy_c = rasterio.transform.xy(transform, rr, cc)
    _, kk = (
        cKDTree(cells[["x", "y"]].to_numpy()).query(np.column_stack([dx_c, dy_c]))
        if len(rr)
        else (None, np.array([], int))
    )
    diag_by_reach = (
        pd.Series(cells["reach_id"].to_numpy()[kk]).value_counts()
        if len(rr)
        else pd.Series(dtype=int)
    )

    rv = rivers.to_crs(grid["crs"])
    rows = []
    for rid, geom, width in zip(
        rv["reach_id"].map(normalize_reach_id),
        rv.geometry,
        pd.to_numeric(rv["width"], errors="coerce"),
    ):
        c = cells[cells["reach_id"] == rid]
        if c.empty or geom is None:
            continue
        coords = [c for part in getattr(geom, "geoms", [geom]) for c in part.coords]
        (x0, y0), (x1, y1) = coords[0], coords[-1]
        a = (np.degrees(np.arctan2(y1 - y0, x1 - x0)) - rotation) % 90.0
        rows.append(
            {
                "reach_id": rid,
                "n_cells": len(c),
                "gap_median_m": pct(c["gap_m"], 50),
                "angle_deg": float(min(a, 90.0 - a)),
                "width_m": float(width),
                "width_cells": float(width) / dx,
                "n_diagonal_only_cells": int(diag_by_reach.get(rid, 0)),
                "q_calib_m3s": q_calib.get(rid, 0.0),
            }
        )
    t = pd.DataFrame(rows)
    t.to_csv(
        analysis_path(basin_id, __file__, "reaches.csv"),
        index=False,
        float_format="%.4g",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for lo, hi, label in WIDTH_CLASSES:
        s = t[(t["width_cells"] >= lo) & (t["width_cells"] < hi)]
        ax.scatter(
            s["angle_deg"], s["gap_median_m"], s=14, label=f"{label} (n={len(s)})"
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("reach angle to the grid (deg, 0-45)")
    ax.set_ylabel("SFINCS round-1 end level - 1D GVF level (m)")
    ax.set_title(
        f"phase1 1.7 1D-2D gap | {basin_id} | calibration Q | dx {dx:.0f} m | {tag}",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analysis_path(basin_id, __file__, "scatter.png"), dpi=130)
    plt.close(fig)

    flowing = t["q_calib_m3s"] > 0
    rho = t.loc[flowing, ["gap_median_m", "angle_deg", "width_cells"]].corr(
        method="spearman"
    )
    return {
        "basin_id": basin_id,
        "grid_dx_m": dx,
        "calibration_sea_level_m": sea,
        "n_reaches": int(len(t)),
        "gap_median_m": pct(t.loc[flowing, "gap_median_m"], 50),
        "gap_p5_m": pct(t.loc[flowing, "gap_median_m"], 5),
        "gap_p95_m": pct(t.loc[flowing, "gap_median_m"], 95),
        "spearman_gap_angle": float(rho.loc["gap_median_m", "angle_deg"]),
        "spearman_gap_width": float(rho.loc["gap_median_m", "width_cells"]),
        "n_channel_cells": int(channel.sum()),
        "n_diagonal_only_cells": int(len(rr)),
    }


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg, tag = load_config(), git_hash()
    rows = [gap_basin(str(b), cfg, tag) for b in args.basin_ids]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
