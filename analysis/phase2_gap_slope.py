"""Phase 2.0: 1D-2D slope excess per reach against path ratio and width, and the gap along the network.

Per round-1 centreline cell, gap = SFINCS end level - steady 1D level, as in
phase1_gvf_gap.py. Per reach:

  slope excess (m/km): least-squares slope of the gap against the cell's
  distance to the sea (nearest node of its own reach); positive when the gap grows
  upstream, i.e. SFINCS's surface is steeper than the 1D surface;
  path ratio: Manhattan length of the consecutive centreline cells
  (sum of |d row| + |d col|, times dx) over the reach's centreline length; 1
  for a grid-aligned reach, up to sqrt(2) for a diagonal one, more for a
  meander the cells follow in steps;
  width in cells: SWORD width / dx.

A staircase effect of the grid shows as slope excess rising with path ratio
and a gap that grows upstream; a roughness or geometry effect does not depend
on the path ratio. Reaches with fewer than MIN_CELLS cells or no calibration
discharge are left out of the correlations.

Writes per basin 90_phase2_gap_slope_reaches.csv, the same-name _scatter.png and
_along.png to <results_dir>/<basin>/figures/analysis/, and
phase2_gap_slope_summary.csv to <results_dir>/figures/analysis/.

Usage: python analysis/phase2_gap_slope.py [basin_id ...]
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import analysis_path, script_parser
from _phase1 import (
    PHASE1_BASINS,
    basin_paths,
    git_hash,
    load_config,
    pct,
    write_summary,
)
from phase1_gvf_gap import cell_gaps
from scipy.spatial import cKDTree

MIN_CELLS = 10


def reach_table(cells: pd.DataFrame, rivers, grid: dict, q_calib: dict) -> pd.DataFrame:
    from src.river_network import normalize_reach_id

    dx = float(np.hypot(grid["transform"].a, grid["transform"].d))
    rv = rivers.to_crs(grid["crs"])
    length = dict(zip(rv["reach_id"].map(normalize_reach_id), rv.geometry.length))
    width = dict(
        zip(
            rv["reach_id"].map(normalize_reach_id),
            pd.to_numeric(rv["width"], errors="coerce"),
        )
    )
    rows = []
    for rid, c in cells.groupby("reach_id", sort=True):
        c = c.sort_values("along_m")
        steps = np.abs(np.diff(c["row"].to_numpy())) + np.abs(
            np.diff(c["col"].to_numpy())
        )
        ok = np.isfinite(c["gap_m"]) & np.isfinite(c["dist_to_sea_km"])
        slope = (
            float(np.polyfit(c.loc[ok, "dist_to_sea_km"], c.loc[ok, "gap_m"], 1)[0])
            if ok.sum() >= 3 and np.ptp(c.loc[ok, "dist_to_sea_km"]) > 0
            else np.nan
        )
        rows.append(
            {
                "reach_id": rid,
                "n_cells": len(c),
                "gap_median_m": pct(c["gap_m"], 50),
                "slope_excess_m_per_km": slope,
                "path_ratio": float(steps.sum() * dx / length[rid])
                if length.get(rid, 0) > 0
                else np.nan,
                "width_cells": float(width.get(rid, np.nan)) / dx,
                "dist_to_sea_km": float(c["dist_to_sea_km"].median()),
                "q_calib_m3s": q_calib.get(rid, 0.0),
            }
        )
    return pd.DataFrame(rows)


def plot_scatter(
    t: pd.DataFrame, basin_id: str, rho: pd.DataFrame, tag: str, path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, xlabel in (
        (axes[0], "path_ratio", "path ratio (Manhattan cell path / centreline length)"),
        (axes[1], "width_cells", "width (cells)"),
    ):
        sc = ax.scatter(
            t[col],
            t["slope_excess_m_per_km"],
            c=t["gap_median_m"],
            cmap="viridis",
            s=18,
        )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("slope excess, SFINCS - 1D (m/km)")
        ax.set_title(
            f"Spearman {rho.loc['slope_excess_m_per_km', col]:+.2f} (n={len(t)})",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label="reach gap median (m)")
    axes[1].set_xscale("log")
    fig.suptitle(f"phase2 2.0 slope excess per reach | {basin_id} | {tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_along(cells: pd.DataFrame, basin_id: str, tag: str, path) -> None:
    # One line per reach: a binned median across reaches mixes parallel branches.
    lim = max(1.0, float(np.nanpercentile(np.abs(cells["gap_m"]), 99)))
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    fig = plt.figure(figsize=(16, 9))
    ax_map = fig.add_subplot(1, 3, 1)
    ax_gap = fig.add_subplot(2, 3, (2, 3))
    ax_lvl = fig.add_subplot(2, 3, (5, 6), sharex=ax_gap)
    sc = ax_map.scatter(
        cells["x"], cells["y"], c=cells["gap_m"], s=1.5, cmap="RdBu_r", norm=norm
    )
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_title("gap per centreline cell", fontsize=9)
    fig.colorbar(sc, ax=ax_map, shrink=0.6, label="SFINCS - 1D level (m)")
    cmap = plt.get_cmap("RdBu_r")
    for _rid, c in cells.groupby("reach_id", sort=True):
        c = c.sort_values("dist_to_sea_km")
        color = cmap(norm(float(np.nanmedian(c["gap_m"]))))
        lw = 1.4 if c["main_path"].any() else 0.6
        ax_gap.plot(c["dist_to_sea_km"], c["gap_m"], color=color, lw=lw)
        ax_lvl.plot(c["dist_to_sea_km"], c["zs_end_m"], color="tab:blue", lw=lw)
        ax_lvl.plot(c["dist_to_sea_km"], c["level_1d_m"], color="tab:orange", lw=lw)
        ax_lvl.plot(c["dist_to_sea_km"], c["dem_m"], color="0.6", lw=0.5)
    ax_gap.axhline(0, color="k", lw=0.8)
    ax_gap.set_ylabel("SFINCS - 1D level (m)")
    ax_gap.set_title(
        "one line per reach, coloured by its median gap (thick: a crossing's main path)",
        fontsize=9,
    )
    ax_lvl.plot([], [], color="tab:blue", label="SFINCS round-1 end level")
    ax_lvl.plot([], [], color="tab:orange", label="1D GVF level")
    ax_lvl.plot([], [], color="0.6", label="DEM at the centreline cell")
    ax_lvl.set_xlabel("distance to the sea along the reach's own nodes (km)")
    ax_lvl.set_ylabel("level (m)")
    ax_lvl.legend(fontsize=8)
    for ax in (ax_gap, ax_lvl):
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"phase2 2.0 1D-2D gap along the network | {basin_id} | {tag}", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def own_reach_nodes(cells: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Return distance to the sea (km) and main_path per cell from the nearest node of its own reach."""
    from src.river_network import normalize_reach_id

    nodes = nodes.assign(reach_id=nodes["reach_id"].map(normalize_reach_id))
    out = pd.DataFrame(index=cells.index, columns=["dist_to_sea_km", "main_path"])
    fallback = cKDTree(nodes[["x", "y"]].to_numpy())
    for rid, idx in cells.groupby("reach_id").groups.items():
        n = nodes[nodes["reach_id"] == rid]
        tree = cKDTree(n[["x", "y"]].to_numpy()) if len(n) else fallback
        n = n if len(n) else nodes
        _, k = tree.query(cells.loc[idx, ["x", "y"]].to_numpy())
        out.loc[idx, "dist_to_sea_km"] = n["dist_to_sea_m"].to_numpy()[k] / 1000.0
        out.loc[idx, "main_path"] = n["main_path"].astype(bool).to_numpy()[k]
    return out.astype({"dist_to_sea_km": float, "main_path": bool})


def slope_basin(basin_id: str, cfg: dict, tag: str) -> dict:
    cells, grid, _bed, rivers, q_calib, _sea = cell_gaps(basin_id, cfg)
    nodes = pd.read_csv(basin_paths(basin_id)["node_table"])
    cells[["dist_to_sea_km", "main_path"]] = own_reach_nodes(cells, nodes)

    t = reach_table(cells, rivers, grid, q_calib)
    t.to_csv(
        analysis_path(basin_id, __file__, "reaches.csv"),
        index=False,
        float_format="%.4g",
    )
    used = t[(t["n_cells"] >= MIN_CELLS) & (t["q_calib_m3s"] > 0)].dropna(
        subset=["slope_excess_m_per_km", "path_ratio", "width_cells"]
    )
    rho = used[
        ["slope_excess_m_per_km", "path_ratio", "width_cells", "gap_median_m"]
    ].corr(method="spearman")
    plot_scatter(
        used, basin_id, rho, tag, analysis_path(basin_id, __file__, "scatter.png")
    )
    plot_along(cells, basin_id, tag, analysis_path(basin_id, __file__, "along.png"))

    main = cells[cells["main_path"]]
    near = main["dist_to_sea_km"] <= main["dist_to_sea_km"].quantile(0.2)
    far = main["dist_to_sea_km"] >= main["dist_to_sea_km"].quantile(0.8)
    return {
        "basin_id": basin_id,
        "n_reaches_used": int(len(used)),
        "slope_excess_median_m_per_km": pct(used["slope_excess_m_per_km"], 50),
        "path_ratio_median": pct(used["path_ratio"], 50),
        "spearman_slope_excess_path_ratio": float(
            rho.loc["slope_excess_m_per_km", "path_ratio"]
        ),
        "spearman_slope_excess_width": float(
            rho.loc["slope_excess_m_per_km", "width_cells"]
        ),
        "spearman_gap_path_ratio": float(rho.loc["gap_median_m", "path_ratio"]),
        "main_gap_lowest20pct_m": pct(main.loc[near, "gap_m"], 50),
        "main_gap_highest20pct_m": pct(main.loc[far, "gap_m"], 50),
    }


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg, tag = load_config(), git_hash()
    rows = [slope_basin(str(b), cfg, tag) for b in args.basin_ids]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(
            write_summary(
                rows, analysis_path(None, __file__, "summary.csv")
            ).T.to_string()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
