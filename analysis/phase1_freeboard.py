"""Phase 1.1: bankfull freeboard = bank top minus the GVF level at Q_bankfull, for two bed options.

Bank top per node: 90th percentile of rule 06's merged DEM (FathomDEM on
GOCO06s, before rule-10 conditioning) in a band from W/2 to W/2 + 200 m on
each side of the centreline, perpendicular to the reach and +-250 m along it;
the lower of the two banks. water_body nodes and nodes without bankfull flow
are skipped. Options: (a) the stored bed (design_discharge: bankfull), level =
forward_level_m; (b) a mean-Q bed (design_discharge: mean), level =
forward_network at Q_bankfull on that bed.

Writes per basin 90_phase1_freeboard_nodes.csv and _profile_{a,b}.png to
<results_dir>/<basin>/figures/analysis/, and phase1_freeboard_summary.csv to
<results_dir>/figures/analysis/.

Usage: python analysis/phase1_freeboard.py [basin_id ...]
"""

import logging

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
    git_hash,
    load_config,
    network_end_levels,
    node_class,
    pct,
    reach_q_dict,
    round0,
    write_summary,
)
from pyproj import Transformer

log = logging.getLogger(__name__)

BAND_M = 200.0
ACROSS_STEP_M = 25.0
ALONG_HALF_M = 250.0
ALONG_STEP_M = 125.0
BANK_PERCENTILE = 90.0


def node_tangents(net) -> tuple[np.ndarray, np.ndarray]:
    """Return the unit downstream tangent per node, from its neighbours on the same reach."""
    tx, ty = np.zeros(net.n_nodes), np.zeros(net.n_nodes)
    for r in net.reach_ids:
        ids = np.asarray(net.reach_nodes[r])
        if len(ids) < 2:
            continue
        gx, gy = np.gradient(net.node_x[ids]), np.gradient(net.node_y[ids])
        tx[ids], ty[ids] = gx, gy
    norm = np.hypot(tx, ty)
    norm[norm == 0] = 1.0
    return tx / norm, ty / norm


def bank_top(
    dem_path, crs, x, y, tx, ty, width_m
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (left, right, lower) bank-top elevation per point; NaN where the band has no DEM."""
    across = np.arange(0.0, BAND_M + 1e-9, ACROSS_STEP_M)
    along = np.arange(-ALONG_HALF_M, ALONG_HALF_M + 1e-9, ALONG_STEP_M)
    a, c = np.meshgrid(along, across, indexing="ij")
    a, c = a.ravel(), c.ravel()
    nx, ny = -ty, tx  # left normal of the downstream tangent
    out = []
    with rasterio.open(dem_path) as src:
        to_dem = Transformer.from_crs(crs, src.crs, always_xy=True)
        nodata = src.nodata
        for side in (1.0, -1.0):
            off = width_m[:, None] / 2.0 + c[None, :]
            px = x[:, None] + a[None, :] * tx[:, None] + side * off * nx[:, None]
            py = y[:, None] + a[None, :] * ty[:, None] + side * off * ny[:, None]
            dx, dy = to_dem.transform(px.ravel(), py.ravel())
            v = np.array([s[0] for s in src.sample(zip(dx, dy))], dtype=float)
            if nodata is not None:
                v[v == nodata] = np.nan
            v = v.reshape(px.shape)
            with np.errstate(all="ignore"):
                out.append(
                    np.nanpercentile(
                        np.where(np.isfinite(v), v, np.nan), BANK_PERCENTILE, axis=1
                    )
                )
    left, right = out
    return left, right, np.fmin(left, right)


def stats(freeboard: np.ndarray) -> dict[str, float]:
    f = freeboard[np.isfinite(freeboard)]
    return {
        "n_nodes": int(f.size),
        "median_m": pct(f, 50),
        "p5_m": pct(f, 5),
        "p95_m": pct(f, 95),
        "share_below_0": float((f < 0).mean()) if f.size else np.nan,
        "share_above_2m": float((f > 2).mean()) if f.size else np.nan,
    }


def profile_figure(
    t: pd.DataFrame, level_col: str, bed_col: str, title: str, path
) -> None:
    m = t[t["main_path"]].sort_values("dist_to_sea_m")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    km = m["dist_to_sea_m"] / 1000.0
    ax.plot(
        km,
        m["bank_top_m"],
        ".",
        ms=3,
        color="tab:brown",
        label="bank top (P90, lower bank)",
    )
    # One line per reach: the largest-Q paths of several seeds overlap in
    # distance to sea, so a single line would zigzag between them.
    for k, (_, r) in enumerate(m.groupby("reach_id")):
        rk = r["dist_to_sea_m"] / 1000.0
        first = k == 0
        ax.plot(
            rk,
            r["level_m"],
            "-",
            lw=1.0,
            color="0.5",
            label="rule-10 surface" if first else None,
        )
        ax.plot(
            rk,
            r[level_col],
            "-",
            lw=1.4,
            color="tab:blue",
            label="GVF level at Q_bankfull" if first else None,
        )
        ax.plot(rk, r[bed_col], "-", lw=1.2, color="k", label="bed" if first else None)
    ax.set_xlabel("distance to sea along the network (km)")
    ax.set_ylabel("m (GOCO06s)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def freeboard_basin(basin_id: str, cfg: dict, tag: str) -> list[dict]:
    from src.gvf_solver import forward_network

    p = basin_paths(basin_id)
    n = float(cfg["river_processing"]["channel_manning_n"])
    bed_bf, rivers, q_bf, _ = round0(basin_id, cfg, design_discharge="bankfull")
    bed_mean, _, _, _ = round0(basin_id, cfg, design_discharge="mean")
    net = bed_bf.net
    q_bf_reach = reach_q_dict(rivers, q_bf)
    level_b = forward_network(
        net,
        bed_mean.bed_m,
        bed_mean.width_m,
        {r: q_bf_reach.get(r, 0.0) for r in net.reach_ids},
        n,
        network_end_levels(bed_mean),
    )
    stored = pd.read_csv(p["node_table"])
    if len(stored) == net.n_nodes:
        log.info(
            f"{basin_id}: re-estimated vs stored forward_level_m, max |diff| = "
            f"{np.nanmax(np.abs(stored['forward_level_m'].to_numpy() - bed_bf.forward_level_m)):.4f} m"
        )
    tx, ty = node_tangents(net)
    left, right, bank = bank_top(
        p["elevation_merged"],
        net_crs(basin_id),
        net.node_x,
        net.node_y,
        tx,
        ty,
        bed_bf.width_m,
    )
    cls = node_class(bed_bf, basin_id)
    keep = (cls != "water_body") & (
        bed_bf.q_m3s
        > float(cfg["river_processing"]["river_depth_modelling"]["bed"]["q_min_m3s"])
    )
    t = pd.DataFrame(
        {
            "node": np.arange(net.n_nodes),
            "reach_id": np.asarray(net.owner),
            "x": net.node_x,
            "y": net.node_y,
            "dist_to_sea_m": bed_bf.dist_to_sea_m,
            "main_path": bed_bf.main_path,
            "node_class": cls,
            "width_m": bed_bf.width_m,
            "q_bankfull_m3s": bed_bf.q_m3s,
            "bank_left_m": left,
            "bank_right_m": right,
            "bank_top_m": bank,
            "level_m": bed_bf.level_m,
            "bed_a_m": bed_bf.bed_m,
            "level_a_m": bed_bf.forward_level_m,
            "bed_b_m": bed_mean.bed_m,
            "level_b_m": level_b,
            "kept": keep,
        }
    )
    t["freeboard_a_m"] = np.where(keep, t["bank_top_m"] - t["level_a_m"], np.nan)
    t["freeboard_b_m"] = np.where(keep, t["bank_top_m"] - t["level_b_m"], np.nan)
    t.to_csv(
        analysis_path(basin_id, __file__, "nodes.csv"), index=False, float_format="%.3f"
    )
    rows = []
    for opt, label in (
        ("a", "stored bed (design: bankfull)"),
        ("b", "mean-Q bed (design: mean)"),
    ):
        rows.append(
            {
                "basin_id": basin_id,
                "option": opt,
                "bed": label,
                **stats(t[f"freeboard_{opt}_m"].to_numpy()),
            }
        )
        profile_figure(
            t,
            f"level_{opt}_m",
            f"bed_{opt}_m",
            f"phase1 1.1 freeboard | {basin_id} | option {opt}: {label} | main path | {tag}",
            analysis_path(basin_id, __file__, f"profile_{opt}.png"),
        )
    return rows


def net_crs(basin_id: str):
    from src.grid import load_grid_def

    return load_grid_def(basin_paths(basin_id)["sfincs_grid"])["crs"]


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg, tag = load_config(), git_hash()
    rows = [r for b in args.basin_ids for r in freeboard_basin(str(b), cfg, tag)]
    df = write_summary(rows, analysis_path(None, __file__, "summary.csv"))
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
