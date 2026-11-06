"""Channel capacity Q_cap per node for the round-0 bed options.

Q_cap is the discharge at which the steady GVF level on the bed reaches the
bank top (bank_freeboard.bank_top: P90 of rule 06's merged DEM beside the
channel, lower bank); nodes whose bank top is not above the calm sea level
are left out. The network is solved at k x the routed Q_bankfull for
k on a log grid (SCALES), the network ends held at their levels in the bed
solve; per node Q_cap = k* x Q_bankfull, with k* where the level crosses the
bank top, interpolated in log k. Below the smallest k the node is reported
at that k, above the largest at that k.

Bed options: the stored surface (FathomDEM two-scale) with design_discharge
bankfull (the default) and period_median, and surface_options' SWOT single 40 km
surface (figures/analysis/surfaces/swot_single_40km.gpkg) with
period_median; bankfull-FathomDEM and period-median-SWOT also with
bed_method manning_sea_inversion (4.4). Each row also gives the forward check
and the shares of flowing nodes on the slope floor and inverted.

Writes per basin 90_channel_capacity_{nodes.csv,profile.png} to
<results_dir>/<basin>/figures/analysis/ and channel_capacity_summary.csv to
<results_dir>/figures/analysis/.

Usage: python analysis/channel_capacity.py [basin_id ...]
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _basin_inputs import (
    DEFAULT_BASINS,
    basin_paths,
    load_config,
    network_end_levels,
    node_class,
    pct,
    reach_q_dict,
    round0,
    write_summary,
)
from _common import analysis_dir, analysis_path, script_parser
from bank_freeboard import bank_top, net_crs, node_tangents

log = logging.getLogger(__name__)
SCALES = np.geomspace(0.05, 5.0, 25)


def options(basin_id: str) -> dict[str, tuple[object, str, str]]:
    swot = analysis_dir(basin_id) / "surfaces" / "swot_single_40km.gpkg"
    inv = "manning_sea_inversion"
    return {
        "fathom_bankfull": (None, "bankfull", "manning"),
        "fathom_period_median": (None, "period_median", "manning"),
        "swot40_period_median": (swot, "period_median", "manning"),
        "fathom_bankfull_inversion": (None, "bankfull", inv),
        "swot40_period_median_inversion": (swot, "period_median", inv),
    }


def capacity_scale(levels: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Return per node the scale k at which the level (rows: SCALES) first reaches the bank."""
    above = levels >= bank[None, :]
    k = np.full(levels.shape[1], np.nan)
    k[above[0]] = SCALES[0]
    k[~above.any(axis=0)] = SCALES[-1]
    todo = np.flatnonzero(above.any(axis=0) & ~above[0])
    first = np.argmax(above[:, todo], axis=0)
    lo, hi = first - 1, first
    l0, l1 = levels[lo, todo], levels[hi, todo]
    f = np.clip((bank[todo] - l0) / np.where(l1 > l0, l1 - l0, 1.0), 0.0, 1.0)
    k[todo] = np.exp(np.log(SCALES[lo]) + f * (np.log(SCALES[hi]) - np.log(SCALES[lo])))
    return k


def basin(basin_id: str, cfg: dict) -> list[dict]:
    from src.gvf_solver import forward_network

    n = float(cfg["river_processing"]["channel_manning_n"])
    q_min = float(cfg["river_processing"]["river_depth_modelling"]["bed"]["q_min_m3s"])
    rows, nodes, bank, keep = [], {}, None, None
    from src.river_bed import forward_deviation_summary

    for name, (surface, design, method) in options(basin_id).items():
        _, rivers, q_bf, q_mean = round0(
            basin_id, cfg, water_surface=surface, design_discharge="bankfull"
        )
        bed, _, _, _ = round0(
            basin_id,
            cfg,
            water_surface=surface,
            design_discharge=design,
            bed_method=method,
        )
        fwd = forward_deviation_summary(bed)
        flowing = (bed.q_m3s > 0) & ~bed.bathymetry
        net = bed.net
        q_bf_r, q_mean_r = reach_q_dict(rivers, q_bf), reach_q_dict(rivers, q_mean)
        qbf_node = np.array([q_bf_r.get(o, 0.0) for o in net.owner])
        qmean_node = np.array([q_mean_r.get(o, 0.0) for o in net.owner])
        if bank is None:
            tx, ty = node_tangents(net)
            _, _, bank = bank_top(
                basin_paths(basin_id)["elevation_merged"],
                net_crs(basin_id),
                net.node_x,
                net.node_y,
                tx,
                ty,
                bed.width_m,
            )
            # A bank top at or below the sea is open water at a mouth, not a bank.
            keep = (
                (node_class(bed, basin_id) != "water_body")
                & (qbf_node > q_min)
                & np.isfinite(bank)
                & (bank > bed.sea_level_m)
            )
        ends = network_end_levels(bed)
        levels = np.vstack(
            [
                forward_network(
                    net,
                    bed.bed_m,
                    bed.width_m,
                    {r: k * q_bf_r.get(r, 0.0) for r in net.reach_ids},
                    n,
                    ends,
                )
                for k in SCALES
            ]
        )
        k_cap = capacity_scale(levels, bank)
        q_cap = k_cap * qbf_node
        ratio = np.where(keep, k_cap, np.nan)
        below_mean = keep & (q_cap < qmean_node)
        nodes[name] = pd.DataFrame(
            {
                "node": np.arange(net.n_nodes),
                "reach_id": np.asarray(net.owner),
                "dist_to_sea_m": bed.dist_to_sea_m,
                "main_path": bed.main_path,
                "kept": keep,
                "bank_top_m": bank,
                "bed_m": bed.bed_m,
                "q_bankfull_m3s": qbf_node,
                "q_mean_m3s": qmean_node,
                "q_cap_m3s": q_cap,
                "q_cap_over_q_bankfull": ratio,
                "q_cap_below_q_mean": below_mean,
            }
        )
        r = ratio[keep]
        rows.append(
            {
                "basin_id": basin_id,
                "option": name,
                "n_nodes": int(keep.sum()),
                "q_cap_over_q_bf_p10": pct(r, 10),
                "q_cap_over_q_bf_p50": pct(r, 50),
                "q_cap_over_q_bf_p90": pct(r, 90),
                "share_q_cap_below_q_bf": float((r < 1.0).mean()),
                "n_q_cap_below_q_mean": int(below_mean.sum()),
                "share_q_cap_below_q_mean": float(below_mean[keep].mean()),
                "forward_median_m": fwd["river_median_m"],
                "forward_p95_m": fwd["river_p95_m"],
                "on_floor_share": float(bed.on_floor[flowing].mean()),
                "inverted_share": float(bed.inverted[flowing].mean()),
                "depth_p50_m": pct(bed.depth_m[flowing], 50),
                "bed_p50_m": pct(bed.bed_m[flowing], 50),
            }
        )
    pd.concat(nodes, names=["option"]).reset_index(level=0).to_csv(
        analysis_path(basin_id, __file__, "nodes.csv"), index=False, float_format="%.4g"
    )
    _figure(basin_id, nodes)
    return rows


def _figure(basin_id: str, nodes: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = {
        "fathom_bankfull": "0.4",
        "fathom_period_median": "tab:orange",
        "swot40_period_median": "tab:green",
        "fathom_bankfull_inversion": "tab:purple",
        "swot40_period_median_inversion": "tab:olive",
    }
    for name, t in nodes.items():
        m = t[t["main_path"] & t["kept"]]
        for k, (_, g) in enumerate(m.groupby("reach_id")):
            g = g.sort_values("dist_to_sea_m")
            ax.plot(
                g["dist_to_sea_m"] / 1e3,
                g["q_cap_over_q_bankfull"],
                color=colors[name],
                lw=1.3,
                label=name if k == 0 else None,
            )
    m = nodes["fathom_bankfull"]
    m = m[m["main_path"] & m["kept"]]
    ax.plot(
        m["dist_to_sea_m"] / 1e3,
        m["q_mean_m3s"] / m["q_bankfull_m3s"],
        ".",
        ms=2,
        color="tab:red",
        label="Q_mean / Q_bankfull",
    )
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(SCALES[0] * 0.9, SCALES[-1] * 1.1)
    ax.invert_xaxis()
    ax.set_xlabel("distance to sea (km), largest-Q paths")
    ax.set_ylabel("Q_cap / Q_bankfull")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(
        f"channel_capacity | {basin_id} | channel capacity of the round-0 bed options"
    )
    fig.tight_layout()
    fig.savefig(analysis_path(basin_id, __file__, "profile.png"), dpi=130)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=DEFAULT_BASINS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    rows = [r for b in args.basin_ids for r in basin(str(b), cfg)]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
