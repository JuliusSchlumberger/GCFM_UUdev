"""Phase 3.1: SWOT node levels against each stage of the rule-10 surface.

Stages, rebuilt on rule 10's node network from its stored node medians
(src.water_surface.build_water_surface): raw FathomDEM median, sea pin (every
node at or above the calm sea level, mouths at it), the network isotonic fit,
and the smoothed level. SWOT: SWORD v17c wse_obs_p50 on nodes with n_obs >= 10,
shifted from EGM2008 to the model datum (GOCO06s) with
src.raster.compute_geoid_offset_arr, matched to the nearest network node of the
same reach within 500 m.

Writes per basin 90_phase3_surface_stages_{reaches.csv,along.png} to
<results_dir>/<basin>/figures/analysis/ and phase3_surface_stages_summary.csv to
<results_dir>/figures/analysis/.

Usage: python analysis/phase3_surface_stages.py [basin_id ...]
"""

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import analysis_path, script_parser
from _phase1 import (
    PHASE1_BASINS,
    basin_paths,
    calm_sea_level_m,
    load_config,
    pct,
    write_summary,
)
from phase1_swot_coverage import MIN_OBS, _sword, geoid_offset_at
from pyproj import Transformer
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)
MATCH_M = 500.0
STAGES = ["raw", "pinned", "isotonic", "level"]


def stages(basin_id: str, cfg: dict):
    from src.gvf_target import (
        build_node_network,
        enforce_non_increasing,
        main_reaches,
        orient_downstream,
    )
    from src.river_network import build_downstream_adjacency, normalize_reach_id
    from src.water_surface import read_water_surface

    p = basin_paths(basin_id)
    table, _ = read_water_surface(p["water_surface"])
    rivers = gpd.read_file(p["rivers_clean"]).to_crs(table.crs)
    downstream = build_downstream_adjacency(rivers)
    rivers, _ = orient_downstream(rivers, downstream)
    net = build_node_network(
        rivers,
        downstream,
        float(cfg["river_processing"]["water_surface"]["node_spacing_m"]),
    )
    by_node = table.drop_duplicates("node_id").set_index("node_id").sort_index()
    if len(by_node) != net.n_nodes or not np.allclose(
        by_node.geometry.x.to_numpy(), net.node_x
    ):
        raise ValueError(f"{basin_id}: rebuilt node network differs from rule 10's")
    sea = calm_sea_level_m(basin_id)
    raw = by_node["level_raw_m"].to_numpy(float)
    mouths = by_node["is_mouth"].to_numpy(bool)
    pinned = np.maximum(raw, sea)
    pinned[mouths] = sea
    iso = enforce_non_increasing(net, pinned, fixed=mouths)
    out = pd.DataFrame(
        {
            "node": np.arange(net.n_nodes),
            "reach_id": np.asarray(net.owner),
            "x": net.node_x,
            "y": net.node_y,
            "raw": raw,
            "pinned": pinned,
            "isotonic": iso,
            "level": by_node["level_m"].to_numpy(float),
            "dem_m": by_node["dem_m"].to_numpy(float),
            "node_class": by_node["node_class"].to_numpy(object),
        }
    )
    q = dict(
        zip(
            rivers["reach_id"].map(normalize_reach_id),
            pd.to_numeric(
                rivers.get("bankfull_discharge_acc_m3s", rivers["width"]),
                errors="coerce",
            ).fillna(0.0),
        )
    )
    return out, net, main_reaches(net, q), table.crs


def swot_on_nodes(basin_id: str, cfg: dict, nodes: pd.DataFrame, crs) -> pd.DataFrame:
    from src.domain import load_domain
    from src.river_network import normalize_reach_id

    p = basin_paths(basin_id)
    bounds, _, _ = load_domain(p["domain_bbox"], p["domain_gpkg"])
    reach_ids = {int(r) for r in nodes["reach_id"].dropna().unique()}
    sw = _sword(
        cfg,
        "nodes",
        bounds,
        reach_ids,
        ["node_id", "reach_id", "n_obs", "wse_obs_p50", "wse_obs_mad"],
    )
    sw = sw[
        (pd.to_numeric(sw["n_obs"], errors="coerce") >= MIN_OBS)
        & np.isfinite(pd.to_numeric(sw["wse_obs_p50"], errors="coerce"))
    ].copy()
    lon, lat = sw.geometry.x.to_numpy(), sw.geometry.y.to_numpy()
    sw["swot_m"] = sw["wse_obs_p50"].astype(float) + geoid_offset_at(cfg, lon, lat)
    sx, sy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    sw["rid"] = sw["reach_id"].map(normalize_reach_id)
    nodes = nodes.assign(swot_m=np.nan, swot_mad_m=np.nan)
    for rid, g in sw.groupby("rid"):
        own = nodes.index[nodes["reach_id"] == rid]
        if not len(own):
            continue
        d, k = cKDTree(nodes.loc[own, ["x", "y"]].to_numpy()).query(
            np.column_stack([sx[sw["rid"] == rid], sy[sw["rid"] == rid]])
        )
        ok = d <= MATCH_M
        hit = (
            pd.DataFrame(
                {
                    "i": own[k[ok]],
                    "v": g["swot_m"].to_numpy()[ok],
                    "mad": g["wse_obs_mad"].astype(float).to_numpy()[ok],
                }
            )
            .groupby("i")
            .median()
        )
        nodes.loc[hit.index, "swot_m"] = hit["v"]
        nodes.loc[hit.index, "swot_mad_m"] = hit["mad"]
    return nodes


def basin(basin_id: str, cfg: dict) -> dict:
    nodes, net, main, crs = stages(basin_id, cfg)
    nodes = swot_on_nodes(basin_id, cfg, nodes, crs)
    obs = np.isfinite(nodes["swot_m"]) & (nodes["node_class"] != "water_body")
    per_reach = (
        nodes[obs]
        .assign(**{f"d_{s}": nodes[s] - nodes["swot_m"] for s in STAGES})
        .groupby("reach_id")
        .agg(
            n=("swot_m", "size"),
            **{f"{s}_minus_swot_m": (f"d_{s}", "median") for s in STAGES},
            swot_mad_m=("swot_mad_m", "median"),
        )
        .reset_index()
    )
    per_reach.to_csv(
        analysis_path(basin_id, __file__, "reaches.csv"),
        index=False,
        float_format="%.3f",
    )
    _plot(basin_id, nodes, net, main)
    row = {"basin_id": basin_id, "n_nodes_swot": int(obs.sum())}
    for s in STAGES:
        row[f"{s}_minus_swot_p50_m"] = pct((nodes[s] - nodes["swot_m"])[obs], 50)
    row["swot_mad_p50_m"] = pct(nodes["swot_mad_m"][obs], 50)
    row["share_mad_gt_0.5"] = float((nodes["swot_mad_m"][obs] > 0.5).mean())
    return row


def _main_path(net, main: set[str]) -> list[str]:
    # Longest chain of largest-Q reaches, followed downstream from a head.
    upstream = net.upstream()
    heads = [r for r in main if not any(u in main for u in upstream[r])]
    best: list[str] = []
    for h in sorted(heads):
        path = [h]
        while True:
            nxt = [d for d in net.downstream[path[-1]] if d in main and d not in path]
            if not nxt:
                break
            path.append(nxt[0])
        if sum(net.reach_length[r] for r in path) > sum(
            net.reach_length[r] for r in best
        ):
            best = path
    return best


def _plot(basin_id: str, nodes: pd.DataFrame, net, main: set[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    path = _main_path(net, main)
    ids, dist, off = [], [], 0.0
    for r in path:
        ids += list(net.reach_nodes[r])
        dist += list(off + np.asarray(net.reach_chainage[r]))
        off += net.reach_length[r]
    t = nodes.iloc[ids].assign(km=np.asarray(dist) / 1e3)
    for s, st in zip(STAGES, ["0.6", "tab:orange", "tab:green", "k"]):
        ax.plot(t["km"], t[s], color=st, lw=1.2 if s != "level" else 2, label=s)
    ax.plot(
        t["km"],
        t["swot_m"],
        "o",
        ms=3,
        color="tab:blue",
        label="SWOT wse_obs_p50 (GOCO06s)",
    )
    ax.set_xlabel("km along the main path (downstream)")
    ax.set_ylabel("level (m, GOCO06s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{basin_id}: rule-10 surface stages against SWOT")
    fig.tight_layout()
    fig.savefig(analysis_path(basin_id, __file__, "along.png"), dpi=130)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    rows = [basin(str(b), cfg) for b in args.basin_ids]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
