"""Phase 3.6: rule-10 surface options and the round-0 bed hung from each.

Options: surface_source fathom | sword_swot, surface_fit two_scale | single,
design_discharge bankfull | period_median. Each surface is rebuilt with
src.water_surface.build_water_surface on rule 10's stored inputs (rule 06's
merged DEM and land use, SWOT levels from SWORD v17c), written to
<results_dir>/<basin>/figures/analysis/phase3_surfaces/, and rule 11's round 0
(_phase1.round0) is run on it.

Per option: surface minus SWOT on observed nodes; forward check (GVF level at
the design discharge minus the surface); share of nodes on the slope floor;
bankfull freeboard (bank top, phase1_freeboard.bank_top, minus the GVF level at
Q_bankfull on the bed).

Writes per basin 90_phase3_surface_options_{summary.csv,profile.png} to
<results_dir>/<basin>/figures/analysis/ and phase3_surface_options_summary.csv
to <results_dir>/figures/analysis/.

Usage: python analysis/phase3_surface_options.py [basin_id ...]
"""

import logging

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from _common import analysis_dir, analysis_path, script_parser
from _phase1 import (
    PHASE1_BASINS,
    basin_paths,
    calm_sea_level_m,
    load_config,
    network_end_levels,
    node_class,
    pct,
    reach_q_dict,
    round0,
    write_summary,
)
from phase1_freeboard import bank_top, net_crs, node_tangents

log = logging.getLogger(__name__)
SURFACES = {
    "fathom_two_scale": {"surface_source": "fathom", "surface_fit": "two_scale"},
    "swot_two_scale": {"surface_source": "sword_swot", "surface_fit": "two_scale"},
    "swot_single": {"surface_source": "sword_swot", "surface_fit": "single"},
    "swot_single_40km": {
        "surface_source": "sword_swot",
        "surface_fit": "single",
        "single_smoothing_m": 40000,
    },
}
DESIGNS = ("bankfull", "period_median")


def build_surfaces(basin_id: str, cfg: dict) -> dict[str, str]:
    from _common import REPO_ROOT
    from src.domain import load_domain
    from src.io import catalogue_entry, load_catalogue, raw_input_path
    from src.sword import read_swot_node_levels
    from src.water_surface import (
        build_water_surface,
        node_sampler,
        sample_centrelines,
        write_water_surface,
    )

    p = basin_paths(basin_id)
    dom = p["prep"] / "domain"
    merged, landuse = p["elevation_merged"], dom / f"{basin_id}_landuse.tif"
    with rasterio.open(merged) as src:
        dem_crs = src.crs
    rivers = gpd.read_file(p["rivers_clean"]).to_crs(dem_crs)
    sea = calm_sea_level_m(basin_id)
    cat = load_catalogue(REPO_ROOT / cfg["data_catalogue"])
    sword = [
        raw_input_path(cat, "river_network").format(region=r)
        for r in catalogue_entry(cat, "river_network")["regions"]
    ]
    bounds, _, _ = load_domain(p["domain_bbox"], p["domain_gpkg"])
    swot = read_swot_node_levels(
        sword,
        bounds,
        rivers["reach_id"],
        p["diagnostics"] / f"{basin_id}_geoid_offset.tif",
        dem_crs,
    )
    water_code = int(cfg["landuse"]["water_code"])
    out_dir = analysis_dir(basin_id) / "phase3_surfaces"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, opt in SURFACES.items():
        surface, _ = build_water_surface(
            rivers,
            lambda lines: sample_centrelines(lines, merged, landuse, water_code),
            {**cfg["river_processing"]["water_surface"], **opt},
            sea_level_m=sea,
            sample_nodes=node_sampler(
                merged, dom / f"{basin_id}_landuse_channels.tif", dem_crs
            ),
            gebco_floor_m=sea - float(cfg["terrain"]["gebco_max_depth_m"]),
            swot_nodes=swot,
        )
        paths[name] = out_dir / f"{name}.gpkg"
        write_water_surface(surface, paths[name], dem_crs)
    return paths


def basin(basin_id: str, cfg: dict) -> list[dict]:
    from src.gvf_solver import forward_network
    from src.river_bed import forward_deviation_summary
    from src.water_surface import read_water_surface

    n = float(cfg["river_processing"]["channel_manning_n"])
    q_min = float(cfg["river_processing"]["river_depth_modelling"]["bed"]["q_min_m3s"])
    surfaces = build_surfaces(basin_id, cfg)
    rows, profiles, bank = [], {}, None
    for s_name, s_path in surfaces.items():
        table, _ = read_water_surface(s_path)
        by_node = table.drop_duplicates("node_id").set_index("node_id").sort_index()
        swot_nodes = (by_node["level_source"] == "swot").to_numpy()
        _, rivers, q_bf, _ = round0(
            basin_id, cfg, water_surface=s_path, design_discharge="bankfull"
        )
        q_bf_reach = reach_q_dict(rivers, q_bf)
        for design in DESIGNS:
            bed, _, _, _ = round0(
                basin_id, cfg, water_surface=s_path, design_discharge=design
            )
            net = bed.net
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
                keep = (node_class(bed, basin_id) != "water_body") & np.array(
                    [q_bf_reach.get(o, 0.0) > q_min for o in net.owner]
                )
            level_bf = forward_network(
                net,
                bed.bed_m,
                bed.width_m,
                {r: q_bf_reach.get(r, 0.0) for r in net.reach_ids},
                n,
                network_end_levels(bed),
            )
            free = np.where(keep, bank - level_bf, np.nan)
            fwd = forward_deviation_summary(bed)
            river = (bed.q_m3s > 0) & ~bed.bathymetry
            surf_minus_swot = (bed.level_m - by_node["level_source_m"].to_numpy())[
                swot_nodes
            ]
            rows.append(
                {
                    "basin_id": basin_id,
                    "surface": s_name,
                    "design_discharge": design,
                    "share_nodes_swot": float(swot_nodes.mean()),
                    "surface_minus_swot_p50_m": pct(surf_minus_swot, 50),
                    "surface_minus_swot_abs_p95_m": pct(np.abs(surf_minus_swot), 95),
                    "forward_median_m": fwd["river_median_m"],
                    "forward_p95_m": fwd["river_p95_m"],
                    "forward_within_0.5m": fwd["river_within_0.5m"],
                    "on_floor_share": float(bed.on_floor[river].mean()),
                    "depth_p50_m": pct(bed.depth_m[river], 50),
                    "freeboard_p50_m": pct(free, 50),
                    "freeboard_share_below_0": float(
                        (free[np.isfinite(free)] < 0).mean()
                    ),
                }
            )
            profiles[(s_name, design)] = bed
    t = pd.DataFrame(rows)
    t.to_csv(
        analysis_path(basin_id, __file__, "summary.csv"),
        index=False,
        float_format="%.4g",
    )
    _figure(basin_id, profiles, bank, surfaces)
    return rows


def _figure(basin_id, profiles, bank, surfaces) -> None:
    from src.water_surface import read_water_surface

    ref = profiles[("fathom_two_scale", "bankfull")]
    net = ref.net
    owner = np.asarray(net.owner)
    sw, _ = read_water_surface(surfaces["swot_two_scale"])
    sw = sw.drop_duplicates("node_id").set_index("node_id").sort_index()
    obs = (sw["level_source"] == "swot").to_numpy() & ref.main_path
    km_all = ref.dist_to_sea_m / 1e3
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        km_all[obs],
        sw["level_source_m"].to_numpy()[obs],
        ".",
        ms=2.5,
        color="tab:blue",
        alpha=0.6,
        label="SWOT node level (window median)",
    )
    ax.plot(
        km_all[ref.main_path],
        bank[ref.main_path],
        ".",
        ms=2,
        color="tab:brown",
        alpha=0.5,
        label="bank top (P90, lower bank)",
    )
    styles = {
        ("fathom_two_scale", "bankfull"): ("0.4", "FathomDEM two-scale"),
        ("swot_two_scale", "bankfull"): ("tab:orange", "SWOT two-scale"),
        ("swot_single", "period_median"): ("tab:green", "SWOT single fit"),
    }
    # One line per reach: several seeds' largest-Q paths overlap in distance to sea.
    for k, r in enumerate(sorted({o for o, mm in zip(owner, ref.main_path) if mm})):
        ids = np.flatnonzero((owner == r) & ref.main_path)
        ids = ids[np.argsort(-ref.dist_to_sea_m[ids])]
        for (s, d), (c, lab) in styles.items():
            b = profiles[(s, d)]
            ax.plot(
                km_all[ids],
                b.level_m[ids],
                "-",
                color=c,
                lw=1.6,
                label=f"surface: {lab}" if k == 0 else None,
            )
            ax.plot(
                km_all[ids],
                b.bed_m[ids],
                "--",
                color=c,
                lw=1.0,
                label=f"bed: {lab}, {d}" if k == 0 else None,
            )
    ax.invert_xaxis()
    ax.set_xlabel("distance to sea (km), largest-Q paths")
    ax.set_ylabel("m (GOCO06s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"phase3 3.6 | {basin_id} | surface options and round-0 beds")
    fig.tight_layout()
    fig.savefig(analysis_path(basin_id, __file__, "profile.png"), dpi=130)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
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
