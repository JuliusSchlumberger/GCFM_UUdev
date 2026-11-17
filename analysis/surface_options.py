"""Rule-10 surface options and the round-0 bed hung from each, with and without the sea inversion.

Options: surface_source fathom | sword_swot, surface_fit two_scale | single,
design_discharge bankfull | period_median, and bed_method manning_sea_inversion
for the pairs in INVERSION_CASES. Each surface is rebuilt with
src.water_surface.build_water_surface on rule 10's stored inputs (rule 06's
merged DEM and land use, SWOT levels from SWORD v17c), written to
<results_dir>/<basin>/figures/analysis/surfaces/, and rule 11's round 0
(_basin_inputs.round0) is run on it.

Per option: surface minus SWOT on observed nodes; forward check (GVF level at
the design discharge minus the surface); share of nodes on the slope floor;
depth median, p95 and maximum; crest height (rule 11's: the GVF level at the
calibration discharge with the sea at the calibration storm tide, floored at the
coastal protection crest, plus freeboard_m) above the DEM at the river centre;
the 1D Froude number at Q_bankfull x peak_ratio; bankfull freeboard (bank top, bank_freeboard.bank_top, minus the GVF level at
Q_bankfull on the bed).

Writes per basin 90_surface_options_{summary.csv,profile.png} and the
figure's data, 90_surface_options_profile_data.csv, to
<results_dir>/<basin>/figures/analysis/ and surface_options_summary.csv
to <results_dir>/figures/analysis/. --redraw redraws the figure from the
saved profile data without rebuilding anything.

Usage: python analysis/surface_options.py [basin_id ...] [--redraw]
"""

import logging

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from _basin_inputs import (
    DEFAULT_BASINS,
    basin_paths,
    calibration_sea_level_m,
    calm_sea_level_m,
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
from matplotlib.lines import Line2D
from src.figure_style import (
    SURFACE_OPTION_STYLE,
    figure_size,
    panel_letters,
    save_figure,
    style_context,
    write_caption,
)

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
# (surface, design discharge) pairs also run with the sea inversion
INVERSION_CASES = (
    ("fathom_two_scale", "bankfull"),
    ("swot_single_40km", "period_median"),
)
# (surface, design discharge) pairs drawn on the profile figure
PROFILE_OPTIONS = (
    ("fathom_two_scale", "bankfull"),
    ("swot_two_scale", "bankfull"),
    ("swot_single", "period_median"),
)


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
    out_dir = analysis_dir(basin_id) / "surfaces"
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


def basin(basin_id: str, cfg: dict, peak_ratio: float = 1.0) -> list[dict]:
    """Return one row per option; peak_ratio scales Q_bankfull to the event peak for the Froude number."""
    from src.gvf_solver import forward_network
    from src.river_bed import forward_deviation_summary
    from src.water_surface import read_water_surface

    n = float(cfg["river_processing"]["channel_manning_n"])
    q_min = float(cfg["river_processing"]["river_depth_modelling"]["bed"]["q_min_m3s"])
    surfaces = build_surfaces(basin_id, cfg)
    p = basin_paths(basin_id)
    depth_net = gpd.read_file(p["rivers_depth"])
    q_calib = reach_q_dict(depth_net, depth_net["q_calib"].to_numpy(float))
    sea_calib = calibration_sea_level_m(basin_id)
    with xr.open_dataset(p["surge_forcing"], decode_times=False) as ds:
        coastal_crest = (
            float(ds["coastal_protection_crest_m"].values)
            if "coastal_protection_crest_m" in ds
            else 0.0
        )
    freeboard = float(cfg["river_processing"]["river_depth_modelling"]["freeboard_m"])
    rows, profiles, bank = [], {}, None
    for s_name, s_path in surfaces.items():
        table, _ = read_water_surface(s_path)
        by_node = table.drop_duplicates("node_id").set_index("node_id").sort_index()
        swot_nodes = (by_node["level_source"] == "swot").to_numpy()
        _, rivers, q_bf, _ = round0(
            basin_id, cfg, water_surface=s_path, design_discharge="bankfull"
        )
        q_bf_reach = reach_q_dict(rivers, q_bf)
        methods = [
            (design, method)
            for design in DESIGNS
            for method in ("manning", "manning_sea_inversion")
            if method == "manning" or (s_name, design) in INVERSION_CASES
        ]
        for design, method in methods:
            bed, _, _, _ = round0(
                basin_id,
                cfg,
                water_surface=s_path,
                design_discharge=design,
                bed_method=method,
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
            mouths = {int(m) for m in bed.mouths}
            ends = network_end_levels(bed)
            # crest as rule 11 sets it: the level at the calibration discharge
            # with the sea at the calibration storm tide, floored at the coastal crest
            level_calib = forward_network(
                net,
                bed.bed_m,
                bed.width_m,
                {r: q_calib.get(r, 0.0) for r in net.reach_ids},
                n,
                {e: sea_calib if e in mouths else v for e, v in ends.items()},
            )
            crest_above_dem = (
                np.maximum(level_calib, coastal_crest) + freeboard - bed.fathom_level_m
            )
            q_peak = {r: peak_ratio * q_bf_reach.get(r, 0.0) for r in net.reach_ids}
            level_peak = forward_network(
                net,
                bed.bed_m,
                bed.width_m,
                q_peak,
                n,
                {e: bed.sea_level_m if e in mouths else v for e, v in ends.items()},
            )
            h_peak = np.maximum(level_peak - bed.bed_m, 0.1)
            q_node = np.array([q_peak.get(o, 0.0) for o in net.owner])
            froude_peak = q_node / (bed.width_m * h_peak * np.sqrt(9.81 * h_peak))
            fwd = forward_deviation_summary(bed)
            river = (bed.q_m3s > 0) & ~bed.bathymetry
            # depth_m holds the Manning depth; the sea inversion only moves the bed
            depth = bed.level_m - bed.bed_m
            surf_minus_swot = (bed.level_m - by_node["level_source_m"].to_numpy())[
                swot_nodes
            ]
            rows.append(
                {
                    "basin_id": basin_id,
                    "surface": s_name,
                    "design_discharge": design,
                    "bed_method": method,
                    "share_nodes_swot": float(swot_nodes.mean()),
                    "surface_minus_swot_p50_m": pct(surf_minus_swot, 50),
                    "surface_minus_swot_abs_p95_m": pct(np.abs(surf_minus_swot), 95),
                    "forward_median_m": fwd["river_median_m"],
                    "forward_p95_m": fwd["river_p95_m"],
                    "on_floor_share": float(bed.on_floor[river].mean()),
                    "depth_p50_m": pct(depth[river], 50),
                    "depth_p95_m": pct(depth[river], 95),
                    "depth_max_m": pct(depth[river], 100),
                    "crest_above_dem_p50_m": pct(crest_above_dem[river], 50),
                    "crest_above_dem_p95_m": pct(crest_above_dem[river], 95),
                    "froude_peak_p50": pct(froude_peak[river & (q_node > 0)], 50),
                    "froude_peak_max": pct(froude_peak[river & (q_node > 0)], 100),
                    "freeboard_p50_m": pct(free, 50),
                    "freeboard_share_below_0": float(
                        (free[np.isfinite(free)] < 0).mean()
                    ),
                }
            )
            if method == "manning":
                profiles[(s_name, design)] = bed
    t = pd.DataFrame(rows)
    t.to_csv(
        analysis_path(basin_id, __file__, "summary.csv"),
        index=False,
        float_format="%.4g",
    )
    table = profile_table(profiles, bank, surfaces)
    table.to_csv(profile_data_path(basin_id), index=False, float_format="%.4g")
    plot_profile(basin_id, table)
    return rows


def profile_data_path(basin_id: str):
    return analysis_path(basin_id, __file__, "profile_data.csv")


def profile_table(profiles, bank, surfaces) -> pd.DataFrame:
    """Return the profile figure's data: one row per node on the largest-discharge paths.

    Columns: reach_id, dist_to_sea_km, swot_level_m (NaN off SWOT-observed
    nodes), bank_top_m, and <surface>_surface_m / <surface>_bed_m for the
    PROFILE_OPTIONS; levels in m above the GOCO06s geoid.
    """
    from src.water_surface import read_water_surface

    ref = profiles[PROFILE_OPTIONS[0]]
    on_path = np.asarray(ref.main_path)
    sw, _ = read_water_surface(surfaces["swot_two_scale"])
    sw = sw.drop_duplicates("node_id").set_index("node_id").sort_index()
    observed = (sw["level_source"] == "swot").to_numpy()
    table = pd.DataFrame(
        {
            "reach_id": np.asarray(ref.net.owner)[on_path],
            "dist_to_sea_km": ref.dist_to_sea_m[on_path] / 1e3,
            "swot_level_m": np.where(observed, sw["level_source_m"].to_numpy(), np.nan)[
                on_path
            ],
            "bank_top_m": np.asarray(bank)[on_path],
        }
    )
    for s, d in PROFILE_OPTIONS:
        table[f"{s}_surface_m"] = profiles[(s, d)].level_m[on_path]
        table[f"{s}_bed_m"] = profiles[(s, d)].bed_m[on_path]
    return table


def plot_profile(basin_id: str, table: pd.DataFrame) -> None:
    """Plot (a) the surface options with SWOT levels and bank top, (b) the beds hung from them.

    Several seeds' largest-discharge paths overlap in distance to the sea, so
    each reach is its own line. The y axis spans the surfaces and beds; bank-top
    points above it are drawn as triangles on the top edge.
    """
    lines = [f"{s}_{kind}_m" for s, _ in PROFILE_OPTIONS for kind in ("surface", "bed")]
    lo, hi = np.nanmin(table[lines].to_numpy()), np.nanmax(table[lines].to_numpy())
    pad = 0.08 * (hi - lo)
    y_lim = (lo - pad, hi + pad)
    bank = table["bank_top_m"].to_numpy()
    above = bank > y_lim[1]
    path = analysis_path(basin_id, __file__, "profile.png")
    with style_context():
        fig, (ax_s, ax_b) = plt.subplots(
            2,
            1,
            figsize=figure_size(120),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        km = table["dist_to_sea_km"].to_numpy()
        ax_s.plot(km[~above], bank[~above], ".", ms=2, color="0.6", zorder=1)
        ax_s.plot(
            km[above],
            np.full(above.sum(), y_lim[1]),
            "^",
            ms=3,
            color="0.6",
            clip_on=False,
            zorder=1,
        )
        ax_s.plot(km, table["swot_level_m"], ".", ms=2.5, color="0.15", zorder=2)
        for _, reach in table.groupby("reach_id", sort=True):
            reach = reach.sort_values("dist_to_sea_km", ascending=False)
            for s, _ in PROFILE_OPTIONS:
                style = {
                    k: v for k, v in SURFACE_OPTION_STYLE[s].items() if k != "label"
                }
                ax_s.plot(
                    reach["dist_to_sea_km"],
                    reach[f"{s}_surface_m"],
                    linewidth=1.2,
                    **style,
                )
                ax_b.plot(
                    reach["dist_to_sea_km"], reach[f"{s}_bed_m"], linewidth=1.2, **style
                )
        ax_s.set_ylim(*y_lim)
        ax_s.invert_xaxis()
        for ax in (ax_s, ax_b):
            ax.set_ylabel("Elevation (m +GOCO06s)")
        ax_b.set_xlabel("Distance to the sea (km)")
        panel_letters([ax_s, ax_b])
        fig.legend(
            handles=[
                *(
                    Line2D([0], [0], linewidth=1.2, **SURFACE_OPTION_STYLE[s])
                    for s, _ in PROFILE_OPTIONS
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    ms=3,
                    linestyle="none",
                    color="0.15",
                    label="SWOT node level",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    ms=3,
                    linestyle="none",
                    color="0.6",
                    label="Bank top",
                ),
            ],
            loc="outside lower center",
            ncols=5,
        )
        save_figure(fig, path)
        plt.close(fig)

    def _gap(s: str) -> str:
        d = (table[f"{s}_surface_m"] - table["swot_level_m"]).dropna()
        label = SURFACE_OPTION_STYLE[s]["label"]
        return f"{label} {d.median():+.2f} m" if len(d) else f"{label} no SWOT node"

    designs = dict(PROFILE_OPTIONS)
    clipped = (
        f", {int(above.sum())} bank-top point(s) above the axis drawn as triangles on its top edge"
        if above.any()
        else ""
    )
    write_caption(
        path,
        f"River water-surface options of basin {basin_id} and the round-0 river beds hung from "
        "them, along the largest-discharge paths from the seeds to the sea; median surface minus "
        f"SWOT level on observed nodes: {'; '.join(_gap(s) for s, _ in PROFILE_OPTIONS)}.\n"
        "(a) Water surfaces, (b) river beds (round 0, Manning depth at the design discharge: "
        + ", ".join(
            f"{SURFACE_OPTION_STYLE[s]['label']} at {designs[s].replace('_', ' ')}"
            for s, _ in PROFILE_OPTIONS
        )
        + "). Black solid: FathomDEM surface, two-scale fit; green dash-dot: SWOT, two-scale fit; "
        "sky-blue dotted: SWOT, single fit; one line per reach. Dark dots: SWOT node levels "
        "(window median); grey dots: bank top (90th percentile of the lower bank)"
        + clipped
        + ". Elevations in m above the GOCO06s geoid; distance to the sea along the path, "
        "sea on the right.",
    )


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=DEFAULT_BASINS)
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="redraw the profile figure from its saved data only",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.redraw:
        for b in args.basin_ids:
            plot_profile(str(b), pd.read_csv(profile_data_path(str(b))))
        return 0
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
