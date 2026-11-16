"""How far up each forcing mainstem the sea reaches: SFINCS coastal event against the steady GVF estimate.

Per forcing point's mainstem (largest Q_bf at bifurcations), the rise of the
river level at mean discharge over its calm-sea level, three ways:

  SFINCS event: zsmax of the coastal run minus the spin-up's end level
    (surge_reach.py);
  GVF event: steady GVF on rule 11's round-0 bed at Q_mean with the sea at the
    coastal scenario's storm tide (its surge RP and SLR) minus at the calm level
    (check_upstream_boundary.py's solve);
  GVF extreme: the same at SWL_ext, the highest storm tide over every scenario
    and the stress case (the boundary check's reference case).

x_inf: first node, going upstream, where the rise falls below
max(level_tol_m, amplitude_threshold_fraction x the rise at the mouth); the
forcing point passes with margin_km beyond it (testing.upstream_boundary_check).

Writes <basin>_profiles.csv per basin, summary.csv and upstream_boundary.png to
<results_dir>/figures/analysis/upstream_boundary/.

Usage: python analysis/upstream_boundary.py 2433835 3279946 [--coast coast_100_03SLR]
"""

import logging
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from _basin_inputs import basin_paths
from _common import REPO_ROOT, analysis_dir, script_parser, site_path
from _figure_style import add_land, map_axes, panel_letters, save_figure, write_caption
from _upstream_profile import (
    boundary_verdict,
    influence_length,
    mainstem_path,
    path_nodes,
    routed_discharge,
)
from check_river_bed import _config, run_round0
from check_upstream_boundary import (
    config_options,
    extreme_cases,
    forcing_points,
    sea_levels_at,
)
from surge_reach import reach_basin

sys.path.insert(0, str(REPO_ROOT / "workflow"))

log = logging.getLogger(__name__)

CASES = {
    "sfincs_event": ("SFINCS, event", "tab:blue", "o"),
    "gvf_event": ("GVF, event sea level", "tab:orange", "s"),
    "gvf_extreme": ("GVF, extreme sea level", "tab:red", "^"),
}


def scenario_case(name: str) -> tuple[float, float]:
    scenarios = yaml.safe_load(
        (REPO_ROOT / "config" / "scenarios.yml").read_text(encoding="utf-8")
    )
    s = scenarios[name]
    return float(s["surge_rp_yr"]), float(s["slr_m"])


def gvf_profiles(basin_id: str, coast: str) -> pd.DataFrame:
    from src.gvf_solver import forward_network

    cfg = _config()
    opts = config_options()
    frc = site_path("results_dir") / basin_id / "preprocessing_inputs" / "forcing"
    bed, rivers, _f, _s, crs, _p = run_round0(basin_id, cfg, analysis_dir(basin_id))
    net = bed.net
    q_mean = routed_discharge(rivers, frc / "river_forcing.nc", "mean_discharge_m3s")
    q_bf = routed_discharge(rivers, frc / "river_forcing.nc", "bankfull_discharge_m3s")
    ends = [int(net.reach_nodes[r][-1]) for r in net.reach_ids if not net.downstream[r]]
    mouths = {int(m) for m in bed.mouths}
    cases = {
        "event": scenario_case(coast),
        **extreme_cases(opts["stress_surge_rp_yr"], opts["stress_slr_m"]),
    }
    sea = sea_levels_at(
        np.c_[net.node_x[ends], net.node_y[ends]], frc / "surge_forcing.nc", crs, cases
    )
    levels = {
        "calm": sea["calm"],
        "event": sea["cases"]["event"],
        "extreme": sea["swl_ext"],
    }
    n = float(cfg["river_processing"]["channel_manning_n"])
    wl = {
        s: forward_network(
            net,
            bed.bed_m,
            bed.width_m,
            q_mean,
            n,
            {
                e: float(v[i]) if e in mouths else float(bed.level_m[e])
                for i, e in enumerate(ends)
            },
        )
        for s, v in levels.items()
    }
    parts = []
    for p in forcing_points(
        frc / "river_forcing.nc", rivers, crs, net, []
    ).itertuples():
        if p.reach_id not in net.reach_nodes:
            continue
        ids, dist = path_nodes(net, mainstem_path(net.downstream, p.reach_id, q_bf))
        k = int(np.argmin(np.hypot(net.node_x[ids] - p.x, net.node_y[ids] - p.y)))
        ids = ids[k:]
        parts.append(
            pd.DataFrame(
                {
                    "reach_id": p.reach_id,
                    "node": ids,
                    "x": net.node_x[ids],
                    "y": net.node_y[ids],
                    "dz_gvf_event_m": (wl["event"] - wl["calm"])[ids],
                    "dz_gvf_extreme_m": (wl["extreme"] - wl["calm"])[ids],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def basin(basin_id: str, coast: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = _config()["testing"]["upstream_boundary_check"]
    tol, frac = float(test["level_tol_m"]), float(test["amplitude_threshold_fraction"])
    margin_m = float(test["margin_km"]) * 1000
    _summary, sfincs = reach_basin(basin_id, coast)
    sfincs = sfincs.rename(columns={"dz_m": "dz_sfincs_event_m"})
    prof = sfincs.merge(
        gvf_profiles(basin_id, coast), on=["reach_id", "node"], how="left"
    )
    rows = []
    for reach_id, p in prof.groupby("reach_id", sort=False):
        dist = p["dist_from_mouth_km"].to_numpy() * 1000
        row = {"basin_id": basin_id, "reach_id": reach_id, "x_f_km": dist[0] / 1000}
        for case in CASES:
            dz = p[f"dz_{case}_m"].to_numpy(float)
            thr = max(tol, frac * float(dz[-1]))
            x_inf = influence_length(dist, dz, thr)
            ok, _why = boundary_verdict(dist[0], float(dz[0]), x_inf, tol, margin_m)
            row |= {
                f"rise_mouth_{case}_m": float(dz[-1]),
                f"x_inf_{case}_km": x_inf / 1000,
                f"margin_{case}_km": (dist[0] - x_inf) / 1000,
                f"passes_{case}": ok,
            }
        rows.append(row)
    return pd.DataFrame(rows), prof


def write_figure(out, basin_ids: list[str], coast: str) -> None:
    results = {}
    summary = pd.read_csv(out / "summary.csv", dtype={"reach_id": str, "basin_id": str})
    for b in basin_ids:
        prof = pd.read_csv(out / f"{b}_profiles.csv", dtype={"reach_id": str})
        results[b] = (summary[summary["basin_id"] == b].reset_index(drop=True), prof)
    figure(results, out / "upstream_boundary.png", coast)


def figure(results: dict, path, coast: str) -> None:
    from src.grid import load_grid_def

    n_paths = max(len(s) for s, _ in results.values())
    fig = plt.figure(figsize=(6 + 5 * n_paths, 5.2 * len(results)))
    gs = fig.add_gridspec(len(results), 1 + n_paths, width_ratios=[1.4] + [1] * n_paths)
    axes, layout = [], []
    for r, (basin_id, (summary, prof)) in enumerate(results.items()):
        ax = fig.add_subplot(gs[r, 0])
        axes.append(ax)
        p = basin_paths(basin_id)
        rivers = gpd.read_file(p["rivers_clean"]).to_crs(
            load_grid_def(p["sfincs_grid"])["crs"]
        )
        add_land(ax, basin_id)
        rivers.plot(ax=ax, color="0.55", lw=0.6)
        for _, s in summary.iterrows():
            q = prof[prof["reach_id"] == s["reach_id"]]
            ax.plot(q["x"], q["y"], color="k", lw=1.4)
            ax.plot(q["x"].iloc[0], q["y"].iloc[0], "kv", ms=9)
            d = q["dist_from_mouth_km"].to_numpy()
            for case, (_label, col, mk) in CASES.items():
                x_inf = s[f"x_inf_{case}_km"]
                if np.isfinite(x_inf):
                    k = int(np.argmin(np.abs(d - x_inf)))
                    ax.plot(
                        q["x"].iloc[k], q["y"].iloc[k], mk, color=col, ms=8, mec="k"
                    )
        xb, yb = rivers.total_bounds[[0, 2]], rivers.total_bounds[[1, 3]]
        pad = 0.05 * max(np.ptp(xb), np.ptp(yb))
        ax.set_xlim(xb[0] - pad, xb[1] + pad)
        ax.set_ylim(yb[0] - pad, yb[1] + pad)
        map_axes(ax)
        reaches = []
        for c, (_, s) in enumerate(summary.iterrows()):
            b = fig.add_subplot(gs[r, 1 + c])
            axes.append(b)
            q = prof[prof["reach_id"] == s["reach_id"]]
            for case, (_label, col, _mk) in CASES.items():
                b.plot(q["dist_from_mouth_km"], q[f"dz_{case}_m"], color=col, lw=1.4)
                if np.isfinite(s[f"x_inf_{case}_km"]):
                    b.axvline(s[f"x_inf_{case}_km"], color=col, ls="--", lw=1)
            b.axvline(s["x_f_km"], color="k", lw=1.3)
            b.set_xlabel("distance from the mouth along the mainstem (km)")
            b.set_ylabel("rise over the calm-sea level (m)")
            b.grid(alpha=0.3)
            reaches.append(s)
        layout.append((basin_id, reaches))
    panel_letters(axes)
    fig.tight_layout()
    save_figure(fig, path, dpi=130)
    plt.close(fig)
    test = config_options()
    k, lines = 0, []
    for basin_id, reaches in layout:
        map_letter = chr(97 + k)
        prof_letters = [chr(97 + k + 1 + j) for j in range(len(reaches))]
        k += 1 + len(reaches)
        detail = "; ".join(
            f"({lt}) forcing reach {s['reach_id']} at {s['x_f_km']:.1f} km: margin "
            + " / ".join(f"{s[f'margin_{c}_km']:.0f}" for c in CASES)
            + " km"
            for lt, s in zip(prof_letters, reaches)
        )
        lines.append(f"- basin {basin_id}: map ({map_letter}); {detail}")
    write_caption(
        path,
        f"""
How far up the river the sea raises the water level, at mean river discharge: the coastal event {coast} in SFINCS against the steady gradually-varied-flow (GVF) estimate. Values in summary.csv.
Maps: light grey land; grey lines, river network; black lines, the mainstem from each forcing point (black triangle, where the river discharge enters the model) to the sea. Markers show where the sea's influence ends on each mainstem (x_inf: the first point upstream where the rise falls below max({test["level_tol_m"]:g} m, {float(_config()["testing"]["upstream_boundary_check"]["amplitude_threshold_fraction"]):.0%} of the rise at the mouth)): blue circle, SFINCS coastal event (highest level of the run minus the spin-up level); orange square, GVF with the sea at the event's storm-tide level; red triangle, GVF with the sea at the extreme level (the highest storm tide over all scenarios and the stress case RP{test["stress_surge_rp_yr"]:g} + {test["stress_slr_m"]:g} m sea-level rise).
Profiles, one per forcing point: the rise along the mainstem in the same three colours, dashed vertical lines at each x_inf, black vertical line at the forcing point. The boundary passes when the forcing point lies at least {test["margin_km"]:g} km beyond x_inf. Margins below are SFINCS event / GVF event / GVF extreme.
"""
        + "\n".join(lines),
    )


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="+")
    parser.add_argument("--coast", default="coast_100_03SLR")
    parser.add_argument(
        "--figure-only", action="store_true", help="redraw from the written CSVs"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = site_path("results_dir") / "figures" / "analysis" / "upstream_boundary"
    out.mkdir(parents=True, exist_ok=True)
    basin_ids = [str(b) for b in args.basin_ids]
    if not args.figure_only:
        tables = []
        for b in basin_ids:
            summary, prof = basin(b, args.coast)
            prof.to_csv(out / f"{b}_profiles.csv", index=False, float_format="%.3f")
            tables.append(summary)
        table = pd.concat(tables, ignore_index=True)
        table.to_csv(out / "summary.csv", index=False, float_format="%.2f")
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(table.round(2).to_string(index=False))
    write_figure(out, basin_ids, args.coast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
