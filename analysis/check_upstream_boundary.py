"""Check that the river forcing points lie beyond the sea's influence on the river level.

Steady GVF levels on rule 11's round-0 bed over the whole node network
(src.gvf_solver.forward_network; bed rebuilt with the current code by
check_river_bed.run_round0) for Q_mean and Q_bf, with the sea mouths at the
nearest station's calm level and at SWL_ext: the highest storm tide plus
SLR x slr_fingerprint over every scenario in scenarios.yml with a surge RP and
the stress case (stress_surge_rp_yr, stress_slr_m), per station. Along each
forcing point's mainstem (largest Q_bf at
bifurcations): dWL(x) = WL(Q, SWL) - WL(Q, calm), the same bed and discharge in
both, so the bed's offset from rule 10's surface cancels.

- x_inf: first node, going upstream, where dWL < max(level_tol_m,
  amplitude_threshold_fraction x A0), A0 = SWL - calm at the mouth;
- verdict at (Q_mean, SWL_ext): pass if dWL at the forcing point < level_tol_m
  and the forcing point lies margin_km beyond x_inf;
- reported: x_SWL (WL(Q_mean, calm) rises above SWL_ext; rule 10's surface
  runs metres higher at mean flow on steep rivers), the backwater length h/S
  (mouth depth, mean bed slope to the forcing point), the e-folding length of
  dWL, WL(Q_mean, calm) above SWL_ext and MSL, DE2's 10 m rule on it, and
  WL(Q_bf, calm) - rule 10's surface at the forcing point.

Parameters from testing.upstream_boundary_check; the options override them.

Writes to figs/upstream_boundary_check/: <basin>_forcing_points.csv (one row
per forcing point x (Q, SWL) pair), <basin>_profile_<reach>.png per forcing
point, <basin>_upstream_boundary_check.png (map of dWL at Q_mean, SWL_ext).
--extra-reach evaluates a boundary at the upstream end of further reaches.

Usage:
    python analysis/check_upstream_boundary.py [basin_id] [--level-tol-m M] [--margin-km KM]
        [--slr-m M] [--extra-reach ID ...]
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _common import REPO_ROOT, add_basin_id_argument, script_parser, site_path
from _upstream_profile import (
    boundary_verdict,
    efolding_length,
    first_above,
    high_gradient_rule,
    influence_length,
    mainstem_path,
    path_nodes,
    routed_discharge,
)
from check_river_bed import _config, _git_hash, run_round0
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.gvf_solver import forward_network
from src.river_network import normalize_reach_id
from src.surge import lookup_storm_tide_at_rp

log = logging.getLogger(__name__)

OPTION_KEYS = (
    "level_tol_m",
    "margin_km",
    "stress_surge_rp_yr",
    "stress_slr_m",
    "high_gradient_elevation_m",
)
DISCHARGES = {"Q_mean": "mean_discharge_m3s", "Q_bf": "bankfull_discharge_m3s"}
SEA_LEVELS = ("calm", "swl_ext")
REFERENCE = ("Q_mean", "swl_ext")


def config_options() -> dict[str, float]:
    """Return the check's parameters from ``testing.upstream_boundary_check``."""
    test = _config()["testing"]["upstream_boundary_check"]
    return {key: float(test[key]) for key in OPTION_KEYS}


def extreme_cases(
    stress_rp_yr: float, stress_slr_m: float
) -> dict[str, tuple[float, float]]:
    """Return (surge RP, SLR) per label: each scenario with a surge RP (scenarios.yml) and the stress case."""
    import yaml

    scenarios = (
        yaml.safe_load(
            (REPO_ROOT / "config" / "scenarios.yml").read_text(encoding="utf-8")
        )
        or {}
    )
    names: dict[tuple[float, float], list[str]] = {}
    for name, s in scenarios.items():
        rp = s.get("surge_rp_yr")
        if isinstance(rp, (int, float)) and not isinstance(rp, bool):
            names.setdefault((float(rp), float(s["slr_m"])), []).append(name)
    names.setdefault((float(stress_rp_yr), float(stress_slr_m)), []).append("stress")
    return {
        f"RP{rp:g} + {slr:g} m SLR ({n[0]}{f' +{len(n) - 1}' if len(n) > 1 else ''})": (
            rp,
            slr,
        )
        for (rp, slr), n in sorted(names.items())
    }


def sea_levels_at(
    xy: np.ndarray, surge_path: Path, crs, cases: dict[str, tuple[float, float]]
) -> dict:
    """Return per point the nearest station's calm level, each case's level, their maximum (SWL_ext) and its case.

    A case's level is the storm tide at its RP plus its SLR x slr_fingerprint (m, model datum).
    """
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    from src.surge import calm_sea_levels

    with xr.open_dataset(surge_path, decode_times=False) as ds:
        sx, sy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
            ds["longitude"].values, ds["latitude"].values
        )
        calm = np.asarray(calm_sea_levels(ds), float)
        per_case = {
            label: np.asarray(lookup_storm_tide_at_rp(ds, rp, slr), float)
            for label, (rp, slr) in cases.items()
        }
    _, k = cKDTree(np.c_[sx, sy]).query(xy)
    labels = np.array(list(per_case))
    stack = np.vstack([v[k] for v in per_case.values()])
    return {
        "calm": calm[k],
        "swl_ext": stack.max(axis=0),
        "case": labels[stack.argmax(axis=0)],
        "cases": dict(zip(labels, stack)),
    }


def forcing_points(
    river_forcing: Path, rivers: gpd.GeoDataFrame, crs, net, extra_reaches: list[str]
) -> pd.DataFrame:
    """Return the active GloFAS crossings (model CRS x, y) plus a point at the upstream end of each extra reach."""
    from pyproj import Transformer
    from src.river_network import outflow_crossing_mask

    with xr.open_dataset(river_forcing, decode_times=False) as ds:
        active = ds["has_glofas"].values.astype(bool) & ~outflow_crossing_mask(
            ds, rivers
        )
        pts = pd.DataFrame(
            {
                "reach_id": [
                    normalize_reach_id(r) for r in ds["inside_reach_id"].values[active]
                ],
                "lon": ds["longitude"].values[active].astype(float),
                "lat": ds["latitude"].values[active].astype(float),
            }
        )
    pts["x"], pts["y"] = Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    ).transform(pts["lon"].to_numpy(), pts["lat"].to_numpy())
    pts["source"] = "forcing"
    extra = []
    for r in extra_reaches:
        r = normalize_reach_id(r)
        if r not in net.reach_nodes:
            raise ValueError(f"--extra-reach {r} is not in the river network")
        node = int(net.reach_nodes[r][0])
        extra.append(
            {
                "reach_id": r,
                "lon": np.nan,
                "lat": np.nan,
                "x": net.node_x[node],
                "y": net.node_y[node],
                "source": "extra",
            }
        )
    return pd.concat([pts, pd.DataFrame(extra)], ignore_index=True) if extra else pts


def check_basin(
    basin_id: str, out_dir: Path, opts: SimpleNamespace
) -> tuple[pd.DataFrame, list[dict], SimpleNamespace]:
    """Return one row per forcing point x (Q, SWL) pair, the profiles, and the network context for the map."""
    cfg = _config()
    test = cfg["testing"]["upstream_boundary_check"]
    frac = float(test["amplitude_threshold_fraction"])
    cases = extreme_cases(opts.stress_surge_rp_yr, opts.stress_slr_m)
    manning_n = float(cfg["river_processing"]["channel_manning_n"])
    bed, rivers, _forcing, _sea, crs, _paths = run_round0(basin_id, cfg, out_dir)
    net = bed.net
    frc = site_path("results_dir") / basin_id / "preprocessing_inputs" / "forcing"
    reach_q = {
        q: routed_discharge(rivers, frc / "river_forcing.nc", var)
        for q, var in DISCHARGES.items()
    }

    ends = [int(net.reach_nodes[r][-1]) for r in net.reach_ids if not net.downstream[r]]
    mouths = {int(m) for m in bed.mouths}
    sea = sea_levels_at(
        np.c_[net.node_x[ends], net.node_y[ends]], frc / "surge_forcing.nc", crs, cases
    )
    end_index = {e: i for i, e in enumerate(ends)}
    end_levels = {
        s: {
            e: float(sea[s][i]) if e in mouths else float(bed.level_m[e])
            for i, e in enumerate(ends)
        }
        for s in SEA_LEVELS
    }
    wl = {
        (q, s): forward_network(
            net, bed.bed_m, bed.width_m, reach_q[q], manning_n, end_levels[s]
        )
        for q in DISCHARGES
        for s in SEA_LEVELS
    }
    is_mouth = np.array([e in mouths for e in ends])
    for label, level in sea["cases"].items():
        a0 = (level - sea["calm"])[is_mouth]
        log.info(
            f"{label}: {a0.min():.2f}-{a0.max():.2f} m above calm at {is_mouth.sum()} sea mouth(s)"
        )
    log.info(f"SWL_ext set by: {sorted(map(str, set(sea['case'][is_mouth])))}")

    rows, profiles = [], []
    for p in forcing_points(
        frc / "river_forcing.nc", rivers, crs, net, opts.extra_reach
    ).itertuples():
        if p.reach_id not in net.reach_nodes:
            log.warning(
                f"Forcing reach {p.reach_id} has no nodes in the network: skipped"
            )
            continue
        ids, dist = path_nodes(
            net, mainstem_path(net.downstream, p.reach_id, reach_q["Q_bf"])
        )
        k_f = int(np.argmin(np.hypot(net.node_x[ids] - p.x, net.node_y[ids] - p.y)))
        ids, dist = ids[k_f:], dist[k_f:]
        x_f = float(dist[0])
        mouth = int(ids[-1])
        i_end = end_index[mouth]
        calm, swl_ext = float(sea["calm"][i_end]), float(sea["swl_ext"][i_end])
        surface = bed.level_m[ids]
        reference = wl[("Q_mean", "calm")][ids]
        h_mouth = float(bed.level_m[mouth] - bed.bed_m[mouth])
        bed_slope = (
            float(bed.bed_m[ids[0]] - bed.bed_m[mouth]) / x_f if x_f > 0 else np.nan
        )
        common = {
            "basin_id": basin_id,
            "reach_id": p.reach_id,
            "source": p.source,
            "lon": p.lon,
            "lat": p.lat,
            "x_f_km": x_f / 1000,
            "mouth_node": mouth,
            "mouth_is_sea": mouth in mouths,
            "sea_calm_m": calm,
            "swl_ext_m": swl_ext,
            "swl_ext_case": str(sea["case"][i_end]),
            "x_swl_km": first_above(dist, reference, swl_ext) / 1000,
            "x_high_gradient_km": first_above(
                dist, reference, calm + opts.high_gradient_elevation_m
            )
            / 1000,
            "mouth_depth_m": h_mouth,
            "bed_slope_mean": bed_slope,
            "l_b_km": h_mouth / bed_slope / 1000 if bed_slope > 0 else np.inf,
            "surface_at_f_m": float(surface[0]),
            "level_at_f_m": float(reference[0]),
            "above_swl_ext_m": float(reference[0]) - swl_ext,
            "above_msl_m": float(reference[0]) - calm,
            "high_gradient_rule": high_gradient_rule(
                float(reference[0]) - calm, opts.high_gradient_elevation_m
            ),
            "forward_minus_surface_m": float(wl[("Q_bf", "calm")][ids[0]] - surface[0]),
        }
        marks = {}
        for q in DISCHARGES:
            for s in SEA_LEVELS[1:]:
                a0 = float(sea[s][i_end]) - calm
                delta = (wl[(q, s)] - wl[(q, "calm")])[ids]
                threshold = max(opts.level_tol_m, frac * a0)
                x_inf = influence_length(dist, delta, threshold)
                ok, why = boundary_verdict(
                    x_f, float(delta[0]), x_inf, opts.level_tol_m, opts.margin_km * 1000
                )
                rows.append(
                    common
                    | {
                        "discharge": q,
                        "q_forcing_m3s": float(reach_q[q].get(p.reach_id, np.nan)),
                        "sea_level": s,
                        "a0_m": a0,
                        "threshold_m": threshold,
                        "delta_at_f_m": float(delta[0]),
                        "x_inf_km": x_inf / 1000,
                        "l_fit_km": efolding_length(
                            dist, delta, x_inf if np.isfinite(x_inf) else x_f + 1.0
                        )
                        / 1000,
                        "passes": ok,
                        "deciding": why,
                        "reference": (q, s) == REFERENCE,
                    }
                )
                if (q, s) == REFERENCE:
                    marks = {
                        "x_inf": x_inf,
                        "passes": ok,
                        "deciding": why,
                        "delta_f": float(delta[0]),
                    }
        profiles.append(
            {
                "point": p,
                "ids": ids,
                "dist": dist,
                "common": common,
                "marks": marks,
                "wl": {k: v[ids] for k, v in wl.items()},
                "bed": bed.bed_m[ids],
                "surface": surface,
                "fathom": bed.fathom_level_m[ids],
            }
        )
    ctx = SimpleNamespace(
        net=net,
        crs=crs,
        cases=cases,
        sea=sea,
        delta_ref=wl[("Q_mean", "swl_ext")] - wl[("Q_mean", "calm")],
        a0_max=float(np.max(sea["swl_ext"] - sea["calm"])) if len(ends) else 1.0,
        bed=bed,
        reach_q=reach_q,
        manning_n=manning_n,
        frac=frac,
        ends=ends,
        mouths=mouths,
        wl=wl,
        surge_path=frc / "surge_forcing.nc",
    )
    return pd.DataFrame(rows), profiles, ctx


def plot_profile(prof: dict, opts: SimpleNamespace, path: Path, title: str) -> None:
    x = prof["dist"] / 1000
    c, m = prof["common"], prof["marks"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True, height_ratios=[3, 2])
    ax[0].plot(x, prof["fathom"], ".", color="0.6", ms=3, label="FathomDEM node median")
    ax[0].plot(x, prof["surface"], "-", color="tab:blue", lw=2, label="rule 10 surface")
    ax[0].plot(x, prof["bed"], "-", color="k", lw=1, label="bed (rule 11 round 0)")
    colors = {"Q_mean": "tab:green", "Q_bf": "tab:purple"}
    styles = {"calm": "-", "swl_ext": ":"}
    for (q, s), eta in prof["wl"].items():
        ax[0].plot(
            x, eta, styles[s], color=colors[q], lw=1.2, label=f"steady GVF {q}, sea {s}"
        )
    ax[0].axhline(
        c["swl_ext_m"], color="tab:red", lw=0.9, label=f"SWL_ext {c['swl_ext_m']:.2f} m"
    )
    ax[0].axhline(
        c["sea_calm_m"],
        color="navy",
        lw=0.8,
        ls=":",
        label=f"calm sea {c['sea_calm_m']:.2f} m",
    )
    ax[0].axhline(
        c["sea_calm_m"] + opts.high_gradient_elevation_m,
        color="tab:brown",
        lw=0.8,
        ls="-.",
        label=f"MSL + {opts.high_gradient_elevation_m:g} m",
    )
    ax[0].set_ylabel("m (model datum)")
    ax[0].legend(fontsize=7, ncol=3)
    for q in DISCHARGES:
        for s in SEA_LEVELS[1:]:
            ax[1].plot(
                x,
                prof["wl"][(q, s)] - prof["wl"][(q, "calm")],
                styles[s],
                color=colors[q],
                lw=1.2,
                label=f"dWL {q}, {s}",
            )
    ax[1].axhline(
        opts.level_tol_m,
        color="k",
        lw=0.8,
        ls="--",
        label=f"level_tol {opts.level_tol_m:g} m",
    )
    ax[1].set_ylabel("dWL vs calm sea (m)")
    ax[1].set_xlabel("distance from the mouth along the mainstem (km)")
    for a in ax:
        a.axvline(c["x_f_km"], color="k", lw=1.2)
        if np.isfinite(m["x_inf"]):
            a.axvline(m["x_inf"] / 1000, color="tab:red", lw=1, ls="--")
        if np.isfinite(c["x_swl_km"]):
            a.axvline(c["x_swl_km"], color="tab:orange", lw=1, ls="--")
    ax[1].text(c["x_f_km"], ax[1].get_ylim()[1], " forcing point", va="top", fontsize=7)
    if np.isfinite(m["x_inf"]):
        ax[1].text(
            m["x_inf"] / 1000,
            ax[1].get_ylim()[1] * 0.85,
            " x_inf",
            va="top",
            fontsize=7,
            color="tab:red",
        )
    if np.isfinite(c["x_swl_km"]):
        ax[1].text(
            c["x_swl_km"],
            ax[1].get_ylim()[1] * 0.7,
            " x_SWL",
            va="top",
            fontsize=7,
            color="tab:orange",
        )
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].invert_xaxis()
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_map(
    basin_id: str, ctx: SimpleNamespace, profiles: list[dict], path: Path, title: str
) -> None:
    dom = site_path("results_dir") / basin_id / "preprocessing_inputs" / "domain"
    net = ctx.net
    fig, ax = plt.subplots(figsize=(10, 9))
    land = gpd.read_file(dom / f"{basin_id}_land_mask_on_grid.gpkg").to_crs(ctx.crs)
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    gpd.read_file(dom / f"{basin_id}_domain.gpkg").to_crs(ctx.crs).boundary.plot(
        ax=ax, color="k", lw=0.8, zorder=2
    )
    segs, vals = [], []
    for r in net.reach_ids:
        ids = np.asarray(net.reach_nodes[r])
        for a, b in zip(ids[:-1], ids[1:]):
            segs.append(
                [(net.node_x[a], net.node_y[a]), (net.node_x[b], net.node_y[b])]
            )
            vals.append(0.5 * (ctx.delta_ref[a] + ctx.delta_ref[b]))
    norm = Normalize(0.0, max(ctx.a0_max, 0.1))
    lc = LineCollection(segs, cmap="viridis_r", norm=norm, linewidths=2.5, zorder=3)
    lc.set_array(np.asarray(vals))
    ax.add_collection(lc)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis_r"),
        ax=ax,
        shrink=0.6,
        label="dWL at Q_mean, SWL_ext vs calm sea (m)",
    )
    for prof in profiles:
        p, m = prof["point"], prof["marks"]
        color = "#2ECC40" if m["passes"] else "#D94040"
        ax.plot(
            p.x,
            p.y,
            "o" if p.source == "forcing" else "s",
            color=color,
            ms=10,
            mec="white",
            zorder=6,
        )
        ax.annotate(
            f"{p.reach_id}\n{m['deciding']}, dWL {m['delta_f']:.2f} m",
            (p.x, p.y),
            fontsize=7,
            xytext=(6, 6),
            textcoords="offset points",
            zorder=7,
        )
        for key, mk, col in (("x_inf", "x", "tab:red"), ("x_swl", "^", "tab:orange")):
            x_m = m["x_inf"] if key == "x_inf" else prof["common"]["x_swl_km"] * 1000
            if np.isfinite(x_m):
                k = int(np.argmin(np.abs(prof["dist"] - x_m)))
                ax.plot(
                    net.node_x[prof["ids"][k]],
                    net.node_y[prof["ids"][k]],
                    mk,
                    color=col,
                    ms=10,
                    mew=2.5,
                    zorder=6,
                )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2ECC40",
            ms=9,
            label="forcing point: pass",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#D94040",
            ms=9,
            label="forcing point: fail",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="0.5",
            ms=9,
            label="extra boundary (--extra-reach)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color="tab:red",
            ls="none",
            ms=9,
            mew=2.5,
            label="x_inf (Q_mean, SWL_ext)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="tab:orange",
            ls="none",
            ms=9,
            label="x_SWL (calm level at Q_mean = SWL_ext)",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = script_parser(__doc__)
    add_basin_id_argument(parser)
    for key in OPTION_KEYS:
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=float,
            default=None,
            help="default: testing.upstream_boundary_check",
        )
    parser.add_argument(
        "--extra-reach",
        nargs="*",
        default=[],
        help="evaluate a boundary at these reaches' upstream ends",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figs" / "upstream_boundary_check"
    )
    args = parser.parse_args()
    for key, value in config_options().items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    basin_id = str(args.basin_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table, profiles, ctx = check_basin(basin_id, args.out_dir, args)
    tag = _git_hash()
    params = (
        f"level_tol {args.level_tol_m:g} m, margin {args.margin_km:g} km, SWL_ext = max over "
        f"{len(ctx.cases)} cases (stress RP{args.stress_surge_rp_yr:g} + {args.stress_slr_m:g} m SLR) ({tag})"
    )
    for prof in profiles:
        c, m = prof["common"], prof["marks"]
        verdict = "PASS" if m["passes"] else f"FAIL ({m['deciding']})"
        plot_profile(
            prof,
            args,
            args.out_dir / f"{basin_id}_profile_{c['reach_id']}.png",
            f"Basin {basin_id}, {c['source']} reach {c['reach_id']} at {c['x_f_km']:.1f} km: {verdict} at Q_mean, "
            f"SWL_ext; dWL {m['delta_f']:.3f} m, x_inf {m['x_inf'] / 1000:.1f} km\n{params}",
        )
    plot_map(
        basin_id,
        ctx,
        profiles,
        args.out_dir / f"{basin_id}_upstream_boundary_check.png",
        f"Basin {basin_id}: sea influence on the river level at Q_mean, SWL_ext\n{params}",
    )
    csv = args.out_dir / f"{basin_id}_forcing_points.csv"
    table.to_csv(csv, index=False)
    ref = table[table["reference"]]
    show = [
        "reach_id",
        "source",
        "x_f_km",
        "delta_at_f_m",
        "x_inf_km",
        "l_fit_km",
        "x_swl_km",
        "l_b_km",
        "above_swl_ext_m",
        "high_gradient_rule",
        "passes",
        "deciding",
    ]
    pd.set_option("display.width", 250)
    print(ref[show].round(3).to_string(index=False))
    print(f"Written: {csv}")


if __name__ == "__main__":
    main()
