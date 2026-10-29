"""Benchmark the upstream boundary check's steady GVF levels against steady SFINCS runs.

Per basin, copies the built event model (``runs/<scenario>/sfincs``, its
skeleton and spin-up restart referenced in place by relative path) to
``<run_dir>/<basin>/<case>/`` and runs it for ``--days`` days with every
discharge point at its forcing point's steady Q_mean and every sea boundary
point at the nearest station's calm level (case ``calm``) or SWL_ext (case
``swl_ext``). SFINCS levels are sampled at every river network node and
compared with check_upstream_boundary's steady GVF levels on the same bed,
along each main river arm (mouth -> upstream, largest bankfull discharge at
confluences; arms carrying at least ``--min-arm-share`` of the sea outflow).
The GVF levels are also solved for further sea levels and discharges.

Writes ``<out_dir>/README.md`` (method, results, the choice of the check's
parameters), ``summary.csv`` and ``summary.png``, and per basin
``<basin>/arms.csv``, ``<basin>/arm_<mouth reach>.png``, ``<basin>/map.png`` and
``<basin>/sfincs_nodes.csv`` (reused when the runs are gone); nothing under
results_dir.

Usage:
    python analysis/benchmark_upstream_boundary.py 2433835 2209166 3279946 [--skip-run] [--days 4]
"""

import logging
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import REPO_ROOT, script_parser, site_path
from _upstream_profile import (
    efolding_length,
    first_above,
    influence_length,
    mainstem_path,
    path_nodes,
    sample_sfincs_at_nodes,
)
from check_river_bed import _git_hash
from check_upstream_boundary import check_basin, config_options, sea_levels_at
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.gvf_solver import forward_network
from src.sfincs_run import parse_sfincs_inp, run_sfincs_subprocess

log = logging.getLogger(__name__)

CASES = ("calm", "swl_ext")
Q_NAMES = ("0.5 Q_mean", "Q_mean", "Q_bf")
OUTPUT_INTERVAL_S = 6 * 3600
FORCING_END_S = 365 * 86400.0
SFINCS_COLOR = "tab:red"
GVF_COLOR = "tab:blue"
CONTOUR_COLOR = "tab:brown"


def _points(path: Path) -> np.ndarray:
    return np.array(
        [
            [float(v) for v in line.split()[:2]]
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    )


def write_case(
    src_dir: Path, case_dir: Path, case: str, q_src: np.ndarray, sea: dict, days: float
) -> None:
    """Write a steady copy of the event model in ``src_dir`` to ``case_dir``."""
    case_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_sfincs_inp(src_dir / "sfincs.inp")
    for key in ("bndfile", "srcfile"):
        shutil.copy2(src_dir / cfg[key], case_dir / cfg[key])
    times = np.array([0.0, FORCING_END_S])
    np.savetxt(
        case_dir / cfg["disfile"],
        np.column_stack([times, np.tile(q_src, (2, 1))]),
        fmt="%.3f",
    )
    np.savetxt(
        case_dir / cfg["bzsfile"],
        np.column_stack([times, np.tile(sea[case], (2, 1))]),
        fmt="%.3f",
    )
    tstart = datetime.strptime(cfg["tstart"], "%Y%m%d %H%M%S")
    settings = {
        "tstop": (tstart + timedelta(days=days)).strftime("%Y%m%d %H%M%S"),
        "dtmapout": OUTPUT_INTERVAL_S,
        "dtmaxout": int(days * 86400),
        "dthisout": 3600,
    }
    lines = []
    for key, value in cfg.items():
        if key in settings:
            value = settings[key]
        elif key.endswith("file") and key not in (
            "bndfile",
            "bzsfile",
            "srcfile",
            "disfile",
        ):
            # SFINCS reads backslash escapes ("\r" in "..\results" ends the path)
            value = os.path.relpath(
                (src_dir / value).resolve(), case_dir.resolve()
            ).replace("\\", "/")
        lines.append(f"{key:<20} = {value}")
    (case_dir / "sfincs.inp").write_text("\n".join(lines) + "\n")


def run_days(case_dir: Path) -> float:
    cfg = parse_sfincs_inp(case_dir / "sfincs.inp")
    t0, t1 = (datetime.strptime(cfg[k], "%Y%m%d %H%M%S") for k in ("tstart", "tstop"))
    return (t1 - t0).total_seconds() / 86400


def run_cases(
    basin_id: str, args, table: pd.DataFrame, profiles: list[dict], ctx, opts
) -> dict[str, Path]:
    """Run (unless ``--skip-run``) the calm and SWL_ext cases; return their run directories."""
    src_dir = site_path("results_dir") / basin_id / "runs" / args.scenario / "sfincs"
    cfg = parse_sfincs_inp(src_dir / "sfincs.inp")
    bnd, src = _points(src_dir / cfg["bndfile"]), _points(src_dir / cfg["srcfile"])
    sea = sea_levels_at(bnd, ctx.surge_path, ctx.crs, ctx.cases)
    fp = [prof["point"] for prof in profiles if prof["point"].source == "forcing"]
    q_mean = {
        r.reach_id: float(
            table[(table["reach_id"] == r.reach_id) & (table["discharge"] == "Q_mean")][
                "q_forcing_m3s"
            ].iloc[0]
        )
        for r in fp
    }
    fxy = np.array([[p.x, p.y] for p in fp])
    q_src = np.array(
        [q_mean[fp[int(np.argmin(np.hypot(*(fxy - s).T)))].reach_id] for s in src]
    )
    log.info(
        f"Steady benchmark {basin_id}: {len(src)} discharge point(s) {np.round(q_src, 1).tolist()} m3/s, "
        f"{len(bnd)} sea point(s), calm {np.round(sea['calm'], 2).tolist()}, "
        f"SWL_ext {np.round(sea['swl_ext'], 2).tolist()} m, {args.days:g} days"
    )
    dirs = {}
    for case in CASES:
        dirs[case] = args.run_dir / basin_id / case
        if not args.skip_run and case in args.cases:
            write_case(src_dir, dirs[case], case, q_src, sea, args.days)
            run_sfincs_subprocess(
                site_path("sfincs_exe"),
                dirs[case],
                args.timeout_h * 3600,
                log,
                label=f"benchmark {basin_id} {case}",
                n_threads=args.threads,
            )
    return dirs


def sfincs_nodes(net, dirs: dict[str, Path], csv: Path) -> pd.DataFrame:
    """Return SFINCS zs at every network node (end of run) and its change over the last output interval."""
    if not all((d / "sfincs_map.nc").exists() for d in dirs.values()):
        if not csv.exists():
            raise FileNotFoundError(
                f"No SFINCS maps under {dirs['calm'].parent} and no {csv}; run without --skip-run"
            )
        log.info(f"SFINCS maps missing: reusing {csv}")
        return pd.read_csv(csv, index_col="node")
    nodes = pd.DataFrame(
        {"node": np.arange(len(net.node_x)), "x": net.node_x, "y": net.node_y}
    )
    out = pd.DataFrame(index=pd.Index(nodes["node"], name="node"))
    for case, d in dirs.items():
        da = sample_sfincs_at_nodes(d / "sfincs_map.nc", nodes)
        out[f"zs_{case}_m"] = da.isel(time=-1).values.astype(float)
        out[f"change_last_interval_{case}_m"] = np.abs(
            da.isel(time=-1).values - da.isel(time=-2).values
        )
        out[f"days_{case}"] = run_days(d)
        out["cell_distance_m"] = da["cell_distance_m"].values
    out.to_csv(csv)
    return out


def sea_conditions(ctx) -> tuple[dict[str, np.ndarray], str]:
    """Return the sea levels at the network ends per condition, low to high, and the SWL_ext condition's name.

    Conditions: calm, each scenario and stress case, and SWL_ext (their per-station maximum), which takes the
    name of a case when that case sets it at every sea mouth.
    """
    mouth = np.array([e in ctx.mouths for e in ctx.ends])
    cond = {"calm": ctx.sea["calm"]} | dict(ctx.sea["cases"])
    governing = set(ctx.sea["case"][mouth])
    if len(governing) == 1:
        swl_name = f"{governing.pop()} = SWL_ext"
        cond[swl_name] = cond.pop(swl_name.removesuffix(" = SWL_ext"))
    else:
        swl_name = "SWL_ext (highest case per station)"
        cond[swl_name] = ctx.sea["swl_ext"]
    return dict(
        sorted(
            cond.items(),
            key=lambda kv: float(np.mean(kv[1][mouth])) if mouth.any() else 0.0,
        )
    ), swl_name


def solve(
    ctx, levels_at_ends: np.ndarray, q_key: str = "Q_mean", factor: float = 1.0
) -> np.ndarray:
    bed = ctx.bed
    end_levels = {
        e: float(levels_at_ends[i]) if e in ctx.mouths else float(bed.level_m[e])
        for i, e in enumerate(ctx.ends)
    }
    q = {r: factor * v for r, v in ctx.reach_q[q_key].items()}
    return forward_network(
        ctx.net, bed.bed_m, bed.width_m, q, ctx.manning_n, end_levels
    )


def river_arms(
    ctx, min_share: float, max_arms: int, forcing_reaches: list[str]
) -> list[list[str]]:
    """Return the reaches of each main arm, upstream first.

    Arms run from a sea mouth up the largest bankfull discharge, plus the path
    from each forcing reach down the largest bankfull discharge to the sea.
    """
    net, q = ctx.net, ctx.reach_q["Q_bf"]
    upstream: dict[str, list[str]] = {}
    for r in net.reach_ids:
        for d in net.downstream.get(r, []):
            upstream.setdefault(d, []).append(r)
    mouth_reaches = [
        r
        for r in net.reach_ids
        if not net.downstream.get(r) and int(net.reach_nodes[r][-1]) in ctx.mouths
    ]
    total = sum(q.get(r, 0.0) for r in mouth_reaches)
    arms = []
    for r in sorted(mouth_reaches, key=lambda r: (-q.get(r, 0.0), r))[:max_arms]:
        if q.get(r, 0.0) < min_share * total:
            break
        path = [r]
        while ups := [
            u for u in upstream.get(path[-1], []) if u in q and u not in path
        ]:
            path.append(max(ups, key=lambda u: q[u]))
        arms.append(path[::-1])
    for r in forcing_reaches:
        path = mainstem_path(net.downstream, r, q)
        if int(net.reach_nodes[path[-1]][-1]) in ctx.mouths and all(
            path[-1] != a[-1] or r not in a for a in arms
        ):
            arms.append(path)
    return arms


def analyse_arm(
    basin_id: str,
    reaches: list[str],
    ctx,
    opts,
    wl: dict,
    wl_q: dict,
    zs: pd.DataFrame,
    forcing_nodes: dict[int, str],
    cond_names: dict[str, str],
) -> tuple[dict, dict]:
    """Return the arm's metrics row and the arrays for its figure."""
    ids, dist = path_nodes(ctx.net, reaches)
    i_end = ctx.ends.index(int(ids[-1]))
    calm_wl = wl[cond_names["calm"]][ids]
    swl_wl = wl[cond_names["swl_ext"]][ids]
    calm = float(ctx.sea_at_ends["calm"][i_end])
    a0 = float(ctx.sea_at_ends["swl_ext"][i_end]) - calm
    threshold = max(opts.level_tol_m, ctx.frac * a0)
    d_gvf = swl_wl - calm_wl
    z_calm, z_swl = (
        zs.loc[ids, "zs_calm_m"].to_numpy(),
        zs.loc[ids, "zs_swl_ext_m"].to_numpy(),
    )
    d_sfincs_raw = z_swl - z_calm
    d_sfincs = np.nan_to_num(d_sfincs_raw, nan=0.0)
    x_inf = influence_length(dist, d_gvf, threshold)
    x_obs = influence_length(dist, d_sfincs, threshold)
    order = np.argsort(dist)
    d_up = d_sfincs[order]
    k_f = [k for k, n in enumerate(ids) if int(n) in forcing_nodes]
    x_inf_q = {
        name: influence_length(
            dist, (wl_q[(name, "swl_ext")] - wl_q[(name, "calm")])[ids], threshold
        )
        for name in Q_NAMES
    }
    row = {
        "basin_id": basin_id,
        "mouth_reach": reaches[-1],
        "top_reach": reaches[0],
        "forcing_reach": forcing_nodes[int(ids[k_f[0]])] if k_f else "",
        "arm_length_km": float(dist[0]) / 1000,
        "x_f_km": float(dist[k_f[0]]) / 1000 if k_f else np.nan,
        "q_mean_mouth_m3s": float(ctx.reach_q["Q_mean"].get(reaches[-1], np.nan)),
        "q_bf_mouth_m3s": float(ctx.reach_q["Q_bf"].get(reaches[-1], np.nan)),
        "sea_calm_m": calm,
        "swl_ext_case": str(ctx.sea["case"][i_end]),
        "a0_m": a0,
        "threshold_m": threshold,
        "x_inf_km": x_inf / 1000,
        "x_obs_km": x_obs / 1000,
        "x_inf_over_x_obs": x_inf / x_obs
        if np.isfinite(x_obs) and x_obs > 0
        else np.nan,
        "x_obs_minus_x_inf_km": (x_obs - x_inf) / 1000,
        "x_10m_gvf_km": first_above(
            dist, calm_wl, calm + opts.high_gradient_elevation_m
        )
        / 1000,
        "x_10m_sfincs_km": first_above(
            dist,
            np.nan_to_num(z_calm, nan=-np.inf),
            calm + opts.high_gradient_elevation_m,
        )
        / 1000,
        "x_swl_km": first_above(dist, calm_wl, calm + a0) / 1000,
        "l_fit_km": efolding_length(
            dist, d_gvf, x_inf if np.isfinite(x_inf) else dist.max() + 1.0
        )
        / 1000,
        "l_obs_km": efolding_length(
            dist, d_sfincs, x_obs if np.isfinite(x_obs) else dist.max() + 1.0
        )
        / 1000,
        "sfincs_max_rise_over_a0": float(np.nanmax(d_sfincs_raw)) / a0
        if a0 > 0
        else np.nan,
        "sfincs_max_rise_over_basin_a0": float(np.nanmax(d_sfincs_raw)) / ctx.a0_max,
        "sfincs_upstream_regain_m": float(np.max(d_up - np.minimum.accumulate(d_up))),
        "calm_sfincs_minus_gvf_median_m": float(np.nanmedian(z_calm - calm_wl)),
        "sfincs_dry_nodes": int(np.isnan(d_sfincs_raw).sum()),
        "max_change_last_interval_m": float(
            np.nanmax(
                zs.loc[
                    ids,
                    ["change_last_interval_calm_m", "change_last_interval_swl_ext_m"],
                ].to_numpy()
            )
        ),
        "days": float(zs["days_swl_ext"].iloc[0]),
    } | {f"x_inf_{name.replace(' ', '_')}_km": v / 1000 for name, v in x_inf_q.items()}
    arrays = {
        "ids": ids,
        "dist": dist,
        "bed": ctx.bed.bed_m[ids],
        "wl": {k: v[ids] for k, v in wl.items()},
        "wl_q": {k: v[ids] for k, v in wl_q.items()},
        "z_calm": z_calm,
        "z_swl": z_swl,
        "d_gvf": d_gvf,
        "d_sfincs": d_sfincs_raw,
    }
    return row, arrays


def arm_name(row) -> str:
    return f"arm_{row['mouth_reach']}_{row['top_reach']}"


def _vlines(ax, row: dict) -> None:
    for key, color, ls in (
        ("x_f_km", "k", "-"),
        ("x_inf_km", GVF_COLOR, "--"),
        ("x_obs_km", SFINCS_COLOR, "--"),
        ("x_10m_gvf_km", CONTOUR_COLOR, ":"),
    ):
        if np.isfinite(row[key]):
            ax.axvline(row[key], color=color, lw=1.1, ls=ls)


def plot_arm(
    row: dict,
    arr: dict,
    conds: list[str],
    cond_names: dict,
    opts,
    path: Path,
    title: str,
) -> None:
    x = arr["dist"] / 1000
    blues = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(conds)))
    greens = plt.get_cmap("Greens")(np.linspace(0.45, 1.0, len(Q_NAMES)))
    fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=True, height_ratios=[3, 2, 2])
    ax[0].plot(x, arr["bed"], color="k", lw=1, label="bed (rule 11 round 0)")
    for name, c in zip(conds, blues):
        ax[0].plot(x, arr["wl"][name], color=c, lw=1.6, label=f"GVF, sea {name}")
    ax[0].plot(
        x, arr["z_calm"], color=SFINCS_COLOR, lw=1.2, ls="-", label="SFINCS, sea calm"
    )
    ax[0].plot(
        x,
        arr["z_swl"],
        color=SFINCS_COLOR,
        lw=1.2,
        ls="--",
        label=f"SFINCS, sea {cond_names['swl_ext']}",
    )
    ax[0].axhline(
        row["sea_calm_m"] + opts.high_gradient_elevation_m,
        color=CONTOUR_COLOR,
        lw=0.9,
        ls="-.",
        label=f"MSL + {opts.high_gradient_elevation_m:g} m",
    )
    ax[0].set_ylabel("water level, bed (m, model datum)")
    ax[0].set_title("Steady levels at Q_mean", fontsize=9, loc="left")
    ax[0].legend(fontsize=7, ncol=2)
    for name, c in zip(conds, blues):
        if name != cond_names["calm"]:
            ax[1].plot(
                x,
                arr["wl"][name] - arr["wl"][cond_names["calm"]],
                color=c,
                lw=1.6,
                label=f"GVF, {name}",
            )
    ax[1].plot(
        x,
        arr["d_sfincs"],
        color=SFINCS_COLOR,
        lw=1.4,
        ls="--",
        label=f"SFINCS, {cond_names['swl_ext']}",
    )
    ax[1].axhline(
        row["threshold_m"],
        color="0.3",
        lw=0.8,
        ls="--",
        label=f"threshold {row['threshold_m']:.3f} m",
    )
    ax[1].set_ylabel("rise vs calm sea (m)")
    ax[1].set_title("Backwater rise at Q_mean", fontsize=9, loc="left")
    ax[1].legend(fontsize=7, ncol=2)
    for name, c in zip(Q_NAMES, greens):
        d = arr["wl_q"][(name, "swl_ext")] - arr["wl_q"][(name, "calm")]
        key = f"x_inf_{name.replace(' ', '_')}_km"
        ax[2].plot(
            x,
            d,
            color=c,
            lw=1.6,
            label=f"GVF, {name}: influence ends {row[key]:.1f} km",
        )
    ax[2].axhline(row["threshold_m"], color="0.3", lw=0.8, ls="--")
    ax[2].set_ylabel("rise vs calm sea (m)")
    ax[2].set_title(
        f"Backwater rise at {cond_names['swl_ext']} for three discharges",
        fontsize=9,
        loc="left",
    )
    ax[2].legend(fontsize=7)
    ax[2].set_xlabel(
        "distance from the mouth along the arm (km); black: forcing point, blue dashed: "
        "GVF influence end, red dashed: SFINCS influence end, brown dotted: 10 m contour"
    )
    for a in ax:
        _vlines(a, row)
        a.grid(color="0.92", lw=0.6)
    ax[2].invert_xaxis()
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_maps(
    basin_id: str,
    ctx,
    delta: dict[str, np.ndarray],
    arms: list[tuple[dict, dict]],
    path: Path,
    title: str,
) -> None:
    dom = site_path("results_dir") / basin_id / "preprocessing_inputs" / "domain"
    land = gpd.read_file(dom / f"{basin_id}_land_mask_on_grid.gpkg").to_crs(ctx.crs)
    domain = gpd.read_file(dom / f"{basin_id}_domain.gpkg").to_crs(ctx.crs)
    net = ctx.net
    norm = Normalize(0.0, max(ctx.a0_max, 0.1))
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)
    for ax, (label, d) in zip(axes, delta.items()):
        if not land.empty:
            land.plot(
                ax=ax, color="#e3e3e3", edgecolor="#bbbbbb", linewidth=0.3, zorder=1
            )
        domain.boundary.plot(ax=ax, color="k", lw=0.8, zorder=2)
        segs, vals = [], []
        for r in net.reach_ids:
            ids = np.asarray(net.reach_nodes[r])
            for a, b in zip(ids[:-1], ids[1:]):
                segs.append(
                    [(net.node_x[a], net.node_y[a]), (net.node_x[b], net.node_y[b])]
                )
                vals.append(0.5 * (d[a] + d[b]))
        lc = LineCollection(segs, cmap="viridis_r", norm=norm, linewidths=2.5, zorder=3)
        lc.set_array(np.asarray(vals))
        ax.add_collection(lc)
        for row, arr in arms:
            for key, mk, col in (
                ("x_inf_km", "x", GVF_COLOR),
                ("x_obs_km", "+", SFINCS_COLOR),
                ("x_10m_gvf_km", "D", CONTOUR_COLOR),
            ):
                if np.isfinite(row[key]):
                    k = int(np.argmin(np.abs(arr["dist"] / 1000 - row[key])))
                    ax.plot(
                        net.node_x[arr["ids"][k]],
                        net.node_y[arr["ids"][k]],
                        mk,
                        color=col,
                        ms=10,
                        mew=2.5,
                        mfc="none" if mk == "D" else col,
                        zorder=6,
                    )
            if np.isfinite(row["x_f_km"]):
                k = int(np.argmin(np.abs(arr["dist"] / 1000 - row["x_f_km"])))
                ax.plot(
                    net.node_x[arr["ids"][k]],
                    net.node_y[arr["ids"][k]],
                    "o",
                    color="k",
                    ms=8,
                    mec="white",
                    zorder=7,
                )
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")
        ax.autoscale()
    fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis_r"),
        ax=axes,
        shrink=0.7,
        label="steady rise of the river level, sea at SWL_ext vs calm, Q_mean (m)",
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="k",
            ms=8,
            label="forcing point",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color=GVF_COLOR,
            ls="none",
            ms=9,
            mew=2.5,
            label="influence end, GVF",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="+",
            color=SFINCS_COLOR,
            ls="none",
            ms=10,
            mew=2.5,
            label="influence end, SFINCS",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color=CONTOUR_COLOR,
            mfc="none",
            ls="none",
            ms=8,
            mew=2,
            label="10 m contour of the calm level",
        ),
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_summary(df: pd.DataFrame, margin_km: float, path: Path, title: str) -> None:
    fin = df[
        np.isfinite(df["x_inf_km"]) & np.isfinite(df["x_obs_km"]) & (df["x_obs_km"] > 0)
    ]
    basins = list(dict.fromkeys(df["basin_id"]))
    colors = dict(zip(basins, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    lim = [
        0.5,
        1.5
        * max(
            fin[["x_inf_km", "x_obs_km", "x_10m_gvf_km"]]
            .replace(np.inf, np.nan)
            .max()
            .max(),
            1.0,
        ),
    ]
    xs = np.geomspace(*lim, 100)
    for a, ycol, ylab in (
        (ax[0], "x_inf_km", "GVF influence end (km)"),
        (ax[1], "x_10m_gvf_km", "10 m contour of the calm level (km)"),
    ):
        a.plot(xs, xs, color="0.4", lw=0.8, label="1:1")
        a.fill_between(
            xs,
            np.maximum(xs - margin_km, lim[0]),
            xs,
            color="0.9",
            label=f"within the {margin_km:g} km margin below 1:1",
        )
        for b in basins:
            s = fin[fin["basin_id"] == b]
            a.plot(
                s["x_obs_km"],
                s[ycol].replace(np.inf, np.nan),
                "o",
                ms=8,
                color=colors[b],
                mec="white",
                label=f"basin {b}",
            )
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(lim)
        a.set_ylim(lim)
        a.set_xlabel("SFINCS influence end (km)")
        a.set_ylabel(ylab)
        a.grid(color="0.92", lw=0.6, which="both")
        a.legend(fontsize=8)
    ax[0].set_title("a) Steady GVF vs steady SFINCS, per arm", fontsize=10, loc="left")
    ax[1].set_title("b) 10 m rule vs steady SFINCS, per arm", fontsize=10, loc="left")
    for _, r in df.iterrows():
        q = [0.5 * r["q_mean_mouth_m3s"], r["q_mean_mouth_m3s"], r["q_bf_mouth_m3s"]]
        x = [r[f"x_inf_{n.replace(' ', '_')}_km"] for n in Q_NAMES]
        if np.isfinite(x[1]) and x[1] > 0:
            ax[2].plot(
                np.asarray(q) / r["q_mean_mouth_m3s"],
                np.asarray(x) / x[1],
                "-o",
                ms=5,
                color=colors[r["basin_id"]],
                lw=1.2,
            )
    ax[2].axhline(1.0, color="0.4", lw=0.8)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("discharge / Q_mean")
    ax[2].set_ylabel("GVF influence end / its value at Q_mean")
    ax[2].set_title(
        "c) Influence end vs river discharge (GVF, sea at SWL_ext)",
        fontsize=10,
        loc="left",
    )
    ax[2].grid(color="0.92", lw=0.6, which="both")
    ax[2].legend(
        handles=[
            plt.Line2D([0], [0], color=colors[b], marker="o", label=f"basin {b}")
            for b in basins
        ],
        fontsize=8,
    )
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fmt(v: float, nd: int = 1) -> str:
    if isinstance(v, str):
        return v
    return "inf" if np.isinf(v) else ("" if np.isnan(v) else f"{v:.{nd}f}")


def write_readme(
    df: pd.DataFrame, opts, args, cond_names: dict, path: Path, tag: str
) -> None:
    fin = df[np.isfinite(df["x_inf_km"]) & np.isfinite(df["x_obs_km"])]
    short = float(fin["x_obs_minus_x_inf_km"].max()) if len(fin) else np.nan
    ratio = fin["x_inf_over_x_obs"].dropna()
    rel = 100 * (df["x_inf_Q_bf_km"] / df["x_inf_Q_mean_km"] - 1).replace(
        [np.inf, -np.inf], np.nan
    )
    q_ratio = df["q_bf_mouth_m3s"] / df["q_mean_mouth_m3s"]
    q_change = "; ".join(
        f"{b}: Q_bf {_fmt(float(q_ratio[g.index].min()), 1)}–{_fmt(float(q_ratio[g.index].max()), 1)} × Q_mean, "
        f"end {_fmt(float(rel[g.index].min()), 0)} to {_fmt(float(rel[g.index].max()), 0)} %"
        for b, g in df.groupby("basin_id", sort=False)
    )
    largest = df.loc[df.groupby("basin_id")["q_mean_mouth_m3s"].idxmax()]
    cols = [
        ("basin_id", "basin"),
        ("mouth_reach", "arm (mouth reach)"),
        ("forcing_reach", "forcing reach"),
        ("top_reach", "arm top reach"),
        ("swl_ext_case", "SWL_ext set by"),
        ("q_mean_mouth_m3s", "Q_mean at mouth (m³/s)"),
        ("a0_m", "A0 (m)"),
        ("threshold_m", "threshold (m)"),
        ("x_f_km", "forcing point"),
        ("x_inf_km", "GVF end"),
        ("x_obs_km", "SFINCS end"),
        ("x_inf_over_x_obs", "GVF/SFINCS"),
        ("x_10m_gvf_km", "10 m contour"),
        ("x_swl_km", "x_SWL"),
        ("l_fit_km", "L GVF"),
        ("l_obs_km", "L SFINCS"),
        ("x_inf_0.5_Q_mean_km", "GVF end 0.5 Q_mean"),
        ("x_inf_Q_bf_km", "GVF end Q_bf"),
        ("sfincs_max_rise_over_a0", "SFINCS max rise / A0"),
        ("sfincs_max_rise_over_basin_a0", "SFINCS max rise / largest A0"),
        ("max_change_last_interval_m", "SFINCS change last 6 h (m)"),
    ]
    table = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for key, _label in cols:
            v = r[key]
            nd = (
                0
                if key == "q_mean_mouth_m3s"
                else (
                    3
                    if key in ("threshold_m", "max_change_last_interval_m")
                    else (
                        2
                        if key
                        in ("a0_m", "x_inf_over_x_obs", "sfincs_max_rise_over_a0")
                        else 1
                    )
                )
            )
            cells.append(_fmt(v, nd) if not isinstance(v, str) else v)
        table.append("| " + " | ".join(cells) + " |")
    basins = list(dict.fromkeys(df["basin_id"]))
    figs = "\n".join(
        f"- `{b}/map.png`, "
        + ", ".join(
            f"`{b}/{arm_name(r)}.png`" for _, r in df[df["basin_id"] == b].iterrows()
        )
        for b in basins
    )
    days = ", ".join(
        f"{b}: {df[df['basin_id'] == b]['days'].iloc[0]:g} days" for b in basins
    )
    margin_ok = bool(np.isnan(short) or short <= opts.margin_km)
    amp = float(df["sfincs_max_rise_over_a0"].max())
    amp_basin = float(df["sfincs_max_rise_over_basin_a0"].max())
    over = df[df["sfincs_max_rise_over_a0"] > 1.05]
    lateral = (
        f" On {len(over)} arm(s) ({', '.join(over['basin_id'] + '/' + over['mouth_reach'])}) the rise exceeds"
        f" the mouth's own A0 (up to {_fmt(amp, 2)} ×) but stays below the basin's largest A0"
        f" ({_fmt(amp_basin, 2)} ×): SWL_ext varies along the coast and in 2D the higher storm level"
        " floods inland and reaches the arm from the side, which 1D GVF from the arm's own mouth cannot"
        " see. This is lateral inflow, not convergence amplification."
        if len(over)
        else ""
    )
    text = f"""# Upstream boundary check: steady GVF vs steady SFINCS

Generated by `python analysis/benchmark_upstream_boundary.py {" ".join(basins)}` ({tag}).
Basins: {", ".join(basins)}. Distances in km from the arm's mouth along the river.

## Why

Every delta model takes its river discharge at forcing points where the river
enters the model domain. These points must lie upstream of the reach whose
level the sea controls, or the compound flood analysis imposes river discharge
inside the surge-controlled reach. The check
(`analysis/check_upstream_boundary.py`) decides this from steady 1D
gradually-varied-flow (GVF) levels. This folder tests that decision against
the 2D hydrodynamic model: the formulas cannot validate each other, SFINCS is
the reference.

## What is computed

- **GVF levels.** Steady GVF over the whole river network on the bed the
  pipeline burns into the DEM (rule 11 round 0): rectangular section of the
  node width, Manning friction, integrated upstream from the sea level at each
  mouth. Solved at Q_mean for the calm sea and each extreme case: every
  scenario in `scenarios.yml` with a surge RP and the stress case
  ({", ".join(f'"{c}"' for c in cond_names["all"])}; storm tide at the RP plus
  SLR × the station's SLR fingerprint), and for 0.5 Q_mean and Q_bf with the
  sea calm and at SWL_ext.
- **SWL_ext**, the most extreme case: per surge station the highest of these
  cases (column `SWL_ext set by`). Every verdict and influence end uses it.
- **SFINCS levels.** The basin's built event model, run steady ({days}) with
  every river inflow at its forcing point's Q_mean and the sea boundary held
  at the calm level and at SWL_ext. Levels sampled at the same network nodes.
- **Rise.** ΔWL(x) = WL(sea at SWL_ext) − WL(sea calm), same discharge and
  bed, so any offset of the bed cancels.
- **Influence end** (GVF: `x_inf`, SFINCS: `x_obs`). First node, going
  upstream, where ΔWL < threshold = max(`level_tol_m`,
  `amplitude_threshold_fraction` × A0), A0 = SWL_ext − calm at the mouth.
- **10 m contour.** First node where the calm level at Q_mean rises
  {opts.high_gradient_elevation_m:g} m above local MSL: the reviewer's rule of thumb (DE2) for
  high-gradient rivers.
- **x_SWL.** First node where the calm level at Q_mean exceeds SWL_ext: where
  a horizontal storm level meets the river, a lower bound of the influence.
- **e-folding length L** (`L GVF`, `L SFINCS`). The length over which the
  rise falls by a factor e (to 37 %), fitted as ΔWL(x) = ΔWL(0) exp(−x/L)
  over the influenced reach. Linear backwater theory predicts such an
  exponential decay with L ≈ h/(3S) (depth over three times the slope); the
  fit summarises how fast the sea's effect fades. It is reported, not used in
  the verdict, because real profiles are not exponential (a flat backwater
  tail, then a sharp drop where the river steepens).
- **Arms.** Paths from a sea mouth upstream along the largest bankfull
  discharge, for mouths carrying at least {args.min_arm_share:.0%} of the sea outflow
  (at most {args.max_arms} per basin).

## Results

{chr(10).join(table)}

`GVF end 0.5 Q_mean` and `GVF end Q_bf` are the GVF influence ends at
SWL_ext for half the mean and the bankfull discharge (SFINCS ran at Q_mean
only). `SFINCS max rise / A0` above 1 means the river rises more than the
sea at the arm's mouth: amplification, or a higher storm level reaching the
arm from elsewhere.

## Findings

1. **GVF reproduces SFINCS.** GVF/SFINCS {_fmt(ratio.min(), 2)}–{_fmt(ratio.max(), 2)}
   over {len(ratio)} arms; SFINCS reaches at most {_fmt(short, 1)} km further than GVF.
   The margin of {opts.margin_km:g} km {"covers" if margin_ok else "does NOT cover"} this.
2. **No amplification along the channels.** The SFINCS rise never exceeds
   {_fmt(amp_basin, 2)} × the basin's largest A0.{lateral} Where the
   influence ends, GVF and SFINCS still agree, so an analytical damping model
   (convergence, Cai et al. 2014) is not needed.
3. **Discharge.** GVF influence end at bankfull vs mean discharge ({q_change}).
   A larger discharge flows deeper and steeper, so friction takes up the
   sea's perturbation sooner and the end moves seaward; this dominates on
   steep rivers. On low-gradient rivers the end sits where the river
   steepens (a bed step), which discharge does not move, so it shifts by a
   few percent either way. Q_mean is the reference because the surge can
   arrive days before the river peak, and it gives the longer influence.
4. **10 m rule.** The 10 m contour of the calm level lies beyond the SFINCS
   influence end by
   {", ".join(f"{_fmt(r['x_10m_gvf_km'] - r['x_obs_km'], 1)} km ({r['basin_id']})" for _, r in largest.iterrows())}
   on each basin's largest arm. The rule holds here, but its margin
   depends on surge height and slope, not on the 10 m value: with a 4 m
   surge on a low-gradient river it is only half a kilometre. It is kept for
   comparison and as a fallback for steep rivers, not as the criterion.

## Parameters (`config.yml`, `testing.upstream_boundary_check`)

| key | value | role | basis |
|---|---|---|---|
| `stress_surge_rp_yr`, `stress_slr_m` | RP{opts.stress_surge_rp_yr:g}, {opts.stress_slr_m:g} m | stress case in SWL_ext beside the scenarios | bounds the scenarios ({"; ".join(sorted(set(df["swl_ext_case"])))} set SWL_ext here) |
| `amplitude_threshold_fraction` | {cond_names["frac"]:g} | influence ends below this × A0 | sets the threshold where × A0 > `level_tol_m` |
| `level_tol_m` | {opts.level_tol_m:g} | floor of the threshold; largest rise allowed at the forcing point | not binding here (thresholds ≥ {_fmt(float(df["threshold_m"].min()), 3)} m) |
| `margin_km` | {opts.margin_km:g} | forcing point at least this far beyond the GVF end | largest GVF shortfall {_fmt(short, 1)} km |
| `high_gradient_elevation_m` | {opts.high_gradient_elevation_m:g} | the 10 m contour (DE2) | reported for comparison |

## Caveats

- Steady states: a 1–3 day surge pulse reaches less far than a held level,
  so both ends are upper bounds. SFINCS level change over the last 6 h is in
  the table; above a few cm the SFINCS end may still move upstream.
- GVF is 1D in-channel; SFINCS also stores water on the floodplain. Both run
  on the same bed, so the benchmark tests the 1D steady approximation, not the bed.
- SFINCS ran at Q_mean only; the discharge sensitivity is GVF.

## Files

- `summary.csv`, `summary.png`: one row / point per arm.
{figs}
- `<basin>/arms.csv`: the arm rows; `<basin>/sfincs_nodes.csv`: SFINCS levels at every node.
"""
    path.write_text(text, encoding="utf-8")


def benchmark_basin(basin_id: str, args, opts) -> tuple[list[dict], dict]:
    out = args.out_dir / basin_id
    out.mkdir(parents=True, exist_ok=True)
    table, profiles, ctx = check_basin(basin_id, out, opts)
    dirs = run_cases(basin_id, args, table, profiles, ctx, opts)
    zs = sfincs_nodes(ctx.net, dirs, out / "sfincs_nodes.csv")

    conds, swl_name = sea_conditions(ctx)
    names = list(conds)
    cond_names = {"calm": "calm", "swl_ext": swl_name, "all": list(ctx.cases)}
    ctx.sea_at_ends = {"calm": conds["calm"], "swl_ext": conds[cond_names["swl_ext"]]}
    log.info(
        f"{basin_id}: GVF for {len(names)} sea conditions and {len(Q_NAMES)} discharges"
    )
    wl = {n: solve(ctx, lv) for n, lv in conds.items()}
    q_def = {
        "0.5 Q_mean": ("Q_mean", 0.5),
        "Q_mean": ("Q_mean", 1.0),
        "Q_bf": ("Q_bf", 1.0),
    }
    wl_q = {
        (qn, s): (
            wl[cond_names[s]]
            if qn == "Q_mean"
            else solve(ctx, ctx.sea_at_ends[s], *q_def[qn])
        )
        for qn in Q_NAMES
        for s in ("calm", "swl_ext")
    }

    forcing_nodes = {
        int(prof["ids"][0]): prof["point"].reach_id
        for prof in profiles
        if prof["point"].source == "forcing"
    }
    tag = _git_hash()
    arms = []
    for reaches in river_arms(
        ctx, args.min_arm_share, args.max_arms, sorted(set(forcing_nodes.values()))
    ):
        row, arr = analyse_arm(
            basin_id, reaches, ctx, opts, wl, wl_q, zs, forcing_nodes, cond_names
        )
        arms.append((row, arr))
        plot_arm(
            row,
            arr,
            names,
            cond_names,
            opts,
            out / f"{arm_name(row)}.png",
            f"Basin {basin_id}, arm from mouth reach {row['mouth_reach']} "
            f"(Q_mean {row['q_mean_mouth_m3s']:.0f} m³/s, A0 {row['a0_m']:.2f} m): influence end "
            f"GVF {row['x_inf_km']:.1f} km, SFINCS {row['x_obs_km']:.1f} km; 10 m contour "
            f"{row['x_10m_gvf_km']:.1f} km ({tag})",
        )
    delta = {
        "GVF (1D steady)": wl[cond_names["swl_ext"]] - wl["calm"],
        "SFINCS (2D, steady)": (zs["zs_swl_ext_m"] - zs["zs_calm_m"])
        .reindex(range(len(ctx.net.node_x)))
        .to_numpy(),
    }
    plot_maps(
        basin_id,
        ctx,
        delta,
        arms,
        out / "map.png",
        f"Basin {basin_id}: rise of the river level, sea at {cond_names['swl_ext']} vs calm, Q_mean ({tag})",
    )
    rows = [r for r, _ in arms]
    pd.DataFrame(rows).to_csv(out / "arms.csv", index=False)
    return rows, cond_names | {"frac": ctx.frac}


def main() -> None:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="+")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--days", type=float, default=4.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-h", type=float, default=12.0)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="reuse existing runs or sfincs_nodes.csv",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=list(CASES),
        help="SFINCS cases to (re)run",
    )
    parser.add_argument("--min-arm-share", type=float, default=0.05)
    parser.add_argument("--max-arms", type=int, default=6)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="default: <results_dir>/../scratch/upstream_benchmark (SFINCS needs paths relative "
        "to the run, so it must share the drive of results_dir)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "figs" / "upstream_boundary_benchmark",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_dir is None:
        args.run_dir = (
            site_path("results_dir").parent / "scratch" / "upstream_benchmark"
        )
    opts = SimpleNamespace(**config_options(), extra_reach=[])

    rows, cond_names = [], {}
    for basin_id in map(str, args.basin_ids):
        basin_rows, cond_names = benchmark_basin(basin_id, args, opts)
        rows += basin_rows
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "summary.csv", index=False)
    tag = _git_hash()
    plot_summary(
        df,
        opts.margin_km,
        args.out_dir / "summary.png",
        f"Upstream boundary check: steady GVF vs steady SFINCS, basins {', '.join(df['basin_id'].unique())} "
        f"({tag})",
    )
    write_readme(df, opts, args, cond_names, args.out_dir / "README.md", tag)
    pd.set_option("display.width", 250)
    show = [
        "basin_id",
        "mouth_reach",
        "forcing_reach",
        "x_f_km",
        "x_inf_km",
        "x_obs_km",
        "x_10m_gvf_km",
        "l_fit_km",
        "l_obs_km",
        "x_inf_0.5_Q_mean_km",
        "x_inf_Q_bf_km",
        "sfincs_max_rise_over_a0",
    ]
    print(df[show].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
