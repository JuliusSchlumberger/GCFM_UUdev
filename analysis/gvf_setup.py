"""River bed set-up: Froude numbers, advection, the bed method, its long profile and the options tried.

`compute <basin>` writes per basin to <results_dir>/figures/analysis/gvf_setup/<basin>/:

  nodes.csv: rule 11's round-0 bed per node (src.river_bed.estimate_river_bed)
    with the steady GVF level and Froude number Fr = Q / (W h sqrt(g h)) at
    Q_mean, Q_bankfull and the river event's peak (Q_bankfull scaled by the
    event's summed inflow peak over the crossings' summed Q_bankfull), sea at
    the calm level; the bank top (bank_freeboard.bank_top) and the SFINCS grid
    DEM one cell beyond each bank (lower side); the spin-up's end level;
    SWOT node levels (surface_stages.swot_on_nodes);
  gap.csv: rule 11's round-1 1D-2D gap per centreline cell (gvf_gap.cell_gaps);
  surface_options.csv, channel_capacity.csv, slope_floor.csv: the option
    tables of surface_options.py, channel_capacity.py and slope_floor.py;
  froude_sfincs_<scenario>.npz: per wet cell of the event run the largest
    vmax / sqrt(g (zsmax - zb)) over the max-output intervals; only when the
    map holds vmax (sfincs.simulation.storevelmax: 1).

`figures <basin> ...` writes to <results_dir>/figures/analysis/gvf_setup/:
froude_1d.png, froude_sfincs_<scenario>.png, advection.png, concept.png,
long_profile.png, bed_options.png.

Advection off (SFINCS local inertia) drops the convective term, valid at low
Froude numbers: errors are small below about 0.3 and grow above 0.5 (de Almeida
& Bates 2013, WRR 49). The 1D solver keeps the (1 - Fr^2) term of the GVF
equation; 1 / (1 - Fr^2) is how much it steepens the profile against a solve
without it.

Usage:
    python analysis/gvf_setup.py compute 2433835 [--river river_100_03SLR --coast coast_100_03SLR]
    python analysis/gvf_setup.py figures 2433835 3279946 [--river ...] [--fr-ok 0.3 --fr-limit 0.5]
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _basin_inputs import (
    basin_paths,
    calm_sea_level_m,
    git_hash,
    load_config,
    raster_at,
    reach_q_dict,
    round0,
)
from _common import REPO_ROOT, script_parser, site_path
from _upstream_profile import mainstem_path, path_nodes
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial import cKDTree

sys.path.insert(0, str(REPO_ROOT / "workflow"))

log = logging.getLogger(__name__)

G_MS2 = 9.81
DISCHARGES = {"q_mean": "Q_mean", "q_bf": "Q_bankfull", "q_peak": "event peak"}
FR_BOUNDS = [0.0, 0.1, 0.2, "ok", "limit", 1.0, 2.0]
FR_COLOURS = ["#1a9850", "#66bd63", "#a6d96a", "#fdae61", "#d73027", "#67001f"]


def out_dir(basin_id: str | None = None) -> Path:
    d = site_path("results_dir") / "figures" / "analysis" / "gvf_setup"
    d = d / basin_id if basin_id else d
    d.mkdir(parents=True, exist_ok=True)
    return d


def _crossings(basin_id: str, rivers) -> pd.DataFrame:
    from src.river_network import normalize_reach_id, outflow_crossing_mask

    with xr.open_dataset(
        basin_paths(basin_id)["river_forcing"], decode_times=False
    ) as ds:
        active = ds["has_glofas"].values.astype(bool) & ~outflow_crossing_mask(
            ds, rivers
        )
        return pd.DataFrame(
            {
                "reach_id": [
                    normalize_reach_id(r) for r in ds["inside_reach_id"].values[active]
                ],
                "q_bf_m3s": ds["bankfull_discharge_m3s"].values[active].astype(float),
            }
        )


def _beside_channel(path, crs, x, y, tx, ty, width_m, offset_m) -> np.ndarray:
    nx, ny = -ty, tx
    sides = [
        raster_at(
            path,
            x + s * (width_m / 2 + offset_m) * nx,
            y + s * (width_m / 2 + offset_m) * ny,
            crs,
        )
        for s in (1.0, -1.0)
    ]
    return np.fmin(*sides)


def _last_frame_at(map_path, x, y) -> np.ndarray:
    with xr.open_dataset(map_path) as ds:
        active = ds["msk"].values > 0
        n_idx, m_idx = np.nonzero(active)
        _, k = cKDTree(np.c_[ds["x"].values[active], ds["y"].values[active]]).query(
            np.c_[x, y]
        )
        frame = ds["zs"].isel({ds["zs"].dims[0]: -1}).values
    return frame[n_idx[k], m_idx[k]]


def compute_nodes(basin_id: str, cfg: dict, river: str) -> pd.DataFrame:
    from bank_freeboard import bank_top, net_crs, node_tangents
    from src.grid import load_grid_def
    from src.gvf_solver import forward_network
    from surface_stages import swot_on_nodes

    p = basin_paths(basin_id)
    n_manning = float(cfg["river_processing"]["channel_manning_n"])
    bed, rivers, q_design, q_mean = round0(basin_id, cfg)
    net = bed.net
    owner = np.asarray(net.owner)
    calm = calm_sea_level_m(basin_id)
    q_bf = reach_q_dict(rivers, q_design)
    crossings = _crossings(basin_id, rivers)
    dis = np.loadtxt(
        site_path("results_dir") / basin_id / "runs" / river / "sfincs" / "sfincs.dis",
        ndmin=2,
    )
    peak_ratio = float(dis[:, 1:].sum(axis=1).max() / crossings["q_bf_m3s"].sum())
    reach_q = {
        "q_mean": reach_q_dict(rivers, q_mean),
        "q_bf": q_bf,
        "q_peak": {r: q * peak_ratio for r, q in q_bf.items()},
    }
    mouths = {int(m) for m in bed.mouths}
    ends = {
        int(net.reach_nodes[r][-1]): calm
        if int(net.reach_nodes[r][-1]) in mouths
        else float(bed.level_m[net.reach_nodes[r][-1]])
        for r in net.reach_ids
        if not net.downstream[r]
    }
    crs = net_crs(basin_id)
    tx, ty = node_tangents(net)
    _, _, bank = bank_top(
        p["elevation_merged"], crs, net.node_x, net.node_y, tx, ty, bed.width_m
    )
    dx = float(load_grid_def(p["sfincs_grid"])["dx"])
    grid_dem = (
        p["sfincs_grid"].parent / f"{basin_id}_elevation_conditioned_sfincs_grid.tif"
    )
    t = pd.DataFrame(
        {
            "node": np.arange(net.n_nodes),
            "reach_id": owner,
            "x": net.node_x,
            "y": net.node_y,
            "dist_to_sea_m": bed.dist_to_sea_m,
            "fathom_level_m": bed.fathom_level_m,
            "level_m": bed.level_m,
            "slope": bed.slope,
            "depth_slope": bed.depth_slope,
            "on_floor": bed.on_floor,
            "width_m": bed.width_m,
            "depth_m": bed.depth_m,
            "bed_m": bed.bed_m,
            "forward_level_m": bed.forward_level_m,
            "bank_top_m": bank,
            "grid_beside_m": _beside_channel(
                grid_dem, crs, net.node_x, net.node_y, tx, ty, bed.width_m, dx
            ),
            "spinup_end_m": _last_frame_at(
                p["spin_up"] / "sfincs_map.nc", net.node_x, net.node_y
            ),
        }
    )
    for key, q in reach_q.items():
        level = forward_network(net, bed.bed_m, bed.width_m, q, n_manning, ends)
        h = np.maximum(level - bed.bed_m, 0.1)
        qn = np.array([q.get(r, 0.0) for r in owner])
        t[f"level_{key}_m"] = level
        t[f"{key}_m3s"] = qn
        t[f"fr_{key}"] = qn / (bed.width_m * h * np.sqrt(G_MS2 * h))
    t["peak_over_bankfull"] = peak_ratio
    swot = swot_on_nodes(basin_id, cfg, t[["node", "reach_id", "x", "y"]].copy(), crs)
    t["swot_m"] = swot["swot_m"].to_numpy()
    return t


def compute_sfincs_froude(basin_id: str, scenario: str) -> dict | None:
    path = (
        site_path("results_dir")
        / basin_id
        / "runs"
        / scenario
        / "sfincs"
        / "sfincs_map.nc"
    )
    with xr.open_dataset(path) as ds:
        if "vmax" not in ds:
            log.warning(
                f"{basin_id} {scenario}: no vmax in the map (sfincs.simulation.storevelmax: 0)"
            )
            return None
        zb = ds["zb"].values
        tdim = ds["vmax"].dims[0]
        fr = np.full(zb.shape, np.nan)
        hmax = np.full(zb.shape, np.nan)
        for i in range(ds.sizes[tdim]):
            h = ds["zsmax"].isel({tdim: i}).values - zb
            v = ds["vmax"].isel({tdim: i}).values
            with np.errstate(invalid="ignore", divide="ignore"):
                f = np.where(h > 0.1, v / np.sqrt(G_MS2 * np.maximum(h, 0.1)), np.nan)
            fr = np.fmax(fr, f)
            hmax = np.fmax(hmax, h)
        wet = np.isfinite(fr)
        return {
            "x": ds["x"].values[wet],
            "y": ds["y"].values[wet],
            "fr": fr[wet],
            "hmax": hmax[wet],
        }


def compute(basin_id: str, river: str, coast: str) -> None:
    import channel_capacity
    import slope_floor
    import surface_options
    from gvf_gap import cell_gaps

    cfg = load_config()
    out = out_dir(basin_id)
    compute_nodes(basin_id, cfg, river).to_csv(
        out / "nodes.csv", index=False, float_format="%.4g"
    )
    cells = cell_gaps(basin_id, cfg)[0]
    cells[["x", "y", "reach_id", "zs_end_m", "level_1d_m", "gap_m"]].to_csv(
        out / "gap.csv", index=False, float_format="%.3f"
    )
    pd.DataFrame(surface_options.basin(basin_id, cfg)).to_csv(
        out / "surface_options.csv", index=False
    )
    pd.DataFrame(channel_capacity.basin(basin_id, cfg)).to_csv(
        out / "channel_capacity.csv", index=False
    )
    pd.DataFrame(slope_floor.floor_basin(basin_id, cfg)).to_csv(
        out / "slope_floor.csv", index=False
    )
    for scenario in (river, coast):
        fr = compute_sfincs_froude(basin_id, scenario)
        if fr is not None:
            np.savez_compressed(out / f"froude_sfincs_{scenario}.npz", **fr)


def _fr_cmap(fr_ok: float, fr_limit: float):
    bounds = [
        fr_ok if b == "ok" else fr_limit if b == "limit" else b for b in FR_BOUNDS
    ]
    return ListedColormap(FR_COLOURS), BoundaryNorm(bounds, len(FR_COLOURS)), bounds


def _network_lines(t: pd.DataFrame, value: str):
    segs, vals = [], []
    for _, g in t.groupby("reach_id", sort=False):
        g = g.sort_values("dist_to_sea_m", ascending=False)
        xy = g[["x", "y"]].to_numpy()
        v = g[value].to_numpy()
        segs += [xy[i : i + 2] for i in range(len(xy) - 1)]
        vals += list(0.5 * (v[:-1] + v[1:]))
    return segs, np.asarray(vals)


def fig_froude_1d(
    nodes: dict, path: Path, fr_ok: float, fr_limit: float, tag: str
) -> None:
    cmap, norm, bounds = _fr_cmap(fr_ok, fr_limit)
    nb = len(nodes)
    fig, axes = plt.subplots(
        nb,
        len(DISCHARGES) + 1,
        figsize=(4.6 * (len(DISCHARGES) + 1), 4.6 * nb),
        squeeze=False,
    )
    for r, (basin_id, t) in enumerate(nodes.items()):
        for c, (key, label) in enumerate(DISCHARGES.items()):
            ax = axes[r, c]
            segs, vals = _network_lines(t, f"fr_{key}")
            lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.2)
            lc.set_array(vals)
            ax.add_collection(lc)
            ax.autoscale()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            share = float(np.mean(t[f"fr_{key}"] > fr_ok))
            extra = (
                f" (x{t['peak_over_bankfull'].iloc[0]:.2f} Q_bf)"
                if key == "q_peak"
                else ""
            )
            ax.set_title(
                f"{basin_id}: {label}{extra}\n{share:.1%} of nodes above Fr {fr_ok:g}",
                fontsize=9,
            )
        b = axes[r, -1]
        for key, label in DISCHARGES.items():
            fr = np.sort(t[f"fr_{key}"].dropna().to_numpy())
            b.plot(fr, np.linspace(0, 1, fr.size), lw=1.5, label=label)
        b.axvspan(fr_ok, fr_limit, color="#fdae61", alpha=0.25)
        b.axvspan(fr_limit, 2.0, color="#d73027", alpha=0.15)
        b.set_xlim(0, max(0.8, fr_limit + 0.2))
        b.set_xlabel("Froude number")
        b.set_ylabel("share of nodes below")
        p95 = float(np.nanpercentile(t["fr_q_peak"], 95))
        b.set_title(
            f"p95 at the event peak {p95:.2f}: 1/(1-Fr^2) = {1 / (1 - p95**2):.3f}",
            fontsize=9,
        )
        b.grid(alpha=0.3)
        b.legend(fontsize=8, loc="lower right")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(
        sm,
        ax=axes[:, : len(DISCHARGES)].ravel().tolist(),
        ticks=bounds,
        shrink=0.6,
        label=f"Froude number (advection off: fine < {fr_ok:g}, not acceptable > {fr_limit:g})",
    )
    fig.suptitle(
        f"gvf_setup: 1D Froude number on the river bed, sea at the calm level | {tag}",
        fontsize=10,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_froude_sfincs(
    data: dict, scenario: str, path: Path, fr_ok: float, fr_limit: float, tag: str
) -> None:
    cmap, norm, bounds = _fr_cmap(fr_ok, fr_limit)
    nb = len(data)
    fig, axes = plt.subplots(
        2,
        nb,
        figsize=(6 * nb, 10),
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for c, (basin_id, d) in enumerate(data.items()):
        ax = axes[0, c]
        if d is None:
            ax.text(
                0.5,
                0.5,
                f"{basin_id} {scenario}:\nno vmax in the map;\nrerun with sfincs.simulation.storevelmax: 1",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            axes[1, c].axis("off")
            continue
        order = np.argsort(d["fr"])
        ax.scatter(
            d["x"][order],
            d["y"][order],
            c=d["fr"][order],
            cmap=cmap,
            norm=norm,
            s=0.3,
            lw=0,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{basin_id}: max Froude number per wet cell", fontsize=9)
        b = axes[1, c]
        fr = np.sort(d["fr"])
        b.plot(fr, np.linspace(0, 1, fr.size), color="k")
        deep = np.sort(d["fr"][d["hmax"] > 1.0])
        b.plot(
            deep,
            np.linspace(0, 1, deep.size),
            color="tab:blue",
            label="cells deeper than 1 m",
        )
        b.axvspan(fr_ok, fr_limit, color="#fdae61", alpha=0.25)
        b.axvspan(fr_limit, 2.0, color="#d73027", alpha=0.15)
        b.set_xlim(0, 1.5)
        b.set_xlabel("Froude number")
        b.set_title(
            f"above {fr_ok:g}: {np.mean(d['fr'] > fr_ok):.1%} of wet cells, "
            f"{np.mean(deep > fr_ok):.1%} of cells deeper than 1 m",
            fontsize=9,
        )
        b.grid(alpha=0.3)
        b.legend(fontsize=8)
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes[0, :].tolist(),
        ticks=bounds,
        shrink=0.7,
        label="Froude number",
    )
    fig.suptitle(
        f"gvf_setup: SFINCS Froude number, {scenario} (vmax / sqrt(g h) per max-output interval) | {tag}",
        fontsize=10,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_advection(gaps: dict, path: Path, tag: str) -> None:
    syn_path = (
        site_path("results_dir")
        / "figures"
        / "analysis"
        / "advection_channel_summary.csv"
    )
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [3, 2]}
    )
    if syn_path.exists():
        syn = pd.read_csv(syn_path)
        syn = syn[syn["alpha"] == 0.25]
        cases = syn[["layout", "q_m3s"]].drop_duplicates().to_numpy()
        x = np.arange(len(cases))
        for k, (adv, col) in enumerate(((0, "tab:blue"), (1, "tab:red"))):
            v = [
                syn[
                    (syn["layout"] == lay)
                    & (syn["q_m3s"] == q)
                    & (syn["advection"] == adv)
                ]["error_median_m"].squeeze()
                for lay, q in cases
            ]
            a.bar(x + (k - 0.5) * 0.38, v, 0.38, color=col, label=f"advection {adv}")
        a.set_xticks(x)
        a.set_xticklabels([f"{lay}\n{q:.0f} m3/s" for lay, q in cases], fontsize=8)
        a.set_ylabel("median level error vs normal depth (m)")
        a.set_title(
            "synthetic 100 m wide trench on a 100 m grid, alpha 0.25 (advection_channel.py)",
            fontsize=9,
        )
        a.legend(fontsize=8)
        a.grid(alpha=0.3, axis="y")
    else:
        a.text(
            0.5,
            0.5,
            "run analysis/advection_channel.py",
            ha="center",
            transform=a.transAxes,
        )
    data = [g["gap_m"].dropna().to_numpy() for g in gaps.values()]
    b.boxplot(data, showfliers=False)
    b.set_xticks(range(1, len(gaps) + 1))
    b.set_xticklabels(list(gaps), fontsize=8)
    b.axhline(0, color="k", lw=0.8)
    for i, d in enumerate(data, start=1):
        b.annotate(
            f"median {np.median(d):+.2f} m",
            (i, np.median(d)),
            xytext=(28, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
        )
    b.set_ylabel("SFINCS steady level - 1D GVF level (m)")
    b.set_title("1D-2D gap on rule 11's round 1, current settings", fontsize=9)
    b.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"gvf_setup: advection adds head on staircase cell paths | {tag}", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_concept(path: Path, tag: str) -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 201)
    sea = 0.0
    true = sea + 6.0 * (np.maximum(x - 25.0, 0.0) / 75.0) ** 1.4
    raw = np.round(true + rng.normal(0, 0.35, x.size), 0)
    raw = np.maximum(raw, sea - 0.4)
    fitted = true
    slope = np.gradient(fitted, x * 1000)
    floor = 1e-5
    depth = (0.03 * 1500 / (200 * np.sqrt(np.maximum(slope, floor)))) ** 0.6
    bed = fitted - depth
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [3, 2]}
    )
    a.plot(
        x,
        raw,
        ".",
        color="0.6",
        ms=4,
        label="FathomDEM river-centre levels (staircase)",
    )
    a.plot(
        x,
        fitted,
        color="tab:blue",
        lw=2,
        label="fitted surface: sea-pinned, monotone, smoothed",
    )
    a.plot(x, bed, color="k", lw=2, label="bed = surface - h")
    a.fill_between(x, bed, fitted, color="tab:blue", alpha=0.12)
    flat = slope < floor
    a.plot(
        x[flat],
        fitted[flat],
        color="tab:orange",
        lw=4,
        alpha=0.6,
        label="slope below the floor: h at the floor",
    )
    a.axhline(sea, color="tab:cyan", ls="--", lw=1, label="calm sea level")
    a.annotate(
        "h = (n Q_bf / (W sqrt(S)))^(3/5)", (55, float(np.min(bed)) + 1.0), fontsize=10
    )
    a.set_xlabel("distance from the mouth (km), schematic")
    a.set_ylabel("level (m)")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3)
    steps = [
        (
            "Rule 10: water surface",
            "FathomDEM river-centre medians per node;\nmouths pinned at the calm sea level;\nmonotone downstream fit, two-scale smoothing",
        ),
        (
            "Slope S",
            "surface slope along the largest-Q path,\nfloored at slope_floor (1e-5) for the depth",
        ),
        (
            "Depth h",
            "Manning normal depth, rectangular section:\nQ_bankfull routed through the network,\nW SWORD width, n channel_manning_n",
        ),
        ("Bed", "bed = surface - h (rule 11, round 0)"),
        (
            "Forward check",
            "steady GVF on the bed at Q_bankfull\nfrom the network ends: back to the surface?",
        ),
        (
            "SFINCS",
            "bed burned into the subgrid; advection 0, alpha 0.25;\nrule 11 runs it walled to set the coastal crests",
        ),
    ]
    b.axis("off")
    for k, (head, body) in enumerate(steps):
        yk = 1 - k / len(steps)
        b.text(
            0.02,
            yk - 0.02,
            f"{k + 1}. {head}",
            fontsize=10,
            weight="bold",
            va="top",
            transform=b.transAxes,
        )
        b.text(0.06, yk - 0.07, body, fontsize=8.5, va="top", transform=b.transAxes)
        if k < len(steps) - 1:
            b.annotate(
                "",
                (0.03, yk - 1 / len(steps) + 0.005),
                (0.03, yk - 0.05),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="0.5"),
            )
    fig.suptitle(
        f"gvf_setup: how the river bed is set from the observed surface | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _mainstem(basin_id: str, t: pd.DataFrame):
    from src.river_network import normalize_reach_id

    cfg = load_config()
    bed, rivers, q_design, _ = round0(basin_id, cfg)
    q_bf = reach_q_dict(rivers, q_design)
    start = max(
        _crossings(basin_id, rivers)["reach_id"].map(normalize_reach_id),
        key=lambda r: q_bf.get(r, 0.0),
    )
    ids, dist = path_nodes(bed.net, mainstem_path(bed.net.downstream, start, q_bf))
    return t.set_index("node").loc[ids].assign(dist_km=dist / 1000).reset_index()


def fig_long_profile(nodes: dict, path: Path, tag: str) -> None:
    nb = len(nodes)
    fig, axes = plt.subplots(
        2 * nb,
        1,
        figsize=(14, 7.5 * nb),
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 1.3] * nb},
    )
    for r, (basin_id, t) in enumerate(nodes.items()):
        m = _mainstem(basin_id, t)
        a, b = axes[2 * r, 0], axes[2 * r + 1, 0]
        km = m["dist_km"]
        a.plot(
            km,
            m["grid_beside_m"],
            color="tab:brown",
            lw=0.8,
            alpha=0.7,
            label="SFINCS grid DEM, one cell beyond the bank",
        )
        a.plot(
            km,
            m["bank_top_m"],
            ".",
            color="tab:brown",
            ms=3,
            label="bank top (P90 of the DEM band)",
        )
        a.plot(
            km,
            m["fathom_level_m"],
            ".",
            color="0.55",
            ms=3,
            label="FathomDEM river-centre level",
        )
        a.plot(
            km,
            m["swot_m"],
            "x",
            color="tab:purple",
            ms=4,
            label="SWOT node level (p50)",
        )
        a.plot(
            km, m["level_m"], color="tab:blue", lw=2, label="fitted surface (rule 10)"
        )
        a.plot(
            km,
            m["level_q_bf_m"],
            color="tab:cyan",
            lw=1.2,
            ls="--",
            label="GVF at Q_bankfull",
        )
        a.plot(
            km,
            m["level_q_mean_m"],
            color="tab:green",
            lw=1.2,
            ls="--",
            label="GVF at Q_mean, calm sea",
        )
        a.plot(
            km,
            m["spinup_end_m"],
            color="tab:green",
            lw=1.0,
            alpha=0.6,
            label="SFINCS spin-up end (Q_mean, calm sea)",
        )
        a.plot(km, m["bed_m"], color="k", lw=1.6, label="bed")
        a.set_ylim(
            float(np.nanmin(m["bed_m"])) - 2,
            float(np.nanpercentile(m["bank_top_m"], 95)) + 3,
        )
        a.invert_xaxis()
        a.set_ylabel("m (model datum)")
        a.set_title(f"{basin_id}: largest-inflow mainstem", fontsize=9)
        a.grid(alpha=0.3)
        a.legend(fontsize=7, ncol=3, loc="upper right")
        b.semilogy(
            km,
            np.maximum(m["slope"], 1e-8),
            color="tab:blue",
            lw=1,
            label="surface slope",
        )
        b.semilogy(
            km,
            m["depth_slope"],
            color="tab:orange",
            lw=1.5,
            label="slope used for the depth (floored)",
        )
        fl = m["on_floor"].astype(bool)
        b.semilogy(km[fl], m["depth_slope"][fl], "o", color="tab:orange", ms=2)
        b.set_ylabel("slope (-)")
        b.set_ylim(1e-7, 1e-2)
        b2 = b.twinx()
        b2.plot(km, m["depth_m"], color="k", lw=1, label="depth h")
        b2.set_ylabel("depth (m)")
        b.invert_xaxis()
        b.set_xlabel("distance from the mouth (km)")
        b.grid(alpha=0.3)
        h1, l1 = b.get_legend_handles_labels()
        h2, l2 = b2.get_legend_handles_labels()
        b.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    fig.suptitle(
        f"gvf_setup: long profile of the river bed and its inputs | {tag}", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


SURFACE_LABELS = {
    ("fathom_two_scale", "bankfull"): "FathomDEM two-scale, bankfull (default)",
}
CAPACITY_DEFAULT = "fathom_bankfull"


def _dots(
    ax,
    table: pd.DataFrame,
    label_col: str,
    value: str,
    title: str,
    default: str,
    xlim=None,
):
    labels = list(dict.fromkeys(table[label_col]))
    for k, (basin_id, g) in enumerate(table.groupby("basin_id", sort=False)):
        y = [labels.index(v) + (k - 0.5) * 0.25 for v in g[label_col]]
        ax.plot(g[value], y, "o", ms=6, label=str(basin_id))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for tl in ax.get_yticklabels():
        if tl.get_text() == default:
            tl.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    if xlim:
        ax.set_xlim(*xlim)


def fig_bed_options(options: dict, path: Path, tag: str) -> None:
    surf = pd.concat(
        [o["surface"].assign(basin_id=b) for b, o in options.items()], ignore_index=True
    )
    surf["option"] = surf["surface"] + ", " + surf["design_discharge"]
    cap = pd.concat(
        [o["capacity"].assign(basin_id=b) for b, o in options.items()],
        ignore_index=True,
    )
    floor = pd.concat(
        [o["floor"].assign(basin_id=b) for b, o in options.items()], ignore_index=True
    )
    floor["option"] = floor["slope_floor"].map(lambda v: f"floor {v:g}")
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    sd = "fathom_two_scale, bankfull"
    for ax, (col, title) in zip(
        axes[0],
        [
            ("forward_median_m", "forward check: GVF - surface, median (m)"),
            ("forward_p95_m", "forward check, p95 (m)"),
            ("on_floor_share", "share of nodes on the slope floor"),
            ("freeboard_share_below_0", "share of nodes overflowing at Q_bankfull"),
        ],
    ):
        _dots(ax, surf, "option", col, title, sd)
    for ax, (col, title) in zip(
        axes[1],
        [
            ("q_cap_over_q_bf_p50", "channel capacity / Q_bankfull, median"),
            ("share_q_cap_below_q_mean", "share of nodes overflowing at Q_mean"),
            ("depth_p50_m", "depth, median (m)"),
            ("inverted_share", "share of nodes with sea-inverted depth"),
        ],
    ):
        _dots(ax, cap, "option", col, title, CAPACITY_DEFAULT)
    for ax, (col, title) in zip(
        axes[2][:2],
        [
            ("on_floor_share", "share of nodes on the floor"),
            (
                "median_depth_change_affected_pct",
                "depth change on the affected nodes (%)",
            ),
        ],
    ):
        _dots(ax, floor, "option", col, title, "floor 1e-05")
    for ax in axes[2][2:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    axes[2, 2].text(
        0,
        0.9,
        "Rows: surface source x fit x design discharge (surface_options.py);\n"
        "bed method and design discharge (channel_capacity.py);\nslope floor (slope_floor.py).\n"
        "Bold: the configured default.",
        fontsize=9,
        va="top",
        transform=axes[2, 2].transAxes,
    )
    fig.suptitle(
        f"gvf_setup: river bed options against the default | {tag}", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figures(
    basin_ids: list[str], river: str, coast: str, fr_ok: float, fr_limit: float
) -> None:
    tag = git_hash()
    out = out_dir()
    nodes = {
        b: pd.read_csv(out_dir(b) / "nodes.csv", dtype={"reach_id": str})
        for b in basin_ids
    }
    fig_froude_1d(nodes, out / "froude_1d.png", fr_ok, fr_limit, tag)
    for scenario in (river, coast):
        data = {}
        for b in basin_ids:
            f = out_dir(b) / f"froude_sfincs_{scenario}.npz"
            data[b] = dict(np.load(f)) if f.exists() else None
        fig_froude_sfincs(
            data, scenario, out / f"froude_sfincs_{scenario}.png", fr_ok, fr_limit, tag
        )
    gaps = {b: pd.read_csv(out_dir(b) / "gap.csv") for b in basin_ids}
    fig_advection(gaps, out / "advection.png", tag)
    fig_concept(out / "concept.png", tag)
    fig_long_profile(nodes, out / "long_profile.png", tag)
    options = {
        b: {
            "surface": pd.read_csv(out_dir(b) / "surface_options.csv"),
            "capacity": pd.read_csv(out_dir(b) / "channel_capacity.csv"),
            "floor": pd.read_csv(out_dir(b) / "slope_floor.csv"),
        }
        for b in basin_ids
    }
    fig_bed_options(options, out / "bed_options.png", tag)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("action", choices=["compute", "figures"])
    parser.add_argument("basin_ids", nargs="+")
    parser.add_argument("--river", default="river_100_03SLR")
    parser.add_argument("--coast", default="coast_100_03SLR")
    parser.add_argument("--fr-ok", type=float, default=0.3)
    parser.add_argument("--fr-limit", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.action == "compute":
        for b in args.basin_ids:
            compute(str(b), args.river, args.coast)
    else:
        figures(
            [str(b) for b in args.basin_ids],
            args.river,
            args.coast,
            args.fr_ok,
            args.fr_limit,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
