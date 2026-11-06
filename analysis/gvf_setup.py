"""River bed set-up figures: the bed method, Froude numbers, long profiles and the bed alternatives.

`compute <basin>` writes per basin to <results_dir>/figures/analysis/gvf_setup/<basin>/:

  nodes.csv: rule 11's round-0 bed per node (src.river_bed.estimate_river_bed)
    with the steady GVF level and Froude number Fr = Q / (W h sqrt(g h)) at
    Q_mean, Q_bankfull and the river event's peak (Q_bankfull scaled by the
    event's summed inflow peak over the crossings' summed Q_bankfull), sea at
    the calm level; the bank top (bank_freeboard.bank_top) and the SFINCS grid
    DEM one cell beyond each bank (lower side); the spin-up's end level;
    SWOT node levels (surface_stages.swot_on_nodes);
  surface_options.csv, channel_capacity.csv: the alternative beds of
    surface_options.py and channel_capacity.py;
  froude_sfincs_<river>.npz: per wet cell of the river event the largest
    vmax / sqrt(g (zsmax - zb)) over the max-output intervals; only when the
    map holds vmax (sfincs.simulation.storevelmax: 1).

`figures <basin> ...` writes to <results_dir>/figures/analysis/gvf_setup/, each
with a .txt description: concept.png, froude_1d.png, froude_sfincs_<river>.png,
long_profile_<basin>.png, bed_alternatives.png.

Advection off (SFINCS local inertia) drops the convective term, valid at low
Froude numbers: errors are small below about 0.3 and grow above 0.5 (de Almeida
& Bates 2013, WRR 49). The 1D solver keeps the (1 - Fr^2) term of the GVF
equation; 1 / (1 - Fr^2) is how much it steepens the profile against a solve
without it.

Usage:
    python analysis/gvf_setup.py compute 2433835 [--river river_100_03SLR]
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
    load_config,
    raster_at,
    reach_q_dict,
    round0,
)
from _common import REPO_ROOT, script_parser, site_path
from _figure_style import add_land, map_axes, panel_letters, write_caption
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

# Bed alternatives: label, surface_options (surface, design discharge), channel_capacity option
ALTERNATIVES = [
    (
        "A  FathomDEM, two-scale, bankfull (production)",
        ("fathom_two_scale", "bankfull"),
        "fathom_bankfull",
    ),
    (
        "B  FathomDEM, two-scale, recent median Q",
        ("fathom_two_scale", "period_median"),
        "fathom_period_median",
    ),
    ("C  SWOT, two-scale, bankfull", ("swot_two_scale", "bankfull"), None),
    ("D  SWOT, two-scale, recent median Q", ("swot_two_scale", "period_median"), None),
    ("E  SWOT, single 20 km, bankfull", ("swot_single", "bankfull"), None),
    ("F  SWOT, single 20 km, recent median Q", ("swot_single", "period_median"), None),
    ("G  SWOT, single 40 km, bankfull", ("swot_single_40km", "bankfull"), None),
    (
        "H  SWOT, single 40 km, recent median Q",
        ("swot_single_40km", "period_median"),
        "swot40_period_median",
    ),
    ("I  A + sea inversion", None, "fathom_bankfull_inversion"),
    ("J  H + sea inversion", None, "swot40_period_median_inversion"),
]
# criterion: column label, pass test
CRITERIA = {
    "forward_median_m": ("forward check\nmedian (m)", lambda v: abs(v) <= 0.25),
    "forward_p95_m": ("forward check\np95 (m)", lambda v: v <= 0.75),
    "overflow_bf": ("overflowing\nat Q_bankfull", lambda v: v <= 0.10),
    "overflow_mean": ("overflowing\nat Q_mean", lambda v: v <= 0.02),
    "on_floor_share": ("on the\nslope floor", None),
}


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


def compute(basin_id: str, river: str) -> None:
    import channel_capacity
    import surface_options

    cfg = load_config()
    out = out_dir(basin_id)
    compute_nodes(basin_id, cfg, river).to_csv(
        out / "nodes.csv", index=False, float_format="%.4g"
    )
    pd.DataFrame(surface_options.basin(basin_id, cfg)).to_csv(
        out / "surface_options.csv", index=False
    )
    pd.DataFrame(channel_capacity.basin(basin_id, cfg)).to_csv(
        out / "channel_capacity.csv", index=False
    )
    fr = compute_sfincs_froude(basin_id, river)
    if fr is not None:
        np.savez_compressed(out / f"froude_sfincs_{river}.npz", **fr)


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


def _zoom(ax, x, y, margin: float = 0.05) -> None:
    dx, dy = np.ptp(x), np.ptp(y)
    pad = margin * max(dx, dy)
    ax.set_xlim(np.min(x) - pad, np.max(x) + pad)
    ax.set_ylim(np.min(y) - pad, np.max(y) + pad)


def _letters(n: int) -> list[str]:
    return [f"({chr(ord('a') + k)})" for k in range(n)]


def fig_froude_1d(nodes: dict, path: Path, fr_ok: float, fr_limit: float) -> None:
    cmap, norm, bounds = _fr_cmap(fr_ok, fr_limit)
    nb, nc = len(nodes), len(DISCHARGES) + 1
    fig, axes = plt.subplots(nb, nc, figsize=(4.6 * nc, 4.6 * nb), squeeze=False)
    for r, (basin_id, t) in enumerate(nodes.items()):
        for c, key in enumerate(DISCHARGES):
            ax = axes[r, c]
            add_land(ax, basin_id)
            segs, vals = _network_lines(t, f"fr_{key}")
            lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.2)
            lc.set_array(vals)
            ax.add_collection(lc)
            _zoom(ax, t["x"], t["y"])
            map_axes(ax)
        b = axes[r, -1]
        for key, colour in zip(DISCHARGES, ("tab:blue", "tab:orange", "tab:green")):
            fr = np.sort(t[f"fr_{key}"].dropna().to_numpy())
            b.plot(fr, np.linspace(0, 1, fr.size), lw=1.5, color=colour)
        b.axvspan(fr_ok, fr_limit, color="#fdae61", alpha=0.25)
        b.axvspan(fr_limit, 2.0, color="#d73027", alpha=0.15)
        b.set_xlim(0, max(0.8, fr_limit + 0.2))
        b.set_xlabel("Froude number")
        b.set_ylabel("share of nodes below")
        b.grid(alpha=0.3)
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes[:, : len(DISCHARGES)].ravel().tolist(),
        ticks=bounds,
        shrink=0.6,
        label="Froude number",
    )
    panel_letters(axes)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    letters = _letters(nb * nc)
    rows = []
    for r, (basin_id, t) in enumerate(nodes.items()):
        k = r * nc
        shares = ", ".join(
            f"{lab} {np.mean(t[f'fr_{key}'] > fr_ok):.1%}"
            for key, lab in DISCHARGES.items()
        )
        p95 = float(np.nanpercentile(t["fr_q_peak"], 95))
        rows.append(
            f"{letters[k]}-{letters[k + nc - 2]} basin {basin_id}: river network coloured by the 1D Froude "
            f"number at mean discharge, bankfull discharge and the river event's peak "
            f"({t['peak_over_bankfull'].iloc[0]:.2f} x bankfull, all of it kept in the channel). "
            f"{letters[k + nc - 1]} cumulative distribution over the nodes: blue mean discharge, "
            f"orange bankfull, green event peak. Share of nodes above {fr_ok:g}: {shares}. "
            f"At the event peak the 95th percentile is {p95:.2f}, so the 1D solver's (1 - Fr^2) term "
            f"steepens the profile by at most {1 / (1 - p95**2):.2f} against a solve without it."
        )
    write_caption(
        path,
        "1D Froude number Fr = Q / (W h sqrt(g h)) on the river bed, steady GVF solve with the sea at the "
        f"calm level. Greens: Fr below {fr_ok:g}, where dropping SFINCS's advection term (local-inertia "
        f"form) is accurate; orange {fr_ok:g}-{fr_limit:g}: small but growing error; reds above "
        f"{fr_limit:g}: not acceptable (de Almeida & Bates 2013). Same bands shaded in the distribution "
        "panels. Land in light grey.\n\n" + "\n\n".join(rows),
    )


def fig_froude_sfincs(
    data: dict, scenario: str, path: Path, fr_ok: float, fr_limit: float
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
    notes = []
    for c, (basin_id, d) in enumerate(data.items()):
        ax, b = axes[0, c], axes[1, c]
        add_land(ax, basin_id)
        map_axes(ax)
        if d is None:
            b.axis("off")
            notes.append(
                f"basin {basin_id}: no vmax in the map (run made with storevelmax 0); panels empty"
            )
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
        _zoom(ax, d["x"], d["y"])
        fr = np.sort(d["fr"])
        deep = np.sort(d["fr"][d["hmax"] > 1.0])
        b.plot(fr, np.linspace(0, 1, fr.size), color="k")
        b.plot(deep, np.linspace(0, 1, deep.size), color="tab:blue")
        b.axvspan(fr_ok, fr_limit, color="#fdae61", alpha=0.25)
        b.axvspan(fr_limit, 2.0, color="#d73027", alpha=0.15)
        b.set_xlim(0, 1.5)
        b.set_xlabel("Froude number")
        b.set_ylabel("share of cells below")
        b.grid(alpha=0.3)
        notes.append(
            f"basin {basin_id}: above {fr_ok:g} in {np.mean(d['fr'] > fr_ok):.1%} of wet cells and "
            f"{np.mean(deep > fr_ok):.1%} of cells deeper than 1 m"
        )
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes[0, :].tolist(),
        ticks=bounds,
        shrink=0.7,
        label="Froude number",
    )
    panel_letters(np.r_[axes[0], axes[1]])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    letters = _letters(2 * nb)
    write_caption(
        path,
        f"SFINCS Froude number in the river event {scenario}. Top row "
        f"({', '.join(letters[:nb])}, one per basin in the order {', '.join(data)}): per wet cell the "
        "largest vmax / sqrt(g h) over the hourly max-output intervals, h = zsmax - zb; colours as in "
        f"froude_1d (greens below {fr_ok:g}, orange to {fr_limit:g}, reds above). Bottom row "
        f"({', '.join(letters[nb:])}): cumulative distribution over all wet cells (black) and over "
        "cells deeper than 1 m (blue), with the same bands shaded. Cell values mix channel and "
        "floodplain flow on the 100 m grid, so they indicate where flow is fast rather than a "
        "channel Froude number. Land in light grey.\n\n" + "\n".join(notes),
    )


def fig_concept(path: Path) -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 201)
    sea = 0.0
    surface = sea + 6.0 * (np.maximum(x - 25.0, 0.0) / 75.0) ** 1.4
    raw = np.maximum(np.round(surface + rng.normal(0, 0.35, x.size), 0), sea - 0.4)
    slope = np.gradient(surface, x * 1000)
    floor = 1e-5
    depth = (0.03 * 1500 / (200 * np.sqrt(np.maximum(slope, floor)))) ** 0.6
    bed = surface - depth
    # GVF level on the bed: where the slope is floored the bed is shallower than
    # the flat surface needs, so the computed level rises at the floor slope
    flat = slope < floor
    forward = np.maximum(surface, sea + 0.035 * x)
    crest = surface + 1.2 + 0.4 * (x / 100)

    fig, (a, b) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [3, 2]}
    )
    a.plot(x, raw, ".", color="0.6", ms=4)
    a.plot(x, surface, color="tab:blue", lw=2)
    a.fill_between(x, bed, surface, color="tab:blue", alpha=0.10)
    a.plot(x[flat], surface[flat], color="tab:orange", lw=5, alpha=0.5)
    a.plot(x, forward, color="tab:cyan", lw=1.6, ls="--")
    a.plot(x, bed, color="k", lw=2)
    a.plot(x, crest, color="tab:brown", lw=1.5, ls="-.")
    a.axhline(sea, color="tab:cyan", lw=0.8, ls=":")
    xa = 70
    k = int(np.argmin(np.abs(x - xa)))
    a.annotate(
        "", (xa, bed[k]), (xa, surface[k]), arrowprops=dict(arrowstyle="->", lw=1.5)
    )
    a.text(xa + 1.5, 0.5 * (bed[k] + surface[k]), "h", fontsize=12, va="center")
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel("distance from the mouth")
    a.set_ylabel("level")
    marks = [
        (1, 88, raw[np.argmin(np.abs(x - 88))] + 0.9),
        (2, 50, surface[np.argmin(np.abs(x - 50))] + 0.9),
        (3, 12, sea - 1.0),
        (4, xa - 4, 0.5 * (bed[k] + surface[k])),
        (5, 18, forward[np.argmin(np.abs(x - 18))] + 0.8),
        (6, 40, crest[np.argmin(np.abs(x - 40))] + 0.8),
    ]
    for n, xm, ym in marks:
        a.text(
            xm,
            ym,
            str(n),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="circle", facecolor="white", edgecolor="k"),
        )
    steps = [
        (
            "Observed water surface",
            "river-centre water levels read from the terrain\nmodel along the river network",
        ),
        (
            "Fitted surface",
            "levels may only fall towards the sea and end at the\ncalm sea level; small steps are smoothed out",
        ),
        (
            "Water-surface slope",
            "taken from the fitted surface; where the river is\nflat near the sea a minimum slope is used instead",
        ),
        (
            "Depth and bed",
            "depth of uniform flow at bankfull discharge for that\nslope, channel width and roughness; bed = surface - depth",
        ),
        (
            "Check",
            "water levels computed on this bed (gradually varied\nflow) should return the surface; they rise above it\nwhere the minimum slope was used",
        ),
        (
            "Flood model",
            "bed built into the flood model; dikes along the river\nset to the protection standard from a run with\nthe river walled in",
        ),
    ]
    b.axis("off")
    for n, (head, body) in enumerate(steps, start=1):
        yk = 0.92 - 0.92 * (n - 1) / len(steps)
        b.text(
            0.02,
            yk - 0.02,
            str(n),
            transform=b.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="circle", facecolor="white", edgecolor="k"),
        )
        b.text(
            0.10,
            yk - 0.02,
            head,
            transform=b.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        b.text(0.10, yk - 0.07, body, transform=b.transAxes, fontsize=8.5, va="top")
    panel_letters([a, b])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    write_caption(
        path,
        """
How the river bed is set from the observed water surface (schematic, no scale). (a) Long profile from the mouth (left) to the upstream model boundary (right); circled numbers mark the steps listed in (b).
Grey dots: water levels read from the terrain model along the river centre, which come in metre steps (1). Blue line: the fitted surface, sea-pinned, never rising towards the sea and smoothed (2). Orange band: the flat reach near the sea where the surface slope falls below the minimum slope used for the depth (3). Arrow h: the depth of uniform flow at bankfull discharge (Manning), which places the bed (black line) below the surface; blue shading is the channel (4). Dashed cyan line: the water level computed back on that bed with the gradually varied flow equation. It follows the surface where the true slope is used and rises above it on the flat reach, where the minimum slope makes the channel shallower than the flat surface would need (5). Dash-dotted brown line: dike crests along the river, set from a run with the river walled in at the protection-standard discharge (6). Dotted cyan line: calm sea level.
(b) The steps in words.
""",
    )


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


def fig_long_profile(basin_id: str, t: pd.DataFrame, path: Path) -> None:
    m = _mainstem(basin_id, t)
    km = m["dist_km"]
    fig, (a, b) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    a.plot(km, m["grid_beside_m"], color="tab:brown", lw=0.8, alpha=0.7)
    a.plot(km, m["bank_top_m"], ".", color="tab:brown", ms=3)
    a.plot(km, m["fathom_level_m"], ".", color="0.55", ms=3)
    a.plot(km, m["swot_m"], "x", color="tab:purple", ms=4)
    a.plot(km, m["level_m"], color="tab:blue", lw=2)
    a.plot(km, m["level_q_bf_m"], color="tab:cyan", lw=1.2, ls="--")
    a.plot(km, m["level_q_mean_m"], color="tab:green", lw=1.2, ls="--")
    a.plot(km, m["spinup_end_m"], color="tab:green", lw=1.0, alpha=0.6)
    a.plot(km, m["bed_m"], color="k", lw=1.6)
    lo, hi = float(np.nanmin(m["bed_m"])), float(np.nanmax(m["level_m"]))
    # valley walls beside incised reaches would set the scale otherwise
    a.set_ylim(
        lo - 2,
        min(float(np.nanpercentile(m["bank_top_m"], 95)) + 3, hi + 0.4 * (hi - lo)),
    )
    a.set_ylabel("level (m, model datum)")
    a.grid(alpha=0.3)
    b.semilogy(km, np.maximum(m["slope"], 1e-8), color="tab:blue", lw=1)
    b.semilogy(km, m["depth_slope"], color="tab:orange", lw=1.5)
    fl = m["on_floor"].astype(bool)
    b.semilogy(km[fl], m["depth_slope"][fl], "o", color="tab:orange", ms=2)
    b.set_ylabel("slope (-)")
    b.set_ylim(1e-7, 1e-2)
    b2 = b.twinx()
    b2.plot(km, m["depth_m"], color="k", lw=1)
    b2.set_ylabel("depth (m)")
    b.invert_xaxis()
    b.set_xlabel("distance from the mouth along the mainstem (km)")
    b.grid(alpha=0.3)
    panel_letters([a, b])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    floor_km = km[fl]
    floor_txt = (
        f"The slope floor applies over {floor_km.min():.0f}-{floor_km.max():.0f} km from the mouth."
        if fl.any()
        else "No node of this path uses the slope floor."
    )
    write_caption(
        path,
        f"""
Long profile of the river bed and its inputs, basin {basin_id}, along the mainstem of the largest inflow; the mouth is on the right.
(a) Levels. Grey dots: terrain-model (FathomDEM) water level at the river centre. Purple crosses: SWOT median water level per SWORD node. Thick blue line: fitted water surface (rule 10), from which the bed is set. Black line: river bed. Dashed cyan: water level computed on the bed at bankfull discharge (the forward check). Dashed green: computed level at mean discharge with a calm sea; thin green: SFINCS level at the end of the spin-up (mean discharge, calm sea). Brown dots: bank top, the 90th percentile of the terrain within 200 m beyond each bank (lower bank). Thin brown line: SFINCS grid elevation one cell beyond the bank (lower side).
(b) Slope and depth. Thin blue: surface slope. Orange: slope used for the depth, the surface slope floored at 1e-5; orange dots mark nodes on the floor. Black (right axis): depth below the fitted surface. Depth follows h = (n Q / (W sqrt(S)))^(3/5), so where the slope is constant the depth still changes with the channel width W (e.g. the channel widening towards the mouth makes it shallower) and with the discharge Q, which drops at every bifurcation. {floor_txt}
""",
    )


def _alternatives_table(options: dict) -> pd.DataFrame:
    rows = []
    for basin_id, o in options.items():
        surf = o["surface"].set_index(["surface", "design_discharge"])
        cap = o["capacity"].set_index("option")
        for label, skey, ckey in ALTERNATIVES:
            s = surf.loc[skey] if skey is not None else None
            c = cap.loc[ckey] if ckey is not None else None
            src = s if s is not None else c
            rows.append(
                {
                    "basin_id": basin_id,
                    "alternative": label,
                    "forward_median_m": float(src["forward_median_m"]),
                    "forward_p95_m": float(src["forward_p95_m"]),
                    "overflow_bf": float(c["share_q_cap_below_q_bf"])
                    if c is not None
                    else float(s["freeboard_share_below_0"]),
                    "overflow_mean": float(c["share_q_cap_below_q_mean"])
                    if c is not None
                    else np.nan,
                    "on_floor_share": float(src["on_floor_share"]),
                }
            )
    return pd.DataFrame(rows)


def fig_bed_alternatives(options: dict, path: Path) -> None:
    table = _alternatives_table(options)
    table.to_csv(path.with_suffix(".csv"), index=False, float_format="%.3f")
    basins = list(options)
    labels = [a[0] for a in ALTERNATIVES]
    fig, axes = plt.subplots(
        1, len(basins), figsize=(7.5 * len(basins), 6.5), sharey=True, squeeze=False
    )
    colours = {True: "#a6d96a", False: "#f4a582", None: "0.92"}
    for ax, basin_id in zip(axes[0], basins):
        g = table[table["basin_id"] == basin_id].set_index("alternative").loc[labels]
        for j, (col, (_, test)) in enumerate(CRITERIA.items()):
            for i, v in enumerate(g[col]):
                ok = None if test is None or not np.isfinite(v) else bool(test(v))
                ax.add_patch(
                    plt.Rectangle(
                        (j, i), 1, 1, facecolor=colours[ok], edgecolor="white", lw=2
                    )
                )
                text = (
                    "n/a"
                    if not np.isfinite(v)
                    else (f"{v:.2f}" if col.startswith("forward") else f"{v:.0%}")
                )
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=9)
        ax.set_xlim(0, len(CRITERIA))
        ax.set_ylim(len(labels), 0)
        ax.set_xticks(np.arange(len(CRITERIA)) + 0.5)
        ax.set_xticklabels([c[0] for c in CRITERIA.values()], fontsize=8)
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels, fontsize=9)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    panel_letters(axes[0], xy=(-0.02, 1.12))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    write_caption(
        path,
        f"""
Alternative ways to set the river bed, scored against the criteria a bed must meet; one panel per basin ({", ".join(f"({chr(97 + k)}) {b}" for k, b in enumerate(basins))}). Green: criterion met; red: not met; grey: shown for information. Values in the cells; the table is in bed_alternatives.csv.

Alternatives (rows). Each sets the bed as surface minus depth (concept.png) and differs in one or more choices:
- water surface: the terrain model's (FathomDEM) river-centre levels, or SWOT satellite water levels (SWORD v17c node medians), with the terrain model where SWOT has no observation;
- fit: "two-scale" smooths the level and the slope over separate lengths (production); "single 20 km / 40 km" takes both from one fit over that length;
- design discharge: the discharge whose uniform-flow depth sets the bed: bankfull (GloFAS, about the 1.5-year flood; production) or the recent median (GloFAS daily median since 2023-07, the period SWOT observes);
- sea inversion: on flat reaches connected to the sea the depth is not taken from the (floored) slope but scaled with width, h = c W^b, with c chosen so that computed levels from the sea meet the surface upstream.

Criteria (columns):
- forward check: the water level computed back on the bed at the design discharge minus the surface it was set from; median within +-0.25 m and 95th percentile at most 0.75 m;
- overflowing at Q_bankfull / at Q_mean: share of river nodes whose channel capacity (the discharge at which the computed level reaches the bank top) is below the bankfull / mean discharge; at most 10 % / 2 %. A bed too shallow for the mean flow floods the floodplain permanently. n/a: not computed for that alternative;
- on the slope floor: share of nodes whose depth uses the minimum slope instead of the observed one.
""",
    )


def figures(basin_ids: list[str], river: str, fr_ok: float, fr_limit: float) -> None:
    out = out_dir()
    nodes = {
        b: pd.read_csv(out_dir(b) / "nodes.csv", dtype={"reach_id": str})
        for b in basin_ids
    }
    fig_concept(out / "concept.png")
    fig_froude_1d(nodes, out / "froude_1d.png", fr_ok, fr_limit)
    data = {}
    for b in basin_ids:
        f = out_dir(b) / f"froude_sfincs_{river}.npz"
        data[b] = dict(np.load(f)) if f.exists() else None
    fig_froude_sfincs(data, river, out / f"froude_sfincs_{river}.png", fr_ok, fr_limit)
    for b, t in nodes.items():
        fig_long_profile(b, t, out / f"long_profile_{b}.png")
    options = {
        b: {
            "surface": pd.read_csv(out_dir(b) / "surface_options.csv"),
            "capacity": pd.read_csv(out_dir(b) / "channel_capacity.csv"),
        }
        for b in basin_ids
    }
    fig_bed_alternatives(options, out / "bed_alternatives.png")


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("action", choices=["compute", "figures"])
    parser.add_argument("basin_ids", nargs="+")
    parser.add_argument("--river", default="river_100_03SLR")
    parser.add_argument("--fr-ok", type=float, default=0.3)
    parser.add_argument("--fr-limit", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.action == "compute":
        for b in args.basin_ids:
            compute(str(b), args.river)
    else:
        figures([str(b) for b in args.basin_ids], args.river, args.fr_ok, args.fr_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
