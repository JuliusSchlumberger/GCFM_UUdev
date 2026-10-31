"""Check event-run timing: when each cell peaks, and whether the river lag matters.

For every finished event run (<results_dir>/{basin}/runs/{scenario}/sfincs/)
reports the time of each wet cell's maximum water level relative to the surge
peak (from the hourly zsmax envelope), the share of cells peaking in the run's
last 6 h (window too short), the flatness of the river crest (tau95/tau90: hours
before the peak that the summed inflow is within 5/10 % of it) and the
gravity-wave travel time from the main inflow to the open sea boundary along
wet cells at flood stage (sum of dx / sqrt(g h)). The lag is immaterial when
that travel time is below tau95. Cells peaking in the last 6 h that were dry at
the event start are grouped into connected compartments; their hourly volume
(level-based, zs - zb over the cell area) shows whether they are still filling
when the run ends. Writes check_event_timing_summary.csv and _compartments.csv
to <results_dir>/figures/analysis/ and, per basin, two maps
90_check_event_timing_<scenario>_time_of_max.png and _late_compartments.png to
<results_dir>/<basin>/figures/analysis/.

Usage:
    python analysis/check_event_timing.py                  # every basin with an event run
    python analysis/check_event_timing.py 2433835 1248635  # only these basins
    python analysis/check_event_timing.py --scenario noslr
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _common import analysis_path, script_parser, site_path
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

log = logging.getLogger(__name__)

G_MS2 = 9.81
LATE_HR = 6.0  # a maximum this close to the run end means the window may cut it off
MIN_RISE_M = 0.05  # cells whose level rises less than this over the run have no meaningful peak time
WET_M = 0.05
SETTLED_HR = 12.0  # the peak water level everywhere should be reached by this long after the surge peak
MIN_COMPARTMENT_CELLS = 10


def read_inp(path) -> dict:
    out = {}
    for line in open(path):
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_timestamp(value: str) -> pd.Timestamp:
    day, clock = value.split()
    return pd.Timestamp(f"{day}T{clock}")


def hours_before_peak_within(t_hr: np.ndarray, q: np.ndarray, frac: float) -> float:
    tt = np.arange(t_hr[0], t_hr[-1] + 1e-9, 0.25)
    qq = np.interp(tt, t_hr, q)
    ip = int(np.argmax(qq))
    below = np.nonzero(qq[:ip] < frac * qq[ip])[0]
    return float(tt[ip] - tt[below[-1]]) if below.size else np.nan


def _cell_centres(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = ds.corner_x.values, ds.corner_y.values
    xc = 0.25 * (cx[:-1, :-1] + cx[1:, :-1] + cx[:-1, 1:] + cx[1:, 1:])
    yc = 0.25 * (cy[:-1, :-1] + cy[1:, :-1] + cy[:-1, 1:] + cy[1:, 1:])
    return xc, yc


def gravity_wave_travel(
    depth_m: np.ndarray,
    wet: np.ndarray,
    start: tuple[int, int],
    targets: np.ndarray,
    dx_m: float,
):
    """Return (hours, km) of the fastest wet 4-neighbour path from start to any target cell."""
    idx = -np.ones(wet.shape, dtype=np.int64)
    idx[wet] = np.arange(int(wet.sum()))
    rows, cols, weights = [], [], []
    for dr, dc in ((0, 1), (1, 0)):
        a = wet[: wet.shape[0] - dr, : wet.shape[1] - dc] & wet[dr:, dc:]
        r, c = np.nonzero(a)
        h = 0.5 * (depth_m[r, c] + depth_m[r + dr, c + dc])
        rows.append(idx[r, c])
        cols.append(idx[r + dr, c + dc])
        weights.append(dx_m / np.sqrt(G_MS2 * h))
    rows, cols, weights = (
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(weights),
    )
    n = int(wet.sum())
    graph = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    if idx[start] < 0:
        return np.nan, np.nan
    t_s, pred = dijkstra(
        graph, directed=False, indices=int(idx[start]), return_predecessors=True
    )
    tgt = idx[targets & wet]
    tgt = tgt[np.isfinite(t_s[tgt])]
    if not tgt.size:
        return np.nan, np.nan
    best = int(tgt[np.argmin(t_s[tgt])])
    steps, node = 0, best
    while node != idx[start] and node >= 0:
        node = pred[node]
        steps += 1
    return float(t_s[best] / 3600.0), steps * dx_m / 1e3


def _label_compartments(mask: np.ndarray, min_cells: int) -> tuple[np.ndarray, int]:
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    small = sizes < min_cells
    small[0] = False
    labels[small[labels]] = 0
    return ndimage.label(labels > 0)


def _compartment_volumes(
    labels: np.ndarray, n: int, zb: np.ndarray, zs_frames, cell_area_m2: float
) -> np.ndarray:
    """Return the level-based volume (m3) of each compartment per frame, shape (frames, n)."""
    flat = labels.ravel()
    inside = flat > 0
    out = []
    for zs in zs_frames:
        h = np.nan_to_num(np.maximum(zs - zb, 0.0)).ravel()
        out.append(
            np.bincount(flat[inside], weights=h[inside], minlength=n + 1)[1:]
            * cell_area_m2
        )
    return np.vstack(out)


def _compartment_table(
    labels: np.ndarray,
    n: int,
    zb: np.ndarray,
    wet0: np.ndarray,
    zs_frames,
    t_hr: np.ndarray,
    t_surge_hr: float,
    cell_area_m2: float,
) -> pd.DataFrame:
    vol = _compartment_volumes(labels, n, zb, zs_frames, cell_area_m2)
    i_back = int(np.searchsorted(t_hr, t_hr[-1] - LATE_HR))
    v_surge = np.array([np.interp(t_surge_hr, t_hr, vol[:, k]) for k in range(n)])
    near_water = np.isin(
        np.arange(1, n + 1), labels[ndimage.binary_dilation(wet0) & (labels > 0)]
    )
    cells = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame(
            {
                "compartment": np.arange(1, n + 1),
                "cells": cells,
                "area_km2": cells * cell_area_m2 / 1e6,
                "volume_at_surge_m3": v_surge,
                "volume_end_m3": vol[-1],
                "share_gained_after_surge": np.where(
                    vol[-1] > 0, (vol[-1] - v_surge) / vol[-1], np.nan
                ),
                "inflow_last6h_m3s": (vol[-1] - vol[i_back])
                / ((t_hr[-1] - t_hr[i_back]) * 3600.0),
                "mean_rise_last6h_cm_hr": (vol[-1] - vol[i_back])
                / (cells * cell_area_m2)
                / (t_hr[-1] - t_hr[i_back])
                * 100.0,
                "borders_initial_water": near_water,
            }
        )


def check_basin(
    results_dir, basin_id: str, scenario: str
) -> tuple[dict, pd.DataFrame] | None:
    run = results_dir / basin_id / "runs" / scenario / "sfincs"
    if not (run / "sfincs_map.nc").exists():
        return None
    inp = read_inp(run / "sfincs.inp")
    tref, tstart, tstop = (parse_timestamp(inp[k]) for k in ("tref", "tstart", "tstop"))
    offset_hr = (tstart - tref).total_seconds() / 3600.0
    run_hr = (tstop - tstart).total_seconds() / 3600.0

    bzs = np.loadtxt(run / "sfincs.bzs", ndmin=2)
    dis = np.loadtxt(run / "sfincs.dis", ndmin=2)
    t_surge_hr = bzs[np.argmax(bzs[:, 1:].max(axis=1)), 0] / 3600.0 - offset_hr
    q_total = dis[:, 1:].sum(axis=1)
    t_q_hr = dis[:, 0] / 3600.0
    t_river_hr = t_q_hr[np.argmax(q_total)] - offset_hr

    with xr.open_dataset(run / "sfincs_map.nc") as ds:
        zsmax = ds.zsmax.values
        if zsmax.ndim == 2 or zsmax.shape[0] < 2:
            log.warning(
                f"{basin_id}: zsmax has no time axis (dtmaxout >= run length); timing skipped"
            )
            return None
        t_max_hr = (ds.timemax.values - np.datetime64(tstart)) / np.timedelta64(1, "h")
        zb = ds.zb.values
        zs0 = ds.zs.values[0]
        msk = ds.msk.values
        xc, yc = _cell_centres(ds)

    finite = np.isfinite(zsmax)
    peak = np.nanmax(np.where(finite, zsmax, np.nan), axis=0)
    wet = np.isfinite(peak) & (peak - zb > WET_M)
    i_peak = np.argmax(np.where(finite, zsmax, -np.inf), axis=0)
    rel_hr = np.where(wet, t_max_hr[i_peak] - t_surge_hr, np.nan)
    rising = wet & (peak - np.where(np.isfinite(zs0), zs0, zb) > MIN_RISE_M)
    rel = rel_hr[rising]
    by_settled = t_max_hr <= t_surge_hr + SETTLED_HR
    late_gain_m = (
        peak
        - np.nanmax(np.where(finite[by_settled], zsmax[by_settled], -np.inf), axis=0)
    )[rising]
    late_gain_m = np.where(
        np.isfinite(late_gain_m), late_gain_m, peak[rising] - zb[rising]
    )

    src = np.loadtxt(run / "sfincs.src", usecols=(0, 1), ndmin=2)
    main = int(np.argmax(dis[:, 1:].max(axis=0)))
    d2 = (xc - src[main, 0]) ** 2 + (yc - src[main, 1]) ** 2
    start = np.unravel_index(
        np.nanargmin(np.where(np.isfinite(d2), d2, np.inf)), d2.shape
    )
    t_wave_hr, path_km = gravity_wave_travel(
        peak - zb, wet, start, msk == 2, float(inp["dx"])
    )

    tau95 = hours_before_peak_within(t_q_hr, q_total, 0.95)
    rec = {
        "basin_id": basin_id,
        "run_hr": run_hr,
        "surge_peak_hr": t_surge_hr,
        "river_peak_hr": t_river_hr,
        "peak_q_m3s": float(q_total.max()),
        "tau95_hr": tau95,
        "tau90_hr": hours_before_peak_within(t_q_hr, q_total, 0.90),
        "path_km": path_km,
        "wave_travel_hr": t_wave_hr,
        "lag_material": bool(
            np.isfinite(t_wave_hr) and np.isfinite(tau95) and t_wave_hr > tau95
        ),
        "rising_cells": int(rising.sum()),
        "tmax_after_surge_p50_hr": float(np.percentile(rel, 50)),
        "tmax_after_surge_p95_hr": float(np.percentile(rel, 95)),
        "tmax_after_surge_p99_hr": float(np.percentile(rel, 99)),
        "tmax_after_surge_max_hr": float(rel.max()),
        "share_peaking_last_6h": float(
            np.mean(t_max_hr[i_peak][rising] >= t_max_hr[-1] - LATE_HR)
        ),
        "share_rising_5cm_after_settled": float(np.mean(late_gain_m > 0.05)),
        "late_rise_p99_m": float(np.percentile(late_gain_m, 99)),
        "hours_after_surge_in_run": float(t_max_hr[-1] - t_surge_hr),
    }

    late = rising & (t_max_hr[i_peak] >= t_max_hr[-1] - LATE_HR)
    wet0 = np.isfinite(zs0) & (zs0 - zb > WET_M)
    labels, n_comp = _label_compartments(late & ~wet0, MIN_COMPARTMENT_CELLS)
    cell_area_m2 = float(inp["dx"]) * float(inp.get("dy", inp["dx"]))
    with xr.open_dataset(run / "sfincs_map.nc") as ds:
        t_zs_hr = (ds.time.values - np.datetime64(tstart)) / np.timedelta64(1, "h")
        comp = _compartment_table(
            labels,
            n_comp,
            zb,
            wet0,
            (ds.zs[i].values for i in range(ds.sizes["time"])),
            t_zs_hr,
            t_surge_hr,
            cell_area_m2,
        )
    comp.insert(0, "basin_id", basin_id)
    rec |= {
        "late_cells": int(late.sum()),
        "late_share_wet_at_start": float(np.mean(wet0[late])) if late.any() else 0.0,
        "late_compartments": n_comp,
        "late_compartment_area_km2": float(comp["area_km2"].sum()),
        "late_inflow_last6h_m3s": float(comp["inflow_last6h_m3s"].sum()),
        "late_share_gained_after_surge": float(
            1.0 - comp["volume_at_surge_m3"].sum() / comp["volume_end_m3"].sum()
        )
        if comp["volume_end_m3"].sum() > 0
        else np.nan,
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        np.where(rising, rel_hr, np.nan),
        origin="lower",
        cmap="RdBu_r",
        vmin=-24,
        vmax=24,
    )
    ax.plot(start[1], start[0], "k^", ms=8)
    fig.colorbar(im, ax=ax, label="time of max water level - surge peak (h)")
    ax.set_title(
        f"{basin_id} {scenario}: p95 {rec['tmax_after_surge_p95_hr']:+.0f} h, "
        f"wave {t_wave_hr:.1f} h vs tau95 {tau95:.1f} h"
    )
    fig.savefig(
        analysis_path(basin_id, __file__, f"{scenario}_time_of_max.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    rise = np.full(labels.shape, np.nan)
    if n_comp:
        rise_by_label = np.concatenate(
            [[np.nan], comp["mean_rise_last6h_cm_hr"].values]
        )
        rise = np.where(labels > 0, rise_by_label[labels], np.nan)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(np.where(wet0, 1.0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=3)
    im = ax.imshow(rise, origin="lower", cmap="viridis", vmin=0, vmax=3)
    ax.plot(start[1], start[0], "k^", ms=8)
    fig.colorbar(im, ax=ax, label="compartment mean rise, last 6 h (cm/h)")
    ax.set_title(
        f"{basin_id} {scenario}: {n_comp} late compartments, "
        f"{rec['late_compartment_area_km2']:.0f} km2\n"
        f"inflow {rec['late_inflow_last6h_m3s']:.0f} m3/s over the last 6 h"
    )
    fig.savefig(
        analysis_path(basin_id, __file__, f"{scenario}_late_compartments.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)
    return rec, comp


def main() -> None:
    parser = script_parser(__doc__)
    parser.add_argument(
        "basin_ids",
        nargs="*",
        help="basin ids (default: every basin with an event run)",
    )
    parser.add_argument("--scenario", default="default")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results_dir = site_path("results_dir")
    basin_ids = args.basin_ids or sorted(
        p.name for p in results_dir.iterdir() if p.name.isdigit()
    )
    results = [
        r for b in basin_ids if (r := check_basin(results_dir, b, args.scenario))
    ]
    if not results:
        raise SystemExit(
            f"No event runs found for scenario {args.scenario!r} under {results_dir}"
        )
    df = pd.DataFrame([rec for rec, _ in results])
    df.to_csv(
        analysis_path(None, __file__, "summary.csv"), index=False, float_format="%.2f"
    )
    pd.concat([comp for _, comp in results]).to_csv(
        analysis_path(None, __file__, "compartments.csv"),
        index=False,
        float_format="%.3f",
    )
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
