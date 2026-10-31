"""Route a river event's inflow down the mainstem with variable-parameter Muskingum-Cunge and compare the peak travel with SFINCS.

Mainstem: rule 11's walled-travel profile (inflow crossing to the network end),
on the river-bed node table's bed and width. Per node a compound section:
rectangular channel (channel_manning_n) plus, per side, the floodplain DEM
(rule 06's conditioned DEM, rule 09's roughness) sampled on normals out to
FLOODPLAIN_M from the bank, median of three profiles ALONG_M apart. A side
conveys and stores only over the run of samples below the level contiguous
with the channel, and with --crests only once the level is above the lowest
skeleton weir crest within WEIR_RADIUS_M on that side. Bed slope: the walled
calibration surface slope, smoothed over SLOPE_SMOOTH_M, floored at
MIN_SLOPE. Routing: Ponce & Chaganti (1994) three-point VPMC on the node
spacing; no backwater, no tide, no lateral inflow.

Variants: channel (in-bank only), crests (floodplain behind the crests),
open (floodplain without crests). Compared with the SFINCS run's mouth
cross-section (raw, and both series Godin-filtered as in peak_lag.py).

Writes 90_mc_peak_travel_<run>.png and _<run>.csv to
<results_dir>/<basin>/figures/analysis/.

Usage: python analysis/mc_peak_travel.py [basin_id] [--run river_100_03SLR]
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
from _common import analysis_path, script_parser
from _phase1 import basin_paths, git_hash, load_config
from peak_lag import godin, peak_hr
from pyproj import Transformer

log = logging.getLogger(__name__)

FLOODPLAIN_M = 5000.0
STEP_M = 25.0
ALONG_M = 200.0
WEIR_RADIUS_M = 600.0
SLOPE_SMOOTH_M = 5000.0
MIN_SLOPE = 2e-5
FLOODPLAIN_N_FALLBACK = 0.06
DZ_M = 0.05
DT_S = 120.0


def mainstem(basin_id: str) -> pd.DataFrame:
    p = basin_paths(basin_id)
    bed_dir = p["node_table"].parent
    walled = pd.read_csv(bed_dir / f"{basin_id}_walled_travel.csv")
    walled["reach_id"] = walled["reach_id"].astype(str)
    order = {r: k for k, r in enumerate(dict.fromkeys(walled["reach_id"]))}
    nodes = pd.read_csv(p["node_table"])
    nodes["reach_id"] = nodes["reach_id"].astype(str)
    prof = nodes[nodes["reach_id"].isin(order)].copy()
    prof["_k"] = prof["reach_id"].map(order)
    prof = prof.sort_values(["_k", "dist_to_sea_m"], ascending=[True, False])
    prof = prof.drop(columns="_k").reset_index(drop=True)
    if len(prof) != len(walled):
        raise ValueError(
            f"mainstem has {len(prof)} nodes, walled profile {len(walled)}"
        )
    seg = np.hypot(np.diff(prof["x"]), np.diff(prof["y"]))
    prof["s_m"] = np.concatenate([[0.0], np.cumsum(seg)])
    level = pd.Series(walled["level_m"].to_numpy(), index=prof["s_m"])
    n_win = max(int(round(SLOPE_SMOOTH_M / np.median(seg))), 3)
    smooth = level.rolling(n_win, center=True, min_periods=1).mean().to_numpy()
    prof["slope"] = np.maximum(-np.gradient(smooth, prof["s_m"]), MIN_SLOPE)
    gx, gy = np.gradient(prof["x"].to_numpy()), np.gradient(prof["y"].to_numpy())
    norm = np.hypot(gx, gy)
    prof["tx"], prof["ty"] = gx / norm, gy / norm
    return prof


def side_profiles(prof: pd.DataFrame, raster_path, crs) -> np.ndarray:
    """Return (2, n_node, n_sample) raster values on the left/right normals; NaN off-raster."""
    d = np.arange(0.0, FLOODPLAIN_M + 1e-9, STEP_M)
    along = np.array([-ALONG_M, 0.0, ALONG_M])
    x, y = prof["x"].to_numpy(), prof["y"].to_numpy()
    tx, ty = prof["tx"].to_numpy(), prof["ty"].to_numpy()
    half_w = prof["width_m"].to_numpy() / 2.0
    out = np.full((2, len(prof), d.size), np.nan)
    with rasterio.open(raster_path) as src:
        to_r = Transformer.from_crs(crs, src.crs, always_xy=True)
        for k, side in enumerate((1.0, -1.0)):
            off = half_w[:, None, None] + d[None, None, :]
            px = x[:, None, None] + along[None, :, None] * tx[:, None, None]
            py = y[:, None, None] + along[None, :, None] * ty[:, None, None]
            px = px - side * off * ty[:, None, None]
            py = py + side * off * tx[:, None, None]
            rx, ry = to_r.transform(px.ravel(), py.ravel())
            v = np.array([s[0] for s in src.sample(zip(rx, ry))], dtype=float)
            if src.nodata is not None:
                v[v == src.nodata] = np.nan
            out[k] = np.nanmedian(v.reshape(px.shape), axis=1)
    return out


def side_crests(prof: pd.DataFrame, weir_path) -> np.ndarray:
    """Return (2, n_node) lowest weir crest per side within WEIR_RADIUS_M; -inf where none."""
    from src.sfincs_run import read_weir_file

    weirs = read_weir_file(weir_path)
    mid = weirs.geometry.interpolate(0.5, normalized=True)
    wx, wy, wz = mid.x.to_numpy(), mid.y.to_numpy(), weirs["elevation"].to_numpy()
    out = np.full((2, len(prof)), -np.inf)
    for i, r in enumerate(prof.itertuples()):
        dx, dy = wx - r.x, wy - r.y
        near = np.hypot(dx, dy) < WEIR_RADIUS_M + r.width_m / 2.0
        left = r.tx * dy - r.ty * dx > 0
        for k, on_side in enumerate((left, ~left)):
            sel = near & on_side
            if sel.any():
                out[k, i] = wz[sel].min()
    return out


def section_tables(prof, ground, rough, crests, n_channel, variant):
    """Return per node (Q, A, top width, celerity) tables over levels from the bed up."""
    tables = []
    for i, r in enumerate(prof.itertuples()):
        z = r.bed_m + np.arange(DZ_M, 30.0, DZ_M)
        h_c = z - r.bed_m
        a = r.width_m * h_c
        q = a * h_c ** (2.0 / 3.0) * np.sqrt(r.slope) / n_channel
        if variant != "channel":
            for k in range(2):
                g = ground[k, i]
                n_f = np.where(
                    np.isfinite(rough[k, i]), rough[k, i], FLOODPLAIN_N_FALLBACK
                )
                depth = z[:, None] - g[None, :]
                wet = np.cumprod(np.nan_to_num(depth, nan=-1.0) > 0, axis=1) > 0
                if variant == "crests":
                    wet &= (z > crests[k, i])[:, None]
                dw = np.where(wet, depth, 0.0)
                a = a + dw.sum(axis=1) * STEP_M
                q = (
                    q
                    + (dw ** (5.0 / 3.0) / n_f[None, :] * np.sqrt(r.slope)).sum(axis=1)
                    * STEP_M
                )
        q = np.maximum.accumulate(q)
        top = np.gradient(a, z)
        cel = np.gradient(q, a)
        tables.append((q, a, top, np.maximum(cel, 1e-3)))
    return tables


def route(q_in: np.ndarray, dx: np.ndarray, slope: np.ndarray, tables) -> np.ndarray:
    """Return (n_time, n_node) VPMC discharge for inflow q_in at DT_S steps."""
    n_t, n_x = len(q_in), len(dx) + 1
    q = np.empty((n_t, n_x))
    q[0, :] = q_in[0]
    for n in range(n_t - 1):
        q[n + 1, 0] = q_in[n + 1]
        for j in range(n_x - 1):
            qt, _, top, cel = tables[j + 1]
            q_ref = (q[n, j] + q[n, j + 1] + q[n + 1, j]) / 3.0
            c = np.interp(q_ref, qt, cel)
            b = np.interp(q_ref, qt, top)
            cr = c * DT_S / dx[j]
            dd = q_ref / (b * slope[j + 1] * c * dx[j])
            den = 1.0 + cr + dd
            q[n + 1, j + 1] = (
                (-1.0 + cr + dd) * q[n + 1, j]
                + (1.0 + cr - dd) * q[n, j]
                + (1.0 - cr + dd) * q[n, j + 1]
            ) / den
    return q


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_id", nargs="?", default="2433835")
    parser.add_argument("--run", default="river_100_03SLR")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    basin_id, run, tag = str(args.basin_id), args.run, git_hash()
    cfg = load_config()
    p = basin_paths(basin_id)
    from src.grid import load_grid_def

    crs = load_grid_def(p["sfincs_grid"])["crs"]
    prof = mainstem(basin_id)
    dom = p["prep"] / "domain"
    ground = side_profiles(prof, dom / f"{basin_id}_elevation_conditioned.tif", crs)
    rough = side_profiles(prof, dom / f"{basin_id}_roughness.tif", crs)
    crests = side_crests(prof, p["skeleton"] / "sfincs.weir")
    n_channel = float(cfg["river_processing"]["channel_manning_n"])

    run_dir = p["root"] / "runs" / run / "sfincs"
    dis = np.loadtxt(run_dir / "sfincs.dis", ndmin=2)
    t_dis_hr, q_dis = dis[:, 0] / 3600.0, dis[:, 1:].sum(axis=1)
    t_hr = np.arange(t_dis_hr[0], t_dis_hr[-1] + 1e-9, DT_S / 3600.0)
    q_in = np.interp(t_hr, t_dis_hr, q_dis)

    his = xr.open_dataset(run_dir / "sfincs_his.nc", decode_times=False)
    t_his = his["time"].values.astype(float) / 3600.0
    mouth = gpd.read_file(p["skeleton"] / f"{basin_id}_mouths.gpkg").iloc[0]
    names = [str(n).strip() for n in his["crosssection_name"].values.astype(str)]
    q_sf = np.abs(his["crosssection_discharge"].values[:, names.index(mouth["name"])])
    his.close()
    dt_his = float(np.median(np.diff(t_his)))
    q_in_his = np.interp(t_his, t_dis_hr, q_dis)
    # The mouth section sits mouth_upstream_m up the last reach; route to its node.
    s_mouth = prof["s_m"].iloc[-1] - float(
        cfg["sfincs"]["observation_points"]["mouth_upstream_m"]
    )
    j_mouth = int(np.argmin(np.abs(prof["s_m"].to_numpy() - s_mouth)))

    t_in_peak = peak_hr(t_dis_hr, q_dis)
    rows = [
        {
            "model": "SFINCS raw",
            "peak_travel_hr": peak_hr(t_his, q_sf) - t_in_peak,
            "peak_m3s": float(np.nanmax(q_sf)),
        },
        {
            "model": "SFINCS Godin (both filtered)",
            "peak_travel_hr": peak_hr(t_his, godin(q_sf, dt_his))
            - peak_hr(t_his, godin(q_in_his, dt_his)),
            "peak_m3s": float(np.nanmax(godin(q_sf, dt_his))),
        },
    ]
    dx = np.diff(prof["s_m"].to_numpy())
    slope = prof["slope"].to_numpy()
    fig, (ax, ax_c) = plt.subplots(2, 1, figsize=(12, 8))
    ax.plot(t_dis_hr, q_dis, "k-", lw=1.2, label=f"inflow (peak {t_in_peak:.0f} h)")
    ax.plot(t_his, q_sf, color="0.6", lw=0.6, label="SFINCS mouth section (raw)")
    for variant, color in (
        ("channel", "tab:blue"),
        ("crests", "tab:green"),
        ("open", "tab:orange"),
    ):
        tables = section_tables(prof, ground, rough, crests, n_channel, variant)
        q = route(q_in, dx, slope, tables)
        q_m = q[:, j_mouth]
        k_pk = int(np.argmax(q_in))
        q_at_peak = q[k_pk, :]
        cel = np.array(
            [np.interp(qq, tb[0], tb[3]) for qq, tb in zip(q_at_peak, tables)]
        )
        rows.append(
            {
                "model": f"MC {variant}",
                "peak_travel_hr": peak_hr(t_hr, q_m) - t_in_peak,
                "peak_m3s": float(q_m.max()),
                "celerity_median_ms": float(np.median(cel)),
            }
        )
        ax.plot(t_hr, q_m, color=color, lw=1.4, label=f"MC {variant}")
        ax_c.plot(
            (prof["s_m"].iloc[-1] - prof["s_m"]) / 1000.0,
            cel,
            color=color,
            label=f"MC {variant}",
        )
        log.info(f"{variant}: routed")
    ax.set_xlabel("hours from tref")
    ax.set_ylabel("discharge (m3/s)")
    ax.set_xlim(t_in_peak - 72, t_in_peak + 96)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax_c.set_xlabel("distance to the network end (km)")
    ax_c.set_ylabel("celerity at the inflow-peak discharge (m/s)")
    ax_c.invert_xaxis()
    ax_c.grid(alpha=0.3)
    ax_c.legend(fontsize=8)
    fig.suptitle(
        f"Muskingum-Cunge peak travel | {basin_id} | {run} | {tag}", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(analysis_path(basin_id, __file__, f"{run}.png"), dpi=120)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df["attenuation_pct"] = 100.0 * (df["peak_m3s"] / q_dis.max() - 1.0)
    df.to_csv(
        analysis_path(basin_id, __file__, f"{run}.csv"),
        index=False,
        float_format="%.3f",
    )
    print(
        f"inflow peak {q_dis.max():.0f} m3/s at {t_in_peak:.0f} h; mainstem "
        f"{prof['s_m'].iloc[-1] / 1000:.1f} km, mouth node {j_mouth}"
    )
    print(df.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
