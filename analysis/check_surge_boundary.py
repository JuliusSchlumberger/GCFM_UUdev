"""Screen the surge stations forcing the sea boundary and measure SFINCS's own amplification.

The water-level boundary takes each COAST-RP station at its own coordinates
(hydromt keeps stations within boundary_setup.water_level_buffer_m of a
boundary cell, rule 14); SFINCS gives every boundary cell the inverse-distance
mean of its two nearest stations. A station inside a bay carries GTSM's local
amplification out to the open boundary.

Station screen (per station, needs the model skeleton):
- position: signed distance to the delta polygon (rule 07a selects on its
  absolute value) and to the nearest water-level boundary cell, negative
  inside; whether the station's cell is active;
- bed: skeleton bed level and raw GEBCO (local MSL) at the station;
- level: raw storm tide at the RP and its anomaly, the station minus the
  median of all stations (flag > --anomaly-tol-m);
- boundary share: fraction of boundary cells weighted by the station.

Amplification (needs a finished run of --scenario):
- excess = zsmax - P, P the peak of sfincs.bzs interpolated as SFINCS does
  (two nearest points, inverse distance), over active sea cells (land use
  200 on the grid, bed below the calm level); 0 at the boundary; mapped and
  binned by distance from the boundary;
- per station: zsmax and the excess in the station's cell, and the peak lag
  of zs there against the boundary peak (map output interval).

Writes to figs/surge_boundary_check/: <basin>_stations.csv,
<basin>_surge_boundary_check.png, <basin>_amplification_profile.csv/.png;
surge_boundary_summary.csv (one row per basin).

Usage:
    python analysis/check_surge_boundary.py [basin_id ...] [--all] [--scenario NAME]
        [--rp-yr RP] [--anomaly-tol-m M] [--excess-tol-m M] [--profile-bin-km KM]
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from _common import DEFAULT_BASIN_ID, REPO_ROOT, script_parser, site_path
from check_river_bed import _git_hash
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely.geometry import Point

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.io import catalogue_entry, load_catalogue
from src.landuse import LANDUSE_SEA_CODE
from src.surge import calm_sea_levels

log = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "figs" / "surge_boundary_check"
MASK_WL_BOUNDARY = 2
BND_MATCH_TOL_M = 5.0  # sfincs.bnd stores coordinates to 0.1 m
EXCESS_LIM_M = 0.5  # fixed colour scale of the excess map
ZSMAX_BLOCK = 24  # map output steps read at a time


@dataclass
class BasinPaths:
    """Input files of one basin's surge-boundary check."""

    surge_forcing: Path
    domain_bbox: Path
    domain_gpkg: Path
    mask_tif: Path
    dep_tif: Path
    landuse_on_grid: Path
    run_dir: Path

    @classmethod
    def for_basin(cls, results_dir: Path, basin_id: str, scenario: str) -> "BasinPaths":
        b = results_dir / basin_id
        dom = b / "preprocessing_inputs" / "domain"
        return cls(
            surge_forcing=b / "preprocessing_inputs" / "forcing" / "surge_forcing.nc",
            domain_bbox=dom / "domain_bbox.json",
            domain_gpkg=dom / f"{basin_id}_domain.gpkg",
            mask_tif=b / "sfincs_skeleton" / "gis" / "mask.tif",
            dep_tif=b / "sfincs_skeleton" / "gis" / "dep.tif",
            landuse_on_grid=dom / f"{basin_id}_landuse_on_grid.tif",
            run_dir=b / "runs" / scenario / "sfincs",
        )

    def has_run(self) -> bool:
        return all(
            (self.run_dir / f).exists()
            for f in ("sfincs.bnd", "sfincs.bzs", "sfincs_map.nc")
        )


def two_nearest_weights(
    xy: np.ndarray, pts_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two nearest points and their inverse-distance weights for each xy.

    SFINCS's interpolation of boundary points onto boundary cells; one point
    gets weight 1.
    """
    if len(pts_xy) == 1:
        return np.zeros((len(xy), 2), dtype=int), np.tile([1.0, 0.0], (len(xy), 1))
    dist, idx = cKDTree(pts_xy).query(xy, k=2)
    dist = np.maximum(dist, 1e-6)
    w = (1.0 / dist) / (1.0 / dist).sum(axis=1, keepdims=True)
    return idx, w


def boundary_share(idx: np.ndarray, w: np.ndarray, n_points: int) -> np.ndarray:
    """Return each point's weight summed over the boundary cells, as a fraction of them."""
    share = np.bincount(idx.ravel(), weights=w.ravel(), minlength=n_points)
    return share / max(len(idx), 1)


def level_anomaly(levels: np.ndarray) -> np.ndarray:
    """Return each level minus the median of all levels (NaN with fewer than three)."""
    levels = np.asarray(levels, dtype=float)
    if len(levels) < 3:
        return np.full(len(levels), np.nan)
    return levels - np.median(levels)


def binned_profile(
    dist_km: np.ndarray, values: np.ndarray, bin_km: float
) -> pd.DataFrame:
    """Return p10/median/p90 and count of values per distance bin."""
    ok = np.isfinite(dist_km) & np.isfinite(values)
    if not ok.any():
        return pd.DataFrame(columns=["dist_km", "n", "p10_m", "median_m", "p90_m"])
    bins = np.floor(dist_km[ok] / bin_km).astype(int)
    df = pd.DataFrame({"bin": bins, "v": values[ok]})
    g = df.groupby("bin")["v"]
    return pd.DataFrame(
        {
            "dist_km": (g.size().index.values + 0.5) * bin_km,
            "n": g.size().values,
            "p10_m": g.quantile(0.1).values,
            "median_m": g.median().values,
            "p90_m": g.quantile(0.9).values,
        }
    )


def signed_distance_m(points: gpd.GeoSeries, polygon) -> np.ndarray:
    """Return the distance of each point to the polygon's outline, negative inside."""
    outline = polygon.boundary
    return np.array(
        [(-1.0 if polygon.contains(p) else 1.0) * p.distance(outline) for p in points]
    )


def read_bnd(path: Path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)[:, :2]


def read_bzs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0], a[:, 1:]


def sample_gebco(
    gebco_dir: Path | None, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """Return raw GEBCO (m, local MSL) at the points; NaN outside every tile."""
    out = np.full(len(lon), np.nan)
    if gebco_dir is None or not gebco_dir.is_dir():
        log.warning(f"GEBCO not found at {gebco_dir}; gebco_msl_m left NaN")
        return out
    for tif in sorted(gebco_dir.glob("*.tif")):
        with rasterio.open(tif) as src:
            b = src.bounds
            inside = (
                (lon >= b.left)
                & (lon < b.right)
                & (lat > b.bottom)
                & (lat <= b.top)
                & np.isnan(out)
            )
            if inside.any():
                vals = np.array(
                    [v[0] for v in src.sample(zip(lon[inside], lat[inside]))],
                    dtype=float,
                )
                if src.nodata is not None:
                    vals[vals == src.nodata] = np.nan
                out[inside] = vals
    return out


def _cell_index(
    transform, shape, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = rasterio.transform.rowcol(transform, x, y)
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return np.where(ok, rows, 0), np.where(ok, cols, 0), ok


def _cell_centres(transform, shape) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    return transform * (cols + 0.5, rows + 0.5)


def _cell_corners(transform, shape) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[0 : shape[0] + 1, 0 : shape[1] + 1]
    return transform * (cols, rows)


def _map_row_flip(ds: xr.Dataset, active: np.ndarray, transform) -> bool:
    """Return whether sfincs_map.nc rows run opposite to the skeleton mask's."""
    msk = ds["msk"].values > 0
    if msk.shape != active.shape:
        raise ValueError(
            f"sfincs_map.nc grid {msk.shape} != skeleton mask {active.shape}: run and skeleton differ"
        )
    first = np.array([float(ds["corner_x"][0, 0]), float(ds["corner_y"][0, 0])])
    top, bottom = (
        np.array(transform * (0, 0)),
        np.array(transform * (0, active.shape[0])),
    )
    flip = np.hypot(*(first - bottom)) < np.hypot(*(first - top))
    if not np.array_equal(msk[::-1] if flip else msk, active):
        raise ValueError(
            "sfincs_map.nc msk does not match the skeleton mask: run and skeleton differ"
        )
    return bool(flip)


def _zsmax(ds: xr.Dataset, flip: bool) -> np.ndarray:
    da = ds["zsmax"]
    out = None
    for i in range(0, da.sizes["timemax"], ZSMAX_BLOCK):
        block = da.isel(timemax=slice(i, i + ZSMAX_BLOCK)).values
        block = np.where(block < -1000, -np.inf, block)
        bmax = block.max(axis=0)
        out = bmax if out is None else np.maximum(out, bmax)
    out = np.where(np.isfinite(out), out, np.nan)
    return out[::-1, :] if flip else out


def check_basin(
    paths: BasinPaths,
    basin_id: str,
    rp_yr: float,
    slr_m: float,
    gebco_dir: Path | None,
    anomaly_tol_m: float,
    excess_tol_m: float,
    profile_bin_km: float,
) -> dict:
    """Run the station screen and, with a finished run, the amplification check."""
    import json

    crs = json.loads(paths.domain_bbox.read_text())["crs"]
    domain_poly = gpd.read_file(paths.domain_gpkg).to_crs(crs).geometry.union_all()
    with rasterio.open(paths.mask_tif) as src:
        mask = src.read(1)
        transform, shape = src.transform, src.shape
    with rasterio.open(paths.dep_tif) as src:
        dep = src.read(1).astype(float)
        if src.nodata is not None:
            dep[dep == src.nodata] = np.nan
    cx, cy = _cell_centres(transform, shape)
    with rasterio.open(paths.landuse_on_grid) as src:
        lr, lc, lok = _cell_index(src.transform, src.shape, cx.ravel(), cy.ravel())
        rows, cols = np.indices(shape).reshape(2, -1)
        same_grid = (
            src.shape == shape
            and np.array_equal(lc, cols)
            and (np.array_equal(lr, rows) or np.array_equal(lr, shape[0] - 1 - rows))
        )
        if not same_grid:
            log.warning(
                f"{basin_id}: {paths.landuse_on_grid.name} is not on the skeleton grid (stale skeleton?); sampled at cell centres"
            )
        sea = (lok & (src.read(1)[lr, lc] == LANDUSE_SEA_CODE)).reshape(shape)
    mask = np.nan_to_num(mask, nan=0).astype(int)
    active = mask > 0
    bnd_rows, bnd_cols = np.nonzero(mask == MASK_WL_BOUNDARY)
    if not len(bnd_rows):
        raise ValueError(
            f"{paths.mask_tif}: no water-level boundary cells (mask == {MASK_WL_BOUNDARY})"
        )
    bnd_cell_xy = np.column_stack([cx[bnd_rows, bnd_cols], cy[bnd_rows, bnd_cols]])
    pixel_m = float(np.hypot(transform.a, transform.d))

    ds = xr.open_dataset(paths.surge_forcing, decode_times=False)
    if float(rp_yr) not in ds["rp_yr"].values:
        raise ValueError(
            f"--rp-yr {rp_yr} is not tabulated in COAST-RP: {ds['rp_yr'].values.tolist()}"
        )
    lon, lat = ds["longitude"].values, ds["latitude"].values
    raw_level = ds["storm_tide_rp_table"].sel(rp_yr=float(rp_yr)).values.astype(float)
    station_calm_m = calm_sea_levels(ds, slr_m)
    pts = gpd.GeoSeries([Point(xy) for xy in zip(lon, lat)], crs="EPSG:4326").to_crs(
        crs
    )
    st_xy = np.column_stack([pts.x, pts.y])

    r, c, on_grid = _cell_index(transform, shape, st_xy[:, 0], st_xy[:, 1])
    in_active = on_grid & active[r, c]
    dist_bnd_m = cKDTree(bnd_cell_xy).query(st_xy)[0]
    st = pd.DataFrame(
        {
            "station": np.arange(len(lon)),
            "lon": lon,
            "lat": lat,
            "x": st_xy[:, 0],
            "y": st_xy[:, 1],
            "signed_dist_domain_m": signed_distance_m(pts, domain_poly),
            "dist_wl_boundary_m": np.where(in_active, -dist_bnd_m, dist_bnd_m),
            "in_active_cell": in_active,
            "cell_mask": np.where(on_grid, mask[r, c], 0),
            "zb_cell_m": np.where(on_grid, dep[r, c], np.nan),
            "gebco_msl_m": sample_gebco(gebco_dir, lon, lat),
            f"storm_tide_rp{rp_yr:g}_raw_m": raw_level,
            "anomaly_m": level_anomaly(raw_level),
            "mdt_m": ds["mdt"].values,
            "calm_level_m": station_calm_m,
            "hg_match_km": ds["hg_match_km"].values if "hg_match_km" in ds else np.nan,
        }
    )
    st["flag_inside"] = st["in_active_cell"]
    st["flag_anomaly"] = st["anomaly_m"].abs() > anomaly_tol_m

    has_run = paths.has_run()
    excess = None
    profile = pd.DataFrame()
    if has_run:
        bnd_xy = read_bnd(paths.run_dir / "sfincs.bnd")
        bzs_t, bzs = read_bzs(paths.run_dir / "sfincs.bzs")
        d, j = cKDTree(st_xy).query(bnd_xy)
        if (d > BND_MATCH_TOL_M).any():
            raise ValueError(
                f"sfincs.bnd points do not match the surge stations (max offset {d.max():.1f} m)"
            )
        bzs_peak = bzs.max(axis=0)
        st["used_in_bnd"] = False
        st.loc[j, "used_in_bnd"] = True
        st["bzs_peak_m"] = np.nan
        st.loc[j, "bzs_peak_m"] = bzs_peak
        idx, w = two_nearest_weights(bnd_cell_xy, bnd_xy)
        share = boundary_share(idx, w, len(bnd_xy))
        st["boundary_share"] = 0.0
        st.loc[j, "boundary_share"] = share

        with xr.open_dataset(paths.run_dir / "sfincs_map.nc", decode_times=False) as m:
            flip = _map_row_flip(m, active, transform)
            zsmax = _zsmax(m, flip)
            rr = shape[0] - 1 - r if flip else r
            zs_st = (
                m["zs"]
                .isel(n=xr.DataArray(rr, dims="s"), m=xr.DataArray(c, dims="s"))
                .values
            )
            map_t = m["time"].values.astype(float)

        # Bed below the calm level too: stale skeletons keep inland water LC100 calls sea.
        water = (
            active
            & sea
            & (np.nan_to_num(dep, nan=np.inf) < station_calm_m.mean())
            & np.isfinite(zsmax)
        )
        wr, wc = np.nonzero(water)
        cell_xy = np.column_stack([cx[wr, wc], cy[wr, wc]])
        i2, w2 = two_nearest_weights(cell_xy, bnd_xy)
        excess = np.full(shape, np.nan)
        excess[wr, wc] = zsmax[wr, wc] - (w2 * bzs_peak[i2]).sum(axis=1)
        dist_km = (
            distance_transform_edt(mask != MASK_WL_BOUNDARY, sampling=pixel_m) / 1000.0
        )
        profile = binned_profile(dist_km[water], excess[water], profile_bin_km)

        st["zsmax_cell_m"] = np.where(in_active, zsmax[r, c], np.nan)
        st["excess_cell_m"] = np.where(in_active, excess[r, c], np.nan)
        zs_st = np.where(zs_st < -1000, np.nan, zs_st)
        peak_t = np.array(
            [
                map_t[np.nanargmax(zs_st[:, k])]
                if ok and np.isfinite(zs_st[:, k]).any()
                else np.nan
                for k, ok in enumerate(
                    in_active & np.isfinite(st["zsmax_cell_m"].values)
                )
            ]
        )
        st["peak_lag_hr"] = (peak_t - bzs_t[np.argmax(bzs.max(axis=1))]) / 3600.0
        st["flag_excess"] = st["excess_cell_m"].abs() > excess_tol_m

    summary = {
        "basin_id": basin_id,
        "rp_yr": rp_yr,
        "n_stations": len(st),
        "n_in_active_cell": int(st["in_active_cell"].sum()),
        "max_inside_km": float(max(0.0, -st["dist_wl_boundary_m"].min()) / 1000.0),
        "max_abs_anomaly_m": float(np.nanmax(np.abs(st["anomaly_m"])))
        if len(st) > 1
        else np.nan,
        "n_anomaly_flagged": int(st["flag_anomaly"].sum()),
        "has_run": has_run,
    }
    if has_run:
        summary |= {
            "n_used_in_bnd": int(st["used_in_bnd"].sum()),
            "excess_median_m": float(np.nanmedian(excess)),
            "excess_p90_m": float(np.nanpercentile(excess, 90)),
            "excess_max_m": float(np.nanmax(excess)),
            "n_excess_flagged": int(st["flag_excess"].sum()),
        }
    return {
        "stations": st,
        "summary": summary,
        "excess": excess,
        "profile": profile,
        "dep": dep,
        "mask": mask,
        "transform": transform,
        "domain_poly": domain_poly,
    }


def plot_basin(
    res: dict, basin_id: str, scenario: str, rp_yr: float, out_dir: Path, tag: str
) -> list[Path]:
    """Write the station map and level panel, and the excess profile."""
    st, mask, transform = res["stations"], res["mask"], res["transform"]
    kx, ky = _cell_corners(transform, mask.shape)
    cx, cy = _cell_centres(transform, mask.shape)
    fig, (ax_map, ax_lvl) = plt.subplots(
        1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    if res["excess"] is not None:
        im = ax_map.pcolormesh(
            kx, ky, res["excess"], cmap="RdBu_r", vmin=-EXCESS_LIM_M, vmax=EXCESS_LIM_M
        )
        fig.colorbar(
            im, ax=ax_map, label="zsmax - imposed boundary peak (m)", shrink=0.8
        )
        land = np.where((mask > 0) & ~np.isfinite(res["excess"]), 1.0, np.nan)
        ax_map.pcolormesh(kx, ky, land, cmap="Greys", vmin=0, vmax=2, alpha=0.5)
    else:
        im = ax_map.pcolormesh(
            kx,
            ky,
            np.where(mask > 0, res["dep"], np.nan),
            cmap="terrain",
            vmin=-15,
            vmax=15,
        )
        fig.colorbar(im, ax=ax_map, label="bed level (m, GOCO06s)", shrink=0.8)
    bnd = mask == MASK_WL_BOUNDARY
    ax_map.scatter(cx[bnd], cy[bnd], s=2, c="k", label="water-level boundary cells")
    gpd.GeoSeries([res["domain_poly"]]).boundary.plot(ax=ax_map, color="0.4", lw=0.8)
    used = st["used_in_bnd"] if "used_in_bnd" in st else pd.Series(True, index=st.index)
    for flag, mk, lbl in (
        (True, "o", "station (forces boundary)"),
        (False, "X", "station (outside buffer)"),
    ):
        sel = used == flag
        if sel.any():
            ax_map.scatter(
                st.loc[sel, "x"],
                st.loc[sel, "y"],
                marker=mk,
                s=60,
                c=np.where(st.loc[sel, "in_active_cell"], "tab:red", "tab:green"),
                edgecolor="k",
                lw=0.6,
                label=lbl,
                zorder=5,
            )
    for _, s in st.iterrows():
        ax_map.annotate(
            str(int(s.station)),
            (s.x, s.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    xs, ys = np.r_[kx.min(), kx.max(), st.x], np.r_[ky.min(), ky.max(), st.y]
    pad = 0.04 * max(np.ptp(xs), np.ptp(ys))
    ax_map.set_xlim(xs.min() - pad, xs.max() + pad)
    ax_map.set_ylim(ys.min() - pad, ys.max() + pad)
    ax_map.set_title("red = station in an active cell, green = outside")
    ax_map.legend(loc="best", fontsize=7)
    ax_map.set_aspect("equal")

    x = st["station"].values
    raw_col = f"storm_tide_rp{rp_yr:g}_raw_m"
    ax_lvl.bar(
        x - 0.2,
        st[raw_col],
        width=0.4,
        color=np.where(st["flag_anomaly"], "tab:orange", "0.6"),
        label=f"COAST-RP RP{rp_yr:g} storm tide (local MSL)",
    )
    ax_lvl.axhline(
        np.median(st[raw_col]), color="0.3", ls=":", lw=1, label="median of stations"
    )
    if "zsmax_cell_m" in st:
        ax_lvl.scatter(
            x,
            st["bzs_peak_m"] - st["calm_level_m"],
            marker="_",
            s=200,
            c="k",
            label="imposed bzs peak - calm level (incl. SLR)",
        )
        ax_lvl.scatter(
            x,
            st["zsmax_cell_m"] - st["calm_level_m"],
            marker="D",
            s=25,
            c="tab:blue",
            label="SFINCS zsmax at station - calm level (incl. SLR)",
        )
    ax_lvl.set_xticks(x)
    ax_lvl.set_xticklabels(
        [f"{i}\n{d / 1000:+.1f} km" for i, d in zip(x, st["dist_wl_boundary_m"])],
        fontsize=7,
    )
    ax_lvl.set_xlabel(
        "station / distance to the water-level boundary (negative = inside)"
    )
    ax_lvl.set_ylabel("m above local MSL")
    ax_lvl.set_title("orange = anomaly beyond tolerance")
    ax_lvl.legend(fontsize=7)
    fig.suptitle(
        f"surge boundary check | basin {basin_id} | scenario {scenario} | {tag}"
    )
    fig.tight_layout()
    out = [out_dir / f"{basin_id}_surge_boundary_check.png"]
    fig.savefig(out[0], dpi=130)
    plt.close(fig)

    if not res["profile"].empty:
        p = res["profile"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(
            p["dist_km"],
            p["p10_m"],
            p["p90_m"],
            color="tab:blue",
            alpha=0.25,
            label="p10-p90",
        )
        ax.plot(p["dist_km"], p["median_m"], color="tab:blue", label="median")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-EXCESS_LIM_M, EXCESS_LIM_M)
        ax.set_xlabel("distance from the water-level boundary (km)")
        ax.set_ylabel("zsmax - imposed boundary peak (m)")
        ax.set_title(
            f"amplification inside SFINCS | basin {basin_id} | {scenario} | {tag}",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        out.append(out_dir / f"{basin_id}_amplification_profile.png")
        fig.savefig(out[-1], dpi=130)
        plt.close(fig)
    return out


def _scenario(scenario: str) -> tuple[float | None, float]:
    import yaml

    sc = yaml.safe_load(
        (REPO_ROOT / "config" / "scenarios.yml").read_text(encoding="utf-8")
    )
    if scenario not in sc:
        raise KeyError(f"scenario '{scenario}' not in config/scenarios.yml")
    rp = sc[scenario].get("surge_rp_yr")
    return (float(rp) if isinstance(rp, (int, float)) else None), float(
        sc[scenario].get("slr_m") or 0.0
    )


def _gebco_dir() -> Path | None:
    cat = load_catalogue(REPO_ROOT / "config" / "data_catalogue.yml")
    return (
        Path(cat["meta"]["root"])
        / catalogue_entry(cat, "coastal_bathymetry")["file_path"]
    )


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument(
        "basin_ids",
        nargs="*",
        default=[DEFAULT_BASIN_ID],
        help=f"basin ids (default: {DEFAULT_BASIN_ID})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="every basin under results_dir with a model skeleton",
    )
    parser.add_argument(
        "--scenario",
        default="default",
        help="run whose zsmax and sfincs.bzs are used (default: default)",
    )
    parser.add_argument(
        "--rp-yr",
        type=float,
        default=None,
        help="COAST-RP return period of the level screen (default: the scenario's surge_rp_yr, else 100)",
    )
    parser.add_argument(
        "--anomaly-tol-m",
        type=float,
        default=0.10,
        help="flag stations this far from the others' median (default: 0.10)",
    )
    parser.add_argument(
        "--excess-tol-m",
        type=float,
        default=0.10,
        help="flag stations whose cell's zsmax exceeds the imposed peak by this (default: 0.10)",
    )
    parser.add_argument(
        "--profile-bin-km",
        type=float,
        default=1.0,
        help="distance bin of the excess profile (default: 1.0)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results_dir = site_path("results_dir")
    basin_ids = (
        sorted(p.parent.parent.name for p in results_dir.glob("*/sfincs_skeleton/gis"))
        if args.all
        else args.basin_ids
    )
    scenario_rp_yr, slr_m = _scenario(args.scenario)
    rp_yr = args.rp_yr if args.rp_yr is not None else (scenario_rp_yr or 100.0)
    gebco_dir = _gebco_dir()
    tag = _git_hash()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries, failed = [], []
    for basin_id in basin_ids:
        paths = BasinPaths.for_basin(results_dir, basin_id, args.scenario)
        missing = [
            p
            for p in (
                paths.surge_forcing,
                paths.domain_gpkg,
                paths.mask_tif,
                paths.dep_tif,
            )
            if not p.exists()
        ]
        if missing:
            log.error(f"{basin_id}: missing {', '.join(str(p) for p in missing)}")
            failed.append(basin_id)
            continue
        try:
            res = check_basin(
                paths,
                basin_id,
                rp_yr,
                slr_m,
                gebco_dir,
                args.anomaly_tol_m,
                args.excess_tol_m,
                args.profile_bin_km,
            )
        except ValueError as e:
            log.error(f"{basin_id}: {e}")
            failed.append(basin_id)
            continue
        res["stations"].to_csv(
            OUT_DIR / f"{basin_id}_stations.csv", index=False, float_format="%.4f"
        )
        if not res["profile"].empty:
            res["profile"].to_csv(
                OUT_DIR / f"{basin_id}_amplification_profile.csv",
                index=False,
                float_format="%.4f",
            )
        figs = plot_basin(res, basin_id, args.scenario, rp_yr, OUT_DIR, tag)
        if not res["summary"]["has_run"]:
            log.warning(
                f"{basin_id}: no finished run '{args.scenario}'; station screen only"
            )
        summaries.append(res["summary"])
        print(
            f"{basin_id}: "
            + ", ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in res["summary"].items()
                if k != "basin_id"
            )
        )
        for f in figs:
            print(f"  {f}")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            OUT_DIR / "surge_boundary_summary.csv", index=False, float_format="%.4f"
        )
        print(f"Summary: {OUT_DIR / 'surge_boundary_summary.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
