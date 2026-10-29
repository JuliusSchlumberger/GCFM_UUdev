"""River boundary forcing: GloFAS loading, crossing detection, snapping, dataset assembly."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Point, Polygon
from src.extreme_values import EVAResult
from src.raster import latlon_dims
from src.river_network import normalize_reach_id
from src.surge import sinusoidal_wave

log = logging.getLogger(__name__)


def derive_forcing_mode(
    river_rp_yr: float | None | str, surge_rp_yr: float | None
) -> str:
    """Return the SFINCS forcing mode from which design RPs a scenario sets.

    Both -> "compound"; river_rp_yr only -> "river_only"; surge_rp_yr only ->
    "coastal_only"; neither -> ValueError. Used by 00_common.smk's scenario_params.
    """
    if river_rp_yr is not None and surge_rp_yr is not None:
        return "compound"
    if river_rp_yr is not None:
        return "river_only"
    if surge_rp_yr is not None:
        return "coastal_only"
    raise ValueError(
        "both river_rp_yr and surge_rp_yr are null -- no forcing to drive the model with"
    )


def has_downstream_in_domain(rch_id_dn_raw, domain_reach_ids: set[str]) -> bool:
    """Return True if any downstream reach ID in rch_id_dn is present in the domain.

    rch_id_dn is a list-repr string per SWORD v17c (e.g. "[id1, id2]"); a
    whitespace-separated string ("id1 id2") is also accepted.
    """
    s = str(rch_id_dn_raw).strip().strip("[]")
    if not s or s.lower() in ("nan", "none", "<na>"):
        return False
    for token in re.split(r"[,\s]+", s):
        normed = normalize_reach_id(token.strip())
        if normed and normed in domain_reach_ids:
            return True
    return False


def _domain_entry_point(line, domain_poly: Polygon) -> Point | None:
    # First entry along the line, even if it dips in and out several times.
    # Point-only contact (touching the boundary) does not count as entering.
    clipped = line.intersection(domain_poly)
    if clipped.is_empty:
        return None
    parts = getattr(clipped, "geoms", [clipped])
    pieces = [
        piece for piece in parts if piece.geom_type == "LineString" and piece.length > 0
    ]
    if not pieces:
        return None
    return min((Point(piece.coords[0]) for piece in pieces), key=line.project)


def resolve_inside_domain_reaches(
    crossings: gpd.GeoDataFrame,
    river_gdf: gpd.GeoDataFrame,
    domain_poly: Polygon,
    max_hops: int = 30,
) -> gpd.GeoDataFrame:
    """Move each crossing to where the network first enters the domain polygon.

    Walks downstream (up to max_hops) past reaches that only touch the
    boundary. Returns a copy of `crossings` with geometry at the entry point,
    'inside_reach_id' of that reach and its 'width'/'max_width'; on failure the
    crossing point is kept and inside_reach_id is None. All inputs EPSG:4326.
    Reach lines are oriented downstream first (SWORD lines often run upstream),
    so the entry is the upstream end of the part inside the domain.
    """
    import pandas as pd
    from src.gvf_target import orient_downstream
    from src.river_network import build_downstream_adjacency

    river_gdf, _ = orient_downstream(river_gdf, build_downstream_adjacency(river_gdf))
    reach_lookup: dict[str, pd.Series] = {}
    for _, row in river_gdf.iterrows():
        reach_id = normalize_reach_id(row.get("reach_id"))
        if reach_id:
            reach_lookup[reach_id] = row

    topo_attrs = [a for a in ("width", "max_width") if a in river_gdf.columns]

    new_geoms: list = []
    inside_reach_ids: list = []
    inside_attr_vals: dict[str, list] = {a: [] for a in topo_attrs}

    for i, (_, crossing_row) in enumerate(crossings.iterrows()):
        start_id = (
            normalize_reach_id(crossing_row.get("reach_id"))
            if "reach_id" in crossing_row.index
            else None
        )

        current = start_id
        visited: set[str] = set()
        found_geom = None
        found_reach = None

        for _ in range(max_hops):
            if not current or current in visited or current not in reach_lookup:
                break
            visited.add(current)
            reach_row = reach_lookup[current]
            entry_point = _domain_entry_point(reach_row.geometry, domain_poly)
            if entry_point is not None:
                found_geom = entry_point
                found_reach = reach_row
                break
            dn_raw = str(reach_row.get("rch_id_dn", "") or "").strip().strip("[]")
            if not dn_raw or dn_raw.lower() in ("nan", "none", "<na>"):
                break
            current = None
            for token in re.split(r"[,\s]+", dn_raw):
                normed = normalize_reach_id(token.strip())
                if normed:
                    current = normed
                    break
            else:
                break

        if found_geom is not None and found_reach is not None:
            new_geoms.append(found_geom)
            inside_reach_ids.append(normalize_reach_id(found_reach.get("reach_id")))
            for a in topo_attrs:
                inside_attr_vals[a].append(found_reach.get(a))
        else:
            new_geoms.append(crossing_row.geometry)
            inside_reach_ids.append(None)
            for a in topo_attrs:
                inside_attr_vals[a].append(crossing_row.get(a))
            log.warning(
                f"Crossing {i} (reach_id={start_id}): no reach entering the "
                f"domain found within {max_hops} hops; using crossing point "
                f"as fallback"
            )

    result = crossings.copy()
    result["geometry"] = new_geoms
    result["inside_reach_id"] = inside_reach_ids
    for a in topo_attrs:
        result[a] = inside_attr_vals[a]
    return result


def find_best_glofas_cell(
    glofas_clip: xr.Dataset,
    variable: str,
    pt_lon: float,
    pt_lat: float,
    radius_m: float,
    min_mean_q: float,
    utm_crs: str,
) -> tuple[int, int] | None:
    """Return (i_lat, i_lon) of the highest-mean-discharge GloFAS cell within radius_m.

    Distances are Euclidean in utm_crs (e.g. 'EPSG:32636'). min_mean_q in
    m3 s-1; pt_lon/pt_lat in degrees. None if no cell qualifies.
    """
    from pyproj import Transformer

    lat_dim, lon_dim = latlon_dims(glofas_clip)
    time_dim = "valid_time" if "valid_time" in glofas_clip.dims else "time"
    lat_arr = glofas_clip[lat_dim].values  # (n_lat,)
    lon_arr = glofas_clip[lon_dim].values  # (n_lon,)

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    pt_x, pt_y = to_utm.transform(pt_lon, pt_lat)

    lon_g, lat_g = np.meshgrid(lon_arr, lat_arr)  # both (n_lat, n_lon)
    cell_x, cell_y = to_utm.transform(lon_g.ravel(), lat_g.ravel())
    cell_x = cell_x.reshape(lon_g.shape)  # (n_lat, n_lon)
    cell_y = cell_y.reshape(lon_g.shape)

    dist_m = np.sqrt((cell_x - pt_x) ** 2 + (cell_y - pt_y) ** 2)
    within = dist_m <= radius_m  # (n_lat, n_lon)

    if not within.any():
        return None

    mean_discharge_m3s = (
        glofas_clip[variable]
        .mean(dim=time_dim, skipna=True)
        .transpose(lat_dim, lon_dim)
        .values
    )  # (n_lat, n_lon)

    qualified = within & (mean_discharge_m3s >= min_mean_q)
    if not qualified.any():
        return None

    flat_best = int(np.argmax(np.where(qualified, mean_discharge_m3s, -np.inf)))
    i_lat, i_lon = divmod(flat_best, len(lon_arr))
    return i_lat, i_lon


def load_grdc_stations(path: str | Path) -> gpd.GeoDataFrame:
    """Load GRDC-Daily.nc station metadata as points (EPSG:4326).

    Reads only station variables, not the (time, id) runoff_mean array.
    Columns: 'id' plus whichever of 'river_name', 'station_name', 'area',
    'country' exist.
    """
    with xr.open_dataset(path) as ds:
        ids = ds["id"].values
        lons = ds["geo_x"].values.astype(float)
        lats = ds["geo_y"].values.astype(float)
        data: dict = {"id": ids}
        for var in ("river_name", "station_name", "area", "country"):
            if var in ds:
                values = ds[var].values
                data[var] = values.astype(str) if values.dtype.kind == "U" else values

    return gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326"
    )


def find_nearest_grdc_station(
    stations_gdf: gpd.GeoDataFrame,
    pt_lon: float,
    pt_lat: float,
    radius_m: float,
    utm_crs: str,
) -> int | None:
    """Return the 'id' of the nearest GRDC station within radius_m, or None.

    Distances are Euclidean in utm_crs; pt_lon/pt_lat in degrees.
    """
    if stations_gdf.empty:
        return None

    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    pt_x, pt_y = to_utm.transform(pt_lon, pt_lat)
    sta_x, sta_y = to_utm.transform(
        stations_gdf.geometry.x.values, stations_gdf.geometry.y.values
    )

    dist_m = np.sqrt((sta_x - pt_x) ** 2 + (sta_y - pt_y) ** 2)
    i_min = int(np.argmin(dist_m))
    if dist_m[i_min] > radius_m:
        return None
    return int(stations_gdf["id"].values[i_min])


def load_grdc_series(
    path: str | Path, station_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, values) of one GRDC station's daily discharge series.

    The -999.0 missing-value sentinel becomes NaN; NaNs are not dropped.
    """
    with xr.open_dataset(path) as ds:
        times = ds["time"].values
        values = ds["runoff_mean"].sel(id=station_id).values.astype(float)

    return times, np.where(values == -999.0, np.nan, values)


_CROSSING_ATTRS = (
    "width",
    "max_width",
    "main_path_id",
    "dist_out",
    "reach_id",
    "rch_id_dn",
)


def find_boundary_crossings(
    river_gdf: gpd.GeoDataFrame,
    bbox: Polygon,
) -> gpd.GeoDataFrame:
    """Return points where river reaches cross the domain bbox boundary (EPSG:4326).

    Only Point intersections are kept; reaches running along the boundary
    (line intersections) are dropped. May be empty.
    """
    bbox_exterior = bbox.exterior
    available = {col for col in _CROSSING_ATTRS if col in river_gdf.columns}
    crossings: list[dict] = []
    for _, row in river_gdf.iterrows():
        geom = row.geometry
        if geom is None or not geom.intersects(bbox_exterior):
            continue
        intersection = geom.intersection(bbox_exterior)
        geoms = (
            list(intersection.geoms)
            if hasattr(intersection, "geoms")
            else [intersection]
        )
        extra = {col: row.get(col) for col in available}
        crossings.extend(
            {"geometry": pt, **extra}
            for pt in geoms
            if pt.geom_type == "Point" and not pt.is_empty
        )
    if not crossings:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(crossings, crs="EPSG:4326")


def load_glofas_clip(
    glofas_path: Path,
    variable: str,
    bounds_wgs84: tuple[float, float, float, float],
    buffer_deg: float,
) -> xr.Dataset:
    """Load `variable` from GloFAS NetCDF file(s), clipped to the domain bbox.

    glofas_path is a .nc file or a directory of them, concatenated along
    'valid_time' or 'time'. buffer_deg (degrees) widens the clip so crossings
    near the boundary are covered.
    """
    glofas_path = Path(glofas_path)
    nc_files = (
        sorted(glofas_path.glob("*.nc")) if glofas_path.is_dir() else [glofas_path]
    )
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found at {glofas_path}")

    lon_min, lat_min, lon_max, lat_max = bounds_wgs84

    def _spatial_slice(ds: xr.Dataset) -> xr.Dataset:
        lat_dim, lon_dim = latlon_dims(ds)
        lat_vals = ds[lat_dim].values
        lat_desc = lat_vals[0] > lat_vals[-1]
        lat_slice = (
            slice(lat_max + buffer_deg, lat_min - buffer_deg)
            if lat_desc
            else slice(lat_min - buffer_deg, lat_max + buffer_deg)
        )
        return ds[[variable]].sel(
            {
                lat_dim: lat_slice,
                lon_dim: slice(lon_min - buffer_deg, lon_max + buffer_deg),
            }
        )

    # Slice each file before concatenating: open_mfdataset then select makes
    # dask coordinate scattered chunk reads (~95% of runtime when profiled).
    if len(nc_files) > 1:
        with xr.open_dataset(nc_files[0]) as probe:
            time_dim = "valid_time" if "valid_time" in probe.dims else "time"
        parts = []
        for f in nc_files:
            with xr.open_dataset(f) as ds_f:
                parts.append(_spatial_slice(ds_f).load())
        clipped = xr.concat(parts, dim=time_dim).sortby(time_dim)
    else:
        with xr.open_dataset(nc_files[0]) as ds:
            clipped = _spatial_slice(ds).load()

    return clipped


def build_river_dataset(
    crossings: gpd.GeoDataFrame,
    has_glofas: np.ndarray,
    bankfull_discharge_m3s: np.ndarray,
    discharge_rp_table: np.ndarray,
    rps_yr: np.ndarray,
    cell_lon: np.ndarray,
    cell_lat: np.ndarray,
    times: np.ndarray,
    lead_days: float,
    period_hr: float,
    results: list[EVAResult | None],
    bankfull_rp_yr: float,
    mean_discharge_m3s: np.ndarray,
    bias_corrected: np.ndarray,
    grdc_station_id: np.ndarray,
    grdc_correlation: np.ndarray,
    grdc_overlap_days: np.ndarray,
) -> xr.Dataset:
    """Assemble river_forcing.nc: bankfull (AMAX/GEV), POT/GPD RP table, EVA diagnostics.

    The SFINCS discharge series is built later (rule 14,
    build_design_discharge_matrix) from mean_discharge_m3s + a design RP
    looked up in discharge_rp_table, so changing the RP needs no EVA rerun.

    Args:
        discharge_rp_table: (n_crossing, n_rp) discharge from the
            POT/GPD fit (src.extreme_values.gpd_return_value_table); NaN row
            where GPD did not converge.
        rps_yr: Years, discharge_rp_table's second axis.
        times, lead_days, period_hr: Stored for rule 14 to rebuild the
            sinusoidal wave.
        bias_corrected: int8, 1 if the GloFAS cell was GRDC bias-corrected before EVA.
        grdc_station_id: int64, matched GRDC 'id', -1 if none.
        grdc_correlation: Pearson r raw GloFAS vs GRDC over the overlap; NaN if uncorrected.
        grdc_overlap_days: Overlapping valid days used; NaN if uncorrected.
        mean_discharge_m3s: Grand-mean discharge of the (possibly bias-corrected) series;
            used for river_rp_yr: "mean" in config/scenarios.yml.
    """
    n_cross = len(crossings)

    def _col(attr: str, default=np.nan):
        out = np.full(n_cross, default, dtype=float)
        for i, r in enumerate(results):
            if r is not None and np.isfinite(getattr(r, attr, np.nan)):
                out[i] = getattr(r, attr)
        return out

    rp2_lo = np.full(n_cross, np.nan)
    rp2_hi = np.full(n_cross, np.nan)
    rp100_lo = np.full(n_cross, np.nan)
    rp100_hi = np.full(n_cross, np.nan)
    trend_str = np.array(["" for _ in range(n_cross)], dtype=object)
    for i, r in enumerate(results):
        if r is not None:
            rp2_lo[i], rp2_hi[i] = r.q_rp2_ci
            rp100_lo[i], rp100_hi[i] = r.q_rp100_ci
            trend_str[i] = r.trend

    cross_lons = crossings.geometry.x.values if n_cross > 0 else np.array([])
    cross_lats = crossings.geometry.y.values if n_cross > 0 else np.array([])

    # has_glofas = enters_domain AND EVA converged (visible_on_grid does not
    # gate it); other crossings get "" so rule 08 never seeds from them.
    if "inside_reach_id" in crossings.columns:
        inside_reach_ids_arr = np.array(
            [
                (normalize_reach_id(v) or "") if has_glofas[i] else ""
                for i, v in enumerate(crossings["inside_reach_id"].values)
            ],
            dtype="U64",
        )
    else:
        inside_reach_ids_arr = np.full(n_cross, "", dtype="U64")

    return xr.Dataset(
        {
            "discharge_rp_table": (
                ["crossing", "rp_yr"],
                discharge_rp_table,
                {
                    "units": "m3 s-1",
                    "long_name": "discharge return-value table (POT/GPD fit), "
                    "for build-time lookup at an arbitrary design return "
                    "period -- see build_design_discharge_matrix",
                },
            ),
            "has_glofas": (
                ["crossing"],
                has_glofas.astype(np.int8),
                {"long_name": "1 if crossing has a valid GloFAS EVA fit"},
            ),
            "bankfull_discharge_m3s": (
                ["crossing"],
                bankfull_discharge_m3s,
                {
                    "units": "m3 s-1",
                    "long_name": f"bankfull discharge (AMAX/GEV, RP={bankfull_rp_yr:g} yr)",
                },
            ),
            "mean_discharge_m3s": (
                ["crossing"],
                mean_discharge_m3s,
                {
                    "units": "m3 s-1",
                    "long_name": "grand-mean discharge over the full GloFAS "
                    "record (time-mean of this crossing's own, possibly "
                    "bias-corrected, series)",
                },
            ),
            "inside_reach_id": (
                ["crossing"],
                inside_reach_ids_arr,
                {"long_name": "SWORD reach_id of the inside-domain source reach"},
            ),
            # EVA diagnostics
            "bankfull_ci_lower": (["crossing"], rp2_lo, {"units": "m3 s-1"}),
            "bankfull_ci_upper": (["crossing"], rp2_hi, {"units": "m3 s-1"}),
            "flood_ci_lower": (["crossing"], rp100_lo, {"units": "m3 s-1"}),
            "flood_ci_upper": (["crossing"], rp100_hi, {"units": "m3 s-1"}),
            "pot_threshold": (
                ["crossing"],
                _col("pot_threshold"),
                {"units": "m3 s-1", "long_name": "POT threshold"},
            ),
            "pot_r_days": (
                ["crossing"],
                _col("pot_r_days"),
                {"units": "days", "long_name": "declustering window"},
            ),
            "pot_peaks_per_year": (["crossing"], _col("pot_peaks_per_year"), {}),
            "gpd_shape": (
                ["crossing"],
                _col("pot_shape"),
                {"long_name": "GPD shape parameter (tail)"},
            ),
            "gpd_scale": (
                ["crossing"],
                _col("pot_scale"),
                {
                    "long_name": "GPD scale parameter -- with pot_threshold, "
                    "gpd_shape, and pot_peaks_per_year, fully determines the "
                    "fitted return-value curve (see "
                    "src.extreme_values.gpd_return_value_table)"
                },
            ),
            "gev_shape": (
                ["crossing"],
                _col("gev_shape"),
                {"long_name": "GEV shape parameter"},
            ),
            "trend": (
                ["crossing"],
                trend_str.astype(str),
                {"long_name": "Mann-Kendall trend on annual maxima"},
            ),
            "trend_pvalue": (["crossing"], _col("trend_pvalue"), {}),
            "sen_slope": (
                ["crossing"],
                _col("sen_slope"),
                {"units": "m3 s-1 yr-1", "long_name": "Sen's slope (AMAX)"},
            ),
            "glofas_cell_lon": (["crossing"], cell_lon, {"units": "degrees_east"}),
            "glofas_cell_lat": (["crossing"], cell_lat, {"units": "degrees_north"}),
            # GRDC bias-correction provenance
            "bias_corrected": (
                ["crossing"],
                bias_corrected.astype(np.int8),
                {
                    "long_name": "1 if GloFAS discharge was bias-corrected against a GRDC gauge"
                },
            ),
            "grdc_station_id": (
                ["crossing"],
                grdc_station_id.astype(np.int64),
                {
                    "long_name": "Matched GRDC station id (GRDC-Daily.nc 'id'); -1 if none"
                },
            ),
            "grdc_correlation": (
                ["crossing"],
                grdc_correlation,
                {
                    "long_name": "Pearson r, raw GloFAS vs GRDC, on overlap days with "
                    "GRDC at or above bias_correction.tail_percentile"
                },
            ),
            "grdc_overlap_days": (
                ["crossing"],
                grdc_overlap_days,
                {
                    "units": "days",
                    "long_name": "Number of overlapping valid days used for bias correction",
                },
            ),
        },
        coords={
            "longitude": (["crossing"], cross_lons, {"units": "degrees_east"}),
            "latitude": (["crossing"], cross_lats, {"units": "degrees_north"}),
            "time": (["time"], times, {"units": "hours since simulation start"}),
            "rp_yr": (
                ["rp_yr"],
                np.asarray(rps_yr, dtype=float),
                {
                    "units": "yr",
                    "long_name": "return period axis for discharge_rp_table",
                },
            ),
        },
        attrs={
            # For rule 14 to rebuild the same sinusoidal wave.
            "lead_days": lead_days,
            "period_hr": period_hr,
        },
    )


def interpolate_discharge_at_rp(
    discharge_rp_table: np.ndarray,
    rps_yr: np.ndarray,
    target_rp_yr: float,
) -> np.ndarray:
    """Return per-crossing discharge (m^3 s^-1) at target_rp_yr, log-RP interpolated.

    Log-RP is the standard flood-frequency convention; raw-RP interpolation is
    badly skewed over the table's multi-order-of-magnitude span.
    """
    log_rp = np.log(target_rp_yr)
    log_rps_yr = np.log(rps_yr)
    n = discharge_rp_table.shape[0]
    return np.array(
        [np.interp(log_rp, log_rps_yr, discharge_rp_table[i]) for i in range(n)]
    )


def real_flood_event_matrix(
    river_ds: xr.Dataset,
    active: np.ndarray,
    rp_yr: float,
    separation_days: int,
    n_largest: int,
    min_ramp_hr: float,
    max_rise_per_hr: float,
) -> tuple[np.ndarray, dict]:
    """Return river discharge for one return period from real GloFAS flood events.

    Uses river_forcing.nc's discharge_daily (the series the RPs were fitted on,
    bias-corrected where rule 07d corrected it).

    The event is defined by the basin's MAIN inflow -- the active crossing
    with the largest bankfull discharge:

      * its independent flood peaks are the daily maxima at least
        separation_days apart;
      * the event whose peak is nearest the return level is taken and scaled
        so the peak equals it. A return level above every recorded peak
        ("above_record") instead uses the n_largest events, each scaled to
        the return level and averaged day by day -- an average flood shape;
      * every other inflow carries its OWN discharge on the same date(s),
        times the same scale factor: one coherent real event for the whole
        basin, in which only the main inflow is exactly at the return level.

    The peak day sits on the river peak hour (lead_days * 24 + period_hr / 2,
    the synthetic wave's centre; the surge peak in storm-tide mode); daily
    means are placed 24 h apart and interpolated linearly. Before
    attrs["event_start_hr"] the discharge holds the mean (where the spin-up
    ends); it then blends linearly into the event over

        ramp = max(min_ramp_hr, gap / (max_rise_per_hr x peak))

    with gap = main inflow's event discharge at the start minus its mean and
    peak = its return level: a relative limit, so one value suits a 1,000 and
    a 100,000 m3/s river. The ramp is not realistic (the fastest GloFAS rises
    are ~1 % of the 100-yr peak per hour, needing a ramp longer than the
    lead-up); it only avoids a step. All inflows share it; it is capped at the
    time from the event start to the peak.

    Returns:
        ((n_active, n_time) discharge in m3 s-1, info dict: mode, the event
        date(s) and scale factor(s), the main crossing's position in
        `active` order and its return level, the ramp length used).
    """
    import scipy.signal as ss

    times = river_ds["time"].values.astype(float)
    q = river_ds["discharge_daily"].values[active].astype(float)  # (n_active, n_day)
    days = np.datetime64("1970-01-01") + river_ds["day"].values.astype("timedelta64[D]")
    mean_discharge_m3s = river_ds["mean_discharge_m3s"].values[active]
    main = int(np.nanargmax(river_ds["bankfull_discharge_m3s"].values[active]))
    target = float(
        interpolate_discharge_at_rp(
            river_ds["discharge_rp_table"].values[active][[main]],
            river_ds["rp_yr"].values,
            rp_yr,
        )[0]
    )

    peak_hr = (
        float(river_ds.attrs["lead_days"]) * 24.0
        + float(river_ds.attrs["period_hr"]) / 2.0
    )
    start_hr = float(river_ds.attrs.get("event_start_hr", 0.0))
    off_lo = int(np.floor((times[0] - peak_hr) / 24.0)) - 1
    off_hi = int(np.ceil((times[-1] - peak_hr) / 24.0)) + 1
    offsets = np.arange(off_lo, off_hi + 1)

    q_main = q[main]
    peaks, _ = ss.find_peaks(
        np.nan_to_num(q_main, nan=-np.inf), distance=int(separation_days)
    )
    peaks = peaks[(peaks + off_lo >= 0) & (peaks + off_hi < q.shape[1])]
    if peaks.size == 0:
        raise ValueError(
            "real_flood_event_matrix: no flood peak with a full event window in discharge_daily"
        )
    peak_q = q_main[peaks]
    if target > peak_q.max():
        chosen = peaks[np.argsort(peak_q)[-int(n_largest) :]]
        mode = "above_record"
    else:
        chosen = peaks[[int(np.argmin(np.abs(peak_q - target)))]]
        mode = "event"
    scales = target / q_main[chosen]
    daily = np.nanmean(
        np.stack([q[:, p + offsets] * s for p, s in zip(chosen, scales)]), axis=0
    )  # (n_active, n_offsets)

    event = np.stack(
        [
            np.interp(times, peak_hr + 24.0 * offsets, daily[i])
            for i in range(q.shape[0])
        ]
    )
    start_q = float(np.interp(start_hr, times, event[main]))
    gap = max(start_q - float(mean_discharge_m3s[main]), 0.0)
    ramp_hr = max(float(min_ramp_hr), gap / (float(max_rise_per_hr) * target))
    lead_up = max(peak_hr - start_hr, 0.0)
    if ramp_hr > lead_up:
        log.warning(
            f"river event ramp of {ramp_hr:.1f} h (gap {gap:,.0f} m3/s) exceeds the {lead_up:.0f} h "
            f"between the event start and the peak -- capped; raise max_rise_per_hr or the event window"
        )
        ramp_hr = lead_up
    w = (
        np.clip((times - start_hr) / ramp_hr, 0.0, 1.0)
        if ramp_hr > 0
        else (times >= start_hr).astype(float)
    )
    matrix = (1.0 - w) * mean_discharge_m3s[:, None] + w * event
    info = {
        "mode": mode,
        "dates": [str(days[p]) for p in chosen],
        "scales": [float(s) for s in scales],
        "main": main,
        "target": target,
        "start_fraction": start_q / target,
        "ramp_hr": ramp_hr,
    }
    return matrix, info


def build_design_discharge_matrix(
    river_ds: xr.Dataset,
    active: np.ndarray,
    rp_yr: float | None | str,
    event: dict | None = None,
) -> np.ndarray:
    """Return the (n_active, n_time) SFINCS discharge (m3 s-1) for a design RP.

    Built at rule 14 from river_forcing.nc's mean_discharge_m3s + discharge_rp_table,
    so changing the RP needs no EVA rerun (rule 07d). Per active crossing:
      1. Q_f at rp_yr, log-RP interpolated (interpolate_discharge_at_rp).
      2. sinusoidal_wave(mean_discharge_m3s, Q_f, time, lead_days, period_hr). Riverine
         protection is NOT subtracted: the calibrated weirs represent it.

    Args:
        active: Boolean mask over crossings (typically has_glofas).
        rp_yr: Years (a scenario's river_rp_yr). None: river excluded,
            zero discharge. "mean" (case-insensitive): constant mean_discharge_m3s.
        event: boundary_forcings.river.event_hydrograph. Source "glofas_event"
            builds a numeric RP from real GloFAS events
            (real_flood_event_matrix; needs discharge_daily); None or "cosine"
            uses the synthetic wave. Any other source raises ValueError.
    """
    if event is not None and event["source"] not in ("glofas_event", "cosine"):
        raise ValueError(
            "boundary_forcings.river.event_hydrograph.source must be glofas_event or cosine, "
            f"got {event['source']!r}"
        )
    times = river_ds["time"].values
    table = river_ds["discharge_rp_table"].values[active]  # (n_active, n_rp)
    rps_yr = river_ds["rp_yr"].values
    lead_days = float(river_ds.attrs["lead_days"])
    period_hr = float(river_ds.attrs["period_hr"])

    n_active = int(active.sum())
    # Base flow = mean discharge (also the spin-up's), so an RP scenario
    # differs from a "mean" one only by its flood peak.
    mean_discharge_m3s = river_ds["mean_discharge_m3s"].values[active]
    lead_q = mean_discharge_m3s
    if isinstance(rp_yr, str) and rp_yr.strip().lower() == "mean":
        design_q = mean_discharge_m3s.copy()
    elif rp_yr is None:
        design_q = np.zeros(n_active, dtype=float)
        lead_q = design_q
    else:
        if event is not None and event["source"] == "glofas_event" and n_active > 0:
            if "discharge_daily" not in river_ds:
                raise ValueError(
                    "river event_hydrograph.source 'glofas_event' needs discharge_daily in "
                    "river_forcing.nc -- rerun rule get_boundary_forcings"
                )
            discharge_matrix, info = real_flood_event_matrix(
                river_ds,
                active,
                float(rp_yr),
                separation_days=int(event["separation_days"]),
                n_largest=int(event["n_largest"]),
                min_ramp_hr=float(event["min_ramp_hr"]),
                max_rise_per_hr=float(event["max_rise_per_hr"]),
            )
            log.info(
                f"River event RP {float(rp_yr):g}: real GloFAS flood(s) {info['dates']} "
                f"({info['mode']}), main inflow scaled x{[round(s, 3) for s in info['scales']]} to "
                f"{info['target']:,.0f} m3/s; at the event start it is at "
                f"{100 * info['start_fraction']:.0f}% of the peak, ramped in from the mean over "
                f"{info['ramp_hr']:.1f} h"
            )
            return discharge_matrix
        design_q = interpolate_discharge_at_rp(table, rps_yr, rp_yr)

    discharge_matrix = np.full((n_active, len(times)), np.nan)
    for i in range(n_active):
        discharge_matrix[i] = sinusoidal_wave(
            lead_q[i], design_q[i], times, lead_days, period_hr
        )
    return discharge_matrix


def load_forcing_crossings(
    forcing_path: str | Path,
    discharge_variable: str | None = None,
) -> gpd.GeoDataFrame:
    """Load active (has_glofas=1) crossing points (EPSG:4326) from river_forcing.nc.

    decode_times=False: the time unit "hours since simulation start" is not
    CF-parseable. discharge_variable (e.g. 'bankfull_discharge_m3s') is returned
    as column 'bankfull_discharge_m3s'. May be empty.
    """
    with xr.open_dataset(forcing_path, decode_times=False) as ds:
        has_glofas = ds["has_glofas"].values.astype(bool)
        lons = ds["longitude"].values
        lats = ds["latitude"].values
        q_vals = ds[discharge_variable].values if discharge_variable else None
        inside_ids = ds["inside_reach_id"].values if "inside_reach_id" in ds else None

    if not has_glofas.any():
        log.warning("No active GloFAS crossings found in river_forcing.nc")
        cols = {"bankfull_discharge_m3s": []} if discharge_variable else {}
        return gpd.GeoDataFrame(cols, geometry=[], crs="EPSG:4326")

    data = {}
    if discharge_variable and q_vals is not None:
        data["bankfull_discharge_m3s"] = q_vals[has_glofas]
    if inside_ids is not None:
        data["inside_reach_id"] = inside_ids[has_glofas].astype(str)
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(lons[has_glofas], lats[has_glofas]),
        crs="EPSG:4326",
    )


def snap_crossings_to_reaches(
    crossings: gpd.GeoDataFrame,
) -> dict[str, float]:
    """Return {reach_id: bankfull_discharge_m3s} per seed reach from crossings' inside_reach_id.

    inside_reach_id is resolved in rule 07d and is empty for crossings that
    failed enters_domain or EVA convergence. Several crossings on one reach
    keep the largest bankfull_discharge_m3s.
    """
    if crossings.empty:
        return {}

    if "bankfull_discharge_m3s" not in crossings.columns:
        log.warning(
            "snap_crossings_to_reaches: no 'bankfull_discharge_m3s' column — all crossings discarded"
        )
        return {}

    if "inside_reach_id" not in crossings.columns:
        log.warning(
            "snap_crossings_to_reaches: no 'inside_reach_id' column — "
            "re-run rule 07d to regenerate river_forcing.nc"
        )
        return {}

    seed_q: dict[str, float] = {}
    for _, row in crossings.iterrows():
        reach_id = normalize_reach_id(str(row.get("inside_reach_id", "") or ""))
        q = row.get("bankfull_discharge_m3s", np.nan)
        if not reach_id or not np.isfinite(float(q)):
            continue
        existing = seed_q.get(reach_id)
        if existing is None or float(q) > existing:
            seed_q[reach_id] = float(q)

    if not seed_q:
        log.warning("No valid seed reaches found from inside_reach_ids")
    else:
        log.info(f"Mapped {len(crossings)} crossing(s) to {len(seed_q)} seed reach(es)")
    return seed_q


def seed_discharge_by_reach(
    river_ds, q: np.ndarray, active: np.ndarray
) -> dict[str, float]:
    """Return the largest discharge per seed reach ({reach_id: m3/s}).

    q holds one discharge (m3/s) per active crossing of river_ds, in crossing
    order. A dataset without inside_reach_id yields no seeds.
    """
    inside_ids = (
        river_ds["inside_reach_id"].values[active]
        if "inside_reach_id" in river_ds
        else np.full(active.sum(), "", dtype=object)
    )
    crossings = gpd.GeoDataFrame(
        {"bankfull_discharge_m3s": q, "inside_reach_id": inside_ids},
        geometry=gpd.points_from_xy(
            river_ds["longitude"].values[active], river_ds["latitude"].values[active]
        ),
        crs="EPSG:4326",
    )
    return snap_crossings_to_reaches(crossings)
