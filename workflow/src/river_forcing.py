"""River boundary forcing: GloFAS loading, crossing detection, snapping, dataset assembly."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Point, Polygon

from src.surge import sinusoidal_wave
from src.extreme_values import EVAResult

log = logging.getLogger(__name__)


# ── private helpers ───────────────────────────────────────────────────────────


def _norm_reach_id(x) -> str | None:
    """Normalise a nullable-integer reach or path ID to a plain int string."""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "<na>", ""):
        return None
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s if s else None


# ── filter helpers ────────────────────────────────────────────────────────────


def has_downstream_in_domain(rch_id_dn_raw, domain_reach_ids: set[str]) -> bool:
    """Return True if any downstream reach ID in rch_id_dn is present in the domain.

    rch_id_dn is a Python-list-repr string per SWORD v17c's convention
    (e.g. "[id1, id2]"); re.split on comma-or-whitespace is deliberately
    also robust to a plain whitespace-separated string (e.g. "id1 id2"),
    in case a future source uses that convention instead.
    """
    s = str(rch_id_dn_raw).strip().strip("[]")
    if not s or s.lower() in ("nan", "none", "<na>"):
        return False
    for token in re.split(r"[,\s]+", s):
        normed = _norm_reach_id(token.strip())
        if normed and normed in domain_reach_ids:
            return True
    return False


# ── inside-domain source location resolution ─────────────────────────────────


def _domain_entry_point(line, domain_poly: Polygon) -> Point | None:
    """
    Return the point where `line` first enters `domain_poly`, walking from
    the line's start coordinate towards its end. None if `line` never enters
    the domain at all.

    Clips `line` to `domain_poly` (interior, not just the exterior ring) and
    picks whichever piece of the clipped geometry starts earliest along the
    original line (via `line.project()`), so a reach that dips in and out of
    the domain multiple times still resolves to its first entry, not an
    arbitrary one.
    """
    clipped = line.intersection(domain_poly)
    if clipped.is_empty:
        return None
    if clipped.geom_type == "Point":
        return clipped
    if clipped.geom_type == "MultiPoint":
        return min(clipped.geoms, key=line.project)
    if clipped.geom_type == "LineString":
        pieces = [clipped]
    elif clipped.geom_type == "MultiLineString":
        pieces = list(clipped.geoms)
    else:
        return None
    candidates = [Point(piece.coords[0]) for piece in pieces if len(piece.coords) > 0]
    if not candidates:
        return None
    return min(candidates, key=line.project)


def resolve_inside_domain_reaches(
    crossings: gpd.GeoDataFrame,
    river_gdf: gpd.GeoDataFrame,
    domain_poly: Polygon,
    max_hops: int = 30,
) -> gpd.GeoDataFrame:
    """
    Walk downstream from each crossing reach to find the point where the
    river network first actually enters the domain polygon -- i.e. the
    domain boundary clipped through the crossing reach's own geometry, not a
    reach vertex somewhere further downstream. Returns a copy of `crossings`
    with:

    - geometry updated to that domain-entry point (crossing point used as
      fallback when no reach within max_hops ever enters the domain)
    - inside_reach_id: reach_id of the reach whose geometry the entry point
      lies on (None on fallback)
    - width, max_width: attributes of that reach (if present in river_gdf),
      overwriting the corresponding attributes of the crossing reach

    Most crossings resolve on the very first hop (the crossing reach itself
    enters the domain at the same point find_boundary_crossings found).
    Walking downstream is only needed for a reach that merely touches the
    domain boundary without any of its length lying inside it (e.g. the
    boundary crossing is its very last vertex) -- in that case, the next
    downstream reach is checked instead.

    Args:
        crossings:   GeoDataFrame from find_boundary_crossings.
        river_gdf:   Full clipped river network GeoDataFrame (EPSG:4326).
        domain_poly: Shapely Polygon of the domain (EPSG:4326).
        max_hops:    Maximum downstream hops before using crossing point as fallback.
    """
    import pandas as pd

    reach_lookup: dict[str, pd.Series] = {}
    for _, row in river_gdf.iterrows():
        rid = _norm_reach_id(row.get("reach_id"))
        if rid:
            reach_lookup[rid] = row

    topo_attrs = [a for a in ("width", "max_width") if a in river_gdf.columns]

    new_geoms: list = []
    inside_reach_ids: list = []
    inside_attr_vals: dict[str, list] = {a: [] for a in topo_attrs}

    for i, (_, crossing_row) in enumerate(crossings.iterrows()):
        start_id = (
            _norm_reach_id(crossing_row.get("reach_id"))
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
                normed = _norm_reach_id(token.strip())
                if normed:
                    current = normed
                    break
            else:
                break

        if found_geom is not None and found_reach is not None:
            new_geoms.append(found_geom)
            inside_reach_ids.append(_norm_reach_id(found_reach.get("reach_id")))
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


# ── GloFAS cell selection ─────────────────────────────────────────────────────


def find_best_glofas_cell(
    glofas_clip: xr.Dataset,
    variable: str,
    pt_lon: float,
    pt_lat: float,
    radius_m: float,
    min_mean_q: float,
    utm_crs: str,
) -> tuple[int, int] | None:
    """
    Find the GloFAS grid cell with the highest mean discharge within a search
    radius, subject to a minimum mean-discharge threshold.

    The source point and all GloFAS cell centres are projected to the domain
    UTM CRS (passed as utm_crs) so that distances are computed in metres via
    plain Euclidean arithmetic — no degree approximation, no haversine.

    Args:
        glofas_clip: Clipped xr.Dataset from load_glofas_clip.
        variable:    Discharge variable name (e.g. 'dis24').
        pt_lon:      Longitude of the inside-domain source point (degrees_east).
        pt_lat:      Latitude of the inside-domain source point (degrees_north).
        radius_m:    Search radius in metres.
        min_mean_q:  Minimum mean discharge to consider a cell (m³ s⁻¹).
        utm_crs:     Projected CRS string (e.g. 'EPSG:32636') used for metric
                     distance computation — typically the domain's auto_utm CRS.

    Returns:
        (i_lat, i_lon) index tuple of the selected cell, or None if no
        qualifying cell is found within the radius.
    """
    from pyproj import Transformer

    lat_dim = "latitude" if "latitude" in glofas_clip.dims else "lat"
    lon_dim = "longitude" if "longitude" in glofas_clip.dims else "lon"
    time_dim = "valid_time" if "valid_time" in glofas_clip.dims else "time"
    lat_arr = glofas_clip[lat_dim].values  # (n_lat,)
    lon_arr = glofas_clip[lon_dim].values  # (n_lon,)

    # Project source point and every cell centre to UTM for metric distances
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

    # Time-mean discharge; transpose to guaranteed (n_lat, n_lon) order
    mean_q = (
        glofas_clip[variable]
        .mean(dim=time_dim, skipna=True)
        .transpose(lat_dim, lon_dim)
        .values
    )  # (n_lat, n_lon)

    qualified = within & (mean_q >= min_mean_q)
    if not qualified.any():
        return None

    # Pick cell with highest mean; argmax on flattened then convert to 2-D index
    flat_best = int(np.argmax(np.where(qualified, mean_q, -np.inf)))
    i_lat, i_lon = divmod(flat_best, len(lon_arr))
    return i_lat, i_lon


# ── GRDC gauge matching ───────────────────────────────────────────────────────


def load_grdc_stations(path: str | Path) -> gpd.GeoDataFrame:
    """
    Load GRDC station metadata as a point GeoDataFrame.

    Only the station-dimensioned variables are read (not the (time, id)
    ``runoff_mean`` array), so this is cheap regardless of the time
    dimension's size.

    Args:
        path: Path to GRDC-Daily.nc.

    Returns:
        GeoDataFrame (EPSG:4326), one row per station, geometry =
        Point(geo_x, geo_y). Includes an 'id' column plus whichever of
        'river_name', 'station_name', 'area', 'country' are present.
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
    """
    Find the nearest GRDC station within radius_m of (pt_lon, pt_lat).

    Mirrors find_best_glofas_cell's UTM-projection-distance pattern: both the
    query point and all station locations are projected to utm_crs so
    distance is plain Euclidean metres.

    Args:
        stations_gdf: GeoDataFrame from load_grdc_stations.
        pt_lon, pt_lat: Inside-domain source point (degrees).
        radius_m: Search radius in metres.
        utm_crs: Domain UTM CRS (e.g. 'EPSG:32636').

    Returns:
        Station 'id' (int) of the nearest station within radius, or None if
        stations_gdf is empty or no station is within radius_m.
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
    """
    Load the full daily discharge series for one GRDC station.

    The dataset's -999.0 missing-value sentinel is converted to NaN.

    Args:
        path: Path to GRDC-Daily.nc.
        station_id: GRDC station 'id' (as returned by find_nearest_grdc_station).

    Returns:
        (times, values): times as np.ndarray[datetime64[ns]], values as
        np.ndarray[float64] with -999 -> NaN. NaN-dropping is left to
        downstream alignment (e.g. extreme_values._to_series).
    """
    with xr.open_dataset(path) as ds:
        times = ds["time"].values
        values = ds["runoff_mean"].sel(id=station_id).values.astype(float)

    return times, np.where(values == -999.0, np.nan, values)


# ── boundary crossings ────────────────────────────────────────────────────────

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
    """
    Find points where river reaches intersect the domain bounding-box boundary.

    For each river reach that crosses the bbox exterior, the intersection
    geometry is extracted.  Only Point intersections are retained; Line or
    MultiPoint intersections (reaches running along the boundary) are discarded.

    Args:
        river_gdf: River network GeoDataFrame in EPSG:4326.
        bbox:      Shapely Polygon of the domain bbox in EPSG:4326.

    Returns:
        GeoDataFrame of Point geometries (EPSG:4326) at bbox boundary crossings.
        May be empty if no crossings are found.
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


# ── GloFAS loading ────────────────────────────────────────────────────────────


def load_glofas_clip(
    glofas_path: Path,
    variable: str,
    bounds_wgs84: tuple[float, float, float, float],
    buffer_deg: float,
) -> xr.Dataset:
    """
    Open GloFAS NetCDF file(s), clip spatially to the domain bbox, and load.

    Handles both a single NetCDF file and a directory of files.  When multiple
    files are found, they are combined along the time dimension (detected by
    probing the first file for 'valid_time' or 'time').

    Args:
        glofas_path: Path to a .nc file or directory containing .nc files.
        variable:    Name of the discharge variable to keep (e.g. 'dis24').
        bounds_wgs84: (lon_min, lat_min, lon_max, lat_max) of the domain.
        buffer_deg:  Extra buffer around the domain bbox when clipping (degrees),
                     to ensure crossings near the boundary are covered.

    Returns:
        Clipped xr.Dataset containing only `variable`, loaded into memory.
    """
    glofas_path = Path(glofas_path)
    nc_files = (
        sorted(glofas_path.glob("*.nc")) if glofas_path.is_dir() else [glofas_path]
    )
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found at {glofas_path}")

    lon_min, lat_min, lon_max, lat_max = bounds_wgs84

    def _spatial_slice(ds: xr.Dataset) -> xr.Dataset:
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"
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

    # Slice each file to the small spatial window *before* loading/concatenating,
    # rather than building one big lazy dask dataset via open_mfdataset and only
    # then selecting+loading — for a small spatial window over a large multi-file
    # time series, the latter forces dask to coordinate scattered chunk reads
    # across every file and dominates runtime (profiled at ~95% of this function).
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


# ── river forcing dataset assembly ────────────────────────────────────────────


def build_river_dataset(
    crossings: gpd.GeoDataFrame,
    has_glofas: np.ndarray,
    bankfull_q: np.ndarray,
    discharge_rp_table: np.ndarray,
    return_periods: np.ndarray,
    cell_lon: np.ndarray,
    cell_lat: np.ndarray,
    times: np.ndarray,
    lead_days: float,
    period_hr: float,
    results: list[EVAResult | None],
    rp_bankfull: float,
    bias_corrected: np.ndarray,
    grdc_station_id: np.ndarray,
    grdc_correlation: np.ndarray,
    grdc_overlap_days: np.ndarray,
) -> xr.Dataset:
    """Assemble the river forcing dataset: bankfull discharge (AMAX/GEV),
    a per-crossing return-period discharge table (POT/GPD), and EVA
    diagnostic variables.

    Unlike the previous design, this does NOT build the actual discharge
    timeseries used to force SFINCS -- that's now built at SFINCS-build time
    (rule 13, see build_design_discharge_matrix below) from bankfull_discharge
    + a build-time-configurable return period looked up in discharge_rp_table,
    so changing the design return period no longer requires re-running EVA.

    Args:
        discharge_rp_table: (n_crossing, n_return_period) array -- discharge
            at each of ``return_periods`` from the fitted POT/GPD curve (see
            src.extreme_values.gpd_return_value_table), NaN row where GPD
            didn't converge for that crossing.
        return_periods: Return periods (years) matching discharge_rp_table's
            second axis (src.extreme_values.STANDARD_RETURN_PERIODS_YR).
        times, lead_days, period_hr: stored (not consumed here) so rule 13 can
            reconstruct the same sinusoidal-wave shape at build time --
            see build_design_discharge_matrix.
        bias_corrected: int8 array, 1 if the GloFAS cell feeding this crossing
            was bias-corrected against a GRDC gauge before EVA.
        grdc_station_id: int64 array, matched GRDC station 'id', or -1 if none.
        grdc_correlation: float array, Pearson r (raw GloFAS vs GRDC) over the
            overlap period, or NaN if no correction was applied.
        grdc_overlap_days: float array, number of overlapping valid days used
            for bias correction, or NaN if no correction was applied.
    """
    n_cross = len(crossings)

    # Pull diagnostics into per-crossing arrays (NaN where no result).
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

    # Store inside_reach_id only for crossings that passed all filters.
    # has_glofas=True encapsulates enters_domain AND EVA convergence
    # (visible_on_grid is informational only -- see 07_get_boundary_forcings.py
    # Step 4 -- and no longer gates this); filtered-out crossings get "" so
    # rule 08 never accidentally seeds from them.
    if "inside_reach_id" in crossings.columns:
        inside_reach_ids_arr = np.array(
            [
                (_norm_reach_id(v) or "") if has_glofas[i] else ""
                for i, v in enumerate(crossings["inside_reach_id"].values)
            ],
            dtype="U64",
        )
    else:
        inside_reach_ids_arr = np.full(n_cross, "", dtype="U64")

    return xr.Dataset(
        {
            "discharge_rp_table": (
                ["crossing", "return_period"],
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
            "bankfull_discharge": (
                ["crossing"],
                bankfull_q,
                {
                    "units": "m3 s-1",
                    "long_name": f"bankfull discharge (AMAX/GEV, RP={rp_bankfull:g} yr)",
                },
            ),
            "inside_reach_id": (
                ["crossing"],
                inside_reach_ids_arr,
                {"long_name": "SWORD reach_id of the inside-domain source reach"},
            ),
            # ── EVA diagnostics ──
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
                    "src.extreme_values.gpd_return_value)"
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
            # ── GRDC bias-correction provenance ──
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
                {"long_name": "Pearson r, raw GloFAS vs GRDC over the overlap period"},
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
            "return_period": (
                ["return_period"],
                np.asarray(return_periods, dtype=float),
                {
                    "units": "years",
                    "long_name": "return period axis for discharge_rp_table",
                },
            ),
        },
        attrs={
            # Not consumed here -- stored so rule 13 can reconstruct the same
            # sinusoidal-wave shape at build time (build_design_discharge_matrix)
            # without needing its own copy of these as separate config params.
            "lead_days": lead_days,
            "period_hr": period_hr,
        },
    )


def build_design_discharge_matrix(
    river_ds: xr.Dataset,
    active: np.ndarray,
    design_rp_yr: float,
) -> np.ndarray:
    """
    Build the discharge timeseries actually fed to SFINCS for the given
    design return period, from river_forcing.nc's stored bankfull_discharge +
    discharge_rp_table (+ protection_discharge, if present) -- the SFINCS-
    build-time counterpart to the old rule-07-time sinusoidal_wave call (see
    build_river_dataset). Changing design_rp_yr only requires re-running the
    build (rule 13), not re-running EVA (rule 07).

    Per active crossing:
      1. Look up the design discharge Q_f at design_rp_yr from
         discharge_rp_table, log-RP-interpolated between the two bracketing
         table entries (standard flood-frequency convention; linear
         interpolation in raw RP-space would be badly skewed given the
         table's four-orders-of-magnitude span).
      2. If protection_discharge (Q_p, existing-protection-level correction)
         is present: floor Q_f at bankfull -- protection contains discharge
         up to Q_p without modifying the DEM/floodplain, but the channel
         always carries at least its own bankfull flow (Q_b): Q_f > Q_p ->
         Q_b + (Q_f - Q_p) (overtopped, excess rides on top of bankfull);
         Q_b < Q_f <= Q_p -> Q_b (contained, no flood signal); Q_f <= Q_b ->
         Q_f unchanged.
      3. Build the hydrograph via sinusoidal_wave(bankfull_q, Q_f,
         river_ds.time.values, river_ds.attrs["lead_days"],
         river_ds.attrs["period_hr"]).

    Args:
        river_ds: Opened river_forcing.nc (xr.Dataset).
        active:   Boolean mask, len == river_ds.sizes["crossing"] -- which
                  crossings to build (typically has_glofas).
        design_rp_yr: Return period (years) to build the event at --
                  boundary_setup.design_rp_river_yr.

    Returns:
        (n_active, n_time) np.ndarray, discharge (m3 s-1) per active crossing.
    """
    times = river_ds["time"].values
    bankfull_q = river_ds["bankfull_discharge"].values[active]
    table = river_ds["discharge_rp_table"].values[active]  # (n_active, n_rp)
    table_rps = river_ds["return_period"].values
    lead_days = float(river_ds.attrs["lead_days"])
    period_hr = float(river_ds.attrs["period_hr"])

    n_active = int(active.sum())
    log_rp = np.log(design_rp_yr)
    log_table_rps = np.log(table_rps)
    design_q = np.array(
        [np.interp(log_rp, log_table_rps, table[i]) for i in range(n_active)]
    )

    if "protection_discharge" in river_ds:
        prot_q = river_ds["protection_discharge"].values[active]
        overtopped = design_q > prot_q
        contained = (~overtopped) & (design_q > bankfull_q)
        design_q = np.where(
            overtopped,
            bankfull_q + (design_q - prot_q),
            np.where(contained, bankfull_q, design_q),
        )

    discharge_matrix = np.full((n_active, len(times)), np.nan)
    for i in range(n_active):
        discharge_matrix[i] = sinusoidal_wave(
            bankfull_q[i], design_q[i], times, lead_days, period_hr
        )
    return discharge_matrix


# ── reading from finished river_forcing.nc ────────────────────────────────────


def load_forcing_crossings(
    forcing_path: str | Path,
    discharge_variable: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Load active (has_glofas=1) crossing points from a finished river_forcing.nc.

    The file is opened with decode_times=False because the time coordinate uses
    the non-standard unit string "hours since simulation start" which xarray
    cannot parse as a CF datetime.

    Args:
        forcing_path:       Path to river_forcing.nc produced by rule 07.
        discharge_variable: If provided (e.g. 'bankfull_discharge'), the
                            corresponding values are included in the returned
                            GeoDataFrame as the 'bankfull_q' column.  Pass
                            None when only crossing locations are needed.

    Returns:
        GeoDataFrame with Point geometries (EPSG:4326).  Includes 'bankfull_q'
        column when discharge_variable is supplied.  May be empty if no
        crossings have GloFAS data.
    """
    with xr.open_dataset(forcing_path, decode_times=False) as ds:
        has_glofas = ds["has_glofas"].values.astype(bool)
        lons = ds["longitude"].values
        lats = ds["latitude"].values
        q_vals = ds[discharge_variable].values if discharge_variable else None
        inside_ids = ds["inside_reach_id"].values if "inside_reach_id" in ds else None

    if not has_glofas.any():
        log.warning("No active GloFAS crossings found in river_forcing.nc")
        cols = {"bankfull_q": []} if discharge_variable else {}
        return gpd.GeoDataFrame(cols, geometry=[], crs="EPSG:4326")

    data = {}
    if discharge_variable and q_vals is not None:
        data["bankfull_q"] = q_vals[has_glofas]
    if inside_ids is not None:
        data["inside_reach_id"] = inside_ids[has_glofas].astype(str)
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(lons[has_glofas], lats[has_glofas]),
        crs="EPSG:4326",
    )


# ── crossing reach-id resolution ─────────────────────────────────────────────


def snap_crossings_to_reaches(
    crossings: gpd.GeoDataFrame,
) -> dict[str, float]:
    """
    Build the seed-discharge dict from pre-computed inside_reach_ids.

    The inside_reach_id for each crossing was resolved in rule 07 by
    resolve_inside_domain_reaches and stored in river_forcing.nc.  Only
    crossings that passed all filters (enters_domain, EVA convergence --
    visible_on_grid is informational only, it no longer gates anything)
    carry a non-empty inside_reach_id — they arrive here already filtered
    via load_forcing_crossings.

    When multiple crossings map to the same reach_id only the largest
    bankfull_q is kept.

    Args:
        crossings: GeoDataFrame returned by load_forcing_crossings.  Must have
                   'inside_reach_id' and 'bankfull_q' columns.

    Returns:
        Dict {reach_id_str: bankfull_discharge} with one entry per seed reach.
    """
    if crossings.empty:
        return {}

    if "bankfull_q" not in crossings.columns:
        log.warning(
            "snap_crossings_to_reaches: no 'bankfull_q' column — all crossings discarded"
        )
        return {}

    if "inside_reach_id" not in crossings.columns:
        log.warning(
            "snap_crossings_to_reaches: no 'inside_reach_id' column — "
            "re-run rule 07 to regenerate river_forcing.nc"
        )
        return {}

    seed_q: dict[str, float] = {}
    for _, row in crossings.iterrows():
        rid = _norm_reach_id(str(row.get("inside_reach_id", "") or ""))
        q = row.get("bankfull_q", np.nan)
        if not rid or not np.isfinite(float(q)):
            continue
        existing = seed_q.get(rid)
        if existing is None or float(q) > existing:
            seed_q[rid] = float(q)

    if not seed_q:
        log.warning("No valid seed reaches found from inside_reach_ids")
    else:
        log.info(f"Mapped {len(crossings)} crossing(s) to {len(seed_q)} seed reach(es)")
    return seed_q
