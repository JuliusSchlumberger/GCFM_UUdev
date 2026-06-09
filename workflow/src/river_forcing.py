"""River boundary forcing: GloFAS loading, crossing detection, snapping, dataset assembly."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon

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
    """Return True if any downstream reach ID in rch_id_dn is present in the domain."""
    s = str(rch_id_dn_raw).strip().strip("[]")
    if not s or s.lower() in ("nan", "none", "<na>"):
        return False
    for token in s.split(","):
        normed = _norm_reach_id(token.strip())
        if normed and normed in domain_reach_ids:
            return True
    return False


# ── inside-domain source location resolution ─────────────────────────────────


def resolve_inside_domain_reaches(
    crossings: gpd.GeoDataFrame,
    river_gdf: gpd.GeoDataFrame,
    domain_poly: Polygon,
    max_hops: int = 30,
) -> gpd.GeoDataFrame:
    """
    Walk downstream from each crossing reach to find the reach whose centroid
    lies inside the domain polygon AND whose lakeflag is not 1 (sea/reservoir).
    Returns a copy of `crossings` with:

    - geometry updated to the inside-domain reach centroid (crossing point used
      as fallback when no qualifying reach is found within max_hops)
    - inside_reach_id: reach_id of the found inside-domain reach (None on fallback)
    - width, max_width, lakeflag: attributes of the inside-domain reach (if
      present in river_gdf), overwriting the corresponding attributes of the
      crossing reach

    Reaches inside the domain with lakeflag == 1 are skipped and the walk
    continues downstream until a non-lake/sea reach is found.

    The geometry update ensures that both the visible_on_grid filter (width of
    the inside-domain reach) and the GloFAS radius search use the same
    representative inside-domain position.

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

    topo_attrs = [
        a for a in ("width", "max_width", "lakeflag") if a in river_gdf.columns
    ]

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
            centroid = reach_row.geometry.interpolate(0.5, normalized=True)
            if domain_poly.contains(centroid):
                try:
                    lf = int(float(str(reach_row.get("lakeflag", 0) or 0)))
                except (ValueError, TypeError):
                    lf = 0
                if lf != 1:
                    found_geom = centroid
                    found_reach = reach_row
                    break
                log.debug(
                    f"Crossing {i}: reach {current} inside domain but "
                    f"lakeflag=1 (sea/reservoir); continuing downstream"
                )
            dn_raw = str(reach_row.get("rch_id_dn", "") or "").strip().strip("[]")
            if not dn_raw or dn_raw.lower() in ("nan", "none", "<na>"):
                break
            current = None
            for token in dn_raw.split(","):
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
                f"Crossing {i} (reach_id={start_id}): no inside-domain reach with "
                f"lakeflag≠1 found within {max_hops} hops; using crossing point as fallback"
            )

    result = crossings.copy()
    result["geometry"] = new_geoms
    result["inside_reach_id"] = inside_reach_ids
    for a in topo_attrs:
        result[a] = inside_attr_vals[a]
    return result


# ── DEM elevation filter ─────────────────────────────────────────────────────


def resolve_dem_elevation_reach(
    crossings: gpd.GeoDataFrame,
    river_gdf: gpd.GeoDataFrame,
    elev_threshold: float,
    domain_poly: Polygon,
    max_hops: int = 50,
) -> gpd.GeoDataFrame:
    """
    Walk downstream from the inside-domain reach to find a reach whose
    max_elevation is within the valid DEM range (≤ elev_threshold).

    Should be called after resolve_inside_domain_reaches.  Updates geometry,
    inside_reach_id, and topographic attributes (width, max_width, lakeflag,
    max_elevation) to the newly-found reach.  Adds boolean column
    `within_dem_range` — False when no qualifying reach is found within
    max_hops.

    Args:
        crossings:       GeoDataFrame returned by resolve_inside_domain_reaches.
        river_gdf:       Clipped river network (EPSG:4326) with max_elevation column.
        elev_threshold:  Maximum accepted elevation (clip_elevation_m + buffer_m).
        domain_poly:     Shapely Polygon of the domain (EPSG:4326).
        max_hops:        Maximum downstream hops before marking as out of range.
    """
    import pandas as pd  # noqa: F401 – used via type hints only

    if "max_elevation" not in river_gdf.columns:
        log.warning(
            "river_gdf has no 'max_elevation' column — "
            "DEM elevation filter skipped; all crossings marked within_dem_range=True"
        )
        result = crossings.copy()
        result["within_dem_range"] = True
        return result

    reach_lookup: dict[str, "pd.Series"] = {}
    for _, row in river_gdf.iterrows():
        rid = _norm_reach_id(row.get("reach_id"))
        if rid:
            reach_lookup[rid] = row

    topo_attrs = [
        a
        for a in ("width", "max_width", "lakeflag", "max_elevation")
        if a in river_gdf.columns
    ]

    new_geoms: list = []
    new_inside_ids: list = []
    within_range: list[bool] = []
    inside_attr_vals: dict[str, list] = {a: [] for a in topo_attrs}

    for i, (_, crossing_row) in enumerate(crossings.iterrows()):
        start_id = _norm_reach_id(crossing_row.get("inside_reach_id"))
        if start_id is None:
            start_id = _norm_reach_id(crossing_row.get("reach_id"))

        start_row = reach_lookup.get(start_id) if start_id else None
        if start_row is None:
            new_geoms.append(crossing_row.geometry)
            new_inside_ids.append(crossing_row.get("inside_reach_id"))
            within_range.append(False)
            for a in topo_attrs:
                inside_attr_vals[a].append(crossing_row.get(a))
            if start_id:
                log.warning(
                    f"Crossing {i}: inside_reach_id={start_id} not found in river "
                    f"network — marked outside DEM elevation range"
                )
            continue

        # Fast-path: starting reach already within threshold
        start_elev = float(start_row.get("max_elevation", np.nan) or np.nan)
        if not np.isfinite(start_elev) or start_elev <= elev_threshold:
            new_geoms.append(crossing_row.geometry)
            new_inside_ids.append(crossing_row.get("inside_reach_id"))
            within_range.append(True)
            for a in topo_attrs:
                inside_attr_vals[a].append(crossing_row.get(a))
            continue

        # Walk downstream until we find a reach that is inside the domain,
        # not a lake, and has max_elevation <= elev_threshold.
        current = start_id
        visited: set[str] = set()
        found_geom = None
        found_reach = None

        for _ in range(max_hops):
            if not current or current in visited or current not in reach_lookup:
                break
            visited.add(current)
            reach_row = reach_lookup[current]

            centroid = reach_row.geometry.interpolate(0.5, normalized=True)
            if domain_poly.contains(centroid):
                try:
                    lf = int(float(str(reach_row.get("lakeflag", 0) or 0)))
                except (ValueError, TypeError):
                    lf = 0
                if lf != 1:
                    elev = float(reach_row.get("max_elevation", np.nan) or np.nan)
                    if not np.isfinite(elev) or elev <= elev_threshold:
                        found_geom = centroid
                        found_reach = reach_row
                        break

            dn_raw = str(reach_row.get("rch_id_dn", "") or "").strip().strip("[]")
            if not dn_raw or dn_raw.lower() in ("nan", "none", "<na>"):
                break
            current = None
            for token in dn_raw.split(","):
                normed = _norm_reach_id(token.strip())
                if normed:
                    current = normed
                    break

        if found_geom is not None and found_reach is not None:
            new_geoms.append(found_geom)
            new_inside_ids.append(_norm_reach_id(found_reach.get("reach_id")))
            within_range.append(True)
            for a in topo_attrs:
                inside_attr_vals[a].append(found_reach.get(a))
            found_elev = float(found_reach.get("max_elevation", np.nan) or np.nan)
            log.debug(
                f"Crossing {i}: walked to reach "
                f"{_norm_reach_id(found_reach.get('reach_id'))} "
                f"(max_elevation={found_elev:.1f} m ≤ {elev_threshold:.1f} m)"
            )
        else:
            new_geoms.append(crossing_row.geometry)
            new_inside_ids.append(crossing_row.get("inside_reach_id"))
            within_range.append(False)
            for a in topo_attrs:
                inside_attr_vals[a].append(crossing_row.get(a))
            log.warning(
                f"Crossing {i} (reach_id={_norm_reach_id(crossing_row.get('reach_id'))}): "
                f"no reach with max_elevation ≤ {elev_threshold:.1f} m found within "
                f"{max_hops} hops (start elevation {start_elev:.1f} m) — "
                f"crossing marked outside DEM elevation range"
            )

    result = crossings.copy()
    result["geometry"] = new_geoms
    result["inside_reach_id"] = new_inside_ids
    result["within_dem_range"] = within_range
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
    flood_q: np.ndarray,
    cell_lon: np.ndarray,
    cell_lat: np.ndarray,
    times: np.ndarray,
    lead_days: float,
    period_hr: float,
    results: list[EVAResult | None],
    rp_bankfull: float,
    rp_flood: float,
) -> xr.Dataset:
    """Assemble the river forcing dataset with synthetic sinusoidal waves rising
    from RP=2 (bankfull) to RP=100 (flood), plus EVA diagnostic variables.

    """
    n_cross = len(crossings)
    discharge_matrix = np.full((n_cross, len(times)), np.nan)
    for i in range(n_cross):
        if has_glofas[i]:
            discharge_matrix[i] = sinusoidal_wave(
                bankfull_q[i], flood_q[i], times, lead_days, period_hr
            )

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
    # has_glofas=True encapsulates enters_domain AND visible_on_grid AND
    # within_dem_range AND EVA convergence; filtered-out crossings get ""
    # so rule 05 never accidentally seeds from them.
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
            "discharge": (
                ["crossing", "time"],
                discharge_matrix,
                {"units": "m3 s-1", "long_name": "river discharge"},
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
            "flood_discharge": (
                ["crossing"],
                flood_q,
                {
                    "units": "m3 s-1",
                    "long_name": f"flood discharge (POT/GPD, RP={rp_flood:g} yr)",
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
        },
        coords={
            "longitude": (["crossing"], cross_lons, {"units": "degrees_east"}),
            "latitude": (["crossing"], cross_lats, {"units": "degrees_north"}),
            "time": (["time"], times, {"units": "hours since simulation start"}),
        },
    )


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
        forcing_path:       Path to river_forcing.nc produced by rule 04.
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

    The inside_reach_id for each crossing was resolved in rule 04 by
    resolve_inside_domain_reaches + resolve_dem_elevation_reach and stored in
    river_forcing.nc.  Only crossings that passed all filters (enters_domain,
    visible_on_grid, within_dem_range, EVA convergence) carry a non-empty
    inside_reach_id — they arrive here already filtered via load_forcing_crossings.

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
            "re-run rule 04 to regenerate river_forcing.nc"
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
