"""
estuarine_depth.py — Leuven et al. (2018) estuarine depth model.

For river reaches within the tidal zone (dist_out <= L_e), computes
depth using an O'Brien-type tidal-prism relation for the cross-sectional
area at the mouth (A0 = C * P^alpha) combined with exponential area
convergence inland (A(x) = A0 * exp(-x / L_A)) and SWORD per-reach widths,
giving d(x) = A(x) / W(x).

For reaches outside L_e (or for any basin where no Nienhuis delta is found
within max_match_dist_km), the power-law depth already computed by rule
add_river_depth is kept unchanged.

Reference:
    Leuven, J.R.F.W. et al. (2018). Empirical Assessment Tool for
    Bathymetry, Flow Velocity and Salinity in Estuaries Based on Tidal
    Amplitude and Remotely-Sensed Imagery. Remote Sensing, 10(12), 1915.
    https://doi.org/10.3390/rs10121915

    Nienhuis, J.H. et al. (2018). Future Change to Tide-Influenced Deltas.
    Geophysical Research Letters, 45, 3499-3507.
    https://doi.org/10.1029/2018GL077638
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.river_network import build_downstream_adjacency, normalize_reach_id

log = logging.getLogger(__name__)


# ── Nienhuis data loading ─────────────────────────────────────────────────────


def load_nienhuis(path: str | Path) -> pd.DataFrame:
    """
    Load the Nienhuis et al. (2018) delta characteristics Excel file.

    Returns a DataFrame with cleaned numeric columns and an 'id' column
    that is the original '#' field as a string (handles entries like '6a').

    Args:
        path: Path to delta_characteristics_72_deltas.xlsx.

    Returns:
        DataFrame indexed 0..N with columns:
            id, name, lat, lon, Q_river, L_e, a, w, P, w_mouth_obs
    """
    df = pd.read_excel(path, sheet_name=0, header=0)

    rename = {
        df.columns[0]: "id",
        "Name": "name",
        "Lat": "lat",
        "Lon": "lon",
    }
    for col in df.columns:
        col_clean = str(col).strip()
        if "Q_river" in col_clean or (
            "Q" in col_clean and "river" in col_clean.lower()
        ):
            rename[col] = "Q_river"
        elif col_clean == "L (m)" or col_clean == "L":
            rename[col] = "L_e"
        elif col_clean == "a (m)" or col_clean == "a":
            rename[col] = "a"
        elif col_clean.startswith("w") and "s-1" in col_clean.lower():
            rename[col] = "w"
        elif col_clean == "P (m3)" or col_clean == "P (m³)" or col_clean == "P":
            rename[col] = "P"
        elif "mouth_obs" in col_clean.lower() and "cor" not in col_clean.lower():
            rename[col] = "w_mouth_obs"

    df = df.rename(columns=rename)
    df["id"] = df["id"].astype(str).str.strip()

    numeric_cols = ["lat", "lon", "Q_river", "L_e", "a", "w", "P", "w_mouth_obs"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()

    # Drop rows with missing essential fields
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    log.info(f"load_nienhuis: loaded {len(df)} delta entries from {path}")
    return df


# ── Basin-to-delta matching ───────────────────────────────────────────────────


def match_basin_to_delta(
    delta_polygon_path: str | Path,
    nienhuis_df: pd.DataFrame,
    max_dist_km: float,
) -> pd.Series | None:
    """
    Find the Nienhuis delta whose Lat/Lon falls within the basin's delta
    polygon bounding box, or within max_dist_km of the bbox boundary.

    Uses a buffered bbox check rather than centroid distance: the delta
    point must fall inside bbox OR within max_dist_km of any bbox edge.
    If multiple candidates qualify, the one closest to the bbox boundary
    is chosen.

    Args:
        delta_polygon_path: Path to delta_polygon.gpkg for this basin.
        nienhuis_df:        DataFrame from load_nienhuis().
        max_dist_km:        Maximum distance (km) from the bbox boundary
                            before a delta is considered a non-match.

    Returns:
        Row of nienhuis_df as a pd.Series if a match is found, else None.
    """
    if nienhuis_df.empty:
        return None

    delta_poly = gpd.read_file(delta_polygon_path).to_crs("EPSG:4326")
    bbox = delta_poly.geometry.union_all().envelope

    buffer_deg = max_dist_km / 111.0  # rough conversion km -> degrees
    bbox_buffered = bbox.buffer(buffer_deg)

    candidates = []
    for _, row in nienhuis_df.iterrows():
        pt = Point(row["lon"], row["lat"])
        if bbox_buffered.contains(pt):
            # Distance from the BBOX boundary (0 if inside bbox)
            dist_km = max(0.0, pt.distance(bbox.boundary) * 111.0)
            candidates.append((dist_km, row))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    best_dist, best_row = candidates[0]
    log.info(
        f"match_basin_to_delta: matched delta '{best_row.get('name', best_row['id'])}' "
        f"(dist_to_bbox={best_dist:.1f} km)"
    )
    return best_row


# ── Estuarine depth calculation ───────────────────────────────────────────────


def compute_estuarine_depths(
    rivers: gpd.GeoDataFrame,
    delta_params: pd.Series,
    obrien_C: float,
    obrien_alpha: float,
    convergence_ratio_k: float,
    blend_fraction: float,
    min_depth_m: float,
    width_column: str = "width",
) -> gpd.GeoDataFrame:
    """
    Apply the Leuven et al. (2018) estuarine depth model to all reaches
    within the tidal zone and linearly blend it with the existing power-law
    depth around the convergence length L_A.

    Depth at reach i with dist_out = x:

        A0              = obrien_C * P^obrien_alpha   [O'Brien mouth area, m²]
        L_A             = L_e / convergence_ratio_k   [convergence length, m]
        W_total_mouth   = sum of widths of all mouth reaches (end_reach == 2)
        d(x)            = A0 * exp(-x / L_A) / W_total_mouth

    Using W_total_mouth as the width normaliser rather than the individual
    reach width means: (a) depth is the same for all reaches at the same
    dist_out x -- it is a property of the estuary cross-section, not of the
    individual channel, and (b) the total cross-sectional area is conserved
    (sum of d*w across all distributaries at distance x = A0*exp(-x/L_A)).
    This correctly handles multi-distributary deltas without over-estimating
    depth in individual narrow arms.

    Mouth reaches (end_reach == 2) are the SWORD terminal reaches flowing
    directly into the sea. If the 'end_reach' column is absent or no such
    reaches are found, all estuarine reaches widths are summed as a fallback.

    Blend zone of width 2 * blend_fraction * L_A centred on L_A:
        alpha   = (x - x_low) / (x_high - x_low)   [0 = estuarine, 1 = fluvial]
        d_blend = (1-alpha)*d_estuarine + alpha*d_fluvial

    Args:
        rivers:               GeoDataFrame from rule add_river_depth
                              (already has 'rivdph' from power-law).
        delta_params:         Matched Nienhuis row (from match_basin_to_delta).
        obrien_C, obrien_alpha: O'Brien relation constants (config).
        convergence_ratio_k:  k such that L_A = L_e / k (config).
        blend_fraction:       Fraction of L_A for the transition zone (config).
        min_depth_m:          Depth floor for estuarine reaches (config).
        width_column:         SWORD width column name (used for mouth-width sum).

    Returns:
        Copy of ``rivers`` with 'rivdph' updated for estuarine/blend reaches
        and new columns 'rivdph_estuarine', 'rivdph_powerlaw', 'rivdph_blend_alpha'.
    """
    L_e = float(delta_params["L_e"])
    P = float(delta_params["P"])

    if not np.isfinite(L_e) or L_e <= 0:
        log.warning("compute_estuarine_depths: L_e is missing or <= 0; skipping")
        return rivers.copy()
    if not np.isfinite(P) or P <= 0:
        log.warning(
            "compute_estuarine_depths: P (tidal prism) is missing or <= 0; skipping"
        )
        return rivers.copy()

    A0 = obrien_C * (P**obrien_alpha)
    # NOTE: L_A is the convergence length scale, not the estuary length L_e (changed here because L_e too long for some basins, e.g. Mississippi > 500km)
    L_A = L_e / convergence_ratio_k
    blend_half = blend_fraction * L_A
    x_lo = L_A - blend_half
    x_hi = L_A + blend_half

    # ── W_total_mouth from end_reach == 2 reaches ─────────────────────────────
    # SWORD 'end_reach' flag: 2 = terminal reach flowing into the sea (mouth).
    if "end_reach" in rivers.columns:
        mouth_mask = rivers["end_reach"].astype(float) == 2.0
        n_mouth = int(mouth_mask.sum())
        if n_mouth > 0:
            W_total_mouth = float(
                rivers.loc[mouth_mask, width_column].fillna(0).clip(lower=0).sum()
            )
        else:
            log.warning(
                "compute_estuarine_depths: no end_reach==2 reaches found; using all estuarine reaches as fallback"
            )
            W_total_mouth = 0.0
    else:
        log.warning(
            "compute_estuarine_depths: 'end_reach' column not present; using all estuarine reaches as fallback"
        )
        n_mouth = 0
        W_total_mouth = 0.0

    if W_total_mouth <= 0:
        # Fallback: sum widths of all reaches with dist_out <= x_hi
        if "dist_out" in rivers.columns:
            est_mask = rivers["dist_out"].apply(
                lambda v: np.isfinite(float(v)) and float(v) <= x_hi
                if v is not None
                else False
            )
            W_total_mouth = float(
                rivers.loc[est_mask, width_column].fillna(0).clip(lower=0).sum()
            )
            log.warning(
                f"compute_estuarine_depths: fallback W_total_mouth={W_total_mouth:.1f} m from {est_mask.sum()} estuarine reaches"
            )
        if W_total_mouth <= 0:
            log.warning(
                "compute_estuarine_depths: cannot determine W_total_mouth; skipping"
            )
            return rivers.copy()

    log.info(
        f"compute_estuarine_depths: L_e={L_e / 1000:.1f} km, P={P:.3g} m³, "
        f"A0={A0:.1f} m², L_A={L_A / 1000:.1f} km, "
        f"W_total_mouth={W_total_mouth:.1f} m (from {n_mouth} end_reach==2 reaches), "
        f"blend zone [{x_lo / 1000:.1f}–{x_hi / 1000:.1f} km]"
    )

    out = rivers.copy()
    out["rivdph_powerlaw"] = out["rivdph"].copy()
    out["rivdph_estuarine"] = False
    out["rivdph_blend_alpha"] = np.nan

    for idx, row in out.iterrows():
        dist_out = row.get("dist_out")
        if dist_out is None or not np.isfinite(float(dist_out)):
            continue
        x = float(dist_out)

        if x > x_hi:
            continue  # purely fluvial — keep power-law depth unchanged

        # Depth is uniform across all reaches at the same x: d(x) = A(x) / W_total_mouth
        d_est = float(A0 * np.exp(-x / L_A)) / W_total_mouth
        d_est = max(d_est, min_depth_m)
        d_fl = float(row["rivdph"]) if np.isfinite(float(row["rivdph"])) else d_est

        if x <= x_lo:
            out.at[idx, "rivdph"] = d_est
            out.at[idx, "rivdph_estuarine"] = True
            out.at[idx, "rivdph_blend_alpha"] = 0.0
        else:
            alpha = (x - x_lo) / (x_hi - x_lo)
            d_blend = (1.0 - alpha) * d_est + alpha * d_fl
            out.at[idx, "rivdph"] = max(d_blend, min_depth_m)
            out.at[idx, "rivdph_estuarine"] = True
            out.at[idx, "rivdph_blend_alpha"] = float(alpha)

    n_est = int(out["rivdph_estuarine"].sum())
    n_blend = int((out["rivdph_blend_alpha"] > 0).sum())
    log.info(
        f"compute_estuarine_depths: {n_est}/{len(out)} reaches updated "
        f"({n_est - n_blend} fully estuarine, {n_blend} in blend zone)"
    )
    return out


def enforce_mouth_depth_monotonic(
    rivers: gpd.GeoDataFrame,
    depth_column: str = "rivdph",
    powerlaw_column: str = "rivdph_powerlaw",
) -> gpd.GeoDataFrame:
    """
    Ensure each river mouth's depth is at least as deep as the deeper of:
    its own power-law (Leopold-Maddock) estimate, or its upstream
    neighbour's final depth -- a river mouth shouldn't act as a shallower
    sill than the channel feeding it, or than its own non-tidal hydraulic-
    geometry sizing would suggest.

    Mirrors enforce_mouth_width_monotonic (src.river_network): same mouth
    definition (reaches with no downstream neighbour in-network) and same
    confluence handling (compared against the DEEPEST of multiple upstream
    neighbours). Applied unconditionally, independent of whether the
    estuarine (Leuven et al.) model actually ran -- when it didn't (disabled,
    or no Nienhuis delta matched), depth_column and powerlaw_column are
    already identical, so this reduces to a plain upstream-continuity check;
    when it did, this catches the case where the O'Brien-relation estimate
    (and/or its min_depth_m floor) put a single mouth's depth well below
    what's physically feeding it -- confirmed live (basin 2433835, single-
    mouth Ebro-matched delta): the exponential-decay estimate came out at
    ~0.24 m at the mouth (dist_out=199 m), floored to min_depth_m=0.5 m, vs.
    ~2.3 m in the immediately upstream reach -- a ~1.6 m sill sitting right
    at the model's downstream boundary.

    Args:
        rivers:          River network with 'reach_id', 'rch_id_dn',
                         depth_column, and (optionally) powerlaw_column.
        depth_column:    Final (possibly estuarine-adjusted) depth column
                         to enforce (default 'rivdph').
        powerlaw_column: Pre-estuarine power-law depth column, one of the
                         floor candidates (default 'rivdph_powerlaw'); if
                         absent, only the upstream-neighbour candidate is
                         used.

    Returns:
        Copy of ``rivers`` with mouth depths raised where needed.
    """
    downstream_adj = build_downstream_adjacency(rivers)
    upstream_adj: dict[str, list[str]] = {rid: [] for rid in downstream_adj}
    for rid, dns in downstream_adj.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(rid)

    rivers = rivers.copy()
    rids = rivers["reach_id"].apply(normalize_reach_id)
    depth_by_rid = dict(zip(rids, rivers[depth_column]))
    has_powerlaw = powerlaw_column in rivers.columns

    mouth_ids = [rid for rid, dns in downstream_adj.items() if not dns]
    n_raised = 0
    for rid in mouth_ids:
        ups = upstream_adj.get(rid, [])
        upstream_depth = max(
            (
                depth_by_rid[u]
                for u in ups
                if u in depth_by_rid and pd.notna(depth_by_rid[u])
            ),
            default=None,
        )
        mouth_mask = rids == rid
        mouth_depth = rivers.loc[mouth_mask, depth_column].iloc[0]
        mouth_power = (
            rivers.loc[mouth_mask, powerlaw_column].iloc[0] if has_powerlaw else None
        )
        candidates = [
            v for v in (upstream_depth, mouth_power) if v is not None and pd.notna(v)
        ]
        if not candidates:
            continue
        floor = max(candidates)
        if pd.notna(mouth_depth) and mouth_depth < floor:
            rivers.loc[mouth_mask, depth_column] = floor
            n_raised += 1

    if n_raised:
        log.info(
            f"enforce_mouth_depth_monotonic: raised depth for {n_raised} mouth "
            f"reach(es) to match the deeper of their own power-law estimate "
            f"or their upstream neighbour"
        )
    return rivers
