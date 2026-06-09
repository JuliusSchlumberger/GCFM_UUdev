# Archived: 2026-06-05
# Source: workflow/src/plots.py  (function plot_datum_correction_delta)
#
# Diagnostic map of Δ elevation = MSL_elevation − EGM2008_elevation over the
# model domain.  Was called at the end of 03a_get_elevation.py.
#
# See also:
#   03a_datum_correction_excerpt.py — the call site and delta_correction computation
#   compute_geoid_offset_arr.py     — the geoid synthesis function

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


_PLOT_DPI = 100
_PLOT_MAX_PX = 2_000_000


def _save(fig, output_path):
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_datum_correction_delta(
    delta,
    utm_crs_str,
    wgs84_bounds,
    osm_land_path,
    output_path,
):
    """
    Diverging raster plot of Δ elevation = MSL_elevation − EGM2008_elevation.

    Shows the spatial pattern of the vertical datum correction applied to
    DiluviumDEM: geoid shift (N_EGM2008 − N_GOCO06s) + MDT subtraction.
    Only land pixels (where DiluviumDEM had valid data) are shown; ocean is grey.
    """
    from pyproj import Transformer

    h, w = delta.shape
    lon_min, lat_min, lon_max, lat_max = wgs84_bounds

    try:
        plot_crs = gpd.GeoSeries(
            gpd.points_from_xy([lon_min, lon_max], [lat_min, lat_max]),
            crs="EPSG:4326",
        ).estimate_utm_crs()
    except Exception:
        plot_crs = utm_crs_str

    _trans = Transformer.from_crs("EPSG:4326", plot_crs, always_xy=True)
    _west, _south = _trans.transform(lon_min, lat_min)
    _east, _north = _trans.transform(lon_max, lat_max)
    extent = (_west, _east, _south, _north)

    total = h * w
    factor = max(1, int(np.ceil(np.sqrt(total / _PLOT_MAX_PX))))
    delta_ds = delta[::factor, ::factor]

    land = gpd.read_file(osm_land_path, bbox=(lon_min, lat_min, lon_max, lat_max))

    valid = delta_ds[~np.isnan(delta_ds)]
    vmax = float(np.percentile(np.abs(valid), 95)) if len(valid) > 0 else 1.0
    vmax = max(vmax, 0.01)

    fig, ax = plt.subplots(figsize=(9, 7))
    if not land.empty:
        land.to_crs(plot_crs).plot(
            ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1
        )
    im = ax.imshow(
        delta_ds,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        origin="upper",
        zorder=2,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="both")
    cb.set_label("Δ elevation (m)  [MSL − EGM2008]")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(
        "Vertical datum correction — DiluviumDEM\n"
        "(EGM2008  →  GOCO06s geoid  →  local MSL via MDT_CNES-CLS22)"
    )
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)
