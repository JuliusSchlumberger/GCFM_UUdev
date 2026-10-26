"""Quick-look diagnostic figures: one panel per key variable of a pipeline stage."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from src.plots import save_figure


@dataclass(frozen=True)
class Panel:
    """Fixed rendering of one variable; ``kind`` is raster, vector or series."""

    kind: str
    units: str
    vmin: float | None = None
    vmax: float | None = None
    cmap: str = "viridis"
    column: str | None = None


_ELEV = Panel("raster", "m", -10.0, 20.0, "terrain")
_DEPTH = Panel("raster", "m", 0.0, 3.0, "Blues")
_LANDUSE = Panel("raster", "class", 0.0, 200.0, "tab20")
_MANNING = Panel("raster", "s m-1/3", 0.0, 0.1, "YlGn")
_MASK = Panel("raster", "0/1", 0.0, 1.0, "Greys")
_OUTLINE = Panel("vector", "")
_WIDTH = Panel("vector", "m", 0.0, 2000.0, "viridis", column="width")
_LEVEL = Panel("series", "m", -2.0, 6.0)
_DISCHARGE = Panel("series", "m3 s-1", 0.0, 10000.0)

STAGES: dict[str, dict[str, Panel]] = {
    "01_separate_deltas": {"delta_polygon": _OUTLINE},
    "02_model_domain": {"domain": _OUTLINE},
    "02b_prepare_landuse": {"landuse_source": _LANDUSE},
    "03_land_polygons": {"land_polygons": _OUTLINE},
    "05a_elevation": {"elevation_m": _ELEV},
    "05b_landuse": {"landuse": _LANDUSE, "sea_mask": _MASK},
    "05c_roughness": {"manning": _MANNING},
    "06_river_network": {"river_network": _WIDTH},
    "07_boundary_forcings": {
        "river_discharge": _DISCHARGE,
        "surge_water_level": _LEVEL,
    },
    "07c_storm_tide_hydrographs": {"storm_tide": _LEVEL},
    "08_clean_river_network": {
        "river_network_clean": _WIDTH,
        "outflow_points": _OUTLINE,
    },
    "08b_channel_landuse": {"landuse_channels": _LANDUSE},
    "09_condition_elevation": {"elevation_conditioned_m": _ELEV},
    "09b_grid_align_landuse": {
        "landuse_on_grid": _LANDUSE,
        "roughness_on_grid": _MANNING,
    },
    "10_depth_estimation": {
        "river_burned_dem_m": _ELEV,
        "river_depth_m": Panel("vector", "m", 0.0, 20.0, "plasma", column="depth_m"),
        "weir_crest_m": Panel("vector", "m", 0.0, 10.0, "magma", column="crest_m"),
    },
    "13_build_sfincs_skeleton": {"zsini_m": _ELEV},
    "13c_land_bed_level": {"land_bed_level_m": _ELEV},
    "14_run_spinup": {"max_depth_m": _DEPTH},
    "16_run_event": {
        "max_depth_m": _DEPTH,
        "flooded_area_km2": Panel("series", "km2", 0.0, 500.0),
    },
    "18_adaptation": {"max_flood_depth_m": _DEPTH},
    "18c_attribution_mask": {
        "attribution_mask": Panel("raster", "class", 0.0, 3.0, "tab10")
    },
}


def git_hash(repo: Path | None = None) -> str:
    """Return the short commit hash of ``repo`` (default: this file's repo).

    Returns "nogit" when git is not installed or ``repo`` is not a git checkout.
    """
    cwd = repo or Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    return out.stdout.strip()


def _draw_raster(ax, data, spec: Panel):
    if isinstance(data, xr.DataArray):
        values = np.asarray(data.values, dtype=float)
        x, y = data[data.dims[-1]].values, data[data.dims[-2]].values
        extent = (x.min(), x.max(), y.min(), y.max())
        origin = "lower" if y[0] < y[-1] else "upper"
    else:
        values, extent, origin = np.asarray(data, dtype=float), None, "upper"
    if values.ndim != 2:
        raise ValueError(f"raster panel needs 2D data, got shape {values.shape}")
    image = ax.imshow(
        values,
        cmap=spec.cmap,
        vmin=spec.vmin,
        vmax=spec.vmax,
        extent=extent,
        origin=origin,
    )
    return image


def _square_limits(ax, bounds):
    # Collinear features (a coast-parallel weir, points on one meridian) would
    # otherwise collapse one axis to zero width.
    xmin, ymin, xmax, ymax = bounds
    half = max(xmax - xmin, ymax - ymin, 1.0) * 0.55
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")


def _draw_vector(ax, data, spec: Panel):
    if not isinstance(data, gpd.GeoDataFrame):
        raise TypeError(f"vector panel needs a GeoDataFrame, got {type(data).__name__}")
    if spec.column is None:
        data.plot(ax=ax, facecolor="none", edgecolor="black", markersize=12)
        mappable = None
    elif spec.column not in data.columns:
        raise KeyError(f"vector panel needs column {spec.column!r}")
    else:
        data.plot(
            ax=ax,
            column=spec.column,
            cmap=spec.cmap,
            vmin=spec.vmin,
            vmax=spec.vmax,
            linewidth=2,
        )
        mappable = ax.collections[-1]
    _square_limits(ax, data.total_bounds)
    return mappable


def _draw_series(ax, data, spec: Panel):
    series = data if isinstance(data, (pd.Series, pd.DataFrame)) else pd.Series(data)
    series.plot(ax=ax, linewidth=1.2)
    ax.set_ylim(spec.vmin, spec.vmax)
    ax.set_ylabel(spec.units)
    if isinstance(series, pd.DataFrame):
        ax.legend(fontsize=7)
    return None


_DRAW = {"raster": _draw_raster, "vector": _draw_vector, "series": _draw_series}


def render(
    stage: str,
    panels: dict[str, object],
    scenario: str,
    out_dir: Path,
    commit: str | None = None,
) -> Path:
    """Write ``<out_dir>/<stage>_<scenario>.png`` with one panel per variable of ``stage``.

    ``panels`` maps each variable in ``STAGES[stage]`` to its data: a 2D array or
    DataArray (raster), a GeoDataFrame (vector) or a Series/DataFrame indexed by
    time (series). Colour scales and axis limits are fixed per variable so
    figures from different runs compare directly.
    """
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGES)}")
    specs = STAGES[stage]
    missing, extra = set(specs) - set(panels), set(panels) - set(specs)
    if missing or extra:
        raise KeyError(
            f"{stage}: missing variables {sorted(missing)}, unexpected {sorted(extra)}"
        )

    commit = commit or git_hash()
    fig = Figure(figsize=(4.2 * len(specs), 3.8), layout="constrained")
    axes = fig.subplots(1, len(specs), squeeze=False)[0]
    for ax, (name, spec) in zip(axes, specs.items()):
        mappable = _DRAW[spec.kind](ax, panels[name], spec)
        if mappable is not None:
            fig.colorbar(mappable, ax=ax, shrink=0.8, label=spec.units)
        ax.set_title(name, fontsize=9)
    fig.suptitle(f"{stage} | {scenario} | {commit}", fontsize=10)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stage}_{scenario}.png"
    save_figure(fig, path, dpi=80)
    return path
