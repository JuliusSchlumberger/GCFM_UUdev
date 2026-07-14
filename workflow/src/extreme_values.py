"""Extreme-value analysis for GloFAS river discharge.

Provides per-cell estimation of return-period discharges using pyextremes:

* RP=bankfull (default 2 yr) via Block Maxima / GEV on annual maxima.
* RP=flood (default 100 yr) via Peaks-Over-Threshold / GPD with
  autocorrelation-based declustering and iterative threshold search.
* Mann-Kendall + Sen's-slope trend test on the annual-maxima series.

All EVA parameters are read from a plain dict (``eva_cfg``) matching the
``boundary_forcings.eva`` section of the project config YAML, including the
newer optional knobs ``min_years_high_confidence`` (default 40.0),
``peaks_per_year_min`` (default 1.0), ``threshold_min_pct`` (default 50.0),
and ``deseasonalize_for_decorr``
(default True) — see ``analyse_cell`` / ``_search_threshold`` for how each is
used and what it defaults to when absent from the YAML.

Bias correction against GRDC gauges
------------------------------------
GloFAS is a reanalysis and carries unquantified structural uncertainty.
Where an observed gauge exists on the main stem near the delta apex (e.g.
GRDC — Global Runoff Data Centre) and the overlapping record with GloFAS is
long enough (``boundary_forcings.bias_correction.min_overlap_days``),
``bias_correct_discharge`` applies empirical quantile mapping over the
overlapping period: empirical CDFs of GloFAS and the gauge are built from
the overlap sample, and the full GloFAS record is remapped to match the
gauge's distribution. Values outside the overlap's quantile range (in
particular the upper, flood-relevant tail) are linearly extrapolated using
the outermost quantile-pair slope rather than clamped, so the correction
does not flatten flood peaks that exceed anything seen in the overlap.
Where no gauge is found within
``boundary_forcings.river.bias_correction.grdc_search_radius_km``, or the overlap is too
short, "GloFAS-as-truth" remains the documented limitation (logged by the
caller, not an error).

Public API
----------
EVAResult              – dataclass holding all outputs for one grid-cell run.
analyse_cell(times, values, eva_cfg, label) -> EVAResult
plot_cell_diagnostics(times, values, eva_cfg, output_path, label) -> EVAResult
empirical_quantile_map(source_overlap, target_overlap, source_full, n_quantiles=200) -> np.ndarray
compute_grdc_correlation(glofas_times, glofas_values, grdc_times, grdc_values, min_overlap_days, label, tail_percentile=90.0) -> dict | None
bias_correct_discharge(glofas_times, glofas_values, grdc_times, grdc_values, cfg, label) -> tuple[np.ndarray, dict | None]
plot_bias_correction(glofas_times, glofas_values, diagnostics, output_path, label) -> None
plot_grdc_overview(domain_poly, river_gdf, crossings_gdf, grdc_stations, highlight_crossing_idx, highlight_station_id, diagnostics, output_path, label) -> None
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

from pyextremes import EVA
import pymannkendall as mk

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import Polygon

from src.plots import map_background

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="lmoments3")
# NoDataBlockWarning fires for every empty annual block when errors='coerce' is
# used in get_extremes("BM", ...).  We request that behaviour deliberately, so
# the warning is expected and carries no new information.
warnings.filterwarnings("ignore", message=".*blocks contained no data.*")

# ── Prevent multiprocessing worker-process spawning on Windows ────────────────
# On Windows, Python uses the "spawn" start method for new processes.  Libraries
# like pyextremes / lmoments3 may internally launch worker pools for bootstrap
# CI computation; when those workers start, they try to re-import the calling
# script (__main__), which fails inside a Snakemake execution because 'snakemake'
# is only injected into the parent process.
#
# To avoid this we (a) limit BLAS / OpenMP threading to 1 (prevents many
# thread-based pools) and (b) run our own sequential scipy bootstrap rather
# than pyextremes' alpha-based bootstrap, which is what spawns the pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# ── result container ──────────────────────────────────────────────────────────


@dataclass
class EVAResult:
    """Outcome of extreme-value analysis for a single GloFAS cell."""

    # headline return levels
    q_rp2: float = np.nan
    q_rp100: float = np.nan
    q_rp2_ci: tuple[float, float] = (np.nan, np.nan)
    q_rp100_ci: tuple[float, float] = (np.nan, np.nan)

    # discharge at an arbitrary requested return period (e.g. an existing
    # flood-protection design standard), from the POT/GPD fit -- see
    # analyse_cell's protection_rp argument. NaN unless requested.
    q_protection: float = np.nan

    # POT diagnostics
    pot_threshold: float = np.nan
    pot_r_days: float = np.nan
    pot_decorr_days: float = np.nan
    pot_peaks_per_year: float = np.nan
    pot_n_peaks: int = 0
    pot_threshold_status: str = "none"
    pot_shape: float = np.nan
    pot_scale: float = np.nan
    pot_ks_pvalue: float = np.nan

    # AMAX diagnostics
    amax_n_blocks: int = 0
    gev_shape: float = np.nan

    # trend (on AMAX series)
    trend: str = "none"
    trend_pvalue: float = np.nan
    sen_slope: float = np.nan

    # bookkeeping
    n_valid_days: int = 0
    record_years: float = np.nan
    ok: bool = False
    messages: list[str] = field(default_factory=list)

    def as_flat_dict(self) -> dict:
        d = asdict(self)
        d["q_rp2_ci_lower"], d["q_rp2_ci_upper"] = self.q_rp2_ci
        d["q_rp100_ci_lower"], d["q_rp100_ci_upper"] = self.q_rp100_ci
        d.pop("q_rp2_ci")
        d.pop("q_rp100_ci")
        d.pop("messages")
        return d


# ── config helpers ────────────────────────────────────────────────────────────


def _c(cfg: dict, key: str, default):
    """Safe dict access with a typed default."""
    return cfg.get(key, default)


# ── helpers ───────────────────────────────────────────────────────────────────


def _to_series(times: np.ndarray, values: np.ndarray) -> pd.Series:
    """Build a clean, datetime-indexed, NaN-free, sorted discharge series."""
    s = pd.Series(
        np.asarray(values, dtype=float),
        index=pd.to_datetime(np.asarray(times)),
    )
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s.dropna()


# ── GRDC bias correction ──────────────────────────────────────────────────────


def _overlap_series(
    glofas_times: np.ndarray,
    glofas_values: np.ndarray,
    grdc_times: np.ndarray,
    grdc_values: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """Align a GloFAS cell series and a GRDC station series to shared dates.

    Both inputs are passed through ``_to_series`` (dedup'd, sorted,
    NaN-dropped), then restricted to the intersection of their valid-data
    dates.

    Returns:
        (glofas_overlap, grdc_overlap), two pd.Series sharing the same
        DatetimeIndex, sorted ascending. May be empty if the records do not
        overlap.
    """
    g = _to_series(glofas_times, glofas_values)
    r = _to_series(grdc_times, grdc_values)
    common_idx = g.index.intersection(r.index)
    return g.loc[common_idx], r.loc[common_idx]


def empirical_quantile_map(
    source_overlap: np.ndarray,
    target_overlap: np.ndarray,
    source_full: np.ndarray,
    n_quantiles: int = 200,
) -> np.ndarray:
    """Bias-correct ``source_full`` via empirical quantile mapping.

    Empirical quantiles of ``source_overlap`` and ``target_overlap`` (paired
    by date, same length) are built at ``n_quantiles`` evenly spaced
    probability levels. ``source_full`` (the entire source record, a
    superset of ``source_overlap``) is then mapped through this quantile
    pairing via ``np.interp``.

    ``np.interp`` clamps out-of-range inputs to the boundary output value,
    which would flatten exactly the flood-relevant upper tail when
    ``source_full`` contains values beyond the overlap sample's maximum (or
    minimum). Instead, values outside ``[src_q[0], src_q[-1]]`` are linearly
    extrapolated using the slope of the outermost quantile-pair segment on
    the corresponding side. A degenerate (zero-width) outermost segment
    falls back to an additive offset carried from that boundary.

    The result is floored at zero (discharge non-negativity).

    Args:
        source_overlap: Source (GloFAS) values during the overlap period.
        target_overlap: Target (GRDC) values during the overlap period,
            paired by date with ``source_overlap`` (same length/order).
        source_full: Full-record source values to be corrected (superset of
            ``source_overlap``).
        n_quantiles: Number of evenly spaced probability levels used to build
            the empirical quantile pairing.

    Returns:
        np.ndarray, same shape as ``source_full``.
    """
    source_full = np.asarray(source_full, dtype=float)
    probs = np.linspace(0.0, 1.0, n_quantiles)
    src_q = np.quantile(source_overlap, probs)
    tgt_q = np.quantile(target_overlap, probs)

    corrected = np.interp(source_full, src_q, tgt_q)

    hi_mask = source_full > src_q[-1]
    if hi_mask.any():
        dx, dy = src_q[-1] - src_q[-2], tgt_q[-1] - tgt_q[-2]
        if dx > 0:
            corrected[hi_mask] = tgt_q[-1] + (dy / dx) * (
                source_full[hi_mask] - src_q[-1]
            )
        else:
            corrected[hi_mask] = source_full[hi_mask] + (tgt_q[-1] - src_q[-1])

    lo_mask = source_full < src_q[0]
    if lo_mask.any():
        dx, dy = src_q[1] - src_q[0], tgt_q[1] - tgt_q[0]
        if dx > 0:
            corrected[lo_mask] = tgt_q[0] + (dy / dx) * (
                source_full[lo_mask] - src_q[0]
            )
        else:
            corrected[lo_mask] = source_full[lo_mask] + (tgt_q[0] - src_q[0])

    return np.clip(corrected, 0.0, None)


def compute_grdc_correlation(
    glofas_times: np.ndarray,
    glofas_values: np.ndarray,
    grdc_times: np.ndarray,
    grdc_values: np.ndarray,
    min_overlap_days: int = 0,
    label: str = "",
    tail_percentile: float = 90.0,
) -> dict | None:
    """Tail-focused GloFAS-vs-GRDC overlap correlation diagnostics.

    Unlike ``bias_correct_discharge``, this is independent of
    ``boundary_forcings.bias_correction.enabled`` — it is the basis for the
    EQM fit in ``bias_correct_discharge``, and is also used directly by the
    standalone GRDC-inspection test, which reports the raw correlation
    regardless of whether the correction itself is applied.

    The headline ``correlation_raw`` is restricted to overlap days where GRDC
    discharge is at or above ``tail_percentile`` of its own overlap
    distribution — this pipeline only cares about GloFAS/GRDC agreement at
    flood-relevant high flows, and an all-days Pearson r is dominated by the
    many low/moderate-flow days, which would otherwise mask a poor fit in the
    tail (or vice versa). The all-days correlation is still computed and
    returned as ``correlation_full_raw`` for reference/plotting context.

    Args:
        glofas_times, glofas_values: Full-record GloFAS cell series.
        grdc_times, grdc_values: Full-record GRDC station series (e.g. from
            ``river_forcing.load_grdc_series``, with -999 already mapped to NaN).
        min_overlap_days: Minimum overlapping valid days required; below this,
            ``None`` is returned.
        label: Identifier used in log messages.
        tail_percentile: Percentile (0-100) of the GRDC overlap distribution
            used as the tail threshold for the headline correlation.

    Returns:
        ``None`` if the overlap is shorter than ``min_overlap_days``, else a
        dict with ``grdc_overlap_days``, ``correlation_raw`` (tail-focused),
        ``correlation_full_raw`` (all-days, for reference), ``tail_percentile``,
        ``tail_threshold``, ``tail_n_days``, ``tail_mask``, ``glofas_overlap``,
        ``grdc_overlap``, ``glofas_overlap_times``.
    """
    tag = f"[{label}] " if label else ""
    g_overlap, r_overlap = _overlap_series(
        glofas_times, glofas_values, grdc_times, grdc_values
    )
    n_overlap = len(g_overlap)

    if n_overlap < min_overlap_days:
        log.info(
            f"{tag}GRDC overlap = {n_overlap} d (< {min_overlap_days}) — "
            f"insufficient for correlation"
        )
        return None

    g_vals, r_vals = g_overlap.values, r_overlap.values
    corr_full = float(np.corrcoef(g_vals, r_vals)[0, 1])

    tail_threshold = float(np.percentile(r_vals, tail_percentile))
    tail_mask = r_vals >= tail_threshold
    n_tail = int(tail_mask.sum())
    if n_tail >= 2:
        corr_tail = float(np.corrcoef(g_vals[tail_mask], r_vals[tail_mask])[0, 1])
    else:
        corr_tail = float("nan")
        log.warning(
            f"{tag}fewer than 2 overlap days at/above the p{tail_percentile:g} "
            f"GRDC tail threshold ({n_tail}) — tail correlation undefined"
        )

    return {
        "grdc_overlap_days": n_overlap,
        "correlation_raw": corr_tail,
        "correlation_full_raw": corr_full,
        "tail_percentile": float(tail_percentile),
        "tail_threshold": tail_threshold,
        "tail_n_days": n_tail,
        "tail_mask": tail_mask,
        "glofas_overlap": g_vals,
        "grdc_overlap": r_vals,
        "glofas_overlap_times": g_overlap.index.values,
    }


def bias_correct_discharge(
    glofas_times: np.ndarray,
    glofas_values: np.ndarray,
    grdc_times: np.ndarray,
    grdc_values: np.ndarray,
    cfg: dict,
    label: str = "",
) -> tuple[np.ndarray, dict | None]:
    """Bias-correct a GloFAS cell series against a matched GRDC station.

    Args:
        glofas_times, glofas_values: Full-record GloFAS cell series.
        grdc_times, grdc_values: Full-record GRDC station series (e.g. from
            ``river_forcing.load_grdc_series``, with -999 already mapped to NaN).
        cfg: ``boundary_forcings.bias_correction`` config dict
            (keys: ``enabled``, ``min_overlap_days``, ``tail_percentile``).
        label: Identifier used in log messages.

    Returns:
        (corrected_values, diagnostics):
          corrected_values: np.ndarray, same shape as ``glofas_values``.
              Equal to ``glofas_values`` unchanged if correction is skipped
              (disabled, or overlap below ``min_overlap_days``).
          diagnostics: ``None`` if skipped, else a dict with the keys from
              ``compute_grdc_correlation`` (``correlation_raw``/
              ``correlation_corrected`` are tail-focused, see that function's
              docstring) plus ``correlation_full_corrected`` (all-days, for
              reference) and ``corrected_overlap`` — consumed by
              ``plot_bias_correction`` and by the caller when populating
              ``river_forcing.nc`` provenance variables.
    """
    glofas_values = np.asarray(glofas_values, dtype=float)

    if not bool(_c(cfg, "enabled", True)):
        return glofas_values, None

    min_overlap_days = int(_c(cfg, "min_overlap_days", 730))
    tail_percentile = float(_c(cfg, "tail_percentile", 90.0))
    diagnostics = compute_grdc_correlation(
        glofas_times,
        glofas_values,
        grdc_times,
        grdc_values,
        min_overlap_days,
        label=label,
        tail_percentile=tail_percentile,
    )
    if diagnostics is None:
        return glofas_values, None

    corrected_full = empirical_quantile_map(
        source_overlap=diagnostics["glofas_overlap"],
        target_overlap=diagnostics["grdc_overlap"],
        source_full=glofas_values,
    )

    g_full_series = _to_series(glofas_times, glofas_values)
    overlap_index = pd.DatetimeIndex(diagnostics["glofas_overlap_times"])
    corrected_overlap = (
        pd.Series(corrected_full, index=g_full_series.index).loc[overlap_index].values
    )
    grdc_overlap = diagnostics["grdc_overlap"]
    tail_mask = diagnostics["tail_mask"]
    corr_corrected_full = float(np.corrcoef(corrected_overlap, grdc_overlap)[0, 1])
    if tail_mask.sum() >= 2:
        corr_corrected_tail = float(
            np.corrcoef(corrected_overlap[tail_mask], grdc_overlap[tail_mask])[0, 1]
        )
    else:
        corr_corrected_tail = float("nan")

    diagnostics["correlation_corrected"] = corr_corrected_tail
    diagnostics["correlation_full_corrected"] = corr_corrected_full
    diagnostics["corrected_overlap"] = corrected_overlap

    tag = f"[{label}] " if label else ""
    log.info(
        f"{tag}GRDC bias correction: {diagnostics['grdc_overlap_days']} d overlap "
        f"({diagnostics['tail_n_days']} d at/above p{tail_percentile:g} GRDC tail) — "
        f"tail r: {diagnostics['correlation_raw']:.3f} -> {corr_corrected_tail:.3f}  "
        f"(all-days r: {diagnostics['correlation_full_raw']:.3f} -> {corr_corrected_full:.3f})"
    )
    return corrected_full, diagnostics


def _deseasonalize_for_decorr(s: pd.Series) -> pd.Series:
    """Day-of-year climatology anomaly — for decorrelation estimation only.

    Subtracts the mean discharge per day-of-year (computed from the series
    itself) before the autocorrelation-based decorrelation window is estimated.
    Monsoonal/seasonal persistence otherwise inflates the autocorrelation and
    yields spuriously long decorrelation windows (and thus declustering radii
    ``r``) that reflect the season cycle rather than true event-independence.

    NOTE: returns an anomaly series used *solely* to estimate ``r`` via
    ``estimate_decorrelation_days`` — the original (seasonal) series ``s`` is
    still what POT extraction and all fitting operate on.
    """
    climatology = s.groupby(s.index.dayofyear).transform("mean")
    return s - climatology


def estimate_decorrelation_days(series: pd.Series, cutoff: float, max_lag: int) -> int:
    """Lag (days) at which autocorrelation first drops below ``cutoff``."""
    x = series.values.astype(float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0 or len(x) <= 2:
        return 1
    upper = int(min(max_lag, len(x) - 1))
    for lag in range(1, upper + 1):
        if float(np.dot(x[:-lag], x[lag:])) / denom < cutoff:
            return lag
    return upper


def _rv(eva_instance, rp: float) -> float:
    """Extract a scalar return value from a pyextremes EVA instance.

    pyextremes may return a plain float or a 3-tuple (value, lower, upper)
    depending on the installed version; this helper handles both.
    """
    result = eva_instance.get_return_value(rp)
    if isinstance(result, (tuple, list, np.ndarray)) and len(result) >= 1:
        return float(result[0])
    return float(result)


def gpd_return_value(
    threshold: float,
    scale: float,
    shape: float,
    peaks_per_year: float,
    return_period: float,
) -> float:
    """Discharge for an arbitrary return period from an already-fitted GPD.

    Reconstructs pyextremes' own POT/GPD return-value formula directly from
    the fitted parameters (pot_threshold/pot_scale/pot_shape/pot_peaks_per_year
    -- e.g. as saved in river_forcing.nc), with no need to re-fit or re-touch
    the raw discharge series. Matches _gpd_boot_ci's own formula exactly
    (confirmed against pyextremes.eva.EVA.get_return_value's source: exceedance
    probability = 1/(return_period * peaks_per_year), evaluated via
    scipy.stats.genpareto.ppf on the exceedances distribution -- fit with
    floc=0 -- then shifted back up by the threshold).

    Args:
        threshold:      POT threshold (m³/s) -- pot_threshold.
        scale:          Fitted GPD scale parameter -- pot_scale.
        shape:          Fitted GPD shape parameter (c) -- pot_shape.
        peaks_per_year: Declustered peaks per year at that threshold --
                        pot_peaks_per_year.
        return_period:  Return period (years) to evaluate.

    Returns:
        Discharge (m³/s) for the requested return period, or NaN if any
        input isn't finite.
    """
    inputs = (threshold, scale, shape, peaks_per_year, return_period)
    if not all(np.isfinite(v) for v in inputs):
        return np.nan
    p = max(0.0, 1.0 - 1.0 / (peaks_per_year * return_period))
    return float(threshold + stats.genpareto.ppf(p, shape, scale=scale))


# Standard return-period list stored per crossing in river_forcing.nc
# (discharge_rp_table) -- 5-yr steps from 5 to 1000 yr, then 1000-yr steps to
# 10000 yr, plus the sub-5yr bankfull-adjacent points (1, 1.5, 2 yr). Fixed,
# not configurable: this is a reference table meant to be interpolated at an
# arbitrary requested return period (see build_design_discharge_matrix in
# river_forcing.py), not a set of independently tunable knobs.
STANDARD_RETURN_PERIODS_YR: np.ndarray = np.array(
    sorted(
        set(
            [1.0, 1.5, 2.0]
            + [float(x) for x in range(5, 1001, 5)]
            + [float(x) for x in range(1000, 10001, 1000)]
        )
    )
)


def gpd_return_value_table(
    threshold: float,
    scale: float,
    shape: float,
    peaks_per_year: float,
    return_periods: np.ndarray = STANDARD_RETURN_PERIODS_YR,
) -> np.ndarray:
    """Discharge at every return period in ``return_periods`` from an
    already-fitted GPD -- vectorized sibling of ``gpd_return_value``, same
    formula, no re-fitting.

    Args:
        threshold:      POT threshold (m³/s) -- pot_threshold.
        scale:          Fitted GPD scale parameter -- pot_scale.
        shape:          Fitted GPD shape parameter (c) -- pot_shape.
        peaks_per_year: Declustered peaks per year at that threshold --
                        pot_peaks_per_year.
        return_periods: Return periods (years) to evaluate, default
                        STANDARD_RETURN_PERIODS_YR.

    Returns:
        np.ndarray, same shape as ``return_periods`` -- all-NaN if any of
        threshold/scale/shape/peaks_per_year isn't finite.
    """
    return_periods = np.asarray(return_periods, dtype=float)
    inputs = (threshold, scale, shape, peaks_per_year)
    if not all(np.isfinite(v) for v in inputs):
        return np.full(return_periods.shape, np.nan)
    p = np.clip(1.0 - 1.0 / (peaks_per_year * return_periods), 0.0, None)
    return threshold + stats.genpareto.ppf(p, shape, scale=scale)


# ── sequential bootstrap CIs (no multiprocessing — safe on Windows/Snakemake) ─


def _gev_boot_ci(
    amax: np.ndarray,
    rp: float,
    alpha: float,
    n_boot: int = 200,
    c0: float | None = None,
    loc0: float | None = None,
    scale0: float | None = None,
) -> tuple[float, float]:
    """Bootstrap CI for GEV return level using scipy MLE — no worker processes.

    ``c0``/``loc0``/``scale0`` (the point-estimate fit's parameters) seed the
    optimizer for every resample — bootstrap samples are draws from the same
    series, so they sit close to that optimum and converge in far fewer
    iterations than scipy's generic default starting guess (profiled as the
    dominant cost of analyse_cell). The fitting method is unchanged (still MLE
    on each resample), so estimates match the unseeded version within tolerance.

    Known estimator inconsistency: the point estimate (``q_rp2`` in
    ``analyse_cell``) is fit via ``fit_method`` (L-moments by default), while
    this CI is built from repeated scipy MLE refits. We deliberately do NOT
    switch the point estimate to MLE — for short hydrological records,
    L-moments is generally the more robust/less biased point estimator — so a
    minor mismatch between "where the point estimate sits" and "where the
    bootstrap distribution is centered" is accepted as a known, minor wart.
    """
    rng = np.random.default_rng(42)
    boot: list[float] = []
    fit_args = (c0,) if c0 is not None and np.isfinite(c0) else ()
    fit_kwargs = {}
    if loc0 is not None and np.isfinite(loc0):
        fit_kwargs["loc"] = loc0
    if scale0 is not None and np.isfinite(scale0):
        fit_kwargs["scale"] = scale0
    for _ in range(n_boot):
        sample = rng.choice(amax, size=len(amax), replace=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c, loc, scale = stats.genextreme.fit(sample, *fit_args, **fit_kwargs)
            rl = float(stats.genextreme.ppf(1.0 - 1.0 / rp, c, loc=loc, scale=scale))
            if np.isfinite(rl) and rl > 0:
                boot.append(rl)
        except Exception:
            pass
    if len(boot) < 20:
        return np.nan, np.nan
    lo_p = (1.0 - alpha) / 2.0 * 100.0
    hi_p = (1.0 + alpha) / 2.0 * 100.0
    return float(np.percentile(boot, lo_p)), float(np.percentile(boot, hi_p))


def _gpd_boot_ci(
    exceedances: np.ndarray,
    threshold: float,
    ppy: float,
    rp: float,
    alpha: float,
    n_boot: int = 200,
    c0: float | None = None,
    scale0: float | None = None,
) -> tuple[float, float]:
    """Bootstrap CI for GPD return level using scipy MLE — no worker processes.

    See ``_gev_boot_ci`` — ``c0``/``scale0`` warm-start the optimizer from the
    point-estimate fit's parameters to cut convergence iterations per resample.

    Same known estimator inconsistency as ``_gev_boot_ci``: the point estimate
    (``q_rp100``) comes from an L-moments (``fit_method``) fit, this CI from
    repeated scipy MLE refits — retained deliberately because L-moments is the
    more robust point estimator for short records.
    """
    rng = np.random.default_rng(42)
    boot: list[float] = []
    fit_args = (c0,) if c0 is not None and np.isfinite(c0) else ()
    fit_kwargs = {}
    if scale0 is not None and np.isfinite(scale0):
        fit_kwargs["scale"] = scale0
    for _ in range(n_boot):
        sample = rng.choice(exceedances, size=len(exceedances), replace=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c, _, scale = stats.genpareto.fit(
                    sample, *fit_args, floc=0, **fit_kwargs
                )
            p = max(0.0, 1.0 - 1.0 / (ppy * rp))
            rl = float(threshold + stats.genpareto.ppf(p, c, scale=scale))
            if np.isfinite(rl) and rl > 0:
                boot.append(rl)
        except Exception:
            pass
    if len(boot) < 20:
        return np.nan, np.nan
    lo_p = (1.0 - alpha) / 2.0 * 100.0
    hi_p = (1.0 + alpha) / 2.0 * 100.0
    return float(np.percentile(boot, lo_p)), float(np.percentile(boot, hi_p))


def _search_threshold(
    s: pd.Series,
    r_days: int,
    record_years: float,
    cfg: dict,
    tag: str,
):
    """Iterate thresholds until peaks/year falls in the acceptance band.

    Targets a peaks-per-year (ppy) band of [ppy_min, ppy_max].  The search
    starts at ``threshold_start_pct`` and moves the percentile in the direction
    that brings ppy toward the band each iteration:

    * ppy > ppy_max → raise threshold (fewer peaks, ppy falls)
    * ppy < ppy_min → lower threshold (more peaks, ppy rises)

    The bidirectional logic is critical: a purely upward search (old behaviour)
    could never recover when the starting percentile already yields ppy < ppy_min,
    because raising the threshold only makes ppy smaller.  ``threshold_min_pct``
    (default 50.0) caps how far down the threshold may go.

    Returns (fitted EVA instance, (threshold, peaks_per_year, status)) or
    (last_eva, last_tuple) flagged with the specific rejection reason
    ('reject: ppy<min' or 'reject: ppy>max') if no iteration met the
    criterion, so it's visible (e.g. in the diagnostic plot title) whether
    the threshold search ran out of room lowering the threshold (too few
    independent peaks even at a low percentile -- typical of slow,
    large-basin rivers) or raising it (too many peaks even at a high
    percentile).
    """
    thr_start = float(_c(cfg, "threshold_start_pct", 90.0))
    thr_step = float(_c(cfg, "threshold_step_pct", 1.0))
    thr_max = float(_c(cfg, "threshold_max_pct", 99.5))
    thr_min = float(_c(cfg, "threshold_min_pct", 50.0))
    max_iter = int(_c(cfg, "threshold_max_iter", 30))
    ppy_min = float(_c(cfg, "peaks_per_year_min", 1.0))
    ppy_max = float(_c(cfg, "peaks_per_year_max", 5.0))
    _fit_method = str(_c(cfg, "fit_method", "Lmoments"))  # reserved for future use

    pct = float(np.clip(thr_start, thr_min, thr_max))
    last_eva = last_choice = None

    for it in range(1, max_iter + 1):
        thr = float(np.percentile(s.values, pct))
        eva = EVA(data=s)
        eva.get_extremes("POT", threshold=thr, r=f"{r_days}D")
        ppy = len(eva.extremes) / record_years

        if ppy_min <= ppy <= ppy_max:
            status = "good"
        elif ppy < ppy_min:
            status = "reject: ppy<min"
        else:
            status = "reject: ppy>max"

        log.info(
            f"{tag}threshold iter {it}/{max_iter}: "
            f"pct={pct:.1f} thr={thr:.2f} peaks/yr={ppy:.2f} "
            f"(target [{ppy_min:.1f}, {ppy_max:.1f}]) → {status}"
        )
        last_eva, last_choice = eva, (thr, ppy, status)

        if status == "good":
            return eva, last_choice

        # Step toward the band; stop if we have hit the corresponding bound.
        if ppy > ppy_max:
            next_pct = pct + thr_step
            if next_pct > thr_max:
                break
        else:  # ppy < ppy_min
            next_pct = pct - thr_step
            if next_pct < thr_min:
                break
        pct = next_pct

    log.warning(
        f"{tag}threshold search exhausted {max_iter} iters without reaching "
        f"the [{ppy_min:.1f}, {ppy_max:.1f}] peaks/yr band; "
        f"using last attempt (flagged reject)"
    )
    return last_eva, last_choice


# ── fast AMAX-only entry point ────────────────────────────────────────────────


def analyse_cell_gev_only(
    times: np.ndarray,
    values: np.ndarray,
    eva_cfg: dict,
    label: str = "",
) -> float:
    """AMAX/GEV fit returning the bankfull return-level only.

    Skips POT/GPD, trend test, and bootstrap CIs — use when only the RP=rp_bf
    point estimate is needed and computational speed matters (e.g. validating
    many grid cells).

    Returns np.nan if the fit fails or the series is too short.
    """
    s = _to_series(times, values)
    min_days = int(_c(eva_cfg, "min_days", 9131))  # ≈25 yr — see analyse_cell
    fit_method = str(_c(eva_cfg, "fit_method", "Lmoments"))
    rp_bf = float(_c(eva_cfg, "rp_bf", 2.0))
    tag = f"[{label}] " if label else ""

    if s.size < min_days:
        return np.nan

    try:
        eva_bm = EVA(data=s)
        eva_bm.get_extremes("BM", block_size="365.2425D", errors="coerce")
        if len(eva_bm.extremes) < 3:
            return np.nan
        eva_bm.fit_model(fit_method)
        return _rv(eva_bm, rp_bf)
    except Exception as exc:
        log.debug(f"{tag}AMAX-only fit failed: {exc}")
        return np.nan


# ── full pipeline entry point ─────────────────────────────────────────────────


def analyse_cell(
    times: np.ndarray,
    values: np.ndarray,
    eva_cfg: dict,
    label: str = "",
    protection_rp: float | None = None,
) -> EVAResult:
    """Run the full EVA pipeline on one cell's daily discharge series.

    Args:
        times:         Daily time coordinate array (numpy datetime64 or numeric).
        values:        Discharge values aligned with ``times`` (may contain NaN).
        eva_cfg:       Plain dict from config["boundary_forcings"]["river"]["eva"].
        label:         Optional identifier used in log messages.
        protection_rp: If given, also evaluate the POT/GPD fit at this return
                       period (yr) and store it as ``res.q_protection`` --
                       reuses the already-fitted GPD model, no extra fitting.
                       Used for the existing flood-protection-level
                       correction (top-level protection_levels config);
                       NaN if the POT/GPD fit itself failed.

    Returns:
        EVAResult. On failure of either model, NaN fields remain and ``ok``
        is False — a bad cell never aborts a whole domain.
    """
    # NOTE — this function is bias-correction agnostic: if a GRDC gauge match
    # was found, the caller (07_get_boundary_forcings.py) already replaced
    # ``values`` with the bias-corrected series via bias_correct_discharge()
    # before calling analyse_cell. See the module docstring for details.
    res = EVAResult()
    s = _to_series(times, values)
    res.n_valid_days = int(s.size)
    tag = f"[{label}] " if label else ""

    # ≈25 yr hard floor: RP100 (100-yr return level) extrapolates far beyond a
    # short record's range — GloFAS v4 reanalysis runs from 1979 (~45 yr at most
    # locations), so this is achievable while still leaving headroom above the
    # bare statistical minimum.
    min_days = int(_c(eva_cfg, "min_days", 9131))
    # Soft confidence threshold (yr): below this, EVA still runs but the result
    # is flagged as reduced-confidence rather than skipped outright.
    min_years_hc = float(_c(eva_cfg, "min_years_high_confidence", 40.0))
    fit_method = str(_c(eva_cfg, "fit_method", "Lmoments"))
    rp_bf = float(_c(eva_cfg, "rp_bf", 2.0))
    rp_fl = float(_c(eva_cfg, "rp_fl", 100.0))
    ci_alpha = float(_c(eva_cfg, "ci_alpha", 0.95))
    decorr_cut = float(_c(eva_cfg, "decorr_cut", 1.0 / np.e))
    decorr_max = int(_c(eva_cfg, "decorr_max", 30))
    deseasonalize = bool(_c(eva_cfg, "deseasonalize_for_decorr", True))

    if s.size < min_days:
        msg = (
            f"{tag}only {s.size} valid days "
            f"(< {min_days} ≈ {min_days / 365.2425:.0f} yr); skipping EVA"
        )
        log.warning(msg)
        res.messages.append(msg)
        return res

    record_years = (s.index[-1] - s.index[0]).days / 365.2425
    res.record_years = record_years
    if record_years <= 0:
        res.messages.append(f"{tag}degenerate time span; skipping")
        return res

    if record_years < min_years_hc:
        msg = (
            f"{tag}record length {record_years:.1f} yr < "
            f"{min_years_hc:.0f} yr — return-level estimates carry "
            f"reduced confidence (not skipped, just flagged)"
        )
        log.info(msg)
        res.messages.append(msg)

    # ── AMAX / GEV → RP=rp_bf (bankfull) + trend ────────────────────────────
    #
    # Deliberate dual-estimator design — BM/GEV here for RP2, POT/GPD below for
    # RP100 — is intentional, NOT an inconsistency to "fix":
    #   * The bankfull ≈ 1.5-2 yr recurrence convention (Leopold & Wolman,
    #     geomorphology literature) is *defined* on the annual-maximum series,
    #     so RP2 must be derived from block maxima to honour that definition.
    #   * RP100 sits far enough into the tail that POT/GPD's use of all
    #     over-threshold peaks (not just one value per year) gives a much
    #     better-constrained tail fit than AMAX/GEV would for the same record.
    # Because the two return levels come from two different estimators fit to
    # different extracted samples, comparing them head-to-head (e.g.
    # "RP100 < RP2 ⇒ reject") is an apples-to-oranges check that flags a false
    # symptom — removed in favour of the per-fit sanity checks below.
    try:
        eva_bm = EVA(data=s)
        # errors='coerce': skip empty blocks (e.g. years with gaps in GloFAS data)
        eva_bm.get_extremes("BM", block_size="365.2425D", errors="coerce")
        res.amax_n_blocks = int(len(eva_bm.extremes))
        eva_bm.fit_model(fit_method)
        # Call without alpha — prevents pyextremes from spawning worker processes.
        # CI is computed separately via a sequential scipy bootstrap below.
        # _rv() handles the version-dependent return type (scalar or 3-tuple).
        res.q_rp2 = _rv(eva_bm, rp_bf)
        gev_params = eva_bm.model.fit_parameters
        res.gev_shape = float(gev_params.get("c", np.nan))

        # Per-fit sanity check (advisory — appends a message, does not flip `ok`):
        # |shape| > 0.5 indicates a heavy-tailed/Frechet or strongly bounded
        # Weibull-type fit that is implausible for river discharge and usually
        # signals a poorly conditioned fit (short record, odd AMAX sample, …).
        if np.isfinite(res.gev_shape) and abs(res.gev_shape) > 0.5:
            res.messages.append(
                f"{tag}GEV shape c={res.gev_shape:.3f} outside plausible "
                f"range (|c| > 0.5) — bankfull fit may be poorly constrained"
            )

        amax = eva_bm.extremes.values.astype(float)
        # Caveat: this CI reflects sampling variability of the AMAX series only
        # — it is conditional on the chosen block definition and propagates
        # neither POT-threshold nor declustering-window uncertainty (those only
        # affect q_rp100_ci below, which carries the same caveat).
        res.q_rp2_ci = _gev_boot_ci(
            amax,
            rp_bf,
            ci_alpha,
            c0=gev_params.get("c"),
            loc0=gev_params.get("loc"),
            scale0=gev_params.get("scale"),
        )

        # Mann-Kendall + Sen's slope on the AMAX series — intentionally left
        # as-is (no FDR/multiple-comparisons correction):
        #   * Each delta has only 1-6 boundary-forcing locations, so there is
        #     no multiple-comparisons problem at this scale.
        #   * Annual maxima from consecutive years are close to independent,
        #     which is what MK's test assumes — so the test is appropriate here
        #     even though MK is often misapplied to serially-correlated daily data.
        # A significant trend means fitting a *stationary* GEV/GPD introduces
        # bias (the historical record no longer represents the current/future
        # regime). The trend flag is therefore meant to feed a downstream HUMAN
        # decision (detrend the series / pick a more recent reference period /
        # move to a non-stationary fit) — not to be acted on automatically here.
        if len(amax) >= 4:
            mkr = mk.original_test(amax)
            res.trend = str(mkr.trend)
            res.trend_pvalue = float(mkr.p)
            res.sen_slope = float(mkr.slope)
        else:
            res.messages.append(f"{tag}too few AMAX blocks for trend test")
    except Exception as e:
        msg = f"{tag}AMAX/GEV fit failed: {e}"
        log.warning(msg)
        res.messages.append(msg)

    # ── POT / GPD → RP=rp_fl (flood) ────────────────────────────────────────
    try:
        # The declustering window stays data-driven and per-location (it must
        # NOT be a fixed constant across deltas — flood-wave duration scales
        # with catchment size and flow regime, so a one-size-fits-all window
        # would over- or under-decluster depending on the basin).
        #
        # Optionally estimate it from a deseasonalized anomaly series: raw
        # daily discharge in monsoonal/seasonal regimes has long-range
        # autocorrelation from the seasonal cycle itself (not from individual
        # flood events persisting), which would otherwise inflate the 1/e
        # crossing lag and produce spuriously long decorrelation windows. The
        # deseasonalized series is used ONLY to pick `r_days` — POT extraction
        # and all fitting below still operate on the original series `s`.
        decorr_series = _deseasonalize_for_decorr(s) if deseasonalize else s
        decorr = estimate_decorrelation_days(decorr_series, decorr_cut, decorr_max)
        res.pot_decorr_days = float(decorr)
        r_days = max(1, int(round(decorr)))
        res.pot_r_days = float(r_days)
        log.info(
            f"{tag}declustering: deseasonalized={deseasonalize}  "
            f"decorr ≈ {decorr} d → r = {r_days} d"
        )

        eva_pot, chosen = _search_threshold(s, r_days, record_years, eva_cfg, tag)

        if eva_pot is not None and chosen is not None:
            thr, ppy, status = chosen
            res.pot_threshold = float(thr)
            res.pot_peaks_per_year = float(ppy)
            res.pot_n_peaks = int(len(eva_pot.extremes))
            res.pot_threshold_status = status

            eva_pot.fit_model(fit_method)
            res.q_rp100 = _rv(eva_pot, rp_fl)
            if protection_rp is not None:
                res.q_protection = _rv(eva_pot, protection_rp)
            gpd_params = eva_pot.model.fit_parameters
            # pyextremes picks between "genpareto" and "expon" candidate
            # distributions by AIC (see EVA.fit_model). "expon" has no shape
            # parameter at all -- it IS the GPD with shape fixed at 0 (the
            # c -> 0 limit of the GPD family) -- so a missing "c" key means
            # shape=0 was selected as the better fit, not that fitting failed.
            res.pot_shape = float(gpd_params.get("c", 0.0))
            # Saved alongside pot_threshold/pot_shape/pot_peaks_per_year so a
            # downstream consumer can compute the discharge for an arbitrary
            # return period later without re-fitting -- see gpd_return_value.
            res.pot_scale = float(gpd_params.get("scale", np.nan))

            # Per-fit sanity check (advisory): |shape| > 0.5 is implausible for
            # river-discharge tails and usually signals a poorly conditioned
            # POT fit (too few peaks, badly chosen threshold, …).
            if np.isfinite(res.pot_shape) and abs(res.pot_shape) > 0.5:
                res.messages.append(
                    f"{tag}GPD shape c={res.pot_shape:.3f} outside plausible "
                    f"range (|c| > 0.5) — flood fit may be poorly constrained"
                )

            # pyextremes' POT `.extremes` are ABSOLUTE peak values (not
            # already-thresholded exceedances) — subtract the threshold to get
            # the exceedances a floc=0 GPD must be fit to.
            exceedances = eva_pot.extremes.values.astype(float) - res.pot_threshold
            neg_mask = exceedances < 0
            if neg_mask.any():
                res.messages.append(
                    f"{tag}{int(neg_mask.sum())} POT peak(s) below threshold "
                    f"(negative exceedance) — filtered before GPD bootstrap"
                )
                exceedances = exceedances[~neg_mask]

            # Caveat (see also q_rp2_ci above): this CI is conditional on the
            # chosen POT threshold AND declustering window `r_days` — neither
            # source of uncertainty is propagated into the bootstrap, which
            # only resamples the already-extracted exceedances.
            res.q_rp100_ci = _gpd_boot_ci(
                exceedances,
                res.pot_threshold,
                ppy,
                rp_fl,
                ci_alpha,
                c0=gpd_params.get("c"),
                scale0=gpd_params.get("scale"),
            )
            try:
                res.pot_ks_pvalue = float(eva_pot.test_ks().pvalue)
                if res.pot_ks_pvalue < 0.05:
                    res.messages.append(
                        f"{tag}KS test p={res.pot_ks_pvalue:.3f} < 0.05 — "
                        f"GPD fit does not pass goodness-of-fit at α=0.05"
                    )
            except Exception:
                pass
    except Exception as e:
        msg = f"{tag}POT/GPD fit failed: {e}"
        log.warning(msg)
        res.messages.append(msg)

    # `ok` reflects only "did both estimators converge to a finite value" — we
    # deliberately do NOT cross-compare q_rp100 vs q_rp2 (see the comment above
    # the AMAX/GEV block for why that check was apples-to-oranges). The
    # per-fit sanity messages above (shape parameters, KS p-value) are advisory
    # only and do not affect `ok`.
    res.ok = np.isfinite(res.q_rp2) and np.isfinite(res.q_rp100)
    return res


# ── diagnostic plot ───────────────────────────────────────────────────────────


def plot_cell_diagnostics(
    times: np.ndarray,
    values: np.ndarray,
    eva_cfg: dict,
    output_path: str | Path,
    label: str = "",
) -> EVAResult:
    """Render a 2×2 diagnostic figure for one cell and return its EVAResult."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit_method = str(_c(eva_cfg, "fit_method", "Lmoments"))
    _ci_alpha = float(_c(eva_cfg, "ci_alpha", 0.95))  # reserved for future use
    rp_bf = float(_c(eva_cfg, "rp_bf", 2.0))
    rp_fl = float(_c(eva_cfg, "rp_fl", 100.0))

    res = analyse_cell(times, values, eva_cfg, label=label)
    s = _to_series(times, values)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"GloFAS EVA diagnostics — {label}", fontsize=13, y=0.98)

    # POT (RP=rp_fl)
    try:
        if np.isfinite(res.pot_threshold):
            eva_pot = EVA(data=s)
            eva_pot.get_extremes(
                "POT", threshold=res.pot_threshold, r=f"{int(res.pot_r_days)}D"
            )
            eva_pot.fit_model(fit_method)
            # top-left: discharge timeseries with extracted peaks
            eva_pot.plot_extremes(ax=axes[0, 0], show_clusters=True)
            axes[0, 0].set_title(f"POT extremes & clusters (r={int(res.pot_r_days)}d)")
            # top-right: GPD return-value curve
            eva_pot.plot_return_values(
                return_period=np.logspace(np.log10(1.1), np.log10(500), 100),
                ax=axes[0, 1],
            )
            axes[0, 1].set_title(
                f"POT/GPD  thr={res.pot_threshold:.1f}  r={int(res.pot_r_days)}d  "
                f"{res.pot_peaks_per_year:.1f} pk/yr  [{res.pot_threshold_status}]"
            )
            axes[0, 1].axvline(rp_fl, ls=":", c="k", lw=1)
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f"POT plot failed:\n{e}", ha="center", va="center")

    # AMAX (RP=rp_bf)
    try:
        if res.amax_n_blocks:
            eva_bm = EVA(data=s)
            eva_bm.get_extremes("BM", block_size="365.2425D", errors="coerce")
            eva_bm.fit_model(fit_method)
            # bottom-left: GEV return-value curve
            eva_bm.plot_return_values(
                return_period=np.logspace(np.log10(1.01), np.log10(100), 100),
                ax=axes[1, 0],
            )
            axes[1, 0].set_title(f"AMAX/GEV  ({res.amax_n_blocks} blocks)")
            axes[1, 0].axvline(rp_bf, ls=":", c="k", lw=1)
            # bottom-right: annual-maxima boxplots by decade
            _plot_decade_trend(axes[1, 1], eva_bm.extremes, res)
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f"AMAX plot failed:\n{e}", ha="center", va="center")

    # pyextremes's plot_extremes() (top-left) calls fig.autofmt_xdate()
    # internally, which assumes every subplot shares ONE date x-axis and
    # hides tick labels/xlabel on every row except the last -- but this
    # figure's four axes each have their OWN independent x-axis (time,
    # return period x2, decade), not a shared date axis, so that call
    # incorrectly hides both top-row axes' tick numbers (their xlabel text
    # gets set again by each panel's own plotting call afterwards, which is
    # why only the tick NUMBERS were missing, not the axis labels).
    axes[0, 0].set_xlabel("date-time")
    for row_ax in (axes[0, 0], axes[0, 1]):
        row_ax.xaxis.set_tick_params(labelbottom=True)
        for lbl in row_ax.get_xticklabels():
            lbl.set_visible(True)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    log.info(f"Wrote EVA diagnostic plot: {output_path}")
    return res


def _plot_decade_trend(ax, amax: pd.Series, res: EVAResult) -> None:
    yrs = amax.index.year.values
    vals = amax.values.astype(float)
    start = (yrs.min() // 10) * 10
    edges = np.arange(start, yrs.max() + 11, 10)
    groups, labels = [], []
    for lo in edges[:-1]:
        m = (yrs >= lo) & (yrs < lo + 10)
        if m.sum() >= 2:
            groups.append(vals[m])
            labels.append(f"{lo}s")
    if groups:
        positions = np.arange(len(groups))
        ax.boxplot(groups, positions=positions, widths=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    if np.isfinite(res.sen_slope):
        xline = (np.array([yrs.min(), yrs.max()]) - start) / 10.0 - 0.5
        yline = np.median(vals) + res.sen_slope * (
            np.array([yrs.min(), yrs.max()]) - np.median(yrs)
        )
        ax.plot(xline, yline, "r-", lw=2, label=f"Sen {res.sen_slope:+.2f}/yr")
        ax.legend(loc="best", fontsize=9)
    ax.set_ylabel("Annual-max discharge (m³ s⁻¹)")
    ax.set_title(
        f"AMAX by decade — MK: {res.trend} (p={res.trend_pvalue:.3f})"
        if np.isfinite(res.trend_pvalue)
        else "AMAX by decade"
    )


# ── GRDC diagnostic plot panels (shared by plot_bias_correction and plot_grdc_overview) ──

_SEASON_MAP = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
_SEASON_COLORS = {
    "DJF": "tab:blue",
    "MAM": "tab:green",
    "JJA": "tab:red",
    "SON": "tab:orange",
}


def _plot_raw_scatter(
    ax,
    g_ov: np.ndarray,
    r_ov: np.ndarray,
    r_tail: float,
    r_full: float,
    tail_mask: np.ndarray,
    tail_percentile: float,
    lims: tuple[float, float],
) -> None:
    """
    GloFAS-vs-GRDC scatter with 1:1 line; tail days (>= tail_percentile of
    GRDC) highlighted, with both the tail-focused (headline) and all-days
    Pearson r in the title.
    """
    ax.scatter(
        g_ov[~tail_mask],
        r_ov[~tail_mask],
        s=8,
        alpha=0.35,
        c="steelblue",
        label="Below tail",
    )
    ax.scatter(
        g_ov[tail_mask],
        r_ov[tail_mask],
        s=16,
        alpha=0.8,
        c="crimson",
        label=f"Tail (≥ p{tail_percentile:g})",
    )
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("GloFAS raw (m³ s⁻¹)")
    ax.set_ylabel("GRDC (m³ s⁻¹)")
    ax.set_title(
        f"Raw GloFAS vs GRDC  (tail r = {r_tail:.3f}; all-days r = {r_full:.3f})"
    )
    ax.legend(loc="upper left", fontsize=8)


def _plot_seasonal_scatter(
    ax,
    g_ov: np.ndarray,
    r_ov: np.ndarray,
    t_ov: pd.DatetimeIndex,
    lims: tuple[float, float],
) -> None:
    """GloFAS-vs-GRDC scatter coloured by season, with per-season Pearson r in the legend."""
    seasons = pd.Series(t_ov.month).map(_SEASON_MAP).values
    for s_name, s_color in _SEASON_COLORS.items():
        m = seasons == s_name
        if m.any():
            if m.sum() >= 2:
                r_season = float(np.corrcoef(g_ov[m], r_ov[m])[0, 1])
                s_label = f"{s_name} (r = {r_season:.2f})"
            else:
                s_label = s_name
            ax.scatter(g_ov[m], r_ov[m], s=8, alpha=0.5, c=s_color, label=s_label)
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("GloFAS raw (m³ s⁻¹)")
    ax.set_ylabel("GRDC (m³ s⁻¹)")
    ax.set_title("By season")
    ax.legend(loc="upper left", fontsize=8, ncol=2)


def _plot_overlap_timeseries(
    ax, t_ov: pd.DatetimeIndex, g_ov: np.ndarray, r_ov: np.ndarray, r_tail: float
) -> None:
    """Overlap-period time series of GloFAS raw and GRDC, on twin y-axes."""
    ax.plot(t_ov, g_ov, color="tab:blue", lw=0.7, label="GloFAS raw")
    ax.set_xlabel("Date")
    ax.set_ylabel("GloFAS raw (m³ s⁻¹)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.plot(t_ov, r_ov, color="tab:orange", lw=0.7, label="GRDC")
    ax2.set_ylabel("GRDC (m³ s⁻¹)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax.set_title(f"Overlap period: GloFAS raw vs GRDC  (tail r = {r_tail:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)


# ── GRDC bias-correction diagnostic plot ──────────────────────────────────────


def plot_bias_correction(
    glofas_times: np.ndarray,
    glofas_values: np.ndarray,
    diagnostics: dict,
    output_path: str | Path,
    label: str = "",
) -> None:
    """Render the 2x3 GRDC bias-correction diagnostic figure.

    Panels:
        (0,0) raw GloFAS-vs-GRDC scatter with 1:1 line, tail days (>= the
              configured GRDC tail percentile) highlighted, and both the
              tail-focused (headline) and all-days Pearson r in the title.
        (0,1) same scatter, points coloured by season (DJF/MAM/JJA/SON).
        (0,2) overlap-period time series of GloFAS raw and GRDC, on twin
              y-axes so both series remain visible regardless of scale.
        (1,0) empirical CDFs of GloFAS raw, GRDC, and GloFAS corrected
              (log-x to emphasise the flood-relevant upper tail).
        (1,1) corrected-GloFAS-vs-GRDC scatter (mirrors (0,0)), tail days
              highlighted, tail-focused and all-days Pearson r after
              correction.
        (1,2) full-record time series of raw vs corrected GloFAS, with GRDC
              overlaid on its observation window.

    Args:
        glofas_times, glofas_values: Full-record raw GloFAS cell series (the
            same series passed into ``bias_correct_discharge``).
        diagnostics: Non-``None`` dict returned by ``bias_correct_discharge``,
            with a ``grdc_station_id`` key added by the caller.
        output_path: PNG output path.
        label: Identifier for the figure title and log message.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g_ov = np.asarray(diagnostics["glofas_overlap"], dtype=float)
    r_ov = np.asarray(diagnostics["grdc_overlap"], dtype=float)
    c_ov = np.asarray(diagnostics["corrected_overlap"], dtype=float)
    t_ov = pd.to_datetime(diagnostics["glofas_overlap_times"])
    tail_mask = diagnostics["tail_mask"]
    tail_pct = diagnostics["tail_percentile"]
    r_raw = diagnostics["correlation_raw"]
    r_raw_full = diagnostics["correlation_full_raw"]
    r_corr = diagnostics["correlation_corrected"]
    r_corr_full = diagnostics["correlation_full_corrected"]
    n_overlap = diagnostics["grdc_overlap_days"]
    station_id = diagnostics.get("grdc_station_id", "?")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"GRDC bias correction — {label}  (station {station_id}, {n_overlap} d overlap)",
        fontsize=13,
        y=0.98,
    )

    lims = (0.0, float(max(g_ov.max(), r_ov.max(), c_ov.max())) * 1.05)

    _plot_raw_scatter(
        axes[0, 0], g_ov, r_ov, r_raw, r_raw_full, tail_mask, tail_pct, lims
    )
    _plot_seasonal_scatter(axes[0, 1], g_ov, r_ov, t_ov, lims)
    _plot_overlap_timeseries(axes[0, 2], t_ov, g_ov, r_ov, r_raw)

    # (1,0) empirical CDFs (log-x for upper-tail emphasis)
    ax = axes[1, 0]
    for data, lbl, color in (
        (g_ov, "GloFAS raw", "tab:blue"),
        (r_ov, "GRDC", "tab:orange"),
        (c_ov, "GloFAS corrected", "tab:green"),
    ):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=lbl, color=color, lw=1.5)
    ax.set_xlabel("Discharge (m³ s⁻¹)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Empirical CDFs (overlap period)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xscale("log")

    # (1,1) corrected scatter (mirrors (0,0))
    ax = axes[1, 1]
    ax.scatter(
        c_ov[~tail_mask],
        r_ov[~tail_mask],
        s=8,
        alpha=0.35,
        c="seagreen",
        label="Below tail",
    )
    ax.scatter(
        c_ov[tail_mask],
        r_ov[tail_mask],
        s=16,
        alpha=0.8,
        c="crimson",
        label=f"Tail (≥ p{tail_pct:g})",
    )
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("GloFAS corrected (m³ s⁻¹)")
    ax.set_ylabel("GRDC (m³ s⁻¹)")
    ax.set_title(
        f"Corrected GloFAS vs GRDC  (tail r = {r_corr:.3f}; all-days r = {r_corr_full:.3f})"
    )
    ax.legend(loc="upper left", fontsize=8)

    # (1,2) full-record time series: raw vs corrected, GRDC overlay
    ax = axes[1, 2]
    corrected_full = empirical_quantile_map(
        source_overlap=g_ov,
        target_overlap=r_ov,
        source_full=np.asarray(glofas_values, dtype=float),
    )
    t_full = pd.to_datetime(np.asarray(glofas_times))
    ax.plot(
        t_full, glofas_values, color="tab:blue", lw=0.5, alpha=0.6, label="GloFAS raw"
    )
    ax.plot(
        t_full,
        corrected_full,
        color="tab:green",
        lw=0.5,
        alpha=0.8,
        label="GloFAS corrected",
    )
    ax.plot(t_ov, r_ov, color="tab:orange", lw=0.8, label="GRDC (obs. window)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (m³ s⁻¹)")
    ax.set_title("Full record: raw vs corrected GloFAS")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    log.info(f"Wrote bias-correction diagnostic plot: {output_path}")


# ── GRDC station/correlation overview plot (standalone inspection) ────────────


def plot_grdc_overview(
    domain_poly: Polygon,
    osm_land_path: str,
    river_gdf: gpd.GeoDataFrame,
    crossings_gdf: gpd.GeoDataFrame,
    grdc_stations: gpd.GeoDataFrame,
    highlight_crossing_idx: int,
    highlight_station_id: int,
    diagnostics: dict,
    output_path: str | Path,
    label: str = "",
    water_bodies_path: str | None = None,
) -> None:
    """Render a 2x2 GRDC-vs-GloFAS correlation overview figure.

    Unlike ``plot_bias_correction``, this does not require the bias
    correction itself to have been applied — ``diagnostics`` is the dict
    returned by ``compute_grdc_correlation`` (raw overlap only), so this plot
    can be produced regardless of ``boundary_forcings.bias_correction.enabled``.

    Panels:
        (0,0) domain map: model domain outline, river network, all
              crossing/boundary-forcing points (active vs inactive), and
              GRDC stations in view, with the current crossing/station pair
              highlighted.
        (0,1) raw GloFAS-vs-GRDC scatter with 1:1 line, tail days (>= the
              configured GRDC tail percentile) highlighted, and both the
              tail-focused (headline) and all-days Pearson r in the title.
        (1,0) same scatter, points coloured by season (DJF/MAM/JJA/SON), with
              per-season r in the legend.
        (1,1) overlap-period time series of GloFAS raw and GRDC, on twin
              y-axes.

    Args:
        domain_poly: Model domain polygon (WGS84).
        osm_land_path: Path to the OSM land polygons used as a map background
            (see ``src.plots.map_background``).
        river_gdf: River network GeoDataFrame (WGS84).
        crossings_gdf: One row per river crossing, with a ``has_glofas`` bool
            column and point geometry (WGS84).
        grdc_stations: GRDC station table from ``river_forcing.load_grdc_stations``.
        highlight_crossing_idx: Row index into ``crossings_gdf`` for the
            crossing being inspected.
        highlight_station_id: GRDC station ``id`` matched to that crossing.
        diagnostics: Dict returned by ``compute_grdc_correlation``.
        output_path: PNG output path.
        label: Identifier for the figure title and log message.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g_ov = np.asarray(diagnostics["glofas_overlap"], dtype=float)
    r_ov = np.asarray(diagnostics["grdc_overlap"], dtype=float)
    t_ov = pd.to_datetime(diagnostics["glofas_overlap_times"])
    tail_mask = diagnostics["tail_mask"]
    tail_pct = diagnostics["tail_percentile"]
    r_raw = diagnostics["correlation_raw"]
    r_raw_full = diagnostics["correlation_full_raw"]
    n_overlap = diagnostics["grdc_overlap_days"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"GRDC-GloFAS correlation — {label}  (station {highlight_station_id}, {n_overlap} d overlap)",
        fontsize=13,
        y=0.98,
    )

    # (0,0) domain map
    ax = axes[0, 0]
    station_pt = grdc_stations.loc[
        grdc_stations["id"] == highlight_station_id, "geometry"
    ].iloc[0]
    crossing_pt = crossings_gdf.geometry.iloc[highlight_crossing_idx]

    map_background(
        ax,
        domain_poly,
        osm_land_path,
        margin_frac=0.15,
        water_bodies_path=water_bodies_path,
    )
    bx, by = domain_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=2, zorder=2, label="Model domain")

    # Widen the view if the matched GRDC station falls outside the domain bbox.
    xlim = (min(ax.get_xlim()[0], station_pt.x), max(ax.get_xlim()[1], station_pt.x))
    ylim = (min(ax.get_ylim()[0], station_pt.y), max(ax.get_ylim()[1], station_pt.y))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    map_bounds = (xlim[0], ylim[0], xlim[1], ylim[1])

    river_gdf.plot(ax=ax, color="steelblue", linewidth=0.8, zorder=1)

    if not crossings_gdf.empty:
        active = crossings_gdf["has_glofas"].astype(bool)
        if active.any():
            crossings_gdf[active].plot(
                ax=ax,
                color="limegreen",
                markersize=40,
                zorder=3,
                label="Crossing (active)",
            )
        if (~active).any():
            crossings_gdf[~active].plot(
                ax=ax,
                color="grey",
                markersize=40,
                marker="x",
                zorder=3,
                label="Crossing (inactive)",
            )

    grdc_in_view = grdc_stations.cx[
        map_bounds[0] : map_bounds[2], map_bounds[1] : map_bounds[3]
    ]
    if not grdc_in_view.empty:
        grdc_in_view.plot(
            ax=ax,
            color="purple",
            markersize=70,
            marker="*",
            zorder=3,
            alpha=0.7,
            label="GRDC station",
        )

    ax.scatter(
        [crossing_pt.x],
        [crossing_pt.y],
        s=180,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        zorder=4,
        label="This crossing",
    )
    ax.scatter(
        [station_pt.x],
        [station_pt.y],
        s=180,
        facecolors="none",
        edgecolors="orange",
        linewidths=2,
        zorder=4,
        label="Matched GRDC station",
    )

    ax.set_title("Domain context")
    ax.legend(loc="best", fontsize=7)

    lims = (0.0, float(max(g_ov.max(), r_ov.max())) * 1.05)
    _plot_raw_scatter(
        axes[0, 1], g_ov, r_ov, r_raw, r_raw_full, tail_mask, tail_pct, lims
    )
    _plot_seasonal_scatter(axes[1, 0], g_ov, r_ov, t_ov, lims)
    _plot_overlap_timeseries(axes[1, 1], t_ov, g_ov, r_ov, r_raw)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    log.info(f"Wrote GRDC overview plot: {output_path}")
