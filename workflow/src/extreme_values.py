"""Estimate GloFAS return-period discharges per cell and bias-correct against GRDC.

Bankfull RP (bankfull_rp_yr, 1.5 yr): Block Maxima / GEV on annual maxima. Flood RP
(preview_rp_yr, 100 yr): Peaks-Over-Threshold / GPD with autocorrelation-based
declustering and iterative threshold search. Trend: Mann-Kendall + Sen's
slope on annual maxima. Parameters come from config boundary_forcings.river.eva.

GRDC bias correction: where a gauge overlaps GloFAS long enough
(min_overlap_days), empirical quantile mapping remaps the full GloFAS record
to the gauge distribution; values beyond the overlap's range are linearly
extrapolated, not clamped, so flood peaks are not flattened. Without a gauge
or overlap, GloFAS is taken as truth (logged by the caller, not an error).
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
from pyextremes import EVA
from scipy import stats
from src.io import require_keys
from src.plots import save_figure

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="lmoments3")
# Empty annual blocks are expected with errors='coerce' in get_extremes("BM").
warnings.filterwarnings("ignore", message=".*blocks contained no data.*")

# Windows "spawn" workers (pyextremes/lmoments3 bootstrap pools) re-import
# __main__, which fails under Snakemake ('snakemake' exists only in the
# parent). Hence single-threaded BLAS/OpenMP and our own sequential bootstrap.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


@dataclass
class EVAResult:
    """Outcome of extreme-value analysis for a single GloFAS cell."""

    q_rp2: float = np.nan
    q_rp100: float = np.nan
    q_rp2_ci: tuple[float, float] = (np.nan, np.nan)
    q_rp100_ci: tuple[float, float] = (np.nan, np.nan)

    # POT/GPD discharge at analyse_cell's protection_rp_yr; NaN unless requested.
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
    # Set when the record is unusable by design (too short); the cell is skipped.
    skip_reason: str = ""
    messages: list[str] = field(default_factory=list)


class EVAFitError(Exception):
    """An EVA fit failed for a reason other than a documented skip."""


# Numeric failures of a GEV/GPD fit; anything else (ImportError, ...) propagates.
_FIT_ERRORS = (ValueError, RuntimeError, ArithmeticError)


_BIAS_KEYS = ("enabled", "min_overlap_days", "tail_percentile")
_THRESHOLD_KEYS = (
    "threshold_start_pct",
    "threshold_step_pct",
    "threshold_max_pct",
    "threshold_min_pct",
    "threshold_max_iter",
    "peaks_per_year_min",
    "peaks_per_year_max",
)
_EVA_KEYS = (
    "min_days",
    "n_boot",
    "min_years_high_confidence",
    "fit_method",
    "bankfull_rp_yr",
    "preview_rp_yr",
    "ci_alpha",
    "decorr_cut",
    "decorr_max",
    "deseasonalize_for_decorr",
) + _THRESHOLD_KEYS


def _to_series(times: np.ndarray, values: np.ndarray) -> pd.Series:
    s = pd.Series(
        np.asarray(values, dtype=float),
        index=pd.to_datetime(np.asarray(times)),
    )
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s.dropna()


def _overlap_series(
    glofas_times: np.ndarray,
    glofas_values: np.ndarray,
    grdc_times: np.ndarray,
    grdc_values: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
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
    """Bias-correct source_full by empirical quantile mapping on the overlap sample.

    Quantiles at n_quantiles evenly spaced levels. Values outside the overlap
    range are linearly extrapolated with the outermost quantile-pair slope
    (np.interp would clamp and flatten the flood tail); a zero-width outer
    segment falls back to an additive offset. Floored at zero.
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
    """Compute tail-focused GloFAS-vs-GRDC correlation on the overlap; None if too short.

    Called by bias_correct_discharge, so only when the correction is enabled.
    correlation_raw uses only days with GRDC >= its tail_percentile (0-100) of
    the overlap, since an all-days Pearson r is dominated by low flows;
    correlation_full_raw is all days. GRDC series must have -999 already
    mapped to NaN.
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
    """Bias-correct a GloFAS cell series against a GRDC station; return (values, diagnostics).

    cfg: boundary_forcings.river.bias_correction (enabled, min_overlap_days,
    tail_percentile). Skipped (input returned, diagnostics None) when disabled
    or the overlap is below min_overlap_days.
    """
    require_keys(cfg, _BIAS_KEYS, "boundary_forcings.river.bias_correction")
    glofas_values = np.asarray(glofas_values, dtype=float)

    if not bool(cfg["enabled"]):
        return glofas_values, None

    min_overlap_days = int(cfg["min_overlap_days"])
    tail_percentile = float(cfg["tail_percentile"])
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

    # The mapping is pointwise, so the overlap is corrected directly rather than
    # re-indexed out of the full series (which may hold NaN or duplicate days).
    corrected_overlap = empirical_quantile_map(
        source_overlap=diagnostics["glofas_overlap"],
        target_overlap=diagnostics["grdc_overlap"],
        source_full=diagnostics["glofas_overlap"],
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
    # Used only to estimate the declustering radius r: seasonal persistence
    # would otherwise inflate the autocorrelation. POT and fits use the raw series.
    climatology = s.groupby(s.index.dayofyear).transform("mean")
    return s - climatology


def estimate_decorrelation_days(series: pd.Series, cutoff: float, max_lag: int) -> int:
    """Return the lag (days) at which autocorrelation first drops below cutoff."""
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


def _rv(eva_instance, rp_yr: float) -> float:
    # pyextremes returns a float or (value, lower, upper) depending on version.
    result = eva_instance.get_return_value(rp_yr)
    if isinstance(result, (tuple, list, np.ndarray)) and len(result) >= 1:
        return float(result[0])
    return float(result)


# Return periods of river_forcing.nc's discharge_rp_table, interpolated at
# any requested RP (river_forcing.build_design_discharge_matrix); fixed, not config.
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
    rps_yr: np.ndarray = STANDARD_RETURN_PERIODS_YR,
) -> np.ndarray:
    """Return discharge (m³/s) at each return period (yr) from an already-fitted GPD.

    shape is scipy's c; peaks_per_year is the declustered rate at threshold.
    All-NaN if any parameter is not finite.
    """
    rps_yr = np.asarray(rps_yr, dtype=float)
    inputs = (threshold, scale, shape, peaks_per_year)
    if not all(np.isfinite(v) for v in inputs):
        return np.full(rps_yr.shape, np.nan)
    p = np.clip(1.0 - 1.0 / (peaks_per_year * rps_yr), 0.0, None)
    return threshold + stats.genpareto.ppf(p, shape, scale=scale)


# Sequential bootstrap CIs: no worker processes (safe on Windows/Snakemake).


def _gev_boot_ci(
    amax: np.ndarray,
    rp_yr: float,
    alpha: float,
    n_boot: int,
    c0: float | None = None,
    loc0: float | None = None,
    scale0: float | None = None,
) -> tuple[float, float]:
    # c0/loc0/scale0 (point-estimate fit) warm-start each MLE refit; the fit
    # was profiled as analyse_cell's dominant cost. Known inconsistency: the
    # point estimate uses fit_method (L-moments, more robust for short records)
    # while this CI uses scipy MLE refits; accepted deliberately.
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
            rl = float(stats.genextreme.ppf(1.0 - 1.0 / rp_yr, c, loc=loc, scale=scale))
            if np.isfinite(rl) and rl > 0:
                boot.append(rl)
        except _FIT_ERRORS:
            continue
    if len(boot) < 20:
        return np.nan, np.nan
    lo_p = (1.0 - alpha) / 2.0 * 100.0
    hi_p = (1.0 + alpha) / 2.0 * 100.0
    return float(np.percentile(boot, lo_p)), float(np.percentile(boot, hi_p))


def _gpd_boot_ci(
    exceedances: np.ndarray,
    threshold: float,
    ppy: float,
    rp_yr: float,
    alpha: float,
    n_boot: int,
    c0: float | None = None,
    scale0: float | None = None,
) -> tuple[float, float]:
    # Warm start and L-moments-vs-MLE caveat as in _gev_boot_ci.
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
            p = max(0.0, 1.0 - 1.0 / (ppy * rp_yr))
            rl = float(threshold + stats.genpareto.ppf(p, c, scale=scale))
            if np.isfinite(rl) and rl > 0:
                boot.append(rl)
        except _FIT_ERRORS:
            continue
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
    # Bidirectional search from threshold_start_pct: raise the percentile while
    # ppy > ppy_max, lower it (down to threshold_min_pct) while ppy < ppy_min.
    # Raises EVAFitError if the band is never reached ('reject: ppy<min', too
    # few peaks, typical of slow large rivers, or 'reject: ppy>max').
    thr_start = float(cfg["threshold_start_pct"])
    thr_step = float(cfg["threshold_step_pct"])
    thr_max = float(cfg["threshold_max_pct"])
    thr_min = float(cfg["threshold_min_pct"])
    max_iter = int(cfg["threshold_max_iter"])
    ppy_min = float(cfg["peaks_per_year_min"])
    ppy_max = float(cfg["peaks_per_year_max"])

    pct = float(np.clip(thr_start, thr_min, thr_max))
    last_choice = None

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
        last_choice = (thr, ppy, status)

        if status == "good":
            return eva, last_choice

        if ppy > ppy_max:
            next_pct = pct + thr_step
            if next_pct > thr_max:
                break
        else:  # ppy < ppy_min
            next_pct = pct - thr_step
            if next_pct < thr_min:
                break
        pct = next_pct

    raise EVAFitError(
        f"{tag}POT threshold search ended without reaching the "
        f"[{ppy_min:.1f}, {ppy_max:.1f}] peaks/yr band "
        f"(last: pct={pct:.1f} peaks/yr={last_choice[1]:.2f} → {last_choice[2]}); "
        f"adjust threshold_min_pct/threshold_max_pct or peaks_per_year_min/max"
    )


def analyse_cell(
    times: np.ndarray,
    values: np.ndarray,
    eva_cfg: dict,
    label: str = "",
    protection_rp_yr: float | None = None,
) -> EVAResult:
    """Run the full EVA on one cell's daily discharge series (may contain NaN).

    eva_cfg: config["boundary_forcings"]["river"]["eva"]. values are already
    bias-corrected by the caller where a GRDC match exists. protection_rp_yr
    (yr): also evaluate the fitted POT/GPD there into q_protection (no pipeline
    rule passes it). A record that is too short sets skip_reason; a numeric
    fit failure leaves NaN fields and ok False (eva_active_flags decides).
    Raises EVAFitError if the POT threshold search does not reach the band.
    """
    require_keys(eva_cfg, _EVA_KEYS, "boundary_forcings.river.eva")
    res = EVAResult()
    s = _to_series(times, values)
    res.n_valid_days = int(s.size)
    tag = f"[{label}] " if label else ""

    # ≈25 yr hard floor: RP100 extrapolates far beyond a short record; GloFAS v4
    # reanalysis runs from 1979 (~45 yr at most locations).
    min_days = int(eva_cfg["min_days"])
    # Below this record length (yr) results are flagged reduced-confidence, not skipped.
    min_years_hc = float(eva_cfg["min_years_high_confidence"])
    fit_method = str(eva_cfg["fit_method"])
    bankfull_rp_yr = float(eva_cfg["bankfull_rp_yr"])
    preview_rp_yr = float(eva_cfg["preview_rp_yr"])
    ci_alpha = float(eva_cfg["ci_alpha"])
    n_boot = int(eva_cfg["n_boot"])
    decorr_cut = float(eva_cfg["decorr_cut"])
    decorr_max = int(eva_cfg["decorr_max"])
    deseasonalize = bool(eva_cfg["deseasonalize_for_decorr"])

    if s.size < min_days:
        msg = (
            f"{tag}only {s.size} valid days "
            f"(< {min_days} ≈ {min_days / 365.2425:.0f} yr); skipping EVA"
        )
        log.warning(msg)
        res.messages.append(msg)
        res.skip_reason = "record too short"
        return res

    record_years = (s.index[-1] - s.index[0]).days / 365.2425
    res.record_years = record_years
    if record_years <= 0:
        res.messages.append(f"{tag}degenerate time span; skipping")
        res.skip_reason = "degenerate time span"
        return res

    if record_years < min_years_hc:
        msg = (
            f"{tag}record length {record_years:.1f} yr < "
            f"{min_years_hc:.0f} yr — return-level estimates carry "
            f"reduced confidence (not skipped, just flagged)"
        )
        log.info(msg)
        res.messages.append(msg)

    # AMAX/GEV for bankfull (bankfull_rp_yr), POT/GPD for flood (preview_rp_yr). The bankfull
    # ≈ 1.5-2 yr convention (Leopold & Wolman) is defined on annual maxima;
    # RP100 is better constrained by all over-threshold peaks. The two come
    # from different estimators, so RP100 < RP2 is not a valid sanity check.
    try:
        eva_bm = EVA(data=s)
        # errors='coerce': skip empty blocks (years with gaps in GloFAS data)
        eva_bm.get_extremes("BM", block_size="365.2425D", errors="coerce")
        res.amax_n_blocks = int(len(eva_bm.extremes))
        eva_bm.fit_model(fit_method)
        # No alpha: it makes pyextremes spawn worker processes.
        res.q_rp2 = _rv(eva_bm, bankfull_rp_yr)
        gev_params = eva_bm.model.fit_parameters
        res.gev_shape = float(gev_params.get("c", np.nan))

        # Advisory: |c| > 0.5 is implausible for river discharge and usually
        # signals a poorly conditioned fit.
        if np.isfinite(res.gev_shape) and abs(res.gev_shape) > 0.5:
            res.messages.append(
                f"{tag}GEV shape c={res.gev_shape:.3f} outside plausible "
                f"range (|c| > 0.5) — bankfull fit may be poorly constrained"
            )

        amax = eva_bm.extremes.values.astype(float)
        # CI reflects AMAX sampling variability only, conditional on the block definition.
        res.q_rp2_ci = _gev_boot_ci(
            amax,
            bankfull_rp_yr,
            ci_alpha,
            n_boot,
            c0=gev_params.get("c"),
            loc0=gev_params.get("loc"),
            scale0=gev_params.get("scale"),
        )
        if not np.all(np.isfinite(res.q_rp2_ci)):
            res.messages.append(
                f"{tag}GEV bootstrap CI undefined (too few valid refits)"
            )

        # Mann-Kendall on AMAX without FDR correction: 1-6 locations per delta,
        # and annual maxima are near-independent as MK assumes. A trend biases
        # the stationary fit; the flag informs a human decision, not an
        # automatic correction.
        if len(amax) >= 4:
            mkr = mk.original_test(amax)
            res.trend = str(mkr.trend)
            res.trend_pvalue = float(mkr.p)
            res.sen_slope = float(mkr.slope)
        else:
            res.messages.append(f"{tag}too few AMAX blocks for trend test")
    except _FIT_ERRORS as e:
        msg = f"{tag}AMAX/GEV fit failed: {e}"
        log.warning(msg)
        res.messages.append(msg)

    try:
        # Declustering window is data-driven per location: flood-wave duration
        # scales with catchment size and regime. Optionally estimated on the
        # deseasonalized series, whose 1/e lag is not inflated by the seasonal
        # cycle; POT and fitting still use the original series.
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
            res.q_rp100 = _rv(eva_pot, preview_rp_yr)
            gpd_params = eva_pot.model.fit_parameters
            # pyextremes picks "genpareto" or "expon" by AIC; "expon" is the GPD
            # with c = 0, so a missing "c" means shape 0, not a failed fit.
            res.pot_shape = float(gpd_params.get("c", 0.0))
            # Kept for gpd_return_value_table (any RP without refitting).
            res.pot_scale = float(gpd_params.get("scale", np.nan))
            if protection_rp_yr is not None:
                # Same formula as discharge_rp_table, not pyextremes' own.
                res.q_protection = float(
                    gpd_return_value_table(
                        res.pot_threshold,
                        res.pot_scale,
                        res.pot_shape,
                        res.pot_peaks_per_year,
                        np.array([protection_rp_yr], dtype=float),
                    )[0]
                )

            # Advisory: |c| > 0.5 is implausible for river-discharge tails.
            if np.isfinite(res.pot_shape) and abs(res.pot_shape) > 0.5:
                res.messages.append(
                    f"{tag}GPD shape c={res.pot_shape:.3f} outside plausible "
                    f"range (|c| > 0.5) — flood fit may be poorly constrained"
                )

            # POT .extremes are absolute peaks; a floc=0 GPD needs exceedances.
            exceedances = eva_pot.extremes.values.astype(float) - res.pot_threshold
            neg_mask = exceedances < 0
            if neg_mask.any():
                res.messages.append(
                    f"{tag}{int(neg_mask.sum())} POT peak(s) below threshold "
                    f"(negative exceedance) — filtered before GPD bootstrap"
                )
                exceedances = exceedances[~neg_mask]

            # CI is conditional on the chosen threshold and r_days (not propagated).
            res.q_rp100_ci = _gpd_boot_ci(
                exceedances,
                res.pot_threshold,
                ppy,
                preview_rp_yr,
                ci_alpha,
                n_boot,
                c0=gpd_params.get("c"),
                scale0=gpd_params.get("scale"),
            )
            if not np.all(np.isfinite(res.q_rp100_ci)):
                res.messages.append(
                    f"{tag}GPD bootstrap CI undefined (too few valid refits)"
                )
            try:
                res.pot_ks_pvalue = float(eva_pot.test_ks().pvalue)
                if res.pot_ks_pvalue < 0.05:
                    res.messages.append(
                        f"{tag}KS test p={res.pot_ks_pvalue:.3f} < 0.05 — "
                        f"GPD fit does not pass goodness-of-fit at α=0.05"
                    )
            except _FIT_ERRORS as e:
                res.messages.append(f"{tag}KS goodness-of-fit test failed: {e}")
    except _FIT_ERRORS as e:
        msg = f"{tag}POT/GPD fit failed: {e}"
        log.warning(msg)
        res.messages.append(msg)

    # ok = both estimators finite; advisory messages do not affect it.
    res.ok = np.isfinite(res.q_rp2) and np.isfinite(res.q_rp100)
    return res


def eva_active_flags(evas: list[tuple[str, EVAResult]]) -> list[bool]:
    """Return per-crossing active flags for (label, EVAResult) pairs.

    A documented skip (skip_reason, e.g. record too short) leaves the crossing
    inactive with a warning. Raises EVAFitError naming every crossing whose
    fit failed otherwise, or if EVAs were run and none is usable.
    """
    failed = [(label, r) for label, r in evas if not r.ok and not r.skip_reason]
    if failed:
        details = "\n".join(
            f"  {label}: q_rp2={r.q_rp2:.4g} q_rp100={r.q_rp100:.4g}; "
            + ("; ".join(r.messages) or "no message")
            for label, r in failed
        )
        raise EVAFitError(f"EVA failed for {len(failed)} crossing(s):\n{details}")
    for label, r in evas:
        if not r.ok:
            log.warning(f"  {label}: EVA skipped ({r.skip_reason}) — crossing inactive")
    flags = [bool(r.ok) for _, r in evas]
    if evas and not any(flags):
        raise EVAFitError(
            f"None of {len(evas)} crossing(s) with a GloFAS cell has a usable EVA: "
            + ", ".join(f"{label} ({r.skip_reason})" for label, r in evas)
        )
    return flags


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

    fit_method = str(eva_cfg["fit_method"])
    bankfull_rp_yr = float(eva_cfg["bankfull_rp_yr"])
    preview_rp_yr = float(eva_cfg["preview_rp_yr"])

    res = analyse_cell(times, values, eva_cfg, label=label)
    s = _to_series(times, values)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"GloFAS EVA diagnostics — {label}", fontsize=13, y=0.98)

    # POT (RP=preview_rp_yr)
    try:
        if np.isfinite(res.pot_threshold):
            eva_pot = EVA(data=s)
            eva_pot.get_extremes(
                "POT", threshold=res.pot_threshold, r=f"{int(res.pot_r_days)}D"
            )
            eva_pot.fit_model(fit_method)
            eva_pot.plot_extremes(ax=axes[0, 0], show_clusters=True)
            axes[0, 0].set_title(f"POT extremes & clusters (r={int(res.pot_r_days)}d)")
            eva_pot.plot_return_values(
                return_period=np.logspace(np.log10(1.1), np.log10(500), 100),
                ax=axes[0, 1],
            )
            axes[0, 1].set_title(
                f"POT/GPD  thr={res.pot_threshold:.1f}  r={int(res.pot_r_days)}d  "
                f"{res.pot_peaks_per_year:.1f} pk/yr  [{res.pot_threshold_status}]"
            )
            axes[0, 1].axvline(preview_rp_yr, ls=":", c="k", lw=1)
    except _FIT_ERRORS as e:
        axes[0, 0].text(0.5, 0.5, f"POT plot failed:\n{e}", ha="center", va="center")

    # AMAX (RP=bankfull_rp_yr)
    try:
        if res.amax_n_blocks:
            eva_bm = EVA(data=s)
            eva_bm.get_extremes("BM", block_size="365.2425D", errors="coerce")
            eva_bm.fit_model(fit_method)
            eva_bm.plot_return_values(
                return_period=np.logspace(np.log10(1.01), np.log10(100), 100),
                ax=axes[1, 0],
            )
            axes[1, 0].set_title(f"AMAX/GEV  ({res.amax_n_blocks} blocks)")
            axes[1, 0].axvline(bankfull_rp_yr, ls=":", c="k", lw=1)
            _plot_decade_trend(axes[1, 1], eva_bm.extremes, res)
    except _FIT_ERRORS as e:
        axes[1, 0].text(0.5, 0.5, f"AMAX plot failed:\n{e}", ha="center", va="center")

    # pyextremes' plot_extremes() calls fig.autofmt_xdate(), which assumes a
    # shared date axis and hides top-row tick labels; these axes are independent.
    axes[0, 0].set_xlabel("date-time")
    for row_ax in (axes[0, 0], axes[0, 1]):
        row_ax.xaxis.set_tick_params(labelbottom=True)
        for lbl in row_ax.get_xticklabels():
            lbl.set_visible(True)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path, dpi=130)
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


def plot_bias_correction(
    glofas_times: np.ndarray,
    glofas_values: np.ndarray,
    diagnostics: dict,
    output_path: str | Path,
    label: str = "",
) -> None:
    """Render the 2x3 GRDC bias-correction diagnostic figure.

    diagnostics: non-None result of bias_correct_discharge, plus a
    grdc_station_id key added by the caller.
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

    # Log-x to emphasise the flood-relevant upper tail.
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
    save_figure(fig, output_path, dpi=130)
    log.info(f"Wrote bias-correction diagnostic plot: {output_path}")
