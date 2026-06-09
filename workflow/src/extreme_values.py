"""Extreme-value analysis for GloFAS river discharge.

Provides per-cell estimation of return-period discharges using pyextremes:

* RP=bankfull (default 2 yr) via Block Maxima / GEV on annual maxima.
* RP=flood (default 100 yr) via Peaks-Over-Threshold / GPD with
  autocorrelation-based declustering and iterative threshold search.
* Mann-Kendall + Sen's-slope trend test on the annual-maxima series.

All EVA parameters are read from a plain dict (``eva_cfg``) matching the
``boundary_forcings.eva`` section of the project config YAML, including the
newer optional knobs ``min_years_high_confidence`` (default 40.0),
``peaks_per_year_min`` (default 3.0), ``threshold_min_pct`` (default 50.0),
and ``deseasonalize_for_decorr``
(default True) — see ``analyse_cell`` / ``_search_threshold`` for how each is
used and what it defaults to when absent from the YAML.

ToDo (bias correction) — NOT implemented
-----------------------------------------
GloFAS is currently treated as truth, and reported return levels carry
unquantified structural uncertainty from the GloFAS reanalysis itself. Where
an observed gauge exists on the main stem near the delta apex (e.g. GRDC —
Global Runoff Data Centre), apply empirical quantile mapping over the
overlapping period: build the empirical CDF of GloFAS and of the gauge, and
remap GloFAS values so their distribution matches the gauge's — with
particular attention to the upper tail (flood-relevant quantiles), not just
the bulk distribution. Where no gauge exists, "GloFAS-as-truth" must remain
documented as an explicit limitation of the analysis.

Public API
----------
EVAResult              – dataclass holding all outputs for one grid-cell run.
analyse_cell(times, values, eva_cfg, label) -> EVAResult
plot_cell_diagnostics(times, values, eva_cfg, output_path, label) -> EVAResult
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

from pyextremes import EVA
import pymannkendall as mk

import numpy as np
import pandas as pd
from scipy import stats

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

    # POT diagnostics
    pot_threshold: float = np.nan
    pot_r_days: float = np.nan
    pot_decorr_days: float = np.nan
    pot_peaks_per_year: float = np.nan
    pot_n_peaks: int = 0
    pot_threshold_status: str = "none"
    pot_shape: float = np.nan
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
    (last_eva, last_tuple) flagged 'reject' if no iteration met the criterion.
    """
    thr_start = float(_c(cfg, "threshold_start_pct", 90.0))
    thr_step = float(_c(cfg, "threshold_step_pct", 1.0))
    thr_max = float(_c(cfg, "threshold_max_pct", 99.5))
    thr_min = float(_c(cfg, "threshold_min_pct", 50.0))
    max_iter = int(_c(cfg, "threshold_max_iter", 30))
    ppy_min = float(_c(cfg, "peaks_per_year_min", 3.0))
    ppy_max = float(_c(cfg, "peaks_per_year_max", 5.0))
    _fit_method = str(_c(cfg, "fit_method", "Lmoments"))  # reserved for future use

    pct = float(np.clip(thr_start, thr_min, thr_max))
    last_eva = last_choice = None

    for it in range(1, max_iter + 1):
        thr = float(np.percentile(s.values, pct))
        eva = EVA(data=s)
        eva.get_extremes("POT", threshold=thr, r=f"{r_days}D")
        ppy = len(eva.extremes) / record_years

        status = "good" if ppy_min <= ppy <= ppy_max else "reject"

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
) -> EVAResult:
    """Run the full EVA pipeline on one cell's daily discharge series.

    Args:
        times:   Daily time coordinate array (numpy datetime64 or numeric).
        values:  Discharge values aligned with ``times`` (may contain NaN).
        eva_cfg: Plain dict from config["boundary_forcings"]["eva"].
        label:   Optional identifier used in log messages.

    Returns:
        EVAResult. On failure of either model, NaN fields remain and ``ok``
        is False — a bad cell never aborts a whole domain.
    """
    # NOTE — GloFAS is treated as truth in this function; no gauge-based bias
    # correction is applied. See the module docstring's "ToDo (bias
    # correction)" block for the planned empirical-quantile-mapping approach.
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
            gpd_params = eva_pot.model.fit_parameters
            res.pot_shape = float(gpd_params.get("c", np.nan))

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
