"""
grid_resolution.py -- Compute a per-basin SFINCS main-grid resolution
(rule 08b, optimize_grid_resolution), instead of using a single fixed value
for every basin regardless of its river network or domain size.
"""

from __future__ import annotations

import math

import numpy as np


def compute_optimal_resolution(
    preferred_target_m: float,
    domain_area_m2: float,
    max_active_cells: float,
) -> dict:
    """
    Resolution = max(preferred_target_m, sqrt(domain_area_m2 / max_active_cells)).

    ``preferred_target_m`` is caller-supplied: ``width_p20 / target_cells_per_width``
    when width-based optimization is enabled, or a flat default resolution
    when it's disabled -- either way, the cell-budget cap
    (``sqrt(domain_area_m2 / max_active_cells)``, the coarsest resolution
    keeping the analytic active-cell estimate at or under the budget)
    applies here unconditionally, as a safety net against an absurdly fine
    resolution blowing up cell count/runtime on a large domain.

    Args:
        preferred_target_m: The resolution (m) preferred before considering
            the cell budget.
        domain_area_m2:     Delta domain polygon area (m^2), in its own
            projected (UTM) CRS.
        max_active_cells:   Cap on the analytic active-cell estimate
            (domain_area_m2 / resolution^2).

    Returns:
        Dict with the final resolution plus both candidates and whether the
        cap bound (rather than preferred_target_m) determined the result,
        for logging/diagnostics.
    """
    cell_budget_cap = math.sqrt(domain_area_m2 / max_active_cells)
    resolution = max(preferred_target_m, cell_budget_cap)
    return {
        "resolution": resolution,
        "preferred_target_m": preferred_target_m,
        "cell_budget_cap": cell_budget_cap,
        "estimated_active_cells": domain_area_m2 / resolution**2,
        "capped": cell_budget_cap > preferred_target_m,
    }


def compute_width_p20_target(
    widths: np.ndarray, target_cells_per_width: float
) -> float:
    """
    Width-based preferred resolution target: the 20th-percentile width
    (plain, unweighted quantile) divided by ``target_cells_per_width`` --
    i.e. resolve at least 80% of the given reaches to
    ``target_cells_per_width`` cells across, accepting that the narrowest
    ~20% may be under-resolved. A percentile (not the minimum) avoids one
    anomalously narrow or noisy reach forcing an unnecessarily fine grid
    across the whole domain.

    Args:
        widths: Channel widths (m) of the reaches to consider (already
            filtered by the caller, e.g. to reaches above the discharge
            threshold).
        target_cells_per_width: Desired number of grid cells spanning the
            20th-percentile-width channel.

    Returns:
        Preferred resolution target (m).
    """
    width_p20 = float(np.quantile(widths, 0.20))
    return width_p20 / target_cells_per_width
