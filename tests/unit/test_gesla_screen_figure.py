import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
import inspect_gesla_gauges as gesla  # noqa: E402


def _row(name: str, status: str, reason: str, steps: str) -> dict:
    return {
        "file_name": name,
        "site_name": name.upper(),
        "contributor": "X",
        "status": status,
        "reason": reason,
        "steps": steps,
        "n_years_used": 12,
        "lag_min": 3.0,
        "corr": 0.9,
        "rmse_m": 0.05,
        "trend_mm_per_yr": 1.0,
    }


def test_screen_figure_counts_usable_gauges_and_clipped_months(tmp_path):
    months = pd.date_range("2000-01-01", periods=120, freq="MS")
    md = pd.Series(0.05 * np.sin(np.arange(120)), index=months)
    md.iloc[10] = 0.9  # beyond the fixed axis
    screen = pd.DataFrame(
        [
            _row("a", "usable", "", ""),
            _row("b", "excluded", "timing lag 60 min", "2005-01 +8+-1 cm"),
        ]
    )
    monthly = {"a": (md, []), "b": (md * 0.5, [{"date": months[60]}])}
    out = tmp_path / "screen.png"

    gesla.plot_basin("0", screen, monthly, out)

    assert out.exists() and out.with_suffix(".pdf").exists()
    caption = out.with_suffix(".txt").read_text(encoding="utf-8")
    assert caption.startswith("1 of 2 open-coast GESLA gauge(s)")
    assert "2 month(s) beyond it" in caption  # the spike in both gauges
    assert (
        "(a) A (X): usable" in caption
        and "(b) B (X): excluded (timing lag 60 min)" in caption
    )
