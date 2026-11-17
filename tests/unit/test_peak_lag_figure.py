import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
import peak_lag  # noqa: E402


def test_peak_lag_figure_shows_both_mouth_peaks_and_states_the_lag(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        peak_lag, "analysis_path", lambda _b, _s, q: tmp_path / f"90_peak_lag_{q}"
    )
    t = np.arange(0.0, 200.0, 1.0)
    row = {
        "basin_id": "0",
        "mouth": "mouth_1",
        "coast_run": "coast",
        "river_run": "river",
        "t_surge_boundary_hr": 100.0,
        "t_river_inflow_hr": 80.0,
        "surge_travel_hr": 1.0,
        "river_travel_hr": 12.0,
        "lag_hr": -11.0,
        "zs_surge_mouth_max_m": 1.2,
        "q_river_mouth_peak_m3s": 900.0,
    }
    q = 500 + 400 * np.exp(-(((t - 92) / 20) ** 2))

    peak_lag.plot_mouth(row, t, np.sin(t / 2), t, q, peak_lag.godin(q, 1.0))

    out = tmp_path / "90_peak_lag_coast_river_mouth_1.png"
    assert out.exists() and out.with_suffix(".pdf").exists()
    caption = out.with_suffix(".txt").read_text(encoding="utf-8")
    assert caption.startswith("At mouth mouth_1") and "-11.0 h" in caption
