import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
import surface_options  # noqa: E402


def test_profile_figure_redraws_from_saved_data_and_clips_bank_outliers(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        surface_options,
        "analysis_path",
        lambda _b, _s, q: tmp_path / f"90_surface_options_{q}",
    )
    km = np.linspace(60.0, 0.0, 30)
    table = pd.DataFrame(
        {
            "reach_id": np.repeat([1, 2], 15),
            "dist_to_sea_km": km,
            "swot_level_m": np.where(np.arange(30) % 3 == 0, km / 6, np.nan),
            "bank_top_m": km / 6 + 0.5,
        }
    )
    table.loc[3, "bank_top_m"] = 70.0  # outlier above the surfaces and beds
    for s, _ in surface_options.PROFILE_OPTIONS:
        table[f"{s}_surface_m"] = km / 6
        table[f"{s}_bed_m"] = km / 6 - 5
    table.to_csv(surface_options.profile_data_path("0"), index=False)

    monkeypatch.setattr(sys, "argv", ["surface_options.py", "0", "--redraw"])
    assert surface_options.main() == 0

    out = tmp_path / "90_surface_options_profile.png"
    assert out.exists() and out.with_suffix(".pdf").exists()
    caption = out.with_suffix(".txt").read_text(encoding="utf-8")
    assert "1 bank-top point(s) above the axis" in caption
