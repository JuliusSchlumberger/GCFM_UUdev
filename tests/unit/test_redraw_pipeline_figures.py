import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
import redraw_pipeline_figures as rpf  # noqa: E402


def test_crest_profile_redraw_uses_last_calibration_round(tmp_path, monkeypatch):
    profile_dir = tmp_path / "preprocessing_inputs" / "visuals" / "11_calibration"
    profile_dir.mkdir(parents=True)
    rows = [
        dict(
            round=r, along_path_m=x, dem_m=2.0, river_depth_m=3.0, weir_crest_m=4.0 + r
        )
        for r in (0, 1)
        for x in (0.0, 100.0)
    ]
    pd.DataFrame(rows).to_csv(profile_dir / "round_profiles_seed42.csv", index=False)
    drawn = {}
    monkeypatch.setattr(rpf, "plot_crest_profiles", lambda b, p, out: drawn.update(p))

    rpf.crest_profile_job(tmp_path, "1", tmp_path / "figures")["30 crest profile"]()

    assert list(drawn) == ["42"]
    assert set(drawn["42"]["round"]) == {1}
