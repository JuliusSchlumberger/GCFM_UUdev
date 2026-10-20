"""Smoke test for tools/make_data_table.py on a synthetic catalogue and strategies."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

TOOL = Path(__file__).resolve().parents[2] / "tools" / "make_data_table.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("make_data_table", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CATALOGUE = {
    "meta": {"version": "1"},
    "datasets": [
        {
            "name": "zeta_grid",
            "description": "A synthetic grid. Second sentence is dropped.",
            "source": "Doe (2020) | with a pipe",
            "file_path": "zeta/grid.tif",
            "version": "v2",
        },
        {
            "name": "fathomdem",
            "description": "Tiles.",
            "source": "Uhe et al. (2025)",
            "file_path": "FathomDEM/",
        },
        {
            "name": "alpha_points",
            "description": "Points.",
            "source": "Built with tools/some_builder.py",
            "file_path": "alpha/points.gpkg",
        },
    ],
}


STRATEGIES = {
    "scales": [0.4, 1.0],
    "beta": {
        "measures": {
            "barrier": {"elevation": {"coastal": 1.0}, "locations": "site/b.geojson"},
            "pumps": {"discharge": 5, "locations": "site/auto/pump.geojson"},
        }
    },
    "alpha": {
        "measures": {
            "levee": {"locations": "site/auto/pump.geojson", "merge": False},
            "raising": {"raise_fraction": 1.0},
        }
    },
}


def test_table_has_one_sorted_row_per_dataset_and_marks_missing_fields():
    table = _load_tool().render_table(CATALOGUE).splitlines()
    rows = table[2:]
    assert len(rows) == 3
    assert [r.split("|")[1].strip() for r in rows] == [
        "`alpha_points`",
        "`fathomdem`",
        "`zeta_grid`",
    ]
    assert "`tools/some_builder.py`" in rows[0]
    assert "`tools/download_fathomdem.py`" in rows[1]
    assert "v2" in rows[2] and "Second sentence" not in rows[2]
    assert "Doe (2020) \\| with a pipe" in rows[2]
    assert all("—" in r for r in rows)  # no licence recorded


def test_geometry_table_lists_each_file_once_with_users_and_origin():
    table = _load_tool().render_geometry_table(STRATEGIES).splitlines()
    rows = table[2:]
    assert len(rows) == 2
    auto_row, drawn_row = rows
    assert auto_row.startswith(
        "| `site/auto/pump.geojson` | `alpha/levee`, `beta/pumps` |"
    )
    assert "`tools/build_adaptation_lines.py`" in auto_row
    assert drawn_row.startswith("| `site/b.geojson` | `beta/barrier` |")
    assert "project-specific" in drawn_row


def test_cli_writes_page_and_help_writes_nothing(tmp_path):
    catalogue = tmp_path / "catalogue.yml"
    catalogue.write_text(yaml.safe_dump(CATALOGUE), encoding="utf-8")
    strategies = tmp_path / "strategies.yml"
    strategies.write_text(yaml.safe_dump(STRATEGIES), encoding="utf-8")
    output = tmp_path / "data.md"
    subprocess.run(
        [sys.executable, str(TOOL), "--help"], check=True, capture_output=True
    )
    assert not output.exists()
    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--catalogue",
            str(catalogue),
            "--strategies",
            str(strategies),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    page = output.read_text(encoding="utf-8")
    assert page.startswith("# Input data")
    assert "python tools/make_data_table.py" in page
    assert "## Adaptation geometries" in page
    assert "`site/b.geojson`" in page
