r"""Render docs/data.md from config/data_catalogue.yml and adaptation_strategies.yml.

Usage:
    python tools/make_data_table.py
    python tools/make_data_table.py --catalogue config/data_catalogue.yml \
        --strategies config/adaptation_strategies.yml --output docs/data.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MISSING = "—"
DESCRIPTION_MAX_CHARS = 140

# Datasets whose catalogue entry does not name the tool that obtains or builds them.
OBTAIN_TOOLS = {
    "fathomdem": "tools/download_fathomdem.py",
    "gtsm_surge_residual": "tools/download_cds_waterlevel.py",
    "gtsm_total_water_level": "tools/download_cds_waterlevel.py",
    "land_use_esa_worldcover": "tools/build_esa_worldcover_vrt.py",
    "river_discharge": "tools/download_glofas.py",
    "river_network": "tools/prepare_modified_sword_dataset.py",
}
TOOL_PATTERN = re.compile(r"tools/[A-Za-z0-9_]+\.py")
FILE_PATTERN = re.compile(r"^[^\s{}]+\.[A-Za-z0-9]+$")
# `auto/<name>` geometries are built per basin by the workflow; the rest are drawn by hand.
AUTO_PREFIX = "auto/"
AUTO_BUILT = (
    "built by rule `build_adaptation_lines` into "
    "`<results_dir>/{basin_id}/adaptation_lines/<name>.geojson`"
)
HAND_DRAWN = "project-specific; obtain from the project team"

HEADER = """# Input data

Generated from `config/data_catalogue.yml`; do not edit. Regenerate with
`python tools/make_data_table.py`.

Every `file_path` is relative to `<raw_data_root>`, the `raw_data_root` key of
`config/site.yml` (see [install.md](install.md#site-settings)), for example:

```
raw_data_root: /data/gcfm/raw_data
```

The pipeline reads only `name`, `file_path` and `attributes` of each entry; the
other fields are documentation. "How to obtain" names the script in `tools/` that
downloads or builds the dataset; datasets without one are downloaded from the
source by hand. {missing} marks a field the catalogue does not record.

"""

GEOMETRY_HEADER = """
## Adaptation geometries

Generated from `config/adaptation_strategies.yml`. Measure files needed by the
`adapt` target, relative to `<raw_data_root>`. Only the strategies selected with
`target_strategies` need their files. `auto/<name>` entries are not raw data:
rule `build_adaptation_lines` derives them from the basin's SFINCS skeleton when
an `adapt` job needs them; the others are drawn per basin.

"""

GEOMETRY_COLUMNS = ("file_path", "strategy / measure", "how to obtain")

COLUMNS = (
    "dataset",
    "description",
    "version",
    "source / citation",
    "licence",
    "file_path",
    "how to obtain",
)


def _cell(value: object) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return MISSING
    text = " ".join(str(value).split())
    return text.replace("|", "\\|")


def _short_description(text: object) -> str:
    if not text:
        return MISSING
    flat = " ".join(str(text).split())
    first = re.split(r"(?<=[.!?])\s", flat, maxsplit=1)[0]
    if len(first) > DESCRIPTION_MAX_CHARS:
        first = first[: DESCRIPTION_MAX_CHARS - 1].rstrip() + "…"
    return _cell(first)


def _how_to_obtain(entry: dict) -> str:
    tool = OBTAIN_TOOLS.get(entry["name"])
    if tool is None:
        found = TOOL_PATTERN.search(
            " ".join(str(entry.get(k, "")) for k in ("source", "notes"))
        )
        tool = found.group(0) if found else None
    return f"`{tool}`" if tool else "manual download from the source"


def render_table(catalogue: dict) -> str:
    """Return the markdown table, one row per dataset sorted by name."""
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for entry in sorted(catalogue["datasets"], key=lambda e: e["name"]):
        licence = entry.get("licence", entry.get("license"))
        row = (
            f"`{entry['name']}`",
            _short_description(entry.get("description")),
            _cell(entry.get("version")),
            _cell(entry.get("source")),
            _cell(licence),
            f"`{entry['file_path']}`" if entry.get("file_path") else MISSING,
            _how_to_obtain(entry),
        )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _file_values(value: object):
    if isinstance(value, dict):
        for item in value.values():
            yield from _file_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _file_values(item)
    elif isinstance(value, str) and (
        value.startswith(AUTO_PREFIX) or (FILE_PATTERN.match(value) and "/" in value)
    ):
        yield value


def adaptation_geometries(strategies: dict) -> dict[str, list[str]]:
    """Map every file path in the strategies config to its `strategy/measure` users."""
    users: dict[str, list[str]] = {}
    for family, spec in strategies.items():
        if not isinstance(spec, dict):
            continue
        for measure, params in (spec.get("measures") or {}).items():
            for path in _file_values(params):
                users.setdefault(path, []).append(f"{family}/{measure}")
    return {path: sorted(set(users[path])) for path in sorted(users)}


def render_geometry_table(strategies: dict) -> str:
    """Return the markdown table of adaptation geometries, one row per file."""
    lines = [
        "| " + " | ".join(GEOMETRY_COLUMNS) + " |",
        "|" + "---|" * len(GEOMETRY_COLUMNS),
    ]
    for path, users in adaptation_geometries(strategies).items():
        row = (
            f"`{path}`",
            ", ".join(f"`{u}`" for u in users),
            AUTO_BUILT if path.startswith(AUTO_PREFIX) else HAND_DRAWN,
        )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_page(catalogue: dict, strategies: dict) -> str:
    """Return the full docs/data.md content."""
    return (
        HEADER.format(missing=MISSING)
        + render_table(catalogue)
        + GEOMETRY_HEADER
        + render_geometry_table(strategies)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=REPO_ROOT / "config" / "data_catalogue.yml",
        help="data catalogue to read (default: config/data_catalogue.yml)",
    )
    parser.add_argument(
        "--strategies",
        type=Path,
        default=REPO_ROOT / "config" / "adaptation_strategies.yml",
        help="adaptation strategies to read (default: config/adaptation_strategies.yml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "data.md",
        help="markdown file to write (default: docs/data.md)",
    )
    args = parser.parse_args()

    catalogue = yaml.safe_load(args.catalogue.read_text(encoding="utf-8"))
    strategies = yaml.safe_load(args.strategies.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_page(catalogue, strategies))
    n_geometries = len(adaptation_geometries(strategies))
    print(
        f"wrote {args.output} ({len(catalogue['datasets'])} datasets, "
        f"{n_geometries} adaptation geometries)"
    )


if __name__ == "__main__":
    main()
