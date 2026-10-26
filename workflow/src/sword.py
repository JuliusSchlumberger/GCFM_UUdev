"""Read SWORD v17c reaches into the pipeline's river-network schema.

Entry point: :func:`read_sword_reaches`, used by rule get_river_network (06)
and rule test_bifurcation_calibration_options (11b).

The published v17c ships one GeoPackage per region (layer ``reaches``). The
pipeline expects topology as bracketed id strings (``rch_id_dn``,
``rch_id_up``: "[id1, id2]") plus ``n_rch_dn`` and ``is_mainstem_edge``;
:func:`to_pipeline_schema` derives those from the published
``rch_id_dn_1..4`` / ``rch_id_up_1..4`` (0 = no reach), ``n_rch_down`` and
``is_mainstem`` and keeps every published field, including the SWOT
``*_obs_*`` statistics. ``obstr_type`` 5, undocumented in the v17c variable
reference (0 none, 1 dam, 2 lock, 3 low-head, 4 falls), is read as 0; the
published value stays in ``obstr_type_published``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import box

log = logging.getLogger(__name__)

SWORD_LAYER = "reaches"
# Widths read from the corrections table must match SWORD within this, or the
# reach id is probably wrong (absorbs widths floored to whole metres).
WIDTH_MATCH_TOLERANCE_M = 1.0
UNDOCUMENTED_OBSTR_TYPES = (5,)


def _id_list(frame: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in (f"{prefix}_{k}" for k in range(1, 5)) if c in frame.columns]
    ids = frame[cols].to_numpy(dtype="float64")
    return pd.Series(
        [
            "[" + ", ".join(str(int(v)) for v in row if np.isfinite(v) and v > 0) + "]"
            for row in ids
        ],
        index=frame.index,
    )


def to_pipeline_schema(reaches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add the topology and naming the pipeline expects to published SWORD reaches."""
    out = reaches.copy()
    out["rch_id_dn"] = _id_list(out, "rch_id_dn")
    out["rch_id_up"] = _id_list(out, "rch_id_up")
    if "n_rch_down" in out.columns:
        out["n_rch_dn"] = out["n_rch_down"]
    if "is_mainstem" in out.columns:
        out["is_mainstem_edge"] = out["is_mainstem"]
    if "obstr_type" in out.columns:
        out["obstr_type_published"] = out["obstr_type"]
        out.loc[out["obstr_type"].isin(UNDOCUMENTED_OBSTR_TYPES), "obstr_type"] = 0
    return out


def read_sword_reaches(
    paths: Sequence[str | Path], wgs84_bounds: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Read and clip the reaches of all regional files to a WGS84 box.

    Reaches are clipped to the box (geometry cut at its edge), as rule 06
    always did; regions without reaches in the box contribute nothing.
    """
    parts = []
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"SWORD regional file not found: {path}")
        gdf = pyogrio.read_dataframe(path, layer=SWORD_LAYER, bbox=tuple(wgs84_bounds))
        if len(gdf):
            parts.append(gdf)
    if not parts:
        raise ValueError(
            f"No SWORD reach intersects {wgs84_bounds} in {len(paths)} regional file(s)"
        )
    reaches = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    clip_box = gpd.GeoDataFrame(geometry=[box(*wgs84_bounds)], crs="EPSG:4326").to_crs(
        reaches.crs
    )
    reaches = gpd.clip(reaches, clip_box).reset_index(drop=True)
    return to_pipeline_schema(reaches)


def read_width_corrections(path: str | Path) -> pd.DataFrame:
    """Read the manual width table (reach_id, expected_width_m, width_m, delta, note)."""
    table = pd.read_csv(path, dtype={"reach_id": "int64"}, comment="#")
    missing = {"reach_id", "expected_width_m", "width_m"} - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    if table["reach_id"].duplicated().any():
        raise ValueError(
            f"{path}: duplicate reach_id {table.loc[table.reach_id.duplicated(), 'reach_id'].tolist()}"
        )
    return table


def apply_width_corrections(
    reaches: gpd.GeoDataFrame, table: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Overwrite ``width`` on the corrected reaches present; mark them ``width_source = manual``.

    A present reach whose SWORD width differs from ``expected_width_m`` by
    more than WIDTH_MATCH_TOLERANCE_M raises: the table no longer matches the
    source data.
    """
    out = reaches.copy()
    if "width_source" not in out.columns:
        out["width_source"] = "sword_width"
    rid = out["reach_id"].astype("int64")
    hits = table[table["reach_id"].isin(set(rid))]
    for row in hits.itertuples(index=False):
        sel = rid == row.reach_id
        current = float(out.loc[sel, "width"].iloc[0])
        if abs(current - float(row.expected_width_m)) > WIDTH_MATCH_TOLERANCE_M:
            raise ValueError(
                f"Width correction for reach {row.reach_id}: SWORD width {current:.2f} m, "
                f"expected {row.expected_width_m} m -- table does not match the source data"
            )
        out.loc[sel, "width"] = float(row.width_m)
        out.loc[sel, "width_source"] = "manual"
    if len(hits):
        log.info(
            f"Manual width corrections applied to {len(hits)} reach(es): {hits['reach_id'].tolist()}"
        )
    return out
