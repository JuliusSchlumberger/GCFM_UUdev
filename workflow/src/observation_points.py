"""Observation points from every inflow down the river network and out to the water-level boundary (rule 12a)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.event_timing import sea_path
from src.river_network import normalize_reach_id

COLUMNS = ["kind", "reach_id", "x", "y", "dist_from_mouth_m"]


def downstream_reaches(
    adjacency: dict[str, list[str]], seeds: list[str], max_hops: int
) -> set[str]:
    """Return every reach reachable downstream of the seed reaches, through every arm."""
    found, frontier = set(), [(s, 0) for s in seeds if s]
    while frontier:
        reach, hops = frontier.pop()
        if reach in found or hops > max_hops:
            continue
        found.add(reach)
        frontier.extend((d, hops + 1) for d in adjacency.get(reach, []))
    return found


def river_points(
    nodes: pd.DataFrame, reaches: set[str], spacing_m: float
) -> pd.DataFrame:
    """Return one river point per spacing_m of distance to the sea on each reach.

    nodes: rule 11's river-bed node table (reach_id, x, y, dist_to_sea_m;
    500 m apart). Per reach, the first node of every spacing_m bin of
    dist_to_sea_m, so the points sit on nodes of the 1D profile.
    """
    sel = nodes[nodes["reach_id"].map(normalize_reach_id).isin(reaches)].copy()
    sel["reach_id"] = sel["reach_id"].map(normalize_reach_id)
    sel["_bin"] = np.floor(sel["dist_to_sea_m"].to_numpy(float) / spacing_m)
    sel = sel.sort_values(["reach_id", "dist_to_sea_m"]).drop_duplicates(
        ["reach_id", "_bin"]
    )
    return pd.DataFrame(
        {
            "kind": "river",
            "reach_id": sel["reach_id"].to_numpy(),
            "x": sel["x"].to_numpy(float),
            "y": sel["y"].to_numpy(float),
            "dist_from_mouth_m": sel["dist_to_sea_m"].to_numpy(float),
        }
    )


def nearest_node_distance(nodes: pd.DataFrame, x: float, y: float) -> float:
    """Return dist_to_sea_m of the river-bed node nearest (x, y)."""
    k = int(
        np.argmin(
            np.hypot(nodes["x"].to_numpy(float) - x, nodes["y"].to_numpy(float) - y)
        )
    )
    return float(nodes["dist_to_sea_m"].iloc[k])


def sea_points(
    bed_m: np.ndarray,
    mask: np.ndarray,
    transform,
    sea_level_m: float,
    mouths: list[tuple[str, float, float]],
    spacing_m: float,
) -> pd.DataFrame:
    """Return points every spacing_m along each mouth's fastest wet path to the water-level boundary.

    transform: the model grid's affine (x, y of cell corners); mouths:
    (reach_id, x, y) of each mouth in the model CRS. The path is
    src.event_timing.sea_path at sea_level_m; its last cell (on the boundary)
    is always included. dist_from_mouth_m is the distance seaward along the path.
    """
    dx_m = float(np.hypot(transform.a, transform.d))
    inv = ~transform
    rows = []
    for reach_id, x, y in mouths:
        col, row = inv * (x, y)
        rc = (int(np.floor(row)), int(np.floor(col)))
        if not (0 <= rc[0] < mask.shape[0] and 0 <= rc[1] < mask.shape[1]):
            continue
        path = sea_path(bed_m, mask, sea_level_m, rc, dx_m)
        if not len(path):
            continue
        dist = np.arange(len(path)) * dx_m
        keep = np.unique(
            np.concatenate(
                [
                    np.searchsorted(dist, np.arange(spacing_m, dist[-1], spacing_m)),
                    [len(path) - 1],
                ]
            )
        )
        for k in keep:
            px, py = transform * (path[k, 1] + 0.5, path[k, 0] + 0.5)
            rows.append(
                {
                    "kind": "sea",
                    "reach_id": reach_id,
                    "x": px,
                    "y": py,
                    "dist_from_mouth_m": float(dist[k]),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)
