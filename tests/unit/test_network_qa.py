"""Unit tests for the width, canal, outward-seed and bifurcation-arm checks in src.river_network."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString
from src.river_network import (
    accumulate_discharge,
    build_downstream_adjacency,
    canal_excluded_arms,
    fix_width_max_width_order,
    kahn_order,
    normalize_reach_id,
    outflow_crossing_mask,
    outward_seed_reaches,
    trim_overlapping_bifurcation_arms,
)


def _with(net, **cols):
    net = net.copy()
    for col, values in cols.items():
        net[col] = [values.get(r, 0) for r in net["reach_id"]]
    return net


def _widths(width, max_width):
    return gpd.GeoDataFrame(
        {"reach_id": ["1"], "width": [width], "max_width": [max_width]},
        geometry=[LineString([(0, 0), (1, 0)])],
    )


def test_max_width_fill_value_is_not_swapped_into_width():
    out = fix_width_max_width_order(_widths(389.0, 30.0))
    assert out["width"].iloc[0] == 389.0
    assert out["width_source"].iloc[0] == "sword_width, max_width fill ignored"


def test_genuine_inverted_pair_is_still_swapped():
    out = fix_width_max_width_order(_widths(400.0, 250.0))
    assert out["width"].iloc[0] == 250.0
    assert out["max_width"].iloc[0] == 400.0
    assert out["width_source"].iloc[0] == "sword_max_width"


def test_fill_check_can_be_disabled():
    out = fix_width_max_width_order(_widths(389.0, 30.0), max_width_fill_m=None)
    assert out["width"].iloc[0] == 30.0


def test_canal_arm_gets_no_share_and_river_arm_takes_all(network):
    net = network({"1": {"dn": ["2", "3"]}, "2": {"dn": ["4"]}, "3": {}, "4": {}})
    net = _with(net, lakeflag={"4": 2})  # canal two reaches down the arm "2"
    adj = build_downstream_adjacency(net)
    excluded = canal_excluded_arms(net, adj)
    assert excluded == {"2"}
    q = dict(
        zip(net["reach_id"], accumulate_discharge(net, {"1": 100.0}, adj, excluded))
    )
    assert q["2"] == 0.0 and q["4"] == 0.0
    assert q["3"] == pytest.approx(100.0)


def test_locks_alone_do_not_make_a_canal(network):
    net = network({"1": {"dn": ["2", "3"]}, "2": {}, "3": {}})
    net = _with(net, obstr_type={"2": 2})  # a weir with a lock on a canalised river
    assert canal_excluded_arms(net, build_downstream_adjacency(net)) == set()


def test_all_arms_canal_keeps_width_split(network):
    net = network({"1": {"dn": ["2", "3"]}, "2": {}, "3": {}})
    net = _with(net, lakeflag={"2": 2, "3": 2})
    assert canal_excluded_arms(net, build_downstream_adjacency(net)) == set()


def test_force_exclude_and_force_include(network):
    net = network({"1": {"dn": ["2", "3"]}, "2": {}, "3": {}})
    net = _with(net, lakeflag={"2": 2})
    adj = build_downstream_adjacency(net)
    assert canal_excluded_arms(net, adj, force_include=["2"]) == set()
    assert canal_excluded_arms(net, adj, force_exclude=["3"]) == {"2", "3"}
    assert canal_excluded_arms(net, adj, auto=False) == set()


def test_canal_on_a_confluence_downstream_does_not_count(network):
    """The chain walk stops at the next junction: a canal beyond it is not this arm's."""
    net = network(
        {
            "1": {"dn": ["2", "3"]},
            "2": {"dn": ["5"]},
            "4": {"dn": ["5"]},
            "5": {},
            "3": {},
        }
    )
    net = _with(net, lakeflag={"5": 2})
    assert canal_excluded_arms(net, build_downstream_adjacency(net)) == set()


def test_seed_fed_from_upstream_is_outward(network):
    net = network({"1": {"dn": ["2"]}, "2": {"dn": ["3"]}, "3": {}})
    adj = build_downstream_adjacency(net)
    assert outward_seed_reaches(net, ["1", "2"], adj) == {"2"}


def test_outflow_crossing_mask(network):
    net = network({"1": {}, "2": {}})
    net["seed_flows_out"] = [False, True]
    ds = xr.Dataset(
        {
            "has_glofas": ("crossing", np.array([1, 1, 0])),
            "inside_reach_id": ("crossing", np.array(["1", "2", "2"], dtype=object)),
        }
    )
    assert outflow_crossing_mask(ds, net).tolist() == [False, True, True]
    assert not outflow_crossing_mask(ds, net.drop(columns="seed_flows_out")).any()


def test_kahn_order_puts_every_reach_after_its_upstream_reaches():
    dn = {"a": ["c"], "b": ["c"], "c": ["d", "e"], "d": [], "e": []}
    order, on_cycle = kahn_order(list(dn), dn.__getitem__)
    pos = {n: i for i, n in enumerate(order)}
    assert on_cycle == []
    assert sorted(order) == sorted(dn)
    assert all(pos[u] < pos[d] for u, ds in dn.items() for d in ds)
    assert order[:2] == ["a", "b"]  # ties keep input order


def test_kahn_order_appends_cycle_nodes_last_in_input_order():
    dn = {0: [1], 1: [2], 2: [1], 3: []}
    order, on_cycle = kahn_order(range(4), dn.__getitem__)
    assert on_cycle == [1, 2]
    assert order == [0, 3, 1, 2]


def test_kahn_order_rejects_successor_outside_nodes():
    with pytest.raises(KeyError):
        kahn_order(["a"], lambda n: ["zz"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (float("nan"), None),
        (np.float64("nan"), None),
        (pd.NA, None),
        (pd.NaT, None),
        ("nan", None),
        (" NaN ", None),
        ("None", None),
        ("<NA>", None),
        ("", None),
        ("   ", None),
        (21406100011, "21406100011"),
        (21406100011.0, "21406100011"),
        ("21406100011.0", "21406100011"),
        (" 21406100011 ", "21406100011"),
        ("1e5", "100000"),
        (float("inf"), "inf"),
        ("-inf", "-inf"),
        (" abc ", "abc"),
    ],
)
def test_normalize_reach_id_edge_cases(raw, expected):
    assert normalize_reach_id(raw) == expected


def _bifurcation(arm_a, arm_b, parent=((0, 0), (1000, 0))):
    return gpd.GeoDataFrame(
        {
            "reach_id": ["1", "2", "3"],
            "rch_id_dn": ["[2, 3]", "[]", "[]"],
            "width": [100.0, 80.0, 40.0],
            "reach_length": [
                1000.0,
                LineString(arm_a).length,
                LineString(arm_b).length,
            ],
            "dist_out": [5000.0, 4000.0, 4000.0],
            "hydro_dist_hw": [0.0, 1000.0, -9999.0],
            "pathlen_hw": [1000.0, 3000.0, 3000.0],
        },
        geometry=[LineString(parent), LineString(arm_a), LineString(arm_b)],
        crs="EPSG:32631",
    )


def _by_id(net):
    return net.set_index(net["reach_id"])


def test_shared_arm_stretch_moves_junction_to_its_end():
    net = _bifurcation(
        [(1000, 0), (1050, 0), (2000, 500)], [(1000, 0), (1050, 0), (2000, -500)]
    )
    out = _by_id(trim_overlapping_bifurcation_arms(net))
    assert out.loc["1"].geometry.coords[-1] == (1050.0, 0.0)
    assert out.loc["2"].geometry.coords[0] == (1050.0, 0.0)
    assert out.loc["3"].geometry.coords[0] == (1050.0, 0.0)
    assert out.loc["1", "reach_length"] == pytest.approx(1050.0)
    assert out.loc["2", "reach_length"] == pytest.approx(out.loc["2"].geometry.length)
    assert out.loc["2", "dist_out"] == pytest.approx(3950.0)
    assert out.loc["1", "dist_out"] == 5000.0
    assert out.loc["1", "pathlen_hw"] == pytest.approx(1050.0)
    assert out.loc["2", "hydro_dist_hw"] == pytest.approx(1050.0)
    assert out.loc["3", "hydro_dist_hw"] == -9999.0


def test_crossing_arms_meet_at_the_new_junction_along_the_wider_arm():
    net = _bifurcation(
        [(1000, 0), (1040, 20), (1080, 0), (2000, -500)],
        [(1000, 0), (1040, -20), (1080, 0), (2000, 500)],
    )
    out = _by_id(trim_overlapping_bifurcation_arms(net))
    assert out.loc["1"].geometry.coords[-1] == pytest.approx((1080.0, 0.0))
    assert (1040.0, 20.0) in list(out.loc["1"].geometry.coords)
    for arm in ("2", "3"):
        assert out.loc[arm].geometry.coords[0] == pytest.approx((1080.0, 0.0))


def test_arms_drawn_upstream_keep_their_direction():
    net = _bifurcation(
        [(2000, 500), (1050, 0), (1000, 0)], [(1000, 0), (1050, 0), (2000, -500)]
    )
    out = _by_id(trim_overlapping_bifurcation_arms(net))
    assert out.loc["2"].geometry.coords[-1] == (1050.0, 0.0)
    assert out.loc["2"].geometry.coords[0] == (2000.0, 500.0)


def test_arms_meeting_only_at_their_downstream_ends_are_left_alone():
    net = _bifurcation(
        [(1000, 0), (1500, 300), (2000, 0)], [(1000, 0), (1500, -300), (2000, 0)]
    )
    out = trim_overlapping_bifurcation_arms(net)
    assert out.geometry.geom_equals(net.geometry).all()
    assert out["reach_length"].equals(net["reach_length"])
