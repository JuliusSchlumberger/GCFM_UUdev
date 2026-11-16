"""Unit tests for the node network (src.gvf_target) and the GVF forward solve (src.gvf_solver)."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString
from src.gvf_solver import (
    critical_depth,
    fit_width_depth,
    forward_network,
    friction_slope,
    froude,
    integrate_reach,
    normal_depth,
)
from src.gvf_target import (
    build_node_network,
    detect_structure_edges,
    enforce_non_increasing,
    orient_downstream,
)
from src.river_network import build_downstream_adjacency

N = 0.03


def _reaches(spec: dict[str, dict]) -> gpd.GeoDataFrame:
    """Reaches from {id: {"xy": [(x, y), ...], "dn": [...], "wse": .., "width": ..}}."""
    rows = []
    for reach_id, s in spec.items():
        rows.append(
            {
                "reach_id": reach_id,
                "rch_id_dn": "[" + ", ".join(s.get("dn", [])) + "]",
                "width": float(s.get("width", 100.0)),
                "wse": float(s.get("wse", 0.0)),
                "obstr_type": int(s.get("obstr", 0)),
                "geometry": LineString(s["xy"]),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32631")


def _levels(rivers):
    """Per-reach level from the spec's 'wse' column."""
    return dict(zip(rivers["reach_id"], rivers["wse"].astype(float)))


def test_froude_includes_depth_and_critical_depth_gives_one():
    q, w = 500.0, 100.0
    hc = critical_depth(q, w)
    assert froude(q, w, hc) == pytest.approx(1.0, rel=1e-9)
    assert froude(q, w, 4 * hc) == pytest.approx(1.0 / 8.0, rel=1e-9)


def test_normal_depth_matches_friction_slope():
    h = normal_depth(1000.0, 200.0, N, 1e-4)
    assert friction_slope(1000.0, 200.0, h, N) == pytest.approx(1e-4, rel=1e-5)
    assert normal_depth(0.0, 200.0, N, 1e-4) == 0.0


def _uniform(length=20000.0, s=1e-4, q=1000.0, w=200.0):
    x = np.linspace(0, length, int(length / 500) + 1)
    z = 10.0 - s * x
    hn = normal_depth(q, w, N, s)
    return x, z, hn


def test_uniform_reach_forward_reproduces_normal_depth():
    x, z, hn = _uniform()
    sol = integrate_reach(x, z, 200.0, 1000.0, N, eta_dn=z[-1] + hn)
    assert np.max(np.abs(sol.h - hn)) < 1e-3
    assert not sol.capped.any()


def test_steep_segment_is_capped_at_froude_limit_and_continues():
    x = np.linspace(0, 5000, 11)
    z = 50.0 - 0.02 * x  # 2 % slope, supercritical at normal depth
    sol = integrate_reach(x, z, 50.0, 100.0, N, eta_dn=z[-1] + 0.1, froude_cap=0.95)
    assert sol.capped.all()
    assert np.all(np.isfinite(sol.eta))
    assert froude(100.0, 50.0, sol.h) == pytest.approx(np.full(11, 0.95), rel=1e-6)


def _chain_spec():
    # Rhine-like chain plus a side arm; quantised wse, one reach line reversed.
    return {
        "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2"], "wse": 3.0},
        "2": {
            "xy": [(20000, 0), (10000, 0)],
            "dn": ["3", "4"],
            "wse": 2.1,
        },  # runs upstream
        "3": {"xy": [(20000, 0), (30000, 0)], "wse": 1.7},
        "4": {"xy": [(20000, 0), (30000, 5000)], "wse": 1.8},
    }


def test_orient_downstream_reverses_upstream_lines():
    rivers = _reaches(_chain_spec())
    out, rev = orient_downstream(rivers, build_downstream_adjacency(rivers))
    assert rev == ["2"]
    assert out.geometry.iloc[1].coords[0] == (10000.0, 0.0)


def test_rising_level_is_reported_not_a_structure():
    rivers = _reaches(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2"], "wse": 7.1},
            "2": {"xy": [(10000, 0), (20000, 0)], "wse": 8.3},
        }
    )
    dn = build_downstream_adjacency(rivers)
    edges, rising = detect_structure_edges(rivers, dn, _levels(rivers))
    assert edges == set()
    assert rising == [("1", "2", pytest.approx(1.2))]


def test_lock_with_small_drop_is_a_structure_but_not_without_drop():
    rivers = _reaches(
        {
            "1": {"xy": [(0, 0), (1000, 0)], "dn": ["2"], "wse": 3.7, "obstr": 2},
            "2": {"xy": [(1000, 0), (2000, 0)], "dn": ["3"], "wse": 3.4},
            "3": {"xy": [(2000, 0), (3000, 0)], "wse": 3.4, "obstr": 2},
        }
    )
    dn = build_downstream_adjacency(rivers)
    edges, _ = detect_structure_edges(rivers, dn, _levels(rivers))
    assert edges == {("1", "2")}


def test_enforce_non_increasing_on_a_chain():
    rivers = _reaches({"1": {"xy": [(0, 0), (5000, 0)]}})
    net = build_node_network(rivers, build_downstream_adjacency(rivers), 500.0)
    v = np.array([3.0, 3.1, 2.9, 3.0, 2.0, 2.2, 1.5, 1.5, 1.6, 1.0, 0.9])
    by_node = np.empty(net.n_nodes)
    by_node[net.reach_nodes["1"]] = v
    prof = enforce_non_increasing(net, by_node)[net.reach_nodes["1"]]
    assert np.all(np.diff(prof) <= 1e-12)
    assert np.max(np.abs(prof - v)) < 0.3


def test_fit_width_depth_recovers_power_law():
    w = np.array([50.0, 100.0, 200.0, 400.0, 800.0])
    c, b = fit_width_depth(w, 0.5 * w**0.4)
    assert b == pytest.approx(0.4, abs=1e-9) and c == pytest.approx(0.5, rel=1e-9)
    assert fit_width_depth(w[:2], np.array([3.0, 5.0])) == (4.0, 0.0)


def test_values_at_points_follow_the_reach_line_even_when_reversed():
    from src.river_bed import values_at_points

    rivers = _reaches({"1": {"xy": [(10000, 0), (0, 0)]}})  # line drawn west, flow west
    net = build_node_network(rivers, build_downstream_adjacency(rivers), 500.0)
    per_node = np.empty(net.n_nodes)
    per_node[net.reach_nodes["1"]] = net.reach_chainage[
        "1"
    ]  # chainage from the upstream end
    out = values_at_points(
        net, per_node, ["1", "1", "x"], np.array([9000.0, 2500.0, 0.0]), np.zeros(3)
    )
    assert out[0] == pytest.approx(1000.0) and out[1] == pytest.approx(7500.0)
    assert np.isnan(out[2])


def _uniform_bed(net, s, z0, q, w):
    hn = normal_depth(q, w, N, s)
    bed = z0 - s * net.node_x
    return bed, bed + hn


def test_forward_network_returns_normal_depth_on_a_uniform_confluence():
    s, q, w = 1e-4, 700.0, 150.0
    rivers = _reaches(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["3"], "width": w},
            "2": {"xy": [(0, 1), (10000, 0)], "dn": ["3"], "width": w},
            "3": {"xy": [(10000, 0), (20000, 0)], "width": w},
        }
    )
    net = build_node_network(rivers, build_downstream_adjacency(rivers), 500.0)
    bed, level = _uniform_bed(net, s, 5.0, q, w)
    mouth = int(net.reach_nodes["3"][-1])
    out = forward_network(
        net,
        bed,
        np.full(net.n_nodes, w),
        {"1": q, "2": q, "3": q},
        N,
        {mouth: level[mouth]},
    )
    assert np.max(np.abs(out - level)) < 1e-3


def test_forward_network_backwater_decays_upstream_and_dry_arm_is_flat():
    s, q, w = 1e-4, 700.0, 150.0
    rivers = _reaches(
        {
            "1": {"xy": [(0, 0), (20000, 0)], "dn": ["2", "3"], "width": w},
            "2": {"xy": [(20000, 0), (40000, 0)], "width": w},
            "3": {"xy": [(20000, 0), (30000, 5000)], "width": w},
        }
    )
    net = build_node_network(rivers, build_downstream_adjacency(rivers), 500.0)
    bed, level = _uniform_bed(net, s, 10.0, q, w)
    m2, m3 = int(net.reach_nodes["2"][-1]), int(net.reach_nodes["3"][-1])
    ends = {m2: level[m2] + 1.0, m3: 0.0}
    out = forward_network(
        net, bed, np.full(net.n_nodes, w), {"1": q, "2": q, "3": 0.0}, N, ends
    )
    rise = (out - level)[net.reach_nodes["2"]][::-1]  # from the mouth upstream
    assert rise[0] == pytest.approx(1.0) and np.all(np.diff(rise) < 0)
    dry = out[net.reach_nodes["3"]]
    assert np.allclose(
        dry[1:], dry[-1]
    )  # its first node is the junction, on the flowing level


def test_main_path_slope_at_a_bifurcation_follows_the_larger_arm():
    from src.gvf_target import main_path_slope, main_reaches

    rivers = _reaches(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2", "3"]},
            "2": {"xy": [(10000, 0), (20000, 0)]},
            "3": {"xy": [(10000, 0), (12000, 0)]},  # short steep side arm
        }
    )
    net = build_node_network(rivers, build_downstream_adjacency(rivers), 500.0)
    slope_of = {"1": 1e-4, "2": 1e-4, "3": 1e-3}
    v = np.empty(net.n_nodes)
    j = net.reach_nodes["1"][-1]
    for r in ("1", "2", "3"):
        ch = net.reach_chainage[r]
        start = 2.0 + 1e-4 * 10000 if r == "1" else 2.0
        v[net.reach_nodes[r]] = start - slope_of[r] * ch
    q = {"1": 1000.0, "2": 900.0, "3": 100.0}
    s = main_path_slope(net, v, q)
    assert s[j] == pytest.approx(1e-4, rel=1e-6)
    assert s[net.reach_nodes["3"][2]] == pytest.approx(1e-3, rel=1e-6)
    assert main_reaches(net, q) == {"1", "2"}
