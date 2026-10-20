"""Unit tests for the GVF target builder (src.gvf_target) and solver (src.gvf_solver)."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString
from src.gvf_solver import (
    GvfProblem,
    critical_depth,
    friction_slope,
    froude,
    integrate_reach,
    invert_bed,
    normal_depth,
)
from src.gvf_target import (
    NODE_INVERT,
    NODE_WATER_BODY,
    GvfTargets,
    build_gvf_targets,
    build_node_network,
    detect_structure_edges,
    enforce_non_increasing,
    orient_downstream,
    reach_target_levels,
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


def _targets(
    rivers,
    eta_fn,
    q,
    classes=None,
    dem=None,
    s_scale=None,
    spacing=500.0,
    structures=None,
):
    """GvfTargets with a prescribed target eta_fn(x, y) instead of SWORD wse."""
    dn = build_downstream_adjacency(rivers)
    net = build_node_network(rivers, dn, spacing, structures)
    eta = np.asarray(eta_fn(net.node_x, net.node_y), dtype=float)
    cls = (
        np.full(net.n_nodes, NODE_INVERT, dtype=object)
        if classes is None
        else classes(net)
    )
    return GvfTargets(
        net=net,
        eta_target=eta,
        eta_raw=eta.copy(),
        s_scale=np.full(net.n_nodes, 1e-4) if s_scale is None else s_scale(net),
        node_class=cls,
        dem=np.full(net.n_nodes, np.nan) if dem is None else dem(net),
        reach_q={r: float(q[r]) for r in net.reach_ids},
        reach_width={r: float(w) for r, w in zip(rivers["reach_id"], rivers["width"])},
        reach_level={},
        rising_edges=[],
    )


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


def test_uniform_reach_inversion_returns_manning_depth():
    s, q, w = 1e-4, 1000.0, 200.0
    hn = normal_depth(q, w, N, s)
    rivers = _reaches({"1": {"xy": [(0, 0), (20000, 0)], "width": w}})
    t = _targets(rivers, lambda x, y: 10.0 - s * x + hn, {"1": q})
    p = GvfProblem(t, N, sea_level_m=10.0 - s * 20000 + hn, mouth_reaches={"1"})
    res = invert_bed(p, lsq_apply="all")
    depth = t.eta_target[t.net.slot_node()] - res.z  # beds are per (reach, node) slot
    assert np.max(np.abs(depth / hn - 1.0)) < 0.01
    assert np.max(np.abs(res.residual)) < 0.005


def _backwater_case(noise_m):
    s, q, w, length = 1e-4, 800.0, 150.0, 30000.0
    rivers = _reaches({"1": {"xy": [(0, 0), (length, 0)], "width": w}})
    dn = build_downstream_adjacency(rivers)
    net = build_node_network(rivers, dn, 500.0)
    x = net.node_x
    z_true = -3.0 - s * x  # sea at 0 m is above normal depth: M1 backwater
    sol = integrate_reach(
        net.reach_chainage["1"], z_true[net.reach_nodes["1"]], w, q, N, eta_dn=0.0
    )
    eta_true = np.empty(net.n_nodes)
    eta_true[net.reach_nodes["1"]] = sol.eta
    rng = np.random.default_rng(1)
    eta_obs = eta_true + rng.uniform(-noise_m, noise_m, net.n_nodes)
    end = net.reach_nodes["1"][-1]
    eta_obs[end] = eta_true[end]
    t = _targets(rivers, lambda xx, yy: eta_obs, {"1": q})
    return t, z_true, s


def _bed_roughness(z):
    return float(np.max(np.abs(np.diff(z, 2))))


def test_regularisation_removes_bed_spikes():
    t, z_true, s = _backwater_case(noise_m=0.10)
    p = GvfProblem(t, N, sea_level_m=0.0, mouth_reaches={"1"})
    order = t.net.reach_slots["1"]
    free = invert_bed(
        p, nudging_passes=0, lsq_apply="all", rho_m=0.0, lsq_max_nfev=30
    ).z[order]
    reg = invert_bed(p, nudging_passes=0, lsq_apply="all", rho_m=0.05).z[order]
    assert _bed_roughness(reg) < 0.5 * _bed_roughness(free)
    adverse = -np.diff(reg) / 500.0  # bed fall per metre; negative = rising downstream
    assert np.all(adverse > -10 * s)


def test_manning_n_moves_the_bed_not_the_fit():
    t, _, _ = _backwater_case(noise_m=0.0)
    beds, rmses = [], []
    for n in (0.02, 0.03, 0.05, 0.07):
        res = invert_bed(
            GvfProblem(t, n, sea_level_m=0.0, mouth_reaches={"1"}), lsq_apply="all"
        )
        beds.append(float(np.mean(res.z)))
        rmses.append(float(np.sqrt(np.nanmean(res.residual**2))))
    assert all(a > b for a, b in zip(beds, beds[1:]))  # rougher channel: deeper bed
    assert max(rmses) < 0.02


def test_steep_segment_is_capped_at_froude_limit_and_continues():
    x = np.linspace(0, 5000, 11)
    z = 50.0 - 0.02 * x  # 2 % slope, supercritical at normal depth
    sol = integrate_reach(x, z, 50.0, 100.0, N, eta_dn=z[-1] + 0.1, froude_cap=0.95)
    assert sol.capped.all()
    assert np.all(np.isfinite(sol.eta))
    assert froude(100.0, 50.0, sol.h) == pytest.approx(np.full(11, 0.95), rel=1e-6)


def _manning_target(s=1e-4, z0=5.0):
    return lambda x, y: z0 - s * x


def test_confluence_levels_agree_and_fit():
    rivers = _reaches(
        {
            "1": {"xy": [(0, 5000), (10000, 0)], "dn": ["3"]},
            "2": {"xy": [(0, -5000), (10000, 0)], "dn": ["3"]},
            "3": {"xy": [(10000, 0), (20000, 0)], "width": 150.0},
        }
    )
    t = _targets(rivers, _manning_target(), {"1": 400.0, "2": 300.0, "3": 700.0})
    p = GvfProblem(t, N, sea_level_m=5.0 - 1e-4 * 20000, mouth_reaches={"3"})
    res = invert_bed(p)
    assert np.max(res.solve.junction_mismatch) < 1e-3
    assert np.nanmax(np.abs(res.residual)) < 0.05


@pytest.mark.parametrize("rejoin", [False, True])
def test_bifurcation_and_loop_mismatch_small(rejoin):
    spec = {
        "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2", "3"]},
        "2": {"xy": [(10000, 0), (20000, 3000)], "dn": ["4"] if rejoin else []},
        "3": {"xy": [(10000, 0), (20000, -3000)], "dn": ["4"] if rejoin else []},
    }
    q = {"1": 1000.0, "2": 600.0, "3": 400.0}
    if rejoin:
        spec["4"] = {"xy": [(20000, 3000), (30000, 0)]}
        spec["2"]["xy"] = [(10000, 0), (20000, 3000)]
        spec["3"]["xy"] = [(10000, 0), (20000, 3000)]
        q["4"] = 1000.0
    rivers = _reaches(spec)
    t = _targets(rivers, _manning_target(), q)
    ends = {"4"} if rejoin else {"2", "3"}
    p = GvfProblem(t, N, sea_level_m=float(np.min(t.eta_target)), mouth_reaches=ends)
    res = invert_bed(p, lsq_apply="all")
    assert np.max(res.solve.junction_mismatch) < 0.05
    assert np.nanmax(np.abs(res.residual)) < 0.05


def test_water_body_keeps_target_level_and_dem_bed():
    rivers = _reaches(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2"]},
            "2": {"xy": [(10000, 0), (15000, 0)], "width": 3000.0},
        }
    )

    def classes(net):
        c = np.full(net.n_nodes, NODE_INVERT, dtype=object)
        c[net.reach_nodes["2"][1:]] = NODE_WATER_BODY
        return c

    t = _targets(
        rivers,
        lambda x, y: np.where(x > 10000, 0.2, 0.2 + 1e-4 * (10000 - x)),
        {"1": 500.0, "2": 500.0},
        classes=classes,
        dem=lambda net: np.full(net.n_nodes, -8.0),
    )
    res = invert_bed(GvfProblem(t, N, sea_level_m=0.1, mouth_reaches={"2"}))
    wb = t.node_class == NODE_WATER_BODY
    assert np.allclose(res.z[wb[t.net.slot_node()]], -8.0)
    assert np.allclose(res.solve.eta[wb], 0.2, atol=1e-6)


def test_structure_separates_nodes_and_sets_upstream_level():
    rivers = _reaches(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2"], "wse": 6.0},
            "2": {"xy": [(10000, 0), (20000, 0)], "wse": 3.0},
        }
    )
    dn = build_downstream_adjacency(rivers)
    edges, _ = detect_structure_edges(rivers, dn, reach_target_levels(rivers))
    assert edges == {("1", "2")}
    t = _targets(
        rivers,
        lambda x, y: np.where(
            x < 10000 - 1, 6.0 + 1e-4 * (10000 - x), 3.0 + 1e-4 * (20000 - x)
        ),
        {"1": 500.0, "2": 500.0},
        structures=edges,
    )
    end1 = t.net.reach_nodes["1"][-1]
    assert end1 != t.net.reach_nodes["2"][0]
    p = GvfProblem(t, N, sea_level_m=3.0, mouth_reaches={"2"})
    res = invert_bed(p)
    assert res.solve.eta[end1] == pytest.approx(t.eta_target[end1], abs=0.05)


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


def test_target_builder_monotone_single_junction_value_and_datum(gvf_cfg):
    rivers = _reaches(_chain_spec())
    cfg = gvf_cfg
    t0 = build_gvf_targets(
        rivers,
        {"1": 100.0, "2": 100.0, "3": 60.0, "4": 40.0},
        0.0,
        lambda x, y, w: np.zeros(len(x)),
        lambda x, y: np.full(len(x), np.nan),
        gebco_floor_m=-10.0,
        cfg=cfg,
    )
    for a, b, _ in t0.net.node_edges():
        assert t0.eta_target[a] >= t0.eta_target[b] - 1e-9
    # one node where reach 2 splits into 3 and 4
    j = t0.net.reach_nodes["2"][-1]
    assert t0.net.reach_nodes["3"][0] == j == t0.net.reach_nodes["4"][0]
    # no 0.1 m steps left: largest node-to-node drop well below the quantum
    drops = [t0.eta_target[a] - t0.eta_target[b] for a, b, _ in t0.net.node_edges()]
    assert max(drops) < 0.1
    t1 = build_gvf_targets(
        rivers,
        {"1": 100.0, "2": 100.0, "3": 60.0, "4": 40.0},
        0.25,
        lambda x, y, w: np.zeros(len(x)),
        lambda x, y: np.full(len(x), np.nan),
        gebco_floor_m=-10.0,
        cfg=cfg,
    )
    assert np.allclose(t1.eta_target - t0.eta_target, 0.25)


def test_rising_level_is_reported_not_a_structure():
    rivers = _reaches(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2"], "wse": 7.1},
            "2": {"xy": [(10000, 0), (20000, 0)], "wse": 8.3},
        }
    )
    dn = build_downstream_adjacency(rivers)
    edges, rising = detect_structure_edges(rivers, dn, reach_target_levels(rivers))
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
    edges, _ = detect_structure_edges(rivers, dn, reach_target_levels(rivers))
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


def test_no_flow_reach_gets_width_depth_bed_and_is_ignored_at_junction():
    from src.gvf_target import NODE_NO_FLOW

    rivers = _reaches(
        {
            "1": {"xy": [(0, 0), (10000, 0)], "dn": ["2", "3"]},
            "2": {"xy": [(10000, 0), (20000, 0)]},
            "3": {"xy": [(10000, 0), (20000, 4000)]},  # side arm with no discharge
        }
    )
    t = _targets(rivers, _manning_target(), {"1": 800.0, "2": 800.0, "3": 0.0})
    p = GvfProblem(
        t, N, sea_level_m=float(np.min(t.eta_target)), mouth_reaches={"2", "3"}
    )
    res = invert_bed(p, lsq_apply="all")
    no_flow = p.slot_class == NODE_NO_FLOW
    assert no_flow.sum() == len(t.net.reach_nodes["3"])
    depth = t.eta_target[p.slot_node] - res.z
    flowing_depth = np.median(depth[t.net.reach_slots["2"]])
    assert np.allclose(
        depth[no_flow], flowing_depth, rtol=0.2
    )  # all widths equal: b = 0, c = median
    assert np.isnan(res.residual[no_flow]).all()
    assert np.max(res.solve.junction_mismatch) < 1e-6  # still-water arm not averaged in


def test_fit_width_depth_recovers_power_law():
    from src.gvf_solver import fit_width_depth

    w = np.array([50.0, 100.0, 200.0, 400.0, 800.0])
    c, b = fit_width_depth(w, 0.5 * w**0.4)
    assert b == pytest.approx(0.4, abs=1e-9) and c == pytest.approx(0.5, rel=1e-9)
    assert fit_width_depth(w[:2], np.array([3.0, 5.0])) == (4.0, 0.0)


def test_values_at_points_follow_the_reach_line_even_when_reversed():
    from src.gvf import GvfRun, values_at_points

    rivers = _reaches({"1": {"xy": [(10000, 0), (0, 0)]}})  # line drawn west, flow west
    t = _targets(rivers, lambda x, y: 0.001 * x, {"1": 100.0})
    run = GvfRun(targets=t, problem=None, result=None, mouths=set())  # type: ignore[arg-type]
    per_slot = t.net.reach_chainage[
        "1"
    ].copy()  # value = chainage from the upstream end
    out = values_at_points(
        run, per_slot, ["1", "1", "x"], np.array([9000.0, 2500.0, 0.0]), np.zeros(3)
    )
    assert out[0] == pytest.approx(1000.0) and out[1] == pytest.approx(7500.0)
    assert np.isnan(out[2])
