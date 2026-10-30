"""Synthetic-channel tests for the sea-influence helpers of the upstream boundary check."""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from _upstream_profile import (  # noqa: E402
    boundary_verdict,
    efolding_length,
    first_above,
    high_gradient_rule,
    influence_length,
    path_nodes,
)
from src.gvf_solver import forward_network, normal_depth  # noqa: E402
from src.gvf_target import build_node_network  # noqa: E402

G = 9.81
S, W, Q, N = 1e-4, 200.0, 500.0, 0.03
REACHES = ["a", "b", "c"]


def _network(length_m=60000.0):
    """Straight channel along x to the sea at x = 0 in three reaches, nodes every 500 m."""
    cuts = np.linspace(length_m, 0.0, 4)
    rivers = gpd.GeoDataFrame(
        {"reach_id": REACHES},
        geometry=[LineString([(cuts[i], 0.0), (cuts[i + 1], 0.0)]) for i in range(3)],
        crs="EPSG:32631",
    )
    net = build_node_network(rivers, {"a": ["b"], "b": ["c"], "c": []}, 500.0)
    ids, dist = path_nodes(net, REACHES)
    return net, ids, dist


def _levels(net, bed, sea_m, q=Q):
    end = int(net.reach_nodes["c"][-1])
    return forward_network(
        net, bed, np.full(net.n_nodes, W), {r: q for r in REACHES}, N, {end: sea_m}
    )


def _uniform_bed(net, q=Q):
    h_n = normal_depth(q, W, N, S)
    return S * net.node_x - h_n, h_n


def test_path_distance_follows_the_channel_to_the_mouth():
    net, ids, dist = _network()
    np.testing.assert_allclose(dist, net.node_x[ids], atol=1e-6)
    assert dist[0] == pytest.approx(60000.0) and dist[-1] == 0.0
    assert len(set(ids.tolist())) == len(ids)


def test_small_sea_rise_decays_with_the_linearised_backwater_length():
    net, ids, dist = _network()
    bed, h_n = _uniform_bed(net)
    delta = (_levels(net, bed, 0.05) - _levels(net, bed, 0.0))[ids]
    assert delta[-1] == pytest.approx(0.05, abs=1e-6)
    assert np.all(np.diff(delta) >= -1e-9)  # grows towards the mouth
    # rectangular Manning: S_f ~ h^-k, k = 2 + 4/3 W/(W + 2h); delta ~ exp(-k S x / (h (1 - Fr^2)))
    k = 2.0 + 4.0 / 3.0 * W / (W + 2.0 * h_n)
    fr2 = Q**2 / (G * W**2 * h_n**3)
    l_lin = h_n * (1.0 - fr2) / (k * S)
    l_fit = efolding_length(dist, delta, influence_length(dist, delta, 1e-4))
    assert l_fit == pytest.approx(l_lin, rel=0.1)
    assert h_n / S > l_fit  # h/S is the ~5 % length, longer than the e-folding length


def test_still_water_level_meets_the_surface_at_its_height_over_the_slope():
    net, ids, dist = _network()
    bed, h_n = _uniform_bed(net)
    x = first_above(dist, (bed + h_n)[ids], 1.0)
    assert abs(x - 1.0 / S) <= 500.0
    assert first_above(dist, (bed + h_n)[ids], 1e3) == np.inf


def test_flat_lower_reach_puts_the_influence_on_the_sloping_reach():
    net, ids, dist = _network(length_m=150000.0)
    x = net.node_x
    bed = np.where(x < 20000.0, -5.0, -5.0 + S * (x - 20000.0))
    calm, surge = _levels(net, bed, 0.0), _levels(net, bed, 1.0)
    assert np.isfinite(calm).all() and np.isfinite(surge).all()
    x_inf = influence_length(dist, (surge - calm)[ids], 0.05)
    assert 20000.0 < x_inf < np.inf


def test_large_surge_on_shallow_flow_reaches_further_than_the_linear_estimate():
    net, ids, dist = _network(length_m=150000.0)
    bed, h_n = _uniform_bed(net)
    a0, tol = 5.0, 0.05
    delta = (_levels(net, bed, a0) - _levels(net, bed, 0.0))[ids]
    assert np.isfinite(delta).all()
    k = 2.0 + 4.0 / 3.0 * W / (W + 2.0 * h_n)
    l_lin = h_n * (1.0 - Q**2 / (G * W**2 * h_n**3)) / (k * S)
    assert influence_length(dist, delta, tol) > l_lin * np.log(a0 / tol)


def test_forcing_point_at_the_influence_length_fails_only_with_a_margin():
    assert boundary_verdict(30000.0, 0.01, 30000.0, 0.05, 5000.0) == (False, "distance")
    assert boundary_verdict(30000.0, 0.01, 30000.0, 0.05, 0.0) == (True, "pass")
    assert boundary_verdict(10000.0, 0.2, 30000.0, 0.05, 0.0) == (
        False,
        "level+distance",
    )


def test_mouth_bed_above_the_sea_starts_at_the_bed():
    net, ids, dist = _network()
    bed, h_n = _uniform_bed(net)
    bed = bed + 2.0 * h_n  # bed at the mouth above the sea
    calm, surge = _levels(net, bed, 0.0), _levels(net, bed, 0.5)
    assert np.isfinite(calm).all()
    assert calm[ids[-1]] == pytest.approx(bed[ids[-1]])
    assert np.all((surge - calm)[ids] >= -1e-9)


def test_high_gradient_rule_marks_the_band_around_the_threshold_marginal():
    assert high_gradient_rule(12.0, 10.0) == "pass"
    assert high_gradient_rule(10.5, 10.0) == "marginal"
    assert high_gradient_rule(8.5, 10.0) == "fail"
