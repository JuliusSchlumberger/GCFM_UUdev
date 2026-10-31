"""Unit tests for src.river_network.accumulate_discharge."""

import logging

import numpy as np
import pytest
from src.river_network import accumulate_discharge, build_downstream_adjacency


def _accumulate(net, seeds):
    q = accumulate_discharge(net, seeds, build_downstream_adjacency(net))
    return dict(zip(net["reach_id"], q))


def test_chain_carries_seed_unchanged(network):
    net = network({"1": {"dn": ["2"]}, "2": {"dn": ["3"]}, "3": {}})
    q = _accumulate(net, {"1": 10.0})
    assert q == pytest.approx({"1": 10.0, "2": 10.0, "3": 10.0})


def test_confluence_sums(network):
    net = network({"1": {"dn": ["3"]}, "2": {"dn": ["3"]}, "3": {"dn": ["4"]}, "4": {}})
    q = _accumulate(net, {"1": 10.0, "2": 5.0})
    assert q["3"] == pytest.approx(15.0)
    assert q["4"] == pytest.approx(15.0)


def test_bifurcation_splits_by_width(network):
    net = network(
        {"1": {"dn": ["2", "3"]}, "2": {"width": 300.0}, "3": {"width": 100.0}}
    )
    q = _accumulate(net, {"1": 100.0})
    assert q["2"] == pytest.approx(75.0)
    assert q["3"] == pytest.approx(25.0)


def test_bifurcation_with_zero_widths_splits_equally(network):
    net = network({"1": {"dn": ["2", "3"]}, "2": {"width": 0.0}, "3": {"width": 0.0}})
    q = _accumulate(net, {"1": 100.0})
    assert q["2"] == pytest.approx(50.0)
    assert q["3"] == pytest.approx(50.0)


def test_bifurcate_then_rejoin_conserves_mass(network):
    net = network(
        {
            "1": {"dn": ["2", "3"]},
            "2": {"dn": ["4"], "width": 200.0},
            "3": {"dn": ["4"], "width": 50.0},
            "4": {"dn": ["5"]},
            "5": {},
        }
    )
    q = _accumulate(net, {"1": 100.0})
    assert q["2"] + q["3"] == pytest.approx(100.0)
    assert q["4"] == pytest.approx(100.0)
    assert q["5"] == pytest.approx(100.0)


def test_seed_with_upstream_inflow_adds_its_value_once(network, caplog):
    """The 1248635 Drecht case: a seed reach that is also fed from upstream."""
    net = network({"1": {"dn": ["2"]}, "2": {"dn": ["3"]}, "3": {}})
    with caplog.at_level(logging.WARNING):
        q = _accumulate(net, {"1": 1000.0, "2": 245.0})
    assert q["2"] == pytest.approx(1245.0)
    assert q["3"] == pytest.approx(1245.0)
    assert any("seed reach 2 also has upstream reaches" in m for m in caplog.messages)


def test_reaches_not_downstream_of_a_seed_stay_zero(network):
    net = network({"1": {"dn": ["2"]}, "2": {}, "3": {"dn": ["4"]}, "4": {}})
    q = _accumulate(net, {"1": 10.0})
    assert q["3"] == 0.0
    assert q["4"] == 0.0


def test_long_chain_beyond_old_iteration_limit(network):
    """A chain far longer than 100 reaches still accumulates without a hop limit."""
    n = 250
    net = network(
        {str(i): {"dn": [str(i + 1)] if i < n else []} for i in range(1, n + 1)}
    )
    q = _accumulate(net, {"1": 7.0})
    assert q[str(n)] == pytest.approx(7.0)


def test_mass_conservation_on_random_dag(network):
    rng = np.random.default_rng(0)
    n = 60
    reaches = {}
    for i in range(1, n + 1):
        later = list(range(i + 1, n + 1))
        k = int(rng.integers(0, 3)) if later else 0
        dn = (
            sorted(rng.choice(later, size=min(k, len(later)), replace=False).tolist())
            if k
            else []
        )
        reaches[str(i)] = {
            "dn": [str(d) for d in dn],
            "width": float(rng.uniform(20, 500)),
        }
    net = network(reaches)
    seeds = {str(i): float(rng.uniform(1, 100)) for i in (1, 2, 3, 10, 20)}
    q = _accumulate(net, seeds)
    ends = [r for r, s in reaches.items() if not s["dn"]]
    assert sum(q[e] for e in ends) == pytest.approx(sum(seeds.values()))


def test_directed_cycle_is_reported_not_fatal(network, caplog):
    net = network({"1": {"dn": ["2"]}, "2": {"dn": ["3"]}, "3": {"dn": ["2"]}})
    with caplog.at_level(logging.WARNING):
        q = _accumulate(net, {"1": 10.0})
    assert np.isfinite(list(q.values())).all()
    assert any("directed cycle" in m for m in caplog.messages)
