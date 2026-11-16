import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import plot_dag  # noqa: E402

RULEGRAPH = """digraph snakemake_dag {
    0[label = "all", color = "0.1 0.6 0.85", style="rounded"];
    1[label = "determine_model_domain", color = "0.2 0.6 0.85", style="rounded"];
    2[label = "get_boundary_forcings", color = "0.3 0.6 0.85", style="rounded"];
    3[label = "run_event", color = "0.4 0.6 0.85", style="rounded"];
    1 -> 2
    2 -> 3
    1 -> 3
    3 -> 0
}
"""


def test_rulegraph_parses_rule_names_and_edges():
    ids, edges = plot_dag.parse_rulegraph(RULEGRAPH)
    assert set(ids.values()) == {
        "all",
        "determine_model_domain",
        "get_boundary_forcings",
        "run_event",
    }
    assert ("determine_model_domain", "run_event") in edges
    assert len(edges) == 4


def test_transitive_reduction_drops_only_implied_edges():
    edges = {("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")}
    assert plot_dag.transitive_reduction(edges) == {("a", "b"), ("b", "c"), ("c", "d")}


def test_every_rule_file_number_maps_to_a_stage():
    numbers = plot_dag.rule_numbers()
    assert numbers["get_boundary_forcings"] == "07d"
    assert plot_dag.stage_of("07d") == "Boundary forcing"
    assert plot_dag.stage_of("12c") == "Adaptation"
    stage_names = {s for _, s, _, _ in plot_dag.STAGES}
    assert {plot_dag.stage_of(n) for n in numbers.values()} <= stage_names


def test_styled_rulegraph_clusters_rules_and_marks_sfincs_runs():
    _, edges = plot_dag.parse_rulegraph(RULEGRAPH)
    edges = {(a, b) for a, b in edges if "all" not in (a, b)}
    rules = {"determine_model_domain", "get_boundary_forcings", "run_event"}
    dot = plot_dag.styled_rulegraph(
        rules, plot_dag.transitive_reduction(edges), plot_dag.rule_numbers()
    )
    assert dot.count("subgraph cluster_") == 3
    assert '"determine_model_domain" -> "run_event"' not in dot
    assert "peripheries=2" in dot.split('"run_event" [')[1].split("\n")[0]
