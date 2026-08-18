rule adapt_compare:
    """
    Aggregate baseline/pre/post flood_metrics.csv across every scenario x
    strategy currently in play for this basin (SCENARIOS x STRATEGIES, i.e.
    target_scenarios/target_strategies) into one long-format comparison CSV
    -- basin-scoped, not per-strategy, so it needs `adapt` (both methods)
    to have completed for every scenario/strategy combination first.
    """
    input:
        baseline_metrics = lambda wc: expand(
            results_path("{basin_id}/runs/{scenario}/metrics/flood_metrics.csv"),
            basin_id=wc.basin_id, scenario=SCENARIOS,
        ),
        pre_metrics = lambda wc: expand(
            results_path("{basin_id}/runs/{scenario}/adaptation/pre/{strategy}/flood_metrics.csv"),
            basin_id=wc.basin_id, scenario=SCENARIOS, strategy=STRATEGIES,
        ),
        post_metrics = lambda wc: expand(
            results_path("{basin_id}/runs/{scenario}/adaptation/post/{strategy}/flood_metrics.csv"),
            basin_id=wc.basin_id, scenario=SCENARIOS, strategy=STRATEGIES,
        ),
    output:
        comparison_csv = results_path("{basin_id}/runs/metrics_comparison.csv"),
    log: "logs/{basin_id}/runs/19_adapt_compare.log"
    script: "../scripts/19_adapt_compare.py"
