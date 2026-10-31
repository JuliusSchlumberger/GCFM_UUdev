rule adapt_compare:
    """
    Aggregate baseline/pre/post flood_metrics.csv into one long-format CSV per basin.

    Inputs are the SCENARIOS x STRATEGIES runs in play, but the script globs
    every flood_metrics.csv under the basin's runs/, so results of earlier
    scenarios or strategies still on disk are included too.
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
    threads: rule_threads("adapt_compare")
    log: "logs/{basin_id}/runs/18_adapt_compare.log"
    script: "../scripts/18_adapt_compare.py"
