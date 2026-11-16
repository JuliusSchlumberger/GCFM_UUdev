"""Render Snakemake rule graphs and job DAGs for pipeline targets to figs/dag/.

Run from the project root (`python plot_dag.py`); needs Graphviz `dot`
(conda install graphviz -c conda-forge). Writes <target>_rulegraph.<fmt> (rules
grouped by pipeline stage, indirect edges dropped, with a .txt caption) and
<target>_dag.<fmt> (one node per job; large for many basins).
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

RULES_DIR = Path("workflow/rules")

# (first rule-file number, label, cluster fill, node border); a rule belongs to
# the last stage whose first number is <= its file's number.
STAGES = [
    ("01", "Domain and input data", "#EEF2F6", "#5B7083"),
    ("07", "Boundary forcing", "#E4F1FB", "#2F7EB8"),
    ("08", "River bed and dike crests", "#E6F4E4", "#3E8E41"),
    ("12", "Model set-up", "#FFF4D9", "#B7860B"),
    ("13", "Simulation", "#FDE8DC", "#C4581F"),
    ("16", "Flood metrics", "#F1E6F5", "#7B4B94"),
    ("17", "Adaptation", "#FCE4EC", "#B0386A"),
    ("19", "Analysis", "#EFEFEF", "#666666"),
]
# 12c only serves the adaptation strategies.
STAGE_OVERRIDES = {"12c": "Adaptation"}
SFINCS_RULES = {
    "modelled_depth_estimation",
    "run_spinup",
    "run_event",
    "adapt_run_event_pre",
    "suite_depth_run",
}

NODE_RE = re.compile(r'^\s*(\d+)\s*\[\s*label\s*=\s*"([^"]+)"')
EDGE_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)")


def parse_rulegraph(dot: str) -> tuple[dict[str, str], set[tuple[str, str]]]:
    """Return {node id: rule name} and the (upstream, downstream) rule edges of a rule graph."""
    ids, edges = {}, set()
    for line in dot.splitlines():
        if m := NODE_RE.match(line):
            ids[m.group(1)] = m.group(2)
        elif m := EDGE_RE.match(line):
            edges.add((m.group(1), m.group(2)))
    return ids, {(ids[a], ids[b]) for a, b in edges}


def transitive_reduction(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Drop each edge a -> c that another path a -> ... -> c already implies (acyclic graph)."""
    succ: dict[str, set[str]] = {}
    for a, b in edges:
        succ.setdefault(a, set()).add(b)

    def reachable_without(start: str, skip: str) -> set[str]:
        seen, stack = set(), [n for n in succ.get(start, ()) if n != skip]
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(succ.get(n, ()))
        return seen

    return {(a, b) for a, b in edges if b not in reachable_without(a, b)}


def rule_numbers(rules_dir: Path = RULES_DIR) -> dict[str, str]:
    """Return {rule name: rule-file number} (e.g. '07d') from workflow/rules/*.smk."""
    out = {}
    for smk in sorted(rules_dir.glob("*.smk")):
        number = smk.name.split("_", 1)[0]
        for name in re.findall(
            r"^rule\s+(\w+)\s*:", smk.read_text(encoding="utf-8"), re.M
        ):
            out[name] = number
    return out


def stage_of(number: str) -> str:
    """Return the stage label of a rule-file number."""
    if number in STAGE_OVERRIDES:
        return STAGE_OVERRIDES[number]
    label = STAGES[0][1]
    for first, name, _, _ in STAGES:
        if number[:2] >= first:
            label = name
    return label


def styled_rulegraph(
    rules: set[str],
    edges: set[tuple[str, str]],
    numbers: dict[str, str],
    second_column: str | None = None,
) -> str:
    """Return DOT source: rules clustered by stage, numbered labels, SFINCS runs outlined twice.

    second_column: the stage that starts a second column; its top rule sits level
    with the first rule, arrows into it from the first column do not set rank, and
    all arrows are drawn with right angles.
    """
    style = {name: (fill, border) for _, name, fill, border in STAGES}
    order = [name for _, name, _, _ in STAGES]
    col2 = set()
    if second_column is not None:
        start = order.index(second_column)
        col2 = {
            r for r in rules if order.index(stage_of(numbers.get(r, "99"))) >= start
        }
    by_stage: dict[str, list[str]] = {}
    for rule in sorted(rules, key=lambda r: (numbers.get(r, "99"), r)):
        by_stage.setdefault(stage_of(numbers.get(rule, "99")), []).append(rule)
    lines = [
        "digraph rulegraph {",
        '  graph [rankdir=TB, newrank=true, nodesep=0.25, ranksep=0.4, fontname="Helvetica", '
        'fontsize=13, bgcolor="white", pad=0.2];',
        '  node [shape=box, style="rounded,filled", fillcolor="white", fontname="Helvetica", '
        'fontsize=11, penwidth=1.4, margin="0.12,0.05"];',
        '  edge [color="#6B6B6B", arrowsize=0.6, penwidth=1.0];',
    ]
    for k, (stage, members) in enumerate(
        sorted(
            by_stage.items(), key=lambda kv: min(numbers.get(r, "99") for r in kv[1])
        )
    ):
        fill, border = style[stage]
        lines += [
            f"  subgraph cluster_{k} {{",
            f'    label="{stage}"; labeljust="l"; style="rounded,filled"; fillcolor="{fill}"; '
            f'color="{border}"; fontcolor="{border}";',
        ]
        for rule in members:
            extra = ", peripheries=2" if rule in SFINCS_RULES else ""
            label = f'<<FONT POINT-SIZE="9" COLOR="{border}">{numbers.get(rule, "")}</FONT> {rule}>'
            lines.append(f'    "{rule}" [label={label}, color="{border}"{extra}];')
        lines.append("  }")
    for a, b in sorted(edges):
        if a not in col2 and b in col2:
            # written downward with the head at its tail: with orthogonal splines dot
            # then runs it up the gap between the columns
            lines.append(f'  "{b}" -> "{a}" [dir=back, constraint=false];')
        else:
            lines.append(f'  "{a}" -> "{b}";')
    if col2 and col2 != rules:
        has_pred = {b for _, b in edges}
        has_col2_pred = {b for a, b in edges if a in col2}
        level = sorted(r for r in rules - col2 if r not in has_pred)[:1]
        level += sorted(r for r in col2 if r not in has_col2_pred)
        lines.append("  {rank=same; " + " ".join(f'"{r}";' for r in level) + "}")
        lines.insert(1, "  graph [splines=ortho];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def caption(
    target: str, stages: list[str], n_rules: int, n_edges: int, n_dropped: int
) -> str:
    """Return the figure description written next to a styled rule graph."""
    return (
        f"Snakemake rule graph of target `{target}`: {n_rules} rules, {n_edges} dependencies. "
        "Each box is a rule, labelled with its rule-file number (workflow/rules/NN_*.smk) and "
        "name; an arrow points from a rule to a rule that reads its output. Boxes are grouped "
        f"into pipeline stages (shaded panels: {', '.join(stages)}). A double outline marks "
        "rules that run SFINCS. Arrows implied by a longer path are left out "
        f"({n_dropped} of {n_edges + n_dropped}), so an arrow means 'directly after', not "
        "'only input'; the target rule itself is not drawn."
    )


def snakemake_graph(target: str, graph_flag: str, config: list[str]) -> str | None:
    """Return Snakemake's DOT output for target, or None when Snakemake fails."""
    # --forceall shows every rule even when outputs exist; --cores is required by 00_common.
    cmd = ["snakemake", target, graph_flag, "--forceall", "--quiet", "--cores", "1"]
    if config:
        cmd += ["--config", *config]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [warning] snakemake exited with code {result.returncode}:")
        print("  " + "\n  ".join(result.stderr.strip().splitlines()[-5:]))
        return None
    return result.stdout


def render(dot: str, out_file: Path, fmt: str) -> bool:
    """Render DOT source to out_file with Graphviz; return success."""
    result = subprocess.run(
        ["dot", f"-T{fmt}", "-o", str(out_file)],
        input=dot,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [warning] dot rendering failed:\n{result.stderr.strip()}")
        return False
    print(f"  Written: {out_file}")
    return True


def main() -> None:
    """Render the graphs for every requested target."""
    parser = argparse.ArgumentParser(
        description="Plot Snakemake DAGs for pipeline targets."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["preprocess", "build"],
        metavar="TARGET",
        help="Snakemake targets to visualise (default: preprocess build)",
    )
    parser.add_argument(
        "--out-dir",
        default="figs/dag",
        metavar="DIR",
        help="Output directory (default: figs/dag)",
    )
    parser.add_argument(
        "--format",
        default="svg",
        choices=["svg", "png", "pdf"],
        metavar="FMT",
        help="Output format: svg, png, or pdf (default: svg)",
    )
    parser.add_argument(
        "--no-dag",
        action="store_true",
        help="Skip the full per-job DAG (only generate rulegraph)",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=1,
        choices=[1, 2],
        help="Rule-graph columns; 2 starts a second column at the Model set-up stage (default: 1)",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Passed to snakemake --config, e.g. target_strategies=\"['advance_1']\"",
    )
    args = parser.parse_args()

    if not shutil.which("snakemake"):
        sys.exit("Error: 'snakemake' not found. Activate your conda environment first.")
    if not shutil.which("dot"):
        sys.exit(
            "Error: Graphviz 'dot' not found.\nInstall it with:  conda install graphviz -c conda-forge"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    numbers = rule_numbers()

    failed: list[str] = []
    for target in args.targets:
        print(f"\nTarget: {target}")
        raw = snakemake_graph(target, "--rulegraph", args.config)
        out_file = out_dir / f"{target}_rulegraph.{args.format}"
        if raw is not None:
            ids, edges = parse_rulegraph(raw)
            rules = set(ids.values()) - {target}
            edges = {(a, b) for a, b in edges if a != target and b != target}
            reduced = transitive_reduction(edges)
            dot = styled_rulegraph(
                rules, reduced, numbers, "Model set-up" if args.columns == 2 else None
            )
            if render(dot, out_file, args.format):
                stages = [
                    s
                    for _, s, _, _ in STAGES
                    if any(stage_of(numbers.get(r, "99")) == s for r in rules)
                ]
                out_file.with_suffix(".txt").write_text(
                    caption(
                        target,
                        stages,
                        len(rules),
                        len(reduced),
                        len(edges) - len(reduced),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                continue
        failed.append(f"{target} rulegraph")
    for target in [] if args.no_dag else args.targets:
        raw = snakemake_graph(target, "--dag", args.config)
        if raw is None or not render(
            raw, out_dir / f"{target}_dag.{args.format}", args.format
        ):
            failed.append(f"{target} dag")

    if failed:
        sys.exit(f"\nFailed: {', '.join(failed)}")
    print(f"\nDone. Open the .{args.format} files in {out_dir}/")


if __name__ == "__main__":
    main()
