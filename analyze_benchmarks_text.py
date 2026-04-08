"""Print a text-only analysis of Steiner benchmark results.

This script is a lightweight alternative to `analyze_benchmarks.py`.
It does not generate plots or an HTML report. Instead, it prints the main
aggregates and comparisons so the person running it can interpret the output
and build their own analysis manually.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from benchmark_utils import find_latest_results_csv, resolve_io_paths, safe_float, safe_int


NUMERIC_FLOAT_FIELDS = {
    "objective",
    "best_bound",
    "mip_gap",
    "runtime_sec",
    "model_runtime_sec",
    "root_lp_bound",
    "root_lp_gap",
    "bb_nodes",
    "simplex_iterations",
    "barrier_iterations",
    "separation_time_sec",
    "separation_share",
    "density",
    "avg_degree",
    "terminal_ratio",
    "time_limit_sec",
}

NUMERIC_INT_FIELDS = {
    "status_code",
    "sol_count",
    "root_lp_callback_calls",
    "num_vars",
    "num_constrs",
    "mipnode_calls",
    "mipsol_calls",
    "separation_calls",
    "lazy_cuts_added",
    "n_nodes",
    "n_edges",
    "n_terminals",
    "threads",
    "cuts",
    "seed",
}


def parse_args():
    """Parse command-line options for the text-only analysis script."""
    parser = argparse.ArgumentParser(
        description="Print a text-only analysis of Steiner benchmark results."
    )
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many rows to show in ranked sections.",
    )
    return parser.parse_args()


def load_rows(results_path):
    """Load the raw benchmark CSV and restore numeric and boolean types."""
    rows = []
    with Path(results_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            for field in NUMERIC_FLOAT_FIELDS:
                if field in parsed:
                    parsed[field] = safe_float(parsed[field])
            for field in NUMERIC_INT_FIELDS:
                if field in parsed:
                    parsed[field] = safe_int(parsed[field])
            parsed["uses_separation"] = parsed.get("uses_separation") in ("True", "true", True)
            rows.append(parsed)
    return rows


def group_by(rows, key):
    """Group row dictionaries by a single field name."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def values(rows, field):
    """Collect non-null values for one field from a row sequence."""
    return [row[field] for row in rows if row.get(field) is not None]


def solved_rows(rows):
    """Return only rows with optimal solver status."""
    return [row for row in rows if row.get("status_name") == "optimal"]


def completed_rows(rows):
    """Return rows that did not fail with a Python-side error."""
    return [row for row in rows if not row.get("error")]


def summarize_numeric(series):
    """Compute descriptive statistics for a numeric series."""
    if not series:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stdev": None,
        }
    return {
        "count": len(series),
        "mean": statistics.fmean(series),
        "median": statistics.median(series),
        "min": min(series),
        "max": max(series),
        "stdev": statistics.stdev(series) if len(series) > 1 else 0.0,
    }


def pearson(x_values, y_values):
    """Compute the Pearson correlation coefficient for two numeric lists."""
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return None
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def formulation_summary(rows):
    """Build one aggregate summary row per formulation."""
    grouped = group_by(rows, "formulation_name")
    summary_rows = []
    for formulation, group in sorted(grouped.items()):
        all_runtime = summarize_numeric(values(group, "runtime_sec"))
        optimal = solved_rows(group)
        root_gap = summarize_numeric(values(optimal, "root_lp_gap"))
        bb_nodes = summarize_numeric(values(group, "bb_nodes"))
        sep_time = summarize_numeric(values(group, "separation_time_sec"))
        sep_share = summarize_numeric(values(group, "separation_share"))
        summary_rows.append(
            {
                "formulation_name": formulation,
                "runs": len(group),
                "optimal_runs": sum(1 for row in group if row.get("status_name") == "optimal"),
                "time_limit_runs": sum(1 for row in group if row.get("status_name") == "time_limit"),
                "errored_runs": sum(1 for row in group if row.get("error")),
                "mean_runtime_sec": all_runtime["mean"],
                "median_runtime_sec": all_runtime["median"],
                "min_runtime_sec": all_runtime["min"],
                "max_runtime_sec": all_runtime["max"],
                "mean_root_lp_gap": root_gap["mean"],
                "median_root_lp_gap": root_gap["median"],
                "mean_bb_nodes": bb_nodes["mean"],
                "median_bb_nodes": bb_nodes["median"],
                "mean_separation_time_sec": sep_time["mean"],
                "median_separation_time_sec": sep_time["median"],
                "mean_separation_share": sep_share["mean"],
                "median_separation_share": sep_share["median"],
            }
        )
    return summary_rows


def pairwise_runtime_wins(rows):
    """Compare each pair of formulations instance by instance on runtime."""
    by_instance = group_by(rows, "instance")
    formulations = sorted({row["formulation_name"] for row in rows})
    output = []
    for left in formulations:
        for right in formulations:
            if left == right:
                continue
            left_wins = 0
            right_wins = 0
            ties = 0
            comparable = 0
            for instance_rows in by_instance.values():
                lookup = {row["formulation_name"]: row for row in instance_rows}
                if left not in lookup or right not in lookup:
                    continue
                left_row = lookup[left]
                right_row = lookup[right]
                if left_row.get("runtime_sec") is None or right_row.get("runtime_sec") is None:
                    continue
                comparable += 1
                if left_row["runtime_sec"] < right_row["runtime_sec"]:
                    left_wins += 1
                elif right_row["runtime_sec"] < left_row["runtime_sec"]:
                    right_wins += 1
                else:
                    ties += 1
            output.append(
                {
                    "left_formulation": left,
                    "right_formulation": right,
                    "comparable_instances": comparable,
                    "left_wins": left_wins,
                    "right_wins": right_wins,
                    "ties": ties,
                }
            )
    return output


def instance_winners(rows):
    """Identify the fastest optimal formulation for each instance."""
    winners = []
    by_instance = group_by(rows, "instance")
    for instance, group in sorted(by_instance.items()):
        candidates = [row for row in group if row.get("status_name") == "optimal"]
        if not candidates:
            winner = None
            runtime = None
        else:
            best = min(candidates, key=lambda row: row["runtime_sec"])
            winner = best["formulation_name"]
            runtime = best["runtime_sec"]
        winners.append(
            {
                "instance": instance,
                "winner": winner,
                "winner_runtime_sec": runtime,
            }
        )
    return winners


def feature_correlations(rows):
    """Correlate instance features with selected performance metrics."""
    features = ["n_edges", "n_terminals", "density", "avg_degree", "terminal_ratio"]
    targets = ["runtime_sec", "root_lp_gap", "bb_nodes", "separation_time_sec"]
    grouped = group_by(rows, "formulation_name")
    output = []
    for formulation, group in sorted(grouped.items()):
        for feature in features:
            for target in targets:
                filtered = [
                    row for row in group if row.get(feature) is not None and row.get(target) is not None
                ]
                output.append(
                    {
                        "formulation_name": formulation,
                        "feature": feature,
                        "target": target,
                        "pearson_correlation": pearson(
                            [row[feature] for row in filtered],
                            [row[target] for row in filtered],
                        ),
                        "sample_size": len(filtered),
                    }
                )
    return output


def format_number(value):
    """Format optional numeric values compactly for terminal output."""
    if value is None:
        return "-"
    return f"{value:.6g}"


def print_section(title):
    """Print a section header."""
    print()
    print(title)
    print("=" * len(title))


def print_key_value(label, value):
    """Print one labeled value line."""
    print(f"{label}: {value}")


def print_formulation_summary(summary_rows):
    """Print one compact line per formulation."""
    print_section("Formulation Summary")
    for row in summary_rows:
        print(
            f"{row['formulation_name']}: "
            f"runs={row['runs']}, "
            f"optimal={row['optimal_runs']}, "
            f"time_limit={row['time_limit_runs']}, "
            f"errors={row['errored_runs']}, "
            f"median_runtime={format_number(row['median_runtime_sec'])} s, "
            f"median_root_gap={format_number(row['median_root_lp_gap'])}, "
            f"median_bb_nodes={format_number(row['median_bb_nodes'])}, "
            f"median_sep_time={format_number(row['median_separation_time_sec'])} s, "
            f"median_sep_share={format_number(row['median_separation_share'])}"
        )


def print_pairwise_wins(rows):
    """Print pairwise runtime win/loss counts."""
    print_section("Pairwise Runtime Wins")
    for row in rows:
        print(
            f"{row['left_formulation']} vs {row['right_formulation']}: "
            f"comparable={row['comparable_instances']}, "
            f"left_wins={row['left_wins']}, "
            f"right_wins={row['right_wins']}, "
            f"ties={row['ties']}"
        )


def print_winner_summary(winner_rows):
    """Print how often each formulation is the fastest optimal solver."""
    print_section("Instance Winners")
    counts = defaultdict(int)
    unsolved = 0
    for row in winner_rows:
        if row["winner"] is None:
            unsolved += 1
        else:
            counts[row["winner"]] += 1
    for formulation, count in sorted(counts.items()):
        print(f"{formulation}: {count} wins")
    print(f"no optimal winner: {unsolved}")


def print_top_instances(rows, top_n):
    """Print the slowest solved runs and the runs with the largest root gaps."""
    solved = [row for row in rows if row.get("status_name") == "optimal"]

    print_section(f"Top {top_n} Slowest Solved Runs")
    for row in sorted(
        solved,
        key=lambda row: row["runtime_sec"] if row.get("runtime_sec") is not None else -1,
        reverse=True,
    )[:top_n]:
        print(
            f"{row['instance']} | {row['formulation_name']} | "
            f"runtime={format_number(row['runtime_sec'])} s | "
            f"root_gap={format_number(row.get('root_lp_gap'))} | "
            f"bb_nodes={format_number(row.get('bb_nodes'))}"
        )

    print_section(f"Top {top_n} Largest Root LP Gaps")
    with_gap = [row for row in solved if row.get("root_lp_gap") is not None]
    for row in sorted(with_gap, key=lambda row: row["root_lp_gap"], reverse=True)[:top_n]:
        print(
            f"{row['instance']} | {row['formulation_name']} | "
            f"root_gap={format_number(row['root_lp_gap'])} | "
            f"runtime={format_number(row['runtime_sec'])} s | "
            f"bb_nodes={format_number(row.get('bb_nodes'))}"
        )


def print_correlations(rows, top_n):
    """Print the strongest absolute correlations for each formulation."""
    print_section(f"Top {top_n} Correlations By Formulation")
    grouped = group_by(rows, "formulation_name")
    for formulation, group in sorted(grouped.items()):
        print(formulation)
        ranked = sorted(
            group,
            key=lambda row: abs(row["pearson_correlation"]) if row["pearson_correlation"] is not None else -1,
            reverse=True,
        )
        for row in ranked[:top_n]:
            print(
                f"  {row['feature']} vs {row['target']}: "
                f"corr={format_number(row['pearson_correlation'])}, "
                f"n={row['sample_size']}"
            )


def print_errors(rows, top_n):
    """Print the first few failed runs, if any exist."""
    failures = [row for row in rows if row.get("error")]
    if not failures:
        return
    print_section(f"First {min(top_n, len(failures))} Failures")
    for row in failures[:top_n]:
        print(f"{row['instance']} | {row['formulation_name']} | {row['error']}")


def main():
    """Run the text-only benchmark analysis."""
    args = parse_args()
    _, output_dir = resolve_io_paths(None, args.output_dir)
    raw_dir = output_dir / "raw"
    results_csv = Path(args.results) if args.results else find_latest_results_csv(raw_dir)
    rows = load_rows(results_csv)

    summary_rows = formulation_summary(rows)
    pairwise_rows = pairwise_runtime_wins(rows)
    winner_rows = instance_winners(rows)
    correlation_rows = feature_correlations(rows)

    print_section("Run Overview")
    print_key_value("results_file", results_csv)
    print_key_value("rows", len(rows))
    print_key_value("completed_rows", len(completed_rows(rows)))
    print_key_value("optimal_rows", len(solved_rows(rows)))
    print_key_value("formulations", ", ".join(sorted({row['formulation_name'] for row in rows})))
    print_key_value("instances", len({row["instance"] for row in rows}))

    print_formulation_summary(summary_rows)
    print_pairwise_wins(pairwise_rows)
    print_winner_summary(winner_rows)
    print_top_instances(rows, args.top)
    print_correlations(correlation_rows, args.top)
    print_errors(rows, args.top)


if __name__ == "__main__":
    main()
