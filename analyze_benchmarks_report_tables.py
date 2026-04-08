"""Generate compact, report-ready CSV tables from a benchmark results file.

This script is intended for writing the computational-study section of the
report. Unlike `analyze_benchmarks.py`, it focuses on a small number of summary
tables that are easy to cite directly in LaTeX.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from benchmark_utils import (
    ensure_output_layout,
    find_latest_results_csv,
    resolve_io_paths,
    safe_float,
    safe_int,
    write_csv,
)


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
    """Parse command-line options for the report-table generator."""
    parser = argparse.ArgumentParser(
        description="Generate compact report-ready CSV tables from benchmark results."
    )
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
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
    """Group row dictionaries by one field."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def values(rows, field):
    """Collect non-null values for a single field."""
    return [row[field] for row in rows if row.get(field) is not None]


def solved_rows(rows):
    """Return rows solved to optimality."""
    return [row for row in rows if row.get("status_name") == "optimal"]


def summarize_numeric(series):
    """Compute standard descriptive statistics for a numeric series."""
    if not series:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(series),
        "mean": statistics.fmean(series),
        "median": statistics.median(series),
        "min": min(series),
        "max": max(series),
    }


def pearson(x_values, y_values):
    """Compute the Pearson correlation coefficient for two numeric series."""
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


def make_main_summary_table(rows):
    """Build the main formulation-comparison table used in the report."""
    grouped = group_by(rows, "formulation_name")
    output = []
    for formulation, group in sorted(grouped.items()):
        optimal = solved_rows(group)
        runtime = summarize_numeric(values(group, "runtime_sec"))
        root_gap = summarize_numeric(values(optimal, "root_lp_gap"))
        bb_nodes = summarize_numeric(values(group, "bb_nodes"))
        sep_time = summarize_numeric(values(group, "separation_time_sec"))
        sep_share = summarize_numeric(values(group, "separation_share"))
        output.append(
            {
                "formulation_name": formulation,
                "runs": len(group),
                "optimal_runs": sum(1 for row in group if row.get("status_name") == "optimal"),
                "time_limit_runs": sum(1 for row in group if row.get("status_name") == "time_limit"),
                "median_runtime_sec": runtime["median"],
                "mean_runtime_sec": runtime["mean"],
                "median_root_lp_gap": root_gap["median"],
                "mean_root_lp_gap": root_gap["mean"],
                "median_bb_nodes": bb_nodes["median"],
                "mean_bb_nodes": bb_nodes["mean"],
                "median_separation_time_sec": sep_time["median"],
                "mean_separation_time_sec": sep_time["mean"],
                "median_separation_share": sep_share["median"],
                "mean_separation_share": sep_share["mean"],
            }
        )
    return output


def make_instance_winner_table(rows):
    """Build a compact table of how often each formulation is the fastest."""
    winners = defaultdict(int)
    by_instance = group_by(rows, "instance")
    for group in by_instance.values():
        candidates = [row for row in group if row.get("status_name") == "optimal"]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row["runtime_sec"])
        winners[best["formulation_name"]] += 1
    return [
        {"formulation_name": formulation, "instance_wins": wins}
        for formulation, wins in sorted(winners.items())
    ]


def make_pairwise_table(rows):
    """Build the pairwise runtime-win table."""
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


def correlation_for(group, feature, target):
    """Compute one correlation entry for one formulation group."""
    filtered = [row for row in group if row.get(feature) is not None and row.get(target) is not None]
    return {
        "feature": feature,
        "target": target,
        "pearson_correlation": pearson(
            [row[feature] for row in filtered],
            [row[target] for row in filtered],
        ),
        "sample_size": len(filtered),
    }


def make_report_correlation_table(rows):
    """Build a compact correlation table with one representative feature per effect.

    The full correlation dump can be noisy because `n_edges`, `density`, and
    `avg_degree` carry very similar information on this benchmark set. This table
    keeps only one representative feature for the report narrative.
    """
    grouped = group_by(rows, "formulation_name")
    report_specs = [
        ("directed cut", "runtime_sec", "n_edges"),
        ("directed cut", "root_lp_gap", "n_terminals"),
        ("directed flow", "runtime_sec", "n_terminals"),
        ("directed flow", "root_lp_gap", "n_edges"),
        ("undirected flow", "runtime_sec", "n_terminals"),
        ("undirected flow", "root_lp_gap", "n_edges"),
        ("undirected cut", "separation_time_sec", "n_terminals"),
        ("undirected cut", "root_lp_gap", "n_edges"),
    ]

    output = []
    for formulation, target, feature in report_specs:
        group = grouped[formulation]
        result = correlation_for(group, feature, target)
        output.append(
            {
                "formulation_name": formulation,
                "quantity": target,
                "main_feature": feature,
                "pearson_correlation": result["pearson_correlation"],
                "sample_size": result["sample_size"],
                "note": (
                    "Representative of graph size; density and avg_degree are very similar"
                    if feature == "n_edges"
                    else ""
                ),
            }
        )
    return output


def main():
    """Generate all compact CSV tables for the report."""
    args = parse_args()
    _, output_dir = resolve_io_paths(None, args.output_dir)
    layout = ensure_output_layout(output_dir)
    results_csv = Path(args.results) if args.results else find_latest_results_csv(layout["raw_dir"])
    rows = load_rows(results_csv)

    stem = results_csv.stem
    main_summary = make_main_summary_table(rows)
    winners = make_instance_winner_table(rows)
    pairwise = make_pairwise_table(rows)
    report_corr = make_report_correlation_table(rows)

    summary_path = layout["tables_dir"] / f"report_summary_{stem}.csv"
    winners_path = layout["tables_dir"] / f"report_instance_wins_{stem}.csv"
    pairwise_path = layout["tables_dir"] / f"report_pairwise_runtime_wins_{stem}.csv"
    corr_path = layout["tables_dir"] / f"report_instance_dependence_{stem}.csv"

    write_csv(summary_path, main_summary, list(main_summary[0].keys()) if main_summary else [])
    write_csv(winners_path, winners, list(winners[0].keys()) if winners else [])
    write_csv(pairwise_path, pairwise, list(pairwise[0].keys()) if pairwise else [])
    write_csv(corr_path, report_corr, list(report_corr[0].keys()) if report_corr else [])

    print(f"Analyzed {results_csv}")
    print(f"Saved report summary table to {summary_path}")
    print(f"Saved instance-win table to {winners_path}")
    print(f"Saved pairwise runtime table to {pairwise_path}")
    print(f"Saved instance-dependence table to {corr_path}")


if __name__ == "__main__":
    main()
