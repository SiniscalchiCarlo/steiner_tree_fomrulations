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
    "lp_relaxation_obj",
    "lp_relaxation_time_sec",
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

PALETTE = {
    "undirected cut": "#1f77b4",
    "undirected flow": "#ff7f0e",
    "directed cut": "#2ca02c",
    "directed flow": "#d62728",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Steiner benchmark results.")
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def load_rows(results_path):
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
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def solved_rows(rows):
    return [row for row in rows if row.get("status_name") == "optimal"]


def completed_rows(rows):
    return [row for row in rows if not row.get("error")]


def values(rows, field):
    return [row[field] for row in rows if row.get(field) is not None]


def summarize_numeric(series):
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
                corr = pearson(
                    [row[feature] for row in filtered],
                    [row[target] for row in filtered],
                )
                output.append(
                    {
                        "formulation_name": formulation,
                        "feature": feature,
                        "target": target,
                        "pearson_correlation": corr,
                        "sample_size": len(filtered),
                    }
                )
    return output


def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def quartiles(series):
    sorted_values = sorted(series)
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0], sorted_values[0], sorted_values[0]
    median = statistics.median(sorted_values)
    lower = sorted_values[: n // 2]
    upper = sorted_values[(n + 1) // 2 :]
    q1 = statistics.median(lower) if lower else sorted_values[0]
    q3 = statistics.median(upper) if upper else sorted_values[-1]
    return q1, median, q3


def transform_value(value, log_scale):
    if value is None:
        return None
    if log_scale:
        return math.log10(max(value, 1e-9))
    return value


def make_boxplot_svg(path, grouped_values, title, y_label, log_scale=False):
    width = 960
    height = 540
    margin_left = 90
    margin_right = 40
    margin_top = 60
    margin_bottom = 110

    data = {label: [value for value in values if value is not None] for label, values in grouped_values.items()}
    transformed = [
        transform_value(value, log_scale)
        for values_list in data.values()
        for value in values_list
    ]
    if not transformed:
        return

    y_min = min(transformed)
    y_max = max(transformed)
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def y_pos(value):
        transformed_value = transform_value(value, log_scale)
        ratio = (transformed_value - y_min) / (y_max - y_min)
        return height - margin_bottom - ratio * plot_height

    labels = list(data.keys())
    step = plot_width / max(len(labels), 1)
    box_width = min(70, step * 0.45)

    elements = [
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22">{escape_xml(title)}</text>',
        f'<text x="25" y="{height / 2}" transform="rotate(-90 25,{height / 2})" text-anchor="middle" font-size="16">{escape_xml(y_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" />',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" />',
    ]

    for index, label in enumerate(labels):
        series = data[label]
        if not series:
            continue
        x_center = margin_left + step * index + step / 2
        q1, median, q3 = quartiles(series)
        low = min(series)
        high = max(series)
        color = PALETTE.get(label, "#444")

        elements.extend(
            [
                f'<line x1="{x_center}" y1="{y_pos(low)}" x2="{x_center}" y2="{y_pos(high)}" stroke="{color}" stroke-width="2" />',
                f'<rect x="{x_center - box_width / 2}" y="{y_pos(q3)}" width="{box_width}" height="{max(y_pos(q1) - y_pos(q3), 1)}" fill="{color}" fill-opacity="0.25" stroke="{color}" />',
                f'<line x1="{x_center - box_width / 2}" y1="{y_pos(median)}" x2="{x_center + box_width / 2}" y2="{y_pos(median)}" stroke="{color}" stroke-width="3" />',
                f'<line x1="{x_center - box_width / 4}" y1="{y_pos(low)}" x2="{x_center + box_width / 4}" y2="{y_pos(low)}" stroke="{color}" stroke-width="2" />',
                f'<line x1="{x_center - box_width / 4}" y1="{y_pos(high)}" x2="{x_center + box_width / 4}" y2="{y_pos(high)}" stroke="{color}" stroke-width="2" />',
                f'<text x="{x_center}" y="{height - margin_bottom + 24}" text-anchor="end" transform="rotate(-35 {x_center},{height - margin_bottom + 24})" font-size="14">{escape_xml(label)}</text>',
            ]
        )

    path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        + "".join(elements)
        + "</svg>\n"
    )


def make_scatter_svg(path, rows, x_field, y_field, title, x_label, y_label, log_y=False):
    width = 960
    height = 540
    margin_left = 90
    margin_right = 180
    margin_top = 60
    margin_bottom = 80

    points = [
        row
        for row in rows
        if row.get(x_field) is not None and row.get(y_field) is not None and not row.get("error")
    ]
    if not points:
        return

    x_values = [row[x_field] for row in points]
    y_values = [transform_value(row[y_field], log_y) for row in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_pos(value):
        ratio = (value - x_min) / (x_max - x_min)
        return margin_left + ratio * plot_width

    def y_pos(value):
        transformed_value = transform_value(value, log_y)
        ratio = (transformed_value - y_min) / (y_max - y_min)
        return height - margin_bottom - ratio * plot_height

    elements = [
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22">{escape_xml(title)}</text>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="16">{escape_xml(x_label)}</text>',
        f'<text x="25" y="{height / 2}" transform="rotate(-90 25,{height / 2})" text-anchor="middle" font-size="16">{escape_xml(y_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" />',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" />',
    ]

    legend_y = margin_top + 10
    for formulation, color in PALETTE.items():
        elements.extend(
            [
                f'<rect x="{width - margin_right + 20}" y="{legend_y - 10}" width="14" height="14" fill="{color}" />',
                f'<text x="{width - margin_right + 42}" y="{legend_y + 2}" font-size="14">{escape_xml(formulation)}</text>',
            ]
        )
        legend_y += 24

    for row in points:
        color = PALETTE.get(row["formulation_name"], "#444")
        elements.append(
            f'<circle cx="{x_pos(row[x_field])}" cy="{y_pos(row[y_field])}" r="4" fill="{color}" fill-opacity="0.65" />'
        )

    path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        + "".join(elements)
        + "</svg>\n"
    )


def build_html_report(report_path, results_csv, summary_rows, plot_paths):
    table_rows = "".join(
        "<tr>"
        f"<td>{escape_xml(row['formulation_name'])}</td>"
        f"<td>{row['runs']}</td>"
        f"<td>{row['optimal_runs']}</td>"
        f"<td>{format_number(row['median_runtime_sec'])}</td>"
        f"<td>{format_number(row['median_root_lp_gap'])}</td>"
        f"<td>{format_number(row['median_bb_nodes'])}</td>"
        f"<td>{format_number(row['median_separation_time_sec'])}</td>"
        "</tr>"
        for row in summary_rows
    )
    images = "".join(
        f"<h2>{escape_xml(path.stem.replace('_', ' ').title())}</h2>"
        f"<img src='../plots/{escape_xml(path.name)}' style='max-width:100%;border:1px solid #ddd' />"
        for path in plot_paths
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Steiner Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.4; }}
    table {{ border-collapse: collapse; width: 100%; margin: 24px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Steiner Formulation Benchmark Report</h1>
  <p>Source results: {escape_xml(results_csv.name)}</p>
  <h2>Formulation Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Formulation</th>
        <th>Runs</th>
        <th>Optimal</th>
        <th>Median Runtime (s)</th>
        <th>Median Root LP Gap</th>
        <th>Median B&amp;B Nodes</th>
        <th>Median Separation Time (s)</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
  {images}
</body>
</html>
"""
    report_path.write_text(html)


def format_number(value):
    if value is None:
        return ""
    return f"{value:.6g}"


def main():
    args = parse_args()
    _, output_dir = resolve_io_paths(None, args.output_dir)
    layout = ensure_output_layout(output_dir)
    results_csv = Path(args.results) if args.results else find_latest_results_csv(layout["raw_dir"])
    rows = load_rows(results_csv)

    summary_rows = formulation_summary(rows)
    pairwise_rows = pairwise_runtime_wins(rows)
    winner_rows = instance_winners(rows)
    correlation_rows = feature_correlations(rows)

    summary_csv = layout["tables_dir"] / f"summary_{results_csv.stem}.csv"
    pairwise_csv = layout["tables_dir"] / f"pairwise_runtime_wins_{results_csv.stem}.csv"
    winners_csv = layout["tables_dir"] / f"instance_winners_{results_csv.stem}.csv"
    correlations_csv = layout["tables_dir"] / f"feature_correlations_{results_csv.stem}.csv"

    write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()) if summary_rows else [])
    write_csv(pairwise_csv, pairwise_rows, list(pairwise_rows[0].keys()) if pairwise_rows else [])
    write_csv(winners_csv, winner_rows, list(winner_rows[0].keys()) if winner_rows else [])
    write_csv(
        correlations_csv,
        correlation_rows,
        list(correlation_rows[0].keys()) if correlation_rows else [],
    )

    grouped = group_by(rows, "formulation_name")
    plot_paths = []

    boxplot_specs = [
        ("runtime_boxplot.svg", "runtime_sec", "Runtime by formulation", "Runtime (s)", True),
        ("root_gap_boxplot.svg", "root_lp_gap", "Root LP gap by formulation", "Root LP gap", False),
        ("bb_nodes_boxplot.svg", "bb_nodes", "Branch-and-bound nodes by formulation", "B&B nodes", True),
        (
            "separation_share_boxplot.svg",
            "separation_share",
            "Separation share by formulation",
            "Separation time / runtime",
            False,
        ),
    ]

    for filename, field, title, label, log_scale in boxplot_specs:
        grouped_values = {name: values(group, field) for name, group in grouped.items()}
        path = layout["plots_dir"] / filename
        make_boxplot_svg(path, grouped_values, title, label, log_scale=log_scale)
        plot_paths.append(path)

    scatter_specs = [
        (
            "runtime_vs_edges.svg",
            "n_edges",
            "runtime_sec",
            "Runtime vs edges",
            "Number of edges",
            "Runtime (s)",
            True,
        ),
        (
            "runtime_vs_terminals.svg",
            "n_terminals",
            "runtime_sec",
            "Runtime vs terminals",
            "Number of terminals",
            "Runtime (s)",
            True,
        ),
        (
            "root_gap_vs_runtime.svg",
            "runtime_sec",
            "root_lp_gap",
            "Root LP gap vs runtime",
            "Runtime (s)",
            "Root LP gap",
            False,
        ),
        (
            "separation_vs_runtime.svg",
            "runtime_sec",
            "separation_time_sec",
            "Separation time vs runtime",
            "Runtime (s)",
            "Separation time (s)",
            True,
        ),
    ]

    for filename, x_field, y_field, title, x_label, y_label, log_y in scatter_specs:
        path = layout["plots_dir"] / filename
        make_scatter_svg(path, rows, x_field, y_field, title, x_label, y_label, log_y=log_y)
        plot_paths.append(path)

    report_path = layout["reports_dir"] / f"report_{results_csv.stem}.html"
    build_html_report(report_path, results_csv, summary_rows, plot_paths)

    print(f"Analyzed {results_csv}")
    print(f"Saved summary table to {summary_csv}")
    print(f"Saved pairwise table to {pairwise_csv}")
    print(f"Saved winner table to {winners_csv}")
    print(f"Saved correlation table to {correlations_csv}")
    print(f"Saved HTML report to {report_path}")


if __name__ == "__main__":
    main()
