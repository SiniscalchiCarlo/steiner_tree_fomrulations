"""Analyze raw benchmark outputs and generate tables, plots, and a report."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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

PALETTE = {
    "undirected cut": "#1f77b4",
    "undirected flow": "#ff7f0e",
    "directed cut": "#2ca02c",
    "directed flow": "#d62728",
}


def parse_args():
    """Parse command-line options for the analysis step."""
    parser = argparse.ArgumentParser(description="Analyze Steiner benchmark results.")
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
    """Group row dictionaries by a single field name."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def solved_rows(rows):
    """Return only rows that ended with an optimal solver status."""
    return [row for row in rows if row.get("status_name") == "optimal"]


def completed_rows(rows):
    """Return only rows that did not fail with a Python-side error."""
    return [row for row in rows if not row.get("error")]


def values(rows, field):
    """Collect non-null values of one field from a row sequence."""
    return [row[field] for row in rows if row.get(field) is not None]


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
    """Find the fastest optimal formulation for each instance."""
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
    """Escape text before inserting it into generated SVG or HTML."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def quartiles(series):
    """Return `(q1, median, q3)` for boxplot rendering."""
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
    """Apply the optional log transform used by selected plots."""
    if value is None:
        return None
    if log_scale:
        return math.log10(max(value, 1e-9))
    return value


def _load_font(size):
    """Load a reasonable font, falling back to Pillow's default font."""
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw, text, font):
    """Return width and height of a text string for the given font."""
    left, top, right, bottom = draw.textbbox((0, 0), str(text), font=font)
    return right - left, bottom - top


def _draw_centered_text(draw, xy, text, font, fill):
    """Draw text centered on a point."""
    width, height = _text_size(draw, text, font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), str(text), font=font, fill=fill)


def _draw_rotated_text(image, xy, text, font, fill, angle):
    """Draw rotated text onto an image using a temporary transparent layer."""
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    dummy_draw = ImageDraw.Draw(dummy)
    width, height = _text_size(dummy_draw, text, font)
    text_image = Image.new("RGBA", (width + 8, height + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4, 4), str(text), font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True)
    image.alpha_composite(rotated, (int(xy[0] - rotated.width / 2), int(xy[1] - rotated.height / 2)))


def _hex_to_rgba(color, alpha=255):
    """Convert a hex color string to an RGBA tuple."""
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4)) + (alpha,)


def make_boxplot_png(path, grouped_values, title, y_label, log_scale=False):
    """Render a boxplot directly as PNG."""
    width = 1100
    height = 640
    margin_left = 130
    margin_right = 60
    margin_top = 70
    margin_bottom = 170

    # Missing values are dropped before plotting so each series is numeric only.
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

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(22)
    label_font = _load_font(16)
    tick_font = _load_font(14)

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def y_pos(value):
        transformed_value = transform_value(value, log_scale)
        ratio = (transformed_value - y_min) / (y_max - y_min)
        return height - margin_bottom - ratio * plot_height

    labels = list(data.keys())
    step = plot_width / max(len(labels), 1)
    box_width = min(70, step * 0.45)

    _draw_centered_text(draw, (width / 2, 30), title, title_font, "#000000")
    _draw_rotated_text(image, (40, height / 2), y_label, label_font, "#000000", 90)
    draw.line(
        [(margin_left, margin_top), (margin_left, height - margin_bottom)],
        fill="#333333",
        width=2,
    )
    draw.line(
        [(margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom)],
        fill="#333333",
        width=2,
    )

    # Add a small set of y-axis ticks so the boxplot values are readable when
    # the image is embedded in LaTeX or viewed standalone.
    for index in range(5):
        ratio = index / 4
        transformed_value = y_min + ratio * (y_max - y_min)
        y = height - margin_bottom - ratio * plot_height
        draw.line([(margin_left - 6, y), (margin_left, y)], fill="#333333", width=1)
        label_value = (10 ** transformed_value) if log_scale else transformed_value
        y_text = format_number(label_value)
        text_width, text_height = _text_size(draw, y_text, tick_font)
        draw.text(
            (margin_left - 14 - text_width, y - text_height / 2),
            y_text,
            font=tick_font,
            fill="#000000",
        )

    for index, label in enumerate(labels):
        series = data[label]
        if not series:
            continue
        x_center = margin_left + step * index + step / 2
        q1, median, q3 = quartiles(series)
        low = min(series)
        high = max(series)
        color = PALETTE.get(label, "#444")
        outline = _hex_to_rgba(color, 255)
        fill = _hex_to_rgba(color, 64)

        draw.line([(x_center, y_pos(low)), (x_center, y_pos(high))], fill=outline, width=2)
        draw.rectangle(
            [
                (x_center - box_width / 2, y_pos(q3)),
                (x_center + box_width / 2, y_pos(q1)),
            ],
            fill=fill,
            outline=outline,
            width=2,
        )
        draw.line(
            [(x_center - box_width / 2, y_pos(median)), (x_center + box_width / 2, y_pos(median))],
            fill=outline,
            width=3,
        )
        draw.line(
            [(x_center - box_width / 4, y_pos(low)), (x_center + box_width / 4, y_pos(low))],
            fill=outline,
            width=2,
        )
        draw.line(
            [(x_center - box_width / 4, y_pos(high)), (x_center + box_width / 4, y_pos(high))],
            fill=outline,
            width=2,
        )
        _draw_rotated_text(
            image,
            (x_center, height - margin_bottom + 55),
            label,
            tick_font,
            "#000000",
            35,
        )

    image.convert("RGB").save(path, "PNG")


def make_scatter_png(path, rows, x_field, y_field, title, x_label, y_label, log_y=False):
    """Render a scatter plot directly as PNG."""
    width = 1150
    height = 640
    margin_left = 130
    margin_right = 300
    margin_top = 70
    margin_bottom = 110

    # Rows with Python-side errors are skipped because they do not represent
    # valid solver measurements.
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

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(22)
    label_font = _load_font(16)
    tick_font = _load_font(14)

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_pos(value):
        ratio = (value - x_min) / (x_max - x_min)
        return margin_left + ratio * plot_width

    def y_pos(value):
        transformed_value = transform_value(value, log_y)
        ratio = (transformed_value - y_min) / (y_max - y_min)
        return height - margin_bottom - ratio * plot_height

    _draw_centered_text(draw, (width / 2, 30), title, title_font, "#000000")
    _draw_centered_text(draw, (width / 2, height - 28), x_label, label_font, "#000000")
    _draw_rotated_text(image, (40, height / 2), y_label, label_font, "#000000", 90)
    draw.line(
        [(margin_left, margin_top), (margin_left, height - margin_bottom)],
        fill="#333333",
        width=2,
    )
    draw.line(
        [(margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom)],
        fill="#333333",
        width=2,
    )

    legend_y = margin_top + 20
    for formulation, color in PALETTE.items():
        draw.rectangle(
            [
                (width - margin_right + 25, legend_y - 10),
                (width - margin_right + 39, legend_y + 4),
            ],
            fill=_hex_to_rgba(color, 255),
            outline=_hex_to_rgba(color, 255),
        )
        draw.text((width - margin_right + 48, legend_y - 10), formulation, font=tick_font, fill="#000000")
        legend_y += 28

    for row in points:
        color = PALETTE.get(row["formulation_name"], "#444")
        x = x_pos(row[x_field])
        y = y_pos(row[y_field])
        draw.ellipse(
            [(x - 4, y - 4), (x + 4, y + 4)],
            fill=_hex_to_rgba(color, 166),
            outline=_hex_to_rgba(color, 255),
        )

    # A few simple ticks are enough to make the exported PNG readable.
    for index in range(5):
        ratio = index / 4
        x_value = x_min + ratio * (x_max - x_min)
        y_value = y_min + ratio * (y_max - y_min)
        x = margin_left + ratio * plot_width
        y = height - margin_bottom - ratio * plot_height
        draw.line([(x, height - margin_bottom), (x, height - margin_bottom + 6)], fill="#333333", width=1)
        draw.line([(margin_left - 6, y), (margin_left, y)], fill="#333333", width=1)
        x_text = format_number(x_value)
        y_text = format_number((10 ** y_value) if log_y else y_value)
        xw, _ = _text_size(draw, x_text, tick_font)
        draw.text((x - xw / 2, height - margin_bottom + 14), x_text, font=tick_font, fill="#000000")
        yw, yh = _text_size(draw, y_text, tick_font)
        draw.text((margin_left - 14 - yw, y - yh / 2), y_text, font=tick_font, fill="#000000")

    image.convert("RGB").save(path, "PNG")


def build_html_report(report_path, results_csv, summary_rows, plot_paths):
    """Build a lightweight HTML report that embeds the generated plots."""
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
    """Format optional numeric values for human-readable output."""
    if value is None:
        return ""
    return f"{value:.6g}"


def main():
    """Run the full post-processing pipeline for one benchmark CSV."""
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
        ("runtime_boxplot.png", "runtime_sec", "Runtime by formulation", "Runtime (s)", True),
        ("root_gap_boxplot.png", "root_lp_gap", "Root LP gap by formulation", "Root LP gap", False),
        ("bb_nodes_boxplot.png", "bb_nodes", "Branch-and-bound nodes by formulation", "B&B nodes", True),
        (
            "separation_share_boxplot.png",
            "separation_share",
            "Separation share by formulation",
            "Separation time / runtime",
            False,
        ),
    ]

    # Stable filenames make it easy to inspect the output directory manually.
    for filename, field, title, label, log_scale in boxplot_specs:
        grouped_values = {name: values(group, field) for name, group in grouped.items()}
        path = layout["plots_dir"] / filename
        make_boxplot_png(path, grouped_values, title, label, log_scale=log_scale)
        plot_paths.append(path)

    scatter_specs = [
        (
            "runtime_vs_edges.png",
            "n_edges",
            "runtime_sec",
            "Runtime vs edges",
            "Number of edges",
            "Runtime (s)",
            True,
        ),
        (
            "runtime_vs_terminals.png",
            "n_terminals",
            "runtime_sec",
            "Runtime vs terminals",
            "Number of terminals",
            "Runtime (s)",
            True,
        ),
        (
            "root_gap_vs_runtime.png",
            "runtime_sec",
            "root_lp_gap",
            "Root LP gap vs runtime",
            "Runtime (s)",
            "Root LP gap",
            False,
        ),
        (
            "separation_vs_runtime.png",
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
        make_scatter_png(path, rows, x_field, y_field, title, x_label, y_label, log_y=log_y)
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
