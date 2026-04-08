"""Shared helpers for benchmark execution and post-processing."""

import csv
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_env_file(env_path):
    """Load simple `KEY=VALUE` entries from a local `.env` file."""
    if not env_path.is_file():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def resolve_path_from_env(var_name, default_path, create=False):
    """Resolve a path from an environment variable or a default value."""
    raw_value = os.getenv(var_name)
    path = Path(raw_value).expanduser() if raw_value else Path(default_path)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    else:
        path = path.resolve()

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def resolve_io_paths(instance_dir=None, output_dir=None):
    """Resolve benchmark input/output directories from CLI, `.env`, or defaults."""
    load_env_file(BASE_DIR / ".env")

    resolved_instance_dir = (
        Path(instance_dir).expanduser().resolve()
        if instance_dir
        else resolve_path_from_env("INSTANCE_DIR", BASE_DIR / "test_istances")
    )
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else resolve_path_from_env("TEST_OUTPUT_DIR", BASE_DIR / "test_output", create=True)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    if not resolved_instance_dir.is_dir():
        raise FileNotFoundError(
            f"INSTANCE_DIR does not exist or is not a directory: {resolved_instance_dir}"
        )

    return resolved_instance_dir, resolved_output_dir


def ensure_output_layout(output_dir):
    """Create the standard output subdirectories and return their paths."""
    raw_dir = output_dir / "raw"
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    for path in (raw_dir, tables_dir, plots_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": output_dir,
        "raw_dir": raw_dir,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "reports_dir": reports_dir,
    }


def make_run_id():
    """Return a timestamp-based identifier for one benchmark run."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_instance(path):
    """Import one instance module and extract the expected data objects."""
    spec = importlib.util.spec_from_file_location("instance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.nodes, module.edges, module.terminals, module.edgeCosts


def list_instance_files(instance_dir, limit=None):
    """Return sorted instance `.py` files, optionally truncated."""
    files = sorted(path for path in Path(instance_dir).iterdir() if path.suffix == ".py")
    if limit is not None:
        return files[:limit]
    return files


def compute_instance_features(nodes, edges, terminals):
    """Compute simple graph features recorded in the raw benchmark CSV."""
    node_count = len(nodes)
    edge_count = len(edges)
    terminal_count = len(terminals)
    max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 0
    density = edge_count / max_edges if max_edges else 0.0
    avg_degree = 2 * edge_count / node_count if node_count else 0.0
    terminal_ratio = terminal_count / node_count if node_count else 0.0
    return {
        "n_nodes": node_count,
        "n_edges": edge_count,
        "n_terminals": terminal_count,
        "density": density,
        "avg_degree": avg_degree,
        "terminal_ratio": terminal_ratio,
    }


def safe_float(value):
    """Convert optional CSV text to `float` while preserving empty values."""
    if value in ("", None):
        return None
    return float(value)


def safe_int(value):
    """Convert optional CSV text to `int` while preserving empty values."""
    if value in ("", None):
        return None
    return int(float(value))


def write_json(path, payload):
    """Write JSON with stable formatting for easy inspection and diffing."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows, fieldnames):
    """Write a list of dictionaries to CSV using a fixed column order."""
    with path.open("w", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_latest_results_csv(raw_dir):
    """Return the newest benchmark CSV found in the raw output directory."""
    candidates = sorted(Path(raw_dir).glob("benchmarks_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No benchmark CSV files found in {raw_dir}")
    return candidates[-1]
