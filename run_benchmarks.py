import argparse
import csv
import time
import traceback
from pathlib import Path

from benchmark_utils import (
    compute_instance_features,
    ensure_output_layout,
    list_instance_files,
    load_instance,
    make_run_id,
    resolve_io_paths,
    write_json,
)
from model_solvers import (
    collect_model_metrics,
    get_formulation_registry,
    solve_formulation,
    solve_lp_relaxation,
)


RESULT_FIELDS = [
    "run_id",
    "timestamp",
    "instance",
    "instance_path",
    "formulation_key",
    "formulation_name",
    "status_code",
    "status_name",
    "sol_count",
    "objective",
    "best_bound",
    "mip_gap",
    "runtime_sec",
    "model_runtime_sec",
    "lp_relaxation_obj",
    "lp_relaxation_status",
    "lp_relaxation_time_sec",
    "root_lp_bound",
    "root_lp_gap",
    "root_lp_callback_calls",
    "bb_nodes",
    "simplex_iterations",
    "barrier_iterations",
    "num_vars",
    "num_constrs",
    "mipnode_calls",
    "mipsol_calls",
    "separation_calls",
    "separation_time_sec",
    "separation_share",
    "lazy_cuts_added",
    "n_nodes",
    "n_edges",
    "n_terminals",
    "density",
    "avg_degree",
    "terminal_ratio",
    "time_limit_sec",
    "threads",
    "cuts",
    "seed",
    "uses_separation",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Steiner formulation benchmarks.")
    parser.add_argument("--instance-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cuts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        "--first-n-instances",
        dest="limit",
        type=int,
        default=None,
        help="Run only the first N instance files after sorting.",
    )
    parser.add_argument(
        "--formulations",
        type=str,
        default="uc,uf,dc,df",
        help="Comma-separated list from: uc, uf, dc, df",
    )
    parser.add_argument(
        "--compute-static-lp-relaxation",
        action="store_true",
        help="Also solve the plain LP relaxation of each formulation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures and continue with remaining runs.",
    )
    return parser.parse_args()


def normalize_formulation_keys(formulation_arg):
    requested = [part.strip() for part in formulation_arg.split(",") if part.strip()]
    registry = get_formulation_registry()
    invalid = [key for key in requested if key not in registry]
    if invalid:
        raise ValueError(f"Unknown formulation keys: {', '.join(invalid)}")
    return requested


def benchmark_one(
    formulation_key,
    formulation_meta,
    nodes,
    edges,
    terminals,
    edge_costs,
    args,
):
    lp_relaxation_obj = None
    lp_relaxation_status = None
    lp_relaxation_time_sec = None

    if args.compute_static_lp_relaxation:
        try:
            lp_start = time.perf_counter()
            lp_model = solve_lp_relaxation(
                formulation_key,
                nodes,
                edges,
                terminals,
                edge_costs,
                time_limit=args.time_limit,
                threads=args.threads,
                cuts=args.cuts,
                seed=args.seed,
            )
            lp_relaxation_time_sec = time.perf_counter() - lp_start
            lp_relaxation_status = lp_model.Status
            if lp_model.SolCount > 0:
                lp_relaxation_obj = lp_model.ObjVal
        except NotImplementedError:
            lp_relaxation_obj = None
            lp_relaxation_status = None
            lp_relaxation_time_sec = None

    solve_start = time.perf_counter()
    model = solve_formulation(
        formulation_key,
        nodes,
        edges,
        terminals,
        edge_costs,
        time_limit=args.time_limit,
        threads=args.threads,
        cuts=args.cuts,
        seed=args.seed,
    )
    runtime_sec = time.perf_counter() - solve_start
    metrics = collect_model_metrics(model)
    metrics["runtime_sec"] = runtime_sec
    metrics["lp_relaxation_obj"] = lp_relaxation_obj
    metrics["lp_relaxation_status"] = lp_relaxation_status
    metrics["lp_relaxation_time_sec"] = lp_relaxation_time_sec
    metrics["separation_share"] = (
        metrics["separation_time_sec"] / runtime_sec if runtime_sec else None
    )
    return metrics


def main():
    args = parse_args()
    formulation_keys = normalize_formulation_keys(args.formulations)
    instance_dir, output_dir = resolve_io_paths(args.instance_dir, args.output_dir)
    layout = ensure_output_layout(output_dir)
    run_id = make_run_id()

    registry = get_formulation_registry()

    instance_files = list_instance_files(instance_dir, limit=args.limit)
    results_path = layout["raw_dir"] / f"benchmarks_{run_id}.csv"
    metadata_path = layout["raw_dir"] / f"benchmarks_{run_id}.json"

    config_payload = {
        "run_id": run_id,
        "instance_dir": str(instance_dir),
        "output_dir": str(output_dir),
        "results_csv": str(results_path),
        "time_limit_sec": args.time_limit,
        "threads": args.threads,
        "cuts": args.cuts,
        "seed": args.seed,
        "formulations": formulation_keys,
        "compute_static_lp_relaxation": args.compute_static_lp_relaxation,
        "instance_count": len(instance_files),
    }
    write_json(metadata_path, config_payload)

    total_jobs = len(instance_files) * len(formulation_keys)
    completed_jobs = 0

    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        for instance_path in instance_files:
            nodes, edges, terminals, edge_costs = load_instance(instance_path)
            instance_features = compute_instance_features(nodes, edges, terminals)

            for formulation_key in formulation_keys:
                formulation_meta = registry[formulation_key]
                completed_jobs += 1
                print(
                    f"[{completed_jobs}/{total_jobs}] "
                    f"{instance_path.name} - {formulation_meta['name']}"
                )

                row = {
                    "run_id": run_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "formulation_key": formulation_key,
                    "formulation_name": formulation_meta["name"],
                    "time_limit_sec": args.time_limit,
                    "threads": args.threads,
                    "cuts": args.cuts,
                    "seed": args.seed,
                    "uses_separation": formulation_meta["uses_separation"],
                    "error": None,
                }
                row.update(instance_features)

                try:
                    metrics = benchmark_one(
                        formulation_key,
                        formulation_meta,
                        nodes,
                        edges,
                        terminals,
                        edge_costs,
                        args,
                    )
                    row.update(metrics)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    if not args.continue_on_error:
                        writer.writerow({field: row.get(field) for field in RESULT_FIELDS})
                        handle.flush()
                        raise
                    traceback.print_exc()

                writer.writerow({field: row.get(field) for field in RESULT_FIELDS})
                handle.flush()

    print(f"Saved raw benchmark results to {results_path}")
    print(f"Saved benchmark metadata to {metadata_path}")
if __name__ == "__main__":
    main()
