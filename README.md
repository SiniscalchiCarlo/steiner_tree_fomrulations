## Benchmark workflow

This repository now contains a two-step computational study pipeline:

1. `run_benchmarks.py`
   Runs every selected formulation on every selected instance and writes one raw CSV row per `(instance, formulation)`.

2. `analyze_benchmarks.py`
   Reads a raw benchmark CSV, computes summary tables, and saves SVG plots plus an HTML report.

### Configuration

The scripts read these variables from `.env` when present:

```bash
INSTANCE_DIR=./test_istances
TEST_OUTPUT_DIR=./test_output
```

### Run benchmarks

```bash
python run_benchmarks.py --time-limit 30 --threads 1 --cuts 0 --compute-static-lp-relaxation
```

Useful options:

- `--limit N` to benchmark only the first `N` instances
- `--formulations uc,uf,dc,df` to select formulations
- `--continue-on-error` to keep going after a failed solve

### Analyze results

```bash
python analyze_benchmarks.py
```

By default this analyzes the latest CSV in `TEST_OUTPUT_DIR/raw`.

### Outputs

The scripts create:

- `TEST_OUTPUT_DIR/raw/benchmarks_<run_id>.csv`
- `TEST_OUTPUT_DIR/raw/benchmarks_<run_id>.json`
- `TEST_OUTPUT_DIR/tables/*.csv`
- `TEST_OUTPUT_DIR/plots/*.svg`
- `TEST_OUTPUT_DIR/reports/report_<run_id>.html`

### Measured quantities

The runner records the metrics needed for Exercise 4:

- overall runtime
- root LP bound and root LP gap
- branch-and-bound nodes
- separation calls, cuts added, and separation time
- instance features such as nodes, edges, terminals, density, and terminal ratio
