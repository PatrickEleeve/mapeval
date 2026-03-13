# Benchmark Workspace

This directory contains the standalone benchmark assets used to evaluate model behavior outside the live trading loop.

Layout:

- `test_suites/` - benchmark case definitions
- `evaluators/benchmark_runner.py` - benchmark runner
- `data_collector.py` - helper for collecting benchmark market snapshots

Generated outputs are intentionally not tracked:

- `benchmark/data/market_data_*.json`
- `benchmark/results/*.json`

If you run the benchmark tooling, those directories will be created on demand.
