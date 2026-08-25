# MAPEval Project Summary

## Overview

MAPEval is a package-first Python project for LLM-driven crypto futures trading research.
It supports three operational paths:

- `simulation` — no external orders
- `paper` — live market data with simulated fills
- `live` — real execution path with explicit safety controls

The project now runs primarily from the `mapeval` package under `src/mapeval/`.

## Current Structure

```text
src/
  mapeval/
    main.py
    trading_engine.py
    data_manager.py
    order_executor.py
    api_server.py
    security.py
    ...
    strategies/
      base.py
      llm_strategy.py
      technical_strategy.py
tests/
benchmark/
scripts/
  fix_editable_pth.py
  manual/
README.md
Makefile
pyproject.toml
```

## Runtime Entry Points

Primary entrypoints:

```bash
python -m mapeval --help
make sim
make paper
make live-testnet
make smoke-backtest
make test
make dev-install
```

Installed console entrypoint:

```bash
mapeval --help
```

## Core Modules

- `src/mapeval/main.py` — CLI, session orchestration, mode selection
- `src/mapeval/trading_engine.py` — trading loop, reconciliation, runtime controls
- `src/mapeval/order_executor.py` — simulation/paper/live execution abstraction
- `src/mapeval/binance_futures_client.py` — Binance Futures client and exchange rule normalization
- `src/mapeval/data_manager.py` — real-time and backtest market data handling
- `src/mapeval/api_server.py` — monitoring and runtime control API
- `src/mapeval/security.py` — audit logging, token helpers, read-only guard
- `src/mapeval/strategies/` — strategy interfaces and concrete strategies

## Safety and Operations

Live-path protections currently include:

- explicit `--allow-live-trading`
- explicit testnet/mainnet selection
- confirmation phrase for live execution
- API token protection for REST and WebSocket monitoring
- read-only runtime guard
- kill switch support
- reconciliation loop for `paper` and `live`
- audit logging for control actions

## Development Notes

The repository was migrated from a flat `src/*.py` layout to `src/mapeval/`.
Tests and runtime code now target `mapeval.*` imports.

For local editable installs on macOS, use:

```bash
make dev-install
```

That command installs the package and repairs hidden editable `.pth` files that can break the `mapeval` console script.

## Verification

Local validation commands:

- `python -m mapeval --help`
- `.venv/bin/mapeval --help`
- `make test`
- `make test-cov`

CI runs the suite on Python 3.10, 3.11, and 3.12. Python 3.11 also enforces the
current coverage baseline and publishes the report.

## Remaining Work

The main structural migration is complete. Remaining work is incremental:

- tighten older long-form docs further
- raise coverage from the current baseline, prioritizing execution and exchange boundaries
- pay down existing Ruff findings, then make lint checks blocking in CI
- decide whether benchmark utilities should also move under `mapeval/`
