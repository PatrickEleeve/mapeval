# CLAUDE.md

## Project Overview

MAPEval is a Python trading research project for LLM-driven crypto futures evaluation and execution-path testing.

It supports:

- `simulation`
- `paper`
- `live`
- `backtest`

The runtime code now lives under `src/mapeval/`.

## High-Level Layout

```text
src/
  mapeval/
    main.py
    trading_engine.py
    data_manager.py
    order_executor.py
    api_server.py
    security.py
    strategies/
      base.py
      llm_strategy.py
      technical_strategy.py
tests/
benchmark/
scripts/
```

## Important Commands

```bash
make dev-install
make sim
make paper
make live-testnet
make smoke-backtest
make test
python -m mapeval --help
```

## Development Conventions

- import runtime modules as `mapeval.*`
- keep source changes under `src/mapeval/`
- use `scripts/manual/` only for ad-hoc external smoke checks
- prefer `make dev-install` on macOS so editable install metadata is repaired
- keep safety features intact on `paper` and `live`

## Important Files

- `src/mapeval/main.py` — CLI entrypoint
- `src/mapeval/trading_engine.py` — trading engine
- `src/mapeval/order_executor.py` — order execution abstraction
- `src/mapeval/data_manager.py` — market data layer
- `src/mapeval/api_server.py` — monitoring/control API
- `src/mapeval/security.py` — audit and runtime safety helpers
- `README.md` — quickstart
- `PROJECT_SUMMARY.md` — current architecture summary
- `IMPLEMENTATION_SUMMARY.md` — migration and cleanup summary
