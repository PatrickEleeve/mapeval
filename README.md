# MAPEval Quick Start

MAPEval now has three intended ways to run.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env
make dev-install
python -m mapeval --help
```

## Fastest Commands

If you are running locally, use these:

```bash
make help
make sim
make paper
make live-testnet
make smoke-backtest
make clean
```

If you prefer Docker:

```bash
docker compose up mapeval
docker compose --profile paper up mapeval-paper
docker compose --profile live up mapeval-live-testnet
```

## 1. Simulation

Use this first. It never sends external orders.

```bash
python -m mapeval \
  --non-interactive \
  --execution-mode simulation \
  --duration 1h \
  --symbols BTCUSDT ETHUSDT \
  --llm-provider openai
```

## 2. Paper Trading

Uses live market data and the real execution path, but fills are simulated.

```bash
python -m mapeval \
  --non-interactive \
  --execution-mode paper \
  --enable-api \
  --duration 1h \
  --symbols BTCUSDT ETHUSDT \
  --llm-provider openai
```

Notes:
- If `--enable-api` is set and no `--api-token` is provided, the app generates a token for the current session and prints it once.
- Reconciliation is on by default every 300 seconds. Use `--reconcile-interval 0` to disable it.

## 3. Live Trading

This mode is intentionally strict. Start on testnet first.

```bash
python -m mapeval \
  --non-interactive \
  --execution-mode live \
  --allow-live-trading \
  --binance-testnet \
  --live-confirmation "ENABLE BINANCE TESTNET LIVE" \
  --enable-api \
  --duration 1h \
  --symbols BTCUSDT ETHUSDT \
  --llm-provider openai
```

Requirements:
- `BINANCE_API_KEY` and `BINANCE_API_SECRET` must be set.
- Mainnet needs `--binance-mainnet` and the matching confirmation phrase `ENABLE BINANCE MAINNET LIVE`.

## Offline Smoke Test

Use this when you want a deterministic local sanity check without Binance connectivity:

```bash
make smoke-backtest
```

## Mental Model

- `simulation`: safest, research only
- `paper`: verify trading workflow
- `live`: real orders, strict safety checks

## Manual Diagnostics

Ad-hoc connectivity/provider smoke scripts live under `scripts/manual/`. They are not part of `pytest` and now read credentials from environment variables only.

## Installed CLI

If you install the project as a package, the console entrypoint is:

```bash
mapeval --help
```

From the repository root, `python -m mapeval --help` now works without setting `PYTHONPATH`.

For local editable installs, `make dev-install` also repairs the macOS hidden-`.pth` issue that can break the `mapeval` console script.

## If You Need More

- For deeper architecture details, see `PROJECT_SUMMARY.md`.
- For container usage, see `docker-compose.yml`.
- For local shortcuts, see `Makefile`.
