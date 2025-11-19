# MAPEval Trading Runner

## Overview
MAPEval simulates leveraged futures trades. LLM prompts guide exposure picks. Users hook in live Binance spot data. Logs capture every step.

## Directory Map
- `src/` holds runtime modules.
- `test/` holds utility testers.
- `logs/` stores session archives.
- `.env` keeps private keys.

## Setup
Use a local Python 3.10 shell. Create a virtual environment. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

Create `.env` near the repo root. Place provider keys there.

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
echo "DEEPSEEK_API_KEY=sk-..." >> .env
echo "QWEN_API_KEY=sk-..." >> .env
```

## Run A Session
Export `PYTHONPATH` so imports resolve. Launch the driver.

```bash
cd mapeval
export PYTHONPATH=src
python src/main.py --duration 1h --symbols BTCUSDT ETHUSDT --llm-provider openai
```

Watch the console. Trades and warnings print in real time. Session files land in `logs/`.
Supported providers today: `openai`, `deepseek`, `qwen`.
All timestamps in logs and console output use UTC+0 for reproducibility.

## Inspect Output
Logs hold JSON bundles. Each record stores parameters plus equity history. Use `jq` or notebooks for analysis.

## Test Network Hooks
Quick probes live in `test/`.

```bash
python test/binance_test.py
```

The script checks Binance REST reachability. Network limits may cause warnings. Review them before live runs.

## Key Modules
- `src/llm_agent.py` crafts prompts and parses JSON replies.
- `src/trading_engine.py` applies exposure targets and manages margin.
- `src/data_manager.py` refreshes cached candles and streaming quotes.
- `src/log_manager.py` writes audit artifacts.

Read these files for deeper study. Each module favours small functions for clarity.

## Operational Notes
Huge leverage magnifies losses. Use sandbox keys first. Confirm funding data paths and timeouts. Document changes inside pull requests.
