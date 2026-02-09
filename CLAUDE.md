# CLAUDE.md

## Project Overview

MAPEval is an LLM-driven cryptocurrency futures trading benchmark system. It evaluates how well different LLMs (OpenAI GPT, DeepSeek, Qwen) perform as trading agents on Binance perpetual futures, supporting real-time trading and backtesting.

## Tech Stack

- **Language:** Python 3.10+
- **LLM Integration:** OpenAI SDK (also used for DeepSeek/Qwen via compatible APIs)
- **Market Data:** Binance Futures API + WebSocket
- **UI:** Rich (console + TUI dashboard)
- **Optional:** FastAPI (monitoring API), SQLAlchemy (persistence), Pydantic (config validation), structlog (structured logging)

## Architecture / Data Flow

```text
main.py (CLI + session orchestration)
  ├── trading_engine.py (core trading loop: poll → decide → execute)
  │     ├── llm_agent.py / async_llm_agent.py (LLM trading decisions)
  │     ├── strategies/ (llm, technical, baselines)
  │     ├── order_executor.py (order placement & fills)
  │     ├── risk_manager.py (drawdown, loss limits, cooldowns)
  │     ├── stop_loss_manager.py (ATR-based dynamic stop-loss)
  │     └── position_sizer.py (fixed, pyramid, volatility, Kelly sizing)
  ├── data_manager.py (real-time + backtest market data)
  │     ├── binance_data_source.py / binance_ws.py / binance_futures_client.py
  │     └── cache_manager.py (local data caching)
  ├── reporter.py / tui_reporter.py (real-time stats & TUI dashboard)
  ├── event_bus.py + events.py (internal pub/sub event system)
  └── notifier.py (Telegram, Webhook notifications)
```

## Project Structure

- `src/` — All source code (flat module layout, no package `__init__.py`)
  - **Core**
    - `main.py` — Entry point, CLI argument parsing, session orchestration
    - `config.py` — API keys (from env), agent configs, trading defaults (`TRADING_CONFIG`, `AGENT_CONFIG`)
    - `config_models.py` — Pydantic config models
    - `trading_engine.py` — Core trading loop (poll prices, call strategy, execute trades)
    - `trading_module.py` — Interface bridging LLM outputs with the trading engine
  - **LLM & Strategies**
    - `llm_agent.py` — LLM agent that generates trading decisions
    - `async_llm_agent.py` — Async LLM agent for parallel signal generation
    - `strategies/` — Strategy implementations (`base.py`, `llm_strategy.py`, `technical_strategy.py`)
  - **Market Data**
    - `data_manager.py` — Market data management (real-time + backtest)
    - `binance_data_source.py` — Binance REST data fetching
    - `binance_ws.py` — Binance WebSocket streaming
    - `binance_futures_client.py` — Binance Futures API client
    - `cache_manager.py` — Local data caching for historical market data
    - `mock_data.py` — Mock market data generation for testing
  - **Trading & Risk**
    - `order_executor.py` — Order placement and management
    - `order_models.py` — Order data models
    - `risk_manager.py` — Drawdown, loss limits, cooldowns
    - `stop_loss_manager.py` — Dynamic stop-loss management based on ATR
    - `position_sizer.py` — Position sizing (fixed, pyramid, volatility, Kelly)
    - `portfolio_risk_controller.py` — Portfolio-level risk (margin, exposure caps, turnover)
    - `signal_evaluator.py` — Track and evaluate LLM signal accuracy
    - `adaptive_interval.py` — Adaptive decision interval based on market volatility
  - **Infrastructure**
    - `event_bus.py` / `events.py` — Internal event system
    - `exchange.py` — Exchange abstraction layer
    - `interfaces.py` — Abstract interfaces for dependency injection
    - `rate_limiter.py` — Token-bucket rate limiter with exponential backoff
    - `notifier.py` — Notification system (Telegram, Webhook, composite)
    - `security.py` — API key management, audit logging, read-only guard
    - `tools.py` — Financial utility functions and analytics
    - `factor_evaluator.py` — Factorized benchmark framework (6 trading ability dimensions)
  - **Reporting & Logging**
    - `reporter.py` — Real-time reporting and statistics
    - `tui_reporter.py` — Rich TUI dashboard
    - `log_manager.py` — Session logging
    - `structured_logging.py` — Structured logging via structlog
    - `performance_analyzer.py` — Detailed performance analysis
  - **Optional Persistence & API**
    - `database.py` / `db_models.py` / `repositories.py` — SQLAlchemy DB persistence
    - `api_server.py` — FastAPI monitoring server
  - **Other**
    - `backtester.py` — Backtesting engine (simulated order fills, performance reports)
- `tests/` — Pytest test suite
- `benchmark/` — LLM trading benchmark framework
  - `data_collector.py` — Fetch multi-timeframe kline data from Binance
  - `data/` — Collected market data (JSON)
  - `evaluators/benchmark_runner.py` — Run test cases, query LLM, score against ground truth
  - `test_suites/` — Test case definitions (`signal_recognition.json`, `risk_awareness.json`, `consistency.json`)
  - `results/` — Benchmark result outputs

## Commands

```bash
# Install
pip install -e ".[all]"

# Run (interactive mode)
python src/main.py

# Run (non-interactive, realtime)
python src/main.py --non-interactive --mode realtime --llm-provider openai --duration 1h

# Run (backtest)
python src/main.py --non-interactive --mode backtest --data-path data/benchmark.csv

# Run benchmark evaluation
python benchmark/evaluators/benchmark_runner.py

# Tests
pytest tests/ -v

# Lint & Format
ruff check src/
ruff format src/
```

## Environment Variables

- `OPENAI_API_KEY` — Required for OpenAI provider
- `DEEPSEEK_API_KEY` — Required for DeepSeek provider
- `QWEN_API_KEY` — Required for Qwen provider
- `MAPEVAL_ENV_FILE` — Custom .env file path (optional)

## Key Conventions

- Source files are in `src/` but imported as flat modules (no `src.` prefix)
- Config defaults live in `src/config.py` (`TRADING_CONFIG`, `AGENT_CONFIG`)
- Tests use pytest with `asyncio_mode = "auto"`
- Ruff is used for linting and formatting (line length 100)
- Type hints are used but `disallow_untyped_defs` is off in mypy
- Available strategies: `llm`, `buy_hold`, `ma_crossover`, `random`
- Available modes: `realtime`, `backtest`
