PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
PYTEST ?= $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)
RUN = $(PYTHON) -m mapeval

SIM_ARGS = --non-interactive --execution-mode simulation --duration 1h --symbols BTCUSDT ETHUSDT --llm-provider openai
PAPER_ARGS = --non-interactive --execution-mode paper --enable-api --duration 1h --symbols BTCUSDT ETHUSDT --llm-provider openai
LIVE_TESTNET_ARGS = --non-interactive --execution-mode live --allow-live-trading --binance-testnet --live-confirmation "ENABLE BINANCE TESTNET LIVE" --enable-api --duration 1h --symbols BTCUSDT ETHUSDT --llm-provider openai
SMOKE_BACKTEST_ARGS = --non-interactive --mode backtest --execution-mode simulation --strategy buy_hold --duration 1h --symbols BTCUSDT --data-path sample_financial_data.csv

.PHONY: help dev-install sim paper live-testnet smoke-backtest test clean

help:
	@echo "make dev-install   # editable install + fix macOS hidden .pth issue"
	@echo "make sim           # safe simulation run"
	@echo "make paper         # live data + simulated fills"
	@echo "make live-testnet  # strict live path on Binance testnet"
	@echo "make smoke-backtest# offline smoke test with sample data"
	@echo "make test          # run pytest"
	@echo "make clean         # remove local caches and runtime artifacts"

dev-install:
	$(PYTHON) -m pip install -e '.[dev,api,db,full]'
	$(PYTHON) scripts/fix_editable_pth.py

sim:
	$(RUN) $(SIM_ARGS)

paper:
	$(RUN) $(PAPER_ARGS)

live-testnet:
	$(RUN) $(LIVE_TESTNET_ARGS)

smoke-backtest:
	$(RUN) $(SMOKE_BACKTEST_ARGS)

test:
	$(PYTEST) -q

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find logs -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	rm -f financial_analysis.png python_code.pdf output.txt
