"""Configuration settings for the real-time MAPEval futures trading project."""

API_KEY = "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"
API_KEY2 = "sk-84ba03aa20934aafb137b118dfa60f7c"
AGENT_CONFIG = {
    "model_name": "gpt-5",
    "temperature": 0.2,
}

TRADING_CONFIG = {
    "symbols": [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "SOLUSDT",
        "TRXUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "SUIUSDT",
        "AAVEUSDT",
    ],
    "initial_capital": 100_000.0,
    "poll_interval_seconds": 5.0,
    "decision_interval_seconds": 60.0,
    "history_interval": "1m",
    "history_lookback": 500,
    "max_leverage": 50,
    "duration_seconds": {
        "1h": 60 * 60,
        "6h": 6 * 60 * 60,
        "12h": 12 * 60 * 60,
        "1d": 24 * 60 * 60,
    },
}