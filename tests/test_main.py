"""Tests for CLI safety helpers in main.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from mapeval.main import (
    _compact_symbols,
    _confirm_live_execution,
    _required_live_confirmation,
    _resolve_live_environment,
)


class TestLiveEnvironmentResolution:
    def test_defaults_to_testnet(self):
        assert _resolve_live_environment(False, False) == "testnet"

    def test_explicit_mainnet(self):
        assert _resolve_live_environment(False, True) == "mainnet"

    def test_rejects_conflicting_flags(self):
        with pytest.raises(ValueError):
            _resolve_live_environment(True, True)


class TestStartupFormatting:
    def test_compact_symbols_shortens_long_lists(self):
        rendered = _compact_symbols(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"])
        assert rendered == "BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT +1 more"


class TestLiveConfirmation:
    def test_required_confirmation_phrase(self):
        assert _required_live_confirmation("mainnet") == "ENABLE BINANCE MAINNET LIVE"

    def test_non_interactive_rejects_wrong_phrase(self):
        with pytest.raises(ValueError):
            _confirm_live_execution("testnet", "wrong", True)

    def test_non_interactive_accepts_exact_phrase(self):
        phrase = _required_live_confirmation("testnet")
        _confirm_live_execution("testnet", phrase, True)
