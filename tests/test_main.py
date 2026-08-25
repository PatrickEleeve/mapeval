"""Tests for CLI safety helpers in main.py."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from mapeval.main import (
    _compact_symbols,
    _confirm_live_execution,
    _parse_args,
    _required_live_confirmation,
    _resolve_live_environment,
    _resolve_stop_loss_enabled,
    _validate_mode_combination,
)


class TestLiveEnvironmentResolution:
    def test_defaults_to_testnet(self):
        assert _resolve_live_environment(False, False) == "testnet"

    def test_explicit_mainnet(self):
        assert _resolve_live_environment(False, True) == "mainnet"

    def test_rejects_conflicting_flags(self):
        with pytest.raises(ValueError, match="Choose only one"):
            _resolve_live_environment(True, True)


class TestStartupFormatting:
    def test_compact_symbols_shortens_long_lists(self):
        rendered = _compact_symbols(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"])
        assert rendered == "BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT +1 more"


class TestLiveConfirmation:
    def test_required_confirmation_phrase(self):
        assert _required_live_confirmation("mainnet") == "ENABLE BINANCE MAINNET LIVE"

    def test_non_interactive_rejects_wrong_phrase(self):
        with pytest.raises(ValueError, match="Live trading requires --live-confirmation"):
            _confirm_live_execution("testnet", "wrong", True)

    def test_non_interactive_accepts_exact_phrase(self):
        phrase = _required_live_confirmation("testnet")
        _confirm_live_execution("testnet", phrase, True)


class TestModeValidation:
    def test_backtest_requires_simulation_execution(self):
        with pytest.raises(ValueError, match="only supports"):
            _validate_mode_combination("backtest", "live")

    def test_backtest_accepts_simulation_execution(self):
        _validate_mode_combination("backtest", "simulation")


class TestStopLossArguments:
    def test_stop_loss_defaults_to_mode_resolution(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mapeval"])
        assert _parse_args().stop_loss is None

    def test_stop_loss_can_be_explicitly_toggled(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mapeval", "--stop-loss"])
        assert _parse_args().stop_loss is True
        monkeypatch.setattr(sys, "argv", ["mapeval", "--no-stop-loss"])
        assert _parse_args().stop_loss is False

    def test_mode_defaults_and_explicit_override(self):
        assert _resolve_stop_loss_enabled("paper", None) is True
        assert _resolve_stop_loss_enabled("live", None) is True
        assert _resolve_stop_loss_enabled("simulation", None) is False
        assert _resolve_stop_loss_enabled("paper", False) is False
        assert _resolve_stop_loss_enabled("simulation", True) is True
