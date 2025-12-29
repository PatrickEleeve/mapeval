"""Unit tests for llm_agent.py sanitization logic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from llm_agent import LLMAgent


class TestSanitizeExposures:
    def test_clips_per_symbol_exposure(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT", "ETHUSDT"],
            max_leverage=30.0,
            per_symbol_max_exposure=5.0,
        )
        exposures = {"BTCUSDT": 10.0, "ETHUSDT": -8.0}
        result = agent._sanitize_exposures(exposures)
        assert result is not None
        assert result["BTCUSDT"] == pytest.approx(5.0)
        assert result["ETHUSDT"] == pytest.approx(-5.0)

    def test_clips_delta_from_previous(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT"],
            max_leverage=30.0,
            per_symbol_max_exposure=10.0,
            max_exposure_delta=2.0,
        )
        agent._last_exposures = {"BTCUSDT": 1.0}
        exposures = {"BTCUSDT": 5.0}
        result = agent._sanitize_exposures(exposures)
        assert result is not None
        assert result["BTCUSDT"] == pytest.approx(3.0)

    def test_scales_total_leverage(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT", "ETHUSDT"],
            max_leverage=10.0,
            per_symbol_max_exposure=10.0,
            max_exposure_delta=10.0,
        )
        exposures = {"BTCUSDT": 8.0, "ETHUSDT": 8.0}
        result = agent._sanitize_exposures(exposures)
        assert result is not None
        total = abs(result["BTCUSDT"]) + abs(result["ETHUSDT"])
        assert total == pytest.approx(10.0)

    def test_returns_none_for_non_numeric(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT"],
            max_leverage=10.0,
        )
        exposures = {"BTCUSDT": "invalid"}
        result = agent._sanitize_exposures(exposures)
        assert result is None

    def test_zero_leverage_zeros_all(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT"],
            max_leverage=0.0,
        )
        exposures = {"BTCUSDT": 5.0}
        result = agent._sanitize_exposures(exposures)
        assert result is not None
        assert result["BTCUSDT"] == 0.0

    def test_preserves_sign_direction(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT", "ETHUSDT"],
            max_leverage=10.0,
            per_symbol_max_exposure=5.0,
        )
        exposures = {"BTCUSDT": 3.0, "ETHUSDT": -2.0}
        result = agent._sanitize_exposures(exposures)
        assert result is not None
        assert result["BTCUSDT"] > 0
        assert result["ETHUSDT"] < 0


class TestSanitizeIndicators:
    def test_filters_unsupported_indicators(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT"],
            max_leverage=10.0,
            indicators=["rsi", "unknown_indicator", "macd"],
        )
        assert "rsi" in agent.indicators
        assert "macd" in agent.indicators
        assert "unknown_indicator" not in agent.indicators

    def test_deduplicates_indicators(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT"],
            max_leverage=10.0,
            indicators=["rsi", "RSI", "rsi"],
        )
        assert agent.indicators.count("rsi") == 1


class TestSystemPromptGeneration:
    def test_contains_symbol_list(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT", "ETHUSDT"],
            max_leverage=20.0,
        )
        assert "BTCUSDT" in agent._system_prompt
        assert "ETHUSDT" in agent._system_prompt

    def test_contains_leverage_limits(self):
        agent = LLMAgent(
            api_key="",
            config={},
            symbols=["BTCUSDT"],
            max_leverage=25.0,
            per_symbol_max_exposure=5.0,
        )
        assert "25" in agent._system_prompt
        assert "5" in agent._system_prompt

