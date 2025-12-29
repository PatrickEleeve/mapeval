"""Unit tests for risk_manager.py."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from risk_manager import RiskLimits, RiskManager, RiskState


class TestRiskLimits:
    def test_default_limits(self):
        limits = RiskLimits()
        assert limits.max_drawdown == 0.20
        assert limits.max_daily_loss == 0.05
        assert limits.max_consecutive_losses == 5

    def test_custom_limits(self):
        limits = RiskLimits(max_drawdown=0.15, max_daily_loss=0.03)
        assert limits.max_drawdown == 0.15
        assert limits.max_daily_loss == 0.03


class TestRiskManagerInitialization:
    def test_initialize_sets_peak_equity(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        assert rm.state.peak_equity == 100_000.0
        assert rm.state.daily_starting_equity == 100_000.0

    def test_update_equity_tracks_peak(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        now = datetime.now(timezone.utc)
        
        rm.update_equity(110_000.0, now)
        assert rm.state.peak_equity == 110_000.0
        
        rm.update_equity(105_000.0, now)
        assert rm.state.peak_equity == 110_000.0


class TestDrawdownCheck:
    def test_within_limit(self):
        rm = RiskManager(RiskLimits(max_drawdown=0.20))
        rm.initialize(100_000.0)
        
        ok, msg = rm.check_drawdown(90_000.0)
        assert ok is True
        assert msg is None

    def test_exceeds_limit(self):
        rm = RiskManager(RiskLimits(max_drawdown=0.20))
        rm.initialize(100_000.0)
        
        ok, msg = rm.check_drawdown(75_000.0)
        assert ok is False
        assert "drawdown exceeded" in msg.lower()


class TestDailyLossCheck:
    def test_within_limit(self):
        rm = RiskManager(RiskLimits(max_daily_loss=0.05))
        rm.initialize(100_000.0)
        
        ok, msg = rm.check_daily_loss(97_000.0)
        assert ok is True

    def test_exceeds_limit(self):
        rm = RiskManager(RiskLimits(max_daily_loss=0.05))
        rm.initialize(100_000.0)
        
        ok, msg = rm.check_daily_loss(90_000.0)
        assert ok is False
        assert "daily loss" in msg.lower()


class TestConsecutiveLosses:
    def test_below_limit(self):
        rm = RiskManager(RiskLimits(max_consecutive_losses=5))
        rm.initialize(100_000.0)
        now = datetime.now(timezone.utc)
        
        for _ in range(3):
            rm.record_trade_result(-100.0, now)
        
        ok, msg = rm.check_consecutive_losses()
        assert ok is True

    def test_at_limit(self):
        rm = RiskManager(RiskLimits(max_consecutive_losses=5))
        rm.initialize(100_000.0)
        now = datetime.now(timezone.utc)
        
        for _ in range(5):
            rm.record_trade_result(-100.0, now)
        
        ok, msg = rm.check_consecutive_losses()
        assert ok is False

    def test_reset_on_win(self):
        rm = RiskManager(RiskLimits(max_consecutive_losses=5))
        rm.initialize(100_000.0)
        now = datetime.now(timezone.utc)
        
        for _ in range(4):
            rm.record_trade_result(-100.0, now)
        
        rm.record_trade_result(50.0, now)
        assert rm.state.consecutive_losses == 0


class TestCooldown:
    def test_no_cooldown_without_losses(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        now = datetime.now(timezone.utc)
        
        ok, msg = rm.check_cooldown(now)
        assert ok is True

    def test_cooldown_after_multiple_losses(self):
        rm = RiskManager(RiskLimits(cooldown_after_loss_seconds=300))
        rm.initialize(100_000.0)
        now = datetime.now(timezone.utc)
        
        for _ in range(3):
            rm.record_trade_result(-100.0, now)
        
        ok, msg = rm.check_cooldown(now + timedelta(seconds=60))
        assert ok is False
        assert "cooldown" in msg.lower()

    def test_cooldown_expires(self):
        rm = RiskManager(RiskLimits(cooldown_after_loss_seconds=300))
        rm.initialize(100_000.0)
        now = datetime.now(timezone.utc)
        
        for _ in range(3):
            rm.record_trade_result(-100.0, now)
        
        ok, msg = rm.check_cooldown(now + timedelta(seconds=400))
        assert ok is True


class TestExposureAdjustment:
    def test_no_adjustment_when_healthy(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        
        adjusted = rm.adjust_exposure_for_risk(5.0, 100_000.0)
        assert adjusted == pytest.approx(5.0)

    def test_reduces_on_drawdown(self):
        rm = RiskManager(RiskLimits(max_drawdown=0.20))
        rm.initialize(100_000.0)
        rm.state.peak_equity = 100_000.0
        
        adjusted = rm.adjust_exposure_for_risk(5.0, 85_000.0)
        assert adjusted < 5.0

    def test_reduces_on_consecutive_losses(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        now = datetime.now(timezone.utc)
        
        for _ in range(3):
            rm.record_trade_result(-100.0, now)
        
        adjusted = rm.adjust_exposure_for_risk(5.0, 100_000.0)
        assert adjusted < 5.0


class TestForceClose:
    def test_force_close_on_max_drawdown(self):
        rm = RiskManager(RiskLimits(max_drawdown=0.20))
        rm.initialize(100_000.0)
        
        assert rm.should_force_close(75_000.0, 100_000.0) is True

    def test_force_close_on_equity_floor(self):
        rm = RiskManager(RiskLimits(min_equity_floor=0.10))
        rm.initialize(100_000.0)
        
        assert rm.should_force_close(5_000.0, 100_000.0) is True

    def test_no_force_close_when_healthy(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        
        assert rm.should_force_close(95_000.0, 100_000.0) is False


class TestRiskReport:
    def test_report_structure(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        now = datetime.now(timezone.utc)
        rm.update_equity(95_000.0, now)
        
        report = rm.get_risk_report(95_000.0)
        
        assert "current_equity" in report
        assert "peak_equity" in report
        assert "drawdown" in report
        assert "consecutive_losses" in report
        assert report["current_equity"] == 95_000.0
        assert report["peak_equity"] == 100_000.0
        assert report["drawdown"] == pytest.approx(0.05)

