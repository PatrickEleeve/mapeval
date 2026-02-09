"""Tests for PositionSizer."""

import sys
sys.path.insert(0, "src")

import pytest
from position_sizer import PositionSizer, ScalingStrategy


class TestPositionSizer:
    def setup_method(self):
        self.sizer = PositionSizer(
            max_position_size=5.0,
            pyramid_levels=3,
            level_spacing_pct=0.02,
        )
    
    def test_calculate_base_size_with_volatility(self):
        size = self.sizer.calculate_base_size(
            equity=100000,
            confidence=0.8,
            volatility=0.02,
            max_leverage=10.0,
        )
        
        assert size > 0
        assert size <= self.sizer.max_position_size
    
    def test_calculate_base_size_without_volatility(self):
        size = self.sizer.calculate_base_size(
            equity=100000,
            confidence=0.5,
            volatility=None,
            max_leverage=10.0,
        )
        
        assert size > 0
        assert size <= self.sizer.max_position_size
    
    def test_calculate_base_size_high_confidence(self):
        low_conf_size = self.sizer.calculate_base_size(
            equity=100000,
            confidence=0.3,
            volatility=0.02,
            max_leverage=10.0,
        )
        
        high_conf_size = self.sizer.calculate_base_size(
            equity=100000,
            confidence=0.9,
            volatility=0.02,
            max_leverage=10.0,
        )
        
        assert high_conf_size > low_conf_size
    
    def test_create_pyramid_plan_long(self):
        plan = self.sizer.create_pyramid_plan(
            symbol="BTCUSDT",
            direction="long",
            current_price=100.0,
            base_size=3.0,
        )
        
        assert plan.symbol == "BTCUSDT"
        assert plan.direction == "long"
        assert len(plan.levels) == 3
        assert plan.levels[0].triggered is True
        assert plan.levels[1].triggered is False
        assert plan.total_filled > 0
        assert plan.total_filled < plan.base_size
    
    def test_create_pyramid_plan_short(self):
        plan = self.sizer.create_pyramid_plan(
            symbol="ETHUSDT",
            direction="short",
            current_price=100.0,
            base_size=3.0,
        )
        
        assert plan.direction == "short"
        assert plan.levels[1].entry_price > 100.0
    
    def test_check_pyramid_triggers(self):
        self.sizer.create_pyramid_plan(
            symbol="BTCUSDT",
            direction="long",
            current_price=100.0,
            base_size=3.0,
        )
        
        triggers = self.sizer.check_pyramid_triggers("BTCUSDT", 95.0)
        
        assert len(triggers) >= 1
    
    def test_check_pyramid_triggers_no_trigger(self):
        self.sizer.create_pyramid_plan(
            symbol="BTCUSDT",
            direction="long",
            current_price=100.0,
            base_size=3.0,
        )
        
        triggers = self.sizer.check_pyramid_triggers("BTCUSDT", 105.0)
        
        assert len(triggers) == 0
    
    def test_calculate_scale_out_levels(self):
        levels = self.sizer.calculate_scale_out_levels(
            symbol="BTCUSDT",
            current_position=1.0,
            entry_price=100.0,
            current_price=110.0,
            take_profit_pct=0.05,
        )
        
        assert len(levels) > 0
    
    def test_calculate_scale_out_levels_no_profit(self):
        levels = self.sizer.calculate_scale_out_levels(
            symbol="BTCUSDT",
            current_position=1.0,
            entry_price=100.0,
            current_price=101.0,
            take_profit_pct=0.05,
        )
        
        assert len(levels) == 0
    
    def test_calculate_kelly_size(self):
        kelly = self.sizer.calculate_kelly_size(
            win_rate=0.6,
            avg_win=100.0,
            avg_loss=-50.0,
        )
        
        assert kelly > 0
        assert kelly <= 0.25
    
    def test_calculate_kelly_size_negative_edge(self):
        kelly = self.sizer.calculate_kelly_size(
            win_rate=0.3,
            avg_win=50.0,
            avg_loss=-100.0,
        )
        
        assert kelly == 0
    
    def test_get_recommended_size_fixed(self):
        result = self.sizer.get_recommended_size(
            symbol="BTCUSDT",
            direction="long",
            current_price=100.0,
            equity=100000,
            confidence=0.7,
            volatility=0.02,
            max_leverage=10.0,
            strategy=ScalingStrategy.FIXED,
        )
        
        assert "recommended_size" in result
        assert result["strategy"] == "fixed"
    
    def test_get_recommended_size_pyramid(self):
        result = self.sizer.get_recommended_size(
            symbol="BTCUSDT",
            direction="long",
            current_price=100.0,
            equity=100000,
            confidence=0.7,
            volatility=0.02,
            max_leverage=10.0,
            strategy=ScalingStrategy.PYRAMID_IN,
        )
        
        assert result["pyramid_plan"] is not None
        assert result["recommended_size"] < result["base_size"]
    
    def test_remove_pyramid(self):
        self.sizer.create_pyramid_plan(
            symbol="BTCUSDT",
            direction="long",
            current_price=100.0,
            base_size=3.0,
        )
        
        assert self.sizer.get_pyramid_status("BTCUSDT") is not None
        
        self.sizer.remove_pyramid("BTCUSDT")
        
        assert self.sizer.get_pyramid_status("BTCUSDT") is None
    
    def test_get_pyramid_status(self):
        self.sizer.create_pyramid_plan(
            symbol="BTCUSDT",
            direction="long",
            current_price=100.0,
            base_size=3.0,
        )
        
        status = self.sizer.get_pyramid_status("BTCUSDT")
        
        assert status is not None
        assert status["symbol"] == "BTCUSDT"
        assert status["direction"] == "long"
        assert status["total_levels"] == 3
        assert status["levels_triggered"] == 1
