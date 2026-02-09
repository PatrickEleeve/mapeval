"""Tests for StopLossManager."""

import sys
sys.path.insert(0, "src")

import pytest
import pandas as pd
from stop_loss_manager import StopLossManager, StopType


class TestStopLossManager:
    def setup_method(self):
        self.manager = StopLossManager(atr_multiplier=2.0)
        self.timestamp = pd.Timestamp.now()
    
    def test_calculate_initial_stop_long_with_atr(self):
        stop = self.manager.calculate_initial_stop(
            symbol="BTCUSDT",
            entry_price=100.0,
            position_side="long",
            atr_value=2.0,
            timestamp=self.timestamp,
        )
        assert stop == 96.0
    
    def test_calculate_initial_stop_short_with_atr(self):
        stop = self.manager.calculate_initial_stop(
            symbol="ETHUSDT",
            entry_price=100.0,
            position_side="short",
            atr_value=2.0,
            timestamp=self.timestamp,
        )
        assert stop == 104.0
    
    def test_calculate_initial_stop_without_atr(self):
        stop = self.manager.calculate_initial_stop(
            symbol="BTCUSDT",
            entry_price=100.0,
            position_side="long",
            atr_value=None,
            timestamp=self.timestamp,
        )
        assert stop == 98.0
    
    def test_trailing_stop_updates_on_profit_long(self):
        self.manager.calculate_initial_stop(
            symbol="BTCUSDT",
            entry_price=100.0,
            position_side="long",
            atr_value=2.0,
            timestamp=self.timestamp,
        )
        
        new_stop = self.manager.update_trailing_stop(
            symbol="BTCUSDT",
            current_price=105.0,
            position_side="long",
            atr_value=2.0,
        )
        
        assert new_stop is not None
        assert new_stop > 96.0
    
    def test_trailing_stop_no_update_on_loss(self):
        self.manager.calculate_initial_stop(
            symbol="BTCUSDT",
            entry_price=100.0,
            position_side="long",
            atr_value=2.0,
            timestamp=self.timestamp,
        )
        
        initial_stop = self.manager.get_stop_price("BTCUSDT")
        
        new_stop = self.manager.update_trailing_stop(
            symbol="BTCUSDT",
            current_price=95.0,
            position_side="long",
            atr_value=2.0,
        )
        
        assert new_stop == initial_stop
    
    def test_check_stop_triggered_long(self):
        self.manager.calculate_initial_stop(
            symbol="BTCUSDT",
            entry_price=100.0,
            position_side="long",
            atr_value=2.0,
            timestamp=self.timestamp,
        )
        
        assert not self.manager.check_stop_triggered("BTCUSDT", 98.0, "long")
        assert self.manager.check_stop_triggered("BTCUSDT", 95.0, "long")
    
    def test_check_stop_triggered_short(self):
        self.manager.calculate_initial_stop(
            symbol="ETHUSDT",
            entry_price=100.0,
            position_side="short",
            atr_value=2.0,
            timestamp=self.timestamp,
        )
        
        assert not self.manager.check_stop_triggered("ETHUSDT", 102.0, "short")
        assert self.manager.check_stop_triggered("ETHUSDT", 105.0, "short")
    
    def test_remove_stop(self):
        self.manager.calculate_initial_stop(
            symbol="BTCUSDT",
            entry_price=100.0,
            position_side="long",
            atr_value=2.0,
            timestamp=self.timestamp,
        )
        
        assert self.manager.get_stop_price("BTCUSDT") is not None
        
        self.manager.remove_stop("BTCUSDT")
        
        assert self.manager.get_stop_price("BTCUSDT") is None
    
    def test_get_stop_info(self):
        self.manager.calculate_initial_stop(
            symbol="BTCUSDT",
            entry_price=100.0,
            position_side="long",
            atr_value=2.0,
            timestamp=self.timestamp,
        )
        
        info = self.manager.get_stop_info("BTCUSDT")
        
        assert info is not None
        assert info["symbol"] == "BTCUSDT"
        assert info["stop_price"] == 96.0
        assert info["initial_entry"] == 100.0
        assert info["atr_multiplier"] == 2.0
