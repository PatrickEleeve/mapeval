"""Dynamic stop-loss management based on ATR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum

import pandas as pd


class StopType(Enum):
    FIXED = "fixed"
    ATR_TRAILING = "atr_trailing"
    PERCENTAGE_TRAILING = "percentage_trailing"


@dataclass
class StopLossLevel:
    symbol: str
    stop_price: float
    stop_type: StopType
    atr_multiplier: float
    initial_entry: float
    highest_favorable: float
    created_at: pd.Timestamp


@dataclass 
class StopLossManager:
    atr_multiplier: float = 2.0
    trailing_activation_pct: float = 0.01
    stop_levels: Dict[str, StopLossLevel] = field(default_factory=dict)
    
    def calculate_initial_stop(
        self,
        symbol: str,
        entry_price: float,
        position_side: str,
        atr_value: Optional[float],
        timestamp: pd.Timestamp,
    ) -> float:
        if atr_value is None or atr_value <= 0:
            default_pct = 0.02
            if position_side == "long":
                stop_price = entry_price * (1 - default_pct)
            else:
                stop_price = entry_price * (1 + default_pct)
        else:
            stop_distance = atr_value * self.atr_multiplier
            if position_side == "long":
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
        
        self.stop_levels[symbol] = StopLossLevel(
            symbol=symbol,
            stop_price=stop_price,
            stop_type=StopType.ATR_TRAILING,
            atr_multiplier=self.atr_multiplier,
            initial_entry=entry_price,
            highest_favorable=entry_price,
            created_at=timestamp,
        )
        
        return stop_price
    
    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        position_side: str,
        atr_value: Optional[float],
    ) -> Optional[float]:
        if symbol not in self.stop_levels:
            return None
        
        level = self.stop_levels[symbol]
        
        if position_side == "long":
            if current_price > level.highest_favorable:
                level.highest_favorable = current_price
                
                profit_pct = (current_price - level.initial_entry) / level.initial_entry
                if profit_pct >= self.trailing_activation_pct:
                    if atr_value and atr_value > 0:
                        new_stop = current_price - (atr_value * self.atr_multiplier)
                    else:
                        new_stop = current_price * 0.98
                    
                    if new_stop > level.stop_price:
                        level.stop_price = new_stop
        else:
            if current_price < level.highest_favorable:
                level.highest_favorable = current_price
                
                profit_pct = (level.initial_entry - current_price) / level.initial_entry
                if profit_pct >= self.trailing_activation_pct:
                    if atr_value and atr_value > 0:
                        new_stop = current_price + (atr_value * self.atr_multiplier)
                    else:
                        new_stop = current_price * 1.02
                    
                    if new_stop < level.stop_price:
                        level.stop_price = new_stop
        
        return level.stop_price
    
    def check_stop_triggered(
        self,
        symbol: str,
        current_price: float,
        position_side: str,
    ) -> bool:
        if symbol not in self.stop_levels:
            return False
        
        level = self.stop_levels[symbol]
        
        if position_side == "long":
            return current_price <= level.stop_price
        else:
            return current_price >= level.stop_price
    
    def get_stop_price(self, symbol: str) -> Optional[float]:
        if symbol not in self.stop_levels:
            return None
        return self.stop_levels[symbol].stop_price
    
    def remove_stop(self, symbol: str) -> None:
        if symbol in self.stop_levels:
            del self.stop_levels[symbol]
    
    def get_all_stops(self) -> Dict[str, float]:
        return {sym: level.stop_price for sym, level in self.stop_levels.items()}
    
    def get_stop_info(self, symbol: str) -> Optional[Dict]:
        if symbol not in self.stop_levels:
            return None
        level = self.stop_levels[symbol]
        return {
            "symbol": level.symbol,
            "stop_price": level.stop_price,
            "stop_type": level.stop_type.value,
            "initial_entry": level.initial_entry,
            "highest_favorable": level.highest_favorable,
            "atr_multiplier": level.atr_multiplier,
        }
