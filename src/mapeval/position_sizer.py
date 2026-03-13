"""Position sizing strategies including pyramid scaling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class ScalingStrategy(Enum):
    FIXED = "fixed"
    PYRAMID_IN = "pyramid_in"
    PYRAMID_OUT = "pyramid_out"
    VOLATILITY_SCALED = "volatility_scaled"
    KELLY = "kelly"


@dataclass
class PositionLevel:
    level: int
    entry_price: float
    size_fraction: float
    triggered: bool = False


@dataclass
class PyramidPlan:
    symbol: str
    direction: str
    base_size: float
    levels: List[PositionLevel] = field(default_factory=list)
    total_filled: float = 0.0
    avg_entry_price: float = 0.0


@dataclass
class PositionSizer:
    max_position_size: float = 5.0
    pyramid_levels: int = 3
    level_spacing_pct: float = 0.02
    scaling_factor: float = 0.7
    
    active_pyramids: Dict[str, PyramidPlan] = field(default_factory=dict)
    
    def calculate_base_size(
        self,
        equity: float,
        confidence: float,
        volatility: Optional[float],
        max_leverage: float,
    ) -> float:
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        if volatility is not None and volatility > 0:
            target_risk = 0.02
            vol_adjusted_size = target_risk / volatility
            vol_adjusted_size = min(vol_adjusted_size, max_leverage * 0.5)
        else:
            vol_adjusted_size = max_leverage * 0.3
        
        base_size = vol_adjusted_size * confidence_multiplier
        base_size = min(base_size, self.max_position_size)
        
        return base_size
    
    def create_pyramid_plan(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        base_size: float,
    ) -> PyramidPlan:
        levels = []
        remaining_fraction = 1.0
        
        for i in range(self.pyramid_levels):
            if direction == "long":
                entry_price = current_price * (1 - self.level_spacing_pct * i)
            else:
                entry_price = current_price * (1 + self.level_spacing_pct * i)
            
            if i == 0:
                size_fraction = 1.0 / self.pyramid_levels
            else:
                size_fraction = (remaining_fraction * self.scaling_factor) / (self.pyramid_levels - i)
            
            remaining_fraction -= size_fraction
            
            levels.append(PositionLevel(
                level=i + 1,
                entry_price=entry_price,
                size_fraction=size_fraction,
                triggered=(i == 0),
            ))
        
        plan = PyramidPlan(
            symbol=symbol,
            direction=direction,
            base_size=base_size,
            levels=levels,
            total_filled=base_size * levels[0].size_fraction,
            avg_entry_price=current_price,
        )
        
        self.active_pyramids[symbol] = plan
        return plan
    
    def check_pyramid_triggers(
        self,
        symbol: str,
        current_price: float,
    ) -> List[Tuple[int, float]]:
        if symbol not in self.active_pyramids:
            return []
        
        plan = self.active_pyramids[symbol]
        triggered_levels = []
        
        for level in plan.levels:
            if level.triggered:
                continue
            
            if plan.direction == "long":
                should_trigger = current_price <= level.entry_price
            else:
                should_trigger = current_price >= level.entry_price
            
            if should_trigger:
                level.triggered = True
                add_size = plan.base_size * level.size_fraction
                
                total_before = plan.total_filled
                plan.total_filled += add_size
                plan.avg_entry_price = (
                    (plan.avg_entry_price * total_before + current_price * add_size)
                    / plan.total_filled
                )
                
                triggered_levels.append((level.level, add_size))
        
        return triggered_levels
    
    def calculate_scale_out_levels(
        self,
        symbol: str,
        current_position: float,
        entry_price: float,
        current_price: float,
        take_profit_pct: float = 0.05,
    ) -> List[Tuple[float, float]]:
        if current_position == 0:
            return []
        
        direction = "long" if current_position > 0 else "short"
        
        if direction == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        if profit_pct < take_profit_pct:
            return []
        
        scale_out_levels = []
        remaining = abs(current_position)
        
        targets = [
            (take_profit_pct, 0.25),
            (take_profit_pct * 1.5, 0.25),
            (take_profit_pct * 2.0, 0.25),
        ]
        
        for target_pct, fraction in targets:
            if profit_pct >= target_pct:
                close_size = remaining * fraction
                scale_out_levels.append((target_pct, close_size))
                remaining -= close_size
        
        return scale_out_levels
    
    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_kelly_fraction: float = 0.25,
    ) -> float:
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        win_loss_ratio = abs(avg_win / avg_loss)
        
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        kelly = max(0, min(kelly, max_kelly_fraction))
        
        fractional_kelly = kelly * 0.5
        
        return fractional_kelly
    
    def get_recommended_size(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        equity: float,
        confidence: float,
        volatility: Optional[float],
        max_leverage: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        strategy: ScalingStrategy = ScalingStrategy.VOLATILITY_SCALED,
    ) -> Dict:
        base_size = self.calculate_base_size(
            equity=equity,
            confidence=confidence,
            volatility=volatility,
            max_leverage=max_leverage,
        )
        
        result = {
            "symbol": symbol,
            "direction": direction,
            "strategy": strategy.value,
            "base_size": base_size,
            "recommended_size": base_size,
            "pyramid_plan": None,
            "kelly_fraction": None,
        }
        
        if strategy == ScalingStrategy.PYRAMID_IN:
            plan = self.create_pyramid_plan(
                symbol=symbol,
                direction=direction,
                current_price=current_price,
                base_size=base_size,
            )
            result["recommended_size"] = plan.total_filled
            result["pyramid_plan"] = {
                "levels": len(plan.levels),
                "initial_fill": plan.total_filled,
                "full_size": base_size,
            }
        
        elif strategy == ScalingStrategy.KELLY:
            if all(v is not None for v in [win_rate, avg_win, avg_loss]):
                kelly = self.calculate_kelly_size(win_rate, avg_win, avg_loss)
                result["recommended_size"] = base_size * kelly / 0.25
                result["kelly_fraction"] = kelly
        
        return result
    
    def remove_pyramid(self, symbol: str) -> None:
        if symbol in self.active_pyramids:
            del self.active_pyramids[symbol]
    
    def get_pyramid_status(self, symbol: str) -> Optional[Dict]:
        if symbol not in self.active_pyramids:
            return None
        
        plan = self.active_pyramids[symbol]
        return {
            "symbol": plan.symbol,
            "direction": plan.direction,
            "base_size": plan.base_size,
            "total_filled": plan.total_filled,
            "avg_entry_price": plan.avg_entry_price,
            "levels_triggered": sum(1 for l in plan.levels if l.triggered),
            "total_levels": len(plan.levels),
            "fill_percentage": (plan.total_filled / plan.base_size) * 100 if plan.base_size > 0 else 0,
        }
