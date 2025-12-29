"""Risk management module for trading operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class RiskLimits:
    """Configuration for risk control parameters."""
    
    max_drawdown: float = 0.20
    max_daily_loss: float = 0.05
    max_position_duration_seconds: int = 14400
    max_correlation: float = 0.85
    max_single_loss: float = 0.02
    min_equity_floor: float = 0.10
    max_consecutive_losses: int = 5
    cooldown_after_loss_seconds: int = 300


@dataclass
class RiskState:
    """Track current risk metrics."""
    
    peak_equity: float = 0.0
    daily_starting_equity: float = 0.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    last_loss_time: Optional[datetime] = None
    position_open_times: Dict[str, datetime] = field(default_factory=dict)
    daily_trades: int = 0
    current_date: Optional[str] = None


class RiskManager:
    """Enforce risk limits and provide risk-adjusted position sizing."""
    
    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()
        self.state = RiskState()
        self.violations: List[Dict[str, Any]] = []
    
    def initialize(self, initial_equity: float) -> None:
        self.state.peak_equity = initial_equity
        self.state.daily_starting_equity = initial_equity
        self.state.current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    def update_equity(self, current_equity: float, timestamp: datetime) -> None:
        current_date = timestamp.strftime("%Y-%m-%d")
        if self.state.current_date != current_date:
            self.state.daily_starting_equity = current_equity
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.current_date = current_date
        
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        
        self.state.daily_pnl = current_equity - self.state.daily_starting_equity
    
    def record_trade_result(self, pnl: float, timestamp: datetime) -> None:
        self.state.daily_trades += 1
        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_time = timestamp
        else:
            self.state.consecutive_losses = 0
    
    def record_position_open(self, symbol: str, timestamp: datetime) -> None:
        self.state.position_open_times[symbol] = timestamp
    
    def record_position_close(self, symbol: str) -> None:
        self.state.position_open_times.pop(symbol, None)
    
    def check_drawdown(self, current_equity: float) -> Tuple[bool, Optional[str]]:
        if self.state.peak_equity <= 0:
            return True, None
        
        drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
        if drawdown > self.limits.max_drawdown:
            return False, f"Max drawdown exceeded: {drawdown:.2%} > {self.limits.max_drawdown:.2%}"
        return True, None
    
    def check_daily_loss(self, current_equity: float) -> Tuple[bool, Optional[str]]:
        if self.state.daily_starting_equity <= 0:
            return True, None
        
        daily_loss = (self.state.daily_starting_equity - current_equity) / self.state.daily_starting_equity
        if daily_loss > self.limits.max_daily_loss:
            return False, f"Daily loss limit exceeded: {daily_loss:.2%} > {self.limits.max_daily_loss:.2%}"
        return True, None
    
    def check_equity_floor(self, current_equity: float, initial_equity: float) -> Tuple[bool, Optional[str]]:
        if initial_equity <= 0:
            return True, None
        
        floor = initial_equity * self.limits.min_equity_floor
        if current_equity < floor:
            return False, f"Equity below floor: {current_equity:.2f} < {floor:.2f}"
        return True, None
    
    def check_position_duration(self, symbol: str, current_time: datetime) -> Tuple[bool, Optional[str]]:
        open_time = self.state.position_open_times.get(symbol)
        if open_time is None:
            return True, None
        
        duration = (current_time - open_time).total_seconds()
        if duration > self.limits.max_position_duration_seconds:
            return False, f"{symbol} position held too long: {duration/3600:.1f}h > {self.limits.max_position_duration_seconds/3600:.1f}h"
        return True, None
    
    def check_consecutive_losses(self) -> Tuple[bool, Optional[str]]:
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            return False, f"Consecutive losses: {self.state.consecutive_losses} >= {self.limits.max_consecutive_losses}"
        return True, None
    
    def check_cooldown(self, current_time: datetime) -> Tuple[bool, Optional[str]]:
        if self.state.last_loss_time is None:
            return True, None
        
        elapsed = (current_time - self.state.last_loss_time).total_seconds()
        if self.state.consecutive_losses >= 3 and elapsed < self.limits.cooldown_after_loss_seconds:
            remaining = self.limits.cooldown_after_loss_seconds - elapsed
            return False, f"In cooldown period: {remaining:.0f}s remaining"
        return True, None
    
    def check_order(
        self,
        symbol: str,
        target_exposure: float,
        current_equity: float,
        initial_equity: float,
        current_time: datetime,
    ) -> Tuple[bool, List[str]]:
        violations: List[str] = []
        
        ok, msg = self.check_drawdown(current_equity)
        if not ok:
            violations.append(msg)
        
        ok, msg = self.check_daily_loss(current_equity)
        if not ok:
            violations.append(msg)
        
        ok, msg = self.check_equity_floor(current_equity, initial_equity)
        if not ok:
            violations.append(msg)
        
        ok, msg = self.check_consecutive_losses()
        if not ok:
            violations.append(msg)
        
        ok, msg = self.check_cooldown(current_time)
        if not ok:
            violations.append(msg)
        
        if violations:
            self.violations.append({
                "timestamp": current_time.isoformat(),
                "symbol": symbol,
                "target_exposure": target_exposure,
                "violations": violations,
            })
        
        return len(violations) == 0, violations
    
    def get_position_age_warnings(self, current_time: datetime) -> List[str]:
        warnings = []
        for symbol, open_time in self.state.position_open_times.items():
            ok, msg = self.check_position_duration(symbol, current_time)
            if not ok:
                warnings.append(msg)
        return warnings
    
    def should_force_close(self, current_equity: float, initial_equity: float) -> bool:
        ok_dd, _ = self.check_drawdown(current_equity)
        ok_floor, _ = self.check_equity_floor(current_equity, initial_equity)
        return not ok_dd or not ok_floor
    
    def adjust_exposure_for_risk(
        self,
        target_exposure: float,
        current_equity: float,
        volatility: Optional[float] = None,
    ) -> float:
        if self.state.peak_equity <= 0:
            return target_exposure
        
        drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
        
        if drawdown > self.limits.max_drawdown * 0.5:
            scale = 1.0 - (drawdown / self.limits.max_drawdown)
            scale = max(0.25, min(1.0, scale))
            target_exposure *= scale
        
        if self.state.consecutive_losses >= 2:
            loss_scale = 1.0 - (self.state.consecutive_losses * 0.15)
            loss_scale = max(0.3, loss_scale)
            target_exposure *= loss_scale
        
        if volatility is not None and volatility > 0.05:
            vol_scale = 0.03 / volatility
            vol_scale = max(0.5, min(1.0, vol_scale))
            target_exposure *= vol_scale
        
        return target_exposure
    
    def get_risk_report(self, current_equity: float) -> Dict[str, Any]:
        drawdown = 0.0
        if self.state.peak_equity > 0:
            drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
        
        daily_return = 0.0
        if self.state.daily_starting_equity > 0:
            daily_return = (current_equity - self.state.daily_starting_equity) / self.state.daily_starting_equity
        
        return {
            "current_equity": current_equity,
            "peak_equity": self.state.peak_equity,
            "drawdown": drawdown,
            "drawdown_limit": self.limits.max_drawdown,
            "daily_pnl": self.state.daily_pnl,
            "daily_return": daily_return,
            "daily_loss_limit": self.limits.max_daily_loss,
            "consecutive_losses": self.state.consecutive_losses,
            "daily_trades": self.state.daily_trades,
            "open_positions": len(self.state.position_open_times),
            "violation_count": len(self.violations),
        }

