"""Detailed performance analysis for trading sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics
import math

import pandas as pd


@dataclass
class PerformanceAnalyzer:
    risk_free_rate: float = 0.02
    trading_days_per_year: int = 365
    
    equity_history: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)
    trade_log: List[Dict] = field(default_factory=list)
    decision_log: List[Dict] = field(default_factory=list)
    
    def record_equity(self, timestamp: pd.Timestamp, equity: float) -> None:
        self.equity_history.append((timestamp, equity))
    
    def record_trade(self, trade: Dict) -> None:
        self.trade_log.append(trade)
    
    def record_decision(self, decision: Dict) -> None:
        self.decision_log.append(decision)
    
    def calculate_returns(self) -> List[float]:
        if len(self.equity_history) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.equity_history)):
            prev_equity = self.equity_history[i-1][1]
            curr_equity = self.equity_history[i][1]
            if prev_equity > 0:
                returns.append((curr_equity - prev_equity) / prev_equity)
        
        return returns
    
    def calculate_sharpe_ratio(self) -> Optional[float]:
        returns = self.calculate_returns()
        if len(returns) < 2:
            return None
        
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return None
        
        periods_per_year = self.trading_days_per_year * 24
        annualized_return = avg_return * periods_per_year
        annualized_std = std_return * math.sqrt(periods_per_year)
        
        return (annualized_return - self.risk_free_rate) / annualized_std
    
    def calculate_sortino_ratio(self) -> Optional[float]:
        returns = self.calculate_returns()
        if len(returns) < 2:
            return None
        
        avg_return = statistics.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return None
        
        downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else abs(downside_returns[0])
        
        if downside_std == 0:
            return None
        
        periods_per_year = self.trading_days_per_year * 24
        annualized_return = avg_return * periods_per_year
        annualized_downside_std = downside_std * math.sqrt(periods_per_year)
        
        return (annualized_return - self.risk_free_rate) / annualized_downside_std
    
    def calculate_max_drawdown(self) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        if not self.equity_history:
            return 0.0, None, None
        
        peak = self.equity_history[0][1]
        peak_timestamp = self.equity_history[0][0]
        max_drawdown = 0.0
        drawdown_start = None
        drawdown_end = None
        
        for timestamp, equity in self.equity_history:
            if equity > peak:
                peak = equity
                peak_timestamp = timestamp
            
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                drawdown_start = peak_timestamp
                drawdown_end = timestamp
        
        return max_drawdown, drawdown_start, drawdown_end
    
    def calculate_calmar_ratio(self) -> Optional[float]:
        if not self.equity_history or len(self.equity_history) < 2:
            return None
        
        start_equity = self.equity_history[0][1]
        end_equity = self.equity_history[-1][1]
        
        if start_equity <= 0:
            return None
        
        total_return = (end_equity - start_equity) / start_equity
        
        start_time = self.equity_history[0][0]
        end_time = self.equity_history[-1][0]
        duration_years = (end_time - start_time).total_seconds() / (365.25 * 24 * 3600)
        
        if duration_years <= 0:
            return None
        
        annualized_return = (1 + total_return) ** (1 / duration_years) - 1 if duration_years >= 1 else total_return
        
        max_dd, _, _ = self.calculate_max_drawdown()
        
        if max_dd == 0:
            return None
        
        return annualized_return / max_dd
    
    def calculate_win_rate(self) -> Optional[float]:
        if not self.trade_log:
            return None
        
        wins = sum(1 for t in self.trade_log if t.get("pnl", 0) > 0)
        return wins / len(self.trade_log)
    
    def calculate_profit_factor(self) -> Optional[float]:
        if not self.trade_log:
            return None
        
        gross_profit = sum(t.get("pnl", 0) for t in self.trade_log if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in self.trade_log if t.get("pnl", 0) < 0))
        
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        
        return gross_profit / gross_loss
    
    def calculate_avg_trade_metrics(self) -> Dict:
        if not self.trade_log:
            return {"avg_win": None, "avg_loss": None, "expectancy": None}
        
        wins = [t.get("pnl", 0) for t in self.trade_log if t.get("pnl", 0) > 0]
        losses = [t.get("pnl", 0) for t in self.trade_log if t.get("pnl", 0) < 0]
        
        avg_win = statistics.mean(wins) if wins else None
        avg_loss = statistics.mean(losses) if losses else None
        
        if wins and losses:
            win_rate = len(wins) / len(self.trade_log)
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        else:
            expectancy = None
        
        return {
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "total_trades": len(self.trade_log),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
        }
    
    def analyze_by_symbol(self) -> Dict[str, Dict]:
        symbol_trades = defaultdict(list)
        
        for trade in self.trade_log:
            symbol = trade.get("symbol", "UNKNOWN")
            symbol_trades[symbol].append(trade)
        
        results = {}
        for symbol, trades in symbol_trades.items():
            pnls = [t.get("pnl", 0) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            results[symbol] = {
                "total_trades": len(trades),
                "total_pnl": sum(pnls),
                "win_rate": len(wins) / len(trades) if trades else None,
                "avg_pnl": statistics.mean(pnls) if pnls else None,
            }
        
        return results
    
    def analyze_decision_quality(self) -> Dict:
        if not self.decision_log:
            return {"hold_ratio": None, "avg_confidence": None}
        
        hold_count = sum(1 for d in self.decision_log if d.get("action") == "HOLD")
        rebalance_count = len(self.decision_log) - hold_count
        
        confidences = [d.get("confidence", 0) for d in self.decision_log if d.get("confidence") is not None]
        
        return {
            "total_decisions": len(self.decision_log),
            "hold_decisions": hold_count,
            "rebalance_decisions": rebalance_count,
            "hold_ratio": hold_count / len(self.decision_log) if self.decision_log else None,
            "avg_confidence": statistics.mean(confidences) if confidences else None,
        }
    
    def calculate_turnover(self) -> Optional[float]:
        if len(self.equity_history) < 2 or not self.trade_log:
            return None
        
        total_volume = sum(abs(t.get("notional", 0)) for t in self.trade_log)
        avg_equity = statistics.mean([e[1] for e in self.equity_history])
        
        if avg_equity <= 0:
            return None
        
        start_time = self.equity_history[0][0]
        end_time = self.equity_history[-1][0]
        duration_days = (end_time - start_time).total_seconds() / (24 * 3600)
        
        if duration_days <= 0:
            return None
        
        daily_turnover = (total_volume / avg_equity) / duration_days
        return daily_turnover
    
    def get_full_report(self) -> Dict:
        if not self.equity_history:
            return {"error": "No equity history recorded"}
        
        start_equity = self.equity_history[0][1]
        end_equity = self.equity_history[-1][1]
        total_return = (end_equity - start_equity) / start_equity if start_equity > 0 else 0
        
        max_dd, dd_start, dd_end = self.calculate_max_drawdown()
        
        return {
            "summary": {
                "start_equity": start_equity,
                "end_equity": end_equity,
                "total_return_pct": total_return * 100,
                "max_drawdown_pct": max_dd * 100,
                "drawdown_start": str(dd_start) if dd_start else None,
                "drawdown_end": str(dd_end) if dd_end else None,
            },
            "risk_metrics": {
                "sharpe_ratio": self.calculate_sharpe_ratio(),
                "sortino_ratio": self.calculate_sortino_ratio(),
                "calmar_ratio": self.calculate_calmar_ratio(),
            },
            "trade_metrics": {
                "win_rate": self.calculate_win_rate(),
                "profit_factor": self.calculate_profit_factor(),
                "daily_turnover": self.calculate_turnover(),
                **self.calculate_avg_trade_metrics(),
            },
            "decision_quality": self.analyze_decision_quality(),
            "by_symbol": self.analyze_by_symbol(),
        }
    
    def format_report(self) -> str:
        report = self.get_full_report()
        
        if "error" in report:
            return f"Error: {report['error']}"
        
        lines = [
            "=" * 60,
            "📊 PERFORMANCE REPORT",
            "=" * 60,
            "",
            "📈 SUMMARY",
            f"  Start Equity:    ${report['summary']['start_equity']:,.2f}",
            f"  End Equity:      ${report['summary']['end_equity']:,.2f}",
            f"  Total Return:    {report['summary']['total_return_pct']:+.2f}%",
            f"  Max Drawdown:    {report['summary']['max_drawdown_pct']:.2f}%",
            "",
            "📉 RISK METRICS",
        ]
        
        risk = report["risk_metrics"]
        if risk["sharpe_ratio"] is not None:
            lines.append(f"  Sharpe Ratio:    {risk['sharpe_ratio']:.2f}")
        if risk["sortino_ratio"] is not None:
            lines.append(f"  Sortino Ratio:   {risk['sortino_ratio']:.2f}")
        if risk["calmar_ratio"] is not None:
            lines.append(f"  Calmar Ratio:    {risk['calmar_ratio']:.2f}")
        
        lines.extend(["", "💼 TRADE METRICS"])
        trade = report["trade_metrics"]
        if trade["win_rate"] is not None:
            lines.append(f"  Win Rate:        {trade['win_rate']*100:.1f}%")
        if trade["profit_factor"] is not None:
            lines.append(f"  Profit Factor:   {trade['profit_factor']:.2f}")
        if trade["total_trades"]:
            lines.append(f"  Total Trades:    {trade['total_trades']}")
        if trade["expectancy"] is not None:
            lines.append(f"  Expectancy:      ${trade['expectancy']:.2f}")
        
        lines.extend(["", "🎯 DECISION QUALITY"])
        decision = report["decision_quality"]
        if decision["total_decisions"]:
            lines.append(f"  Total Decisions: {decision['total_decisions']}")
            lines.append(f"  HOLD Ratio:      {decision['hold_ratio']*100:.1f}%")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)
