"""Reporting for real-time leveraged trading."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RealTimeReporter:
    """Collect live statistics, surface warnings, and produce final summaries."""

    print_interval_seconds: float = 60.0
    _last_print_ts: Optional[pd.Timestamp] = field(default=None, init=False)
    ticks: List[Dict[str, Any]] = field(default_factory=list, init=False)
    warnings: List[Dict[str, str]] = field(default_factory=list, init=False)

    def record_tick(self, timestamp: pd.Timestamp, account: Any, prices: Dict[str, float]) -> None:
        snapshot = {
            "timestamp": timestamp,
            "equity": float(account.equity),
            "balance": float(account.balance),
            "unrealized_pnl": float(account.unrealized_pnl),
            "margin_used": float(account.margin_used),
            "available_margin": float(account.available_margin),
            "prices": dict(prices),
        }
        self.ticks.append(snapshot)

        if self._should_print(timestamp):
            open_symbols = ", ".join(
                f"{sym}:{pos.quantity:.4f}@{pos.entry_price:.2f}"
                for sym, pos in getattr(account, "positions", {}).items()
            )
            print(
                f"[{timestamp}] equity={account.equity:.2f} balance={account.balance:.2f} "
                f"unrealized={account.unrealized_pnl:.2f} margin_used={account.margin_used:.2f} "
                f"available_margin={account.available_margin:.2f} positions=[{open_symbols}]"
            )
            self._last_print_ts = timestamp

    def record_warning(self, timestamp: pd.Timestamp, message: str) -> None:
        warning = {"timestamp": timestamp, "message": message}
        self.warnings.append(warning)
        print(f"[WARN {timestamp}] {message}")

    def finalize(
        self,
        equity_history: List[Dict[str, Any]],
        trade_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not equity_history:
            report = {
                "warnings": self.warnings,
                "number_of_trades": len(trade_log),
                "start_equity": None,
                "end_equity": None,
                "total_return": None,
                "max_drawdown": None,
            }
            print("\n=== Final Trading Summary ===")
            print("No equity history captured. Check data connectivity and agent responses.")
            return report

        df = pd.DataFrame(equity_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        equity_series = df["equity"]

        start_equity = float(equity_series.iloc[0])
        end_equity = float(equity_series.iloc[-1])
        duration_seconds = max(
            1.0,
            (df.index[-1] - df.index[0]).total_seconds(),
        )
        total_return = (end_equity / start_equity - 1.0) if start_equity > 0 else 0.0

        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = float(abs(drawdown.min())) if not drawdown.empty else 0.0

        total_realized = sum(float(entry.get("realized_pnl", 0.0)) for entry in trade_log)
        total_unrealized = float(df["unrealized_pnl"].iloc[-1])
        num_trades = len(trade_log)

        winning_trades = [t for t in trade_log if t.get("realized_pnl", 0) > 0]
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0

        equity_returns = equity_series.pct_change().dropna()
        if not equity_returns.empty and equity_returns.std() > 0:
            avg_interval = (df.index[-1] - df.index[0]).total_seconds() / len(df) if len(df) > 1 else 60
            periods_per_year = (365 * 24 * 60 * 60) / avg_interval
            sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * math.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0.0

        seconds_per_year = 365 * 24 * 60 * 60
        if start_equity > 0 and duration_seconds >= 24 * 60 * 60:
            growth = end_equity / start_equity
            try:
                annualized_return = math.exp(math.log(growth) * (seconds_per_year / duration_seconds)) - 1.0
            except ValueError:
                annualized_return = None
        else:
            annualized_return = None

        report = {
            "start_equity": start_equity,
            "end_equity": end_equity,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "number_of_trades": num_trades,
            "total_realized_pnl": total_realized,
            "ending_unrealized_pnl": total_unrealized,
            "warnings": self.warnings,
        }

        print("\n=== Final Trading Summary ===")
        print(f"Start equity: {start_equity:.2f} USDT")
        print(f"End equity:   {end_equity:.2f} USDT")
        print(f"Total return: {total_return * 100:.2f}%")
        if annualized_return is None:
            print("Annualized:   N/A (session shorter than 1 day or invalid growth)")
        else:
            print(f"Annualized:   {annualized_return * 100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Win Rate:     {win_rate * 100:.2f}%")
        print(f"Max drawdown: {max_drawdown * 100:.2f}%")
        print(f"Realized PnL: {total_realized:.2f} USDT")
        print(f"Unrealized PnL: {total_unrealized:.2f} USDT")
        print(f"Trades executed: {num_trades}")
        if self.warnings:
            print("Warnings encountered:")
            for warning in self.warnings:
                print(f"- [{warning['timestamp']}] {warning['message']}")

        return report

    def _should_print(self, timestamp: pd.Timestamp) -> bool:
        if self._last_print_ts is None:
            return True
        delta = (timestamp - self._last_print_ts).total_seconds()
        return delta >= self.print_interval_seconds
