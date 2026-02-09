"""Backtesting engine for strategy evaluation.

Processes historical data bar-by-bar without time.sleep(), simulating
order fills with configurable models. Generates performance reports
using PerformanceAnalyzer.

Usage::

    bt = Backtester(
        strategy=MACrossoverStrategy(short_window=10, long_window=50),
        symbols=["BTCUSDT", "ETHUSDT"],
        initial_capital=100_000,
    )
    result = bt.run(historical_data)
    print(result["performance"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from tools import FinancialTools
from strategies.base import Strategy, StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    initial_capital: float = 100_000.0
    commission_rate: float = 0.0005
    slippage: float = 0.0005
    max_leverage: float = 5.0
    per_symbol_max_exposure: float = 3.0
    bar_interval: str = "1m"
    lookback_bars: int = 200


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    config: BacktestConfig
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    final_equity: float = 0.0
    total_return: float = 0.0
    total_trades: int = 0
    performance: Dict[str, Any] = field(default_factory=dict)
    duration_bars: int = 0

    def summary(self) -> str:
        lines = [
            f"Backtest Result ({self.duration_bars} bars)",
            f"  Initial Capital: {self.config.initial_capital:.2f}",
            f"  Final Equity:    {self.final_equity:.2f}",
            f"  Total Return:    {self.total_return:.2%}",
            f"  Total Trades:    {self.total_trades}",
        ]
        if self.performance:
            lines.append(f"  Sharpe Ratio:    {self.performance.get('sharpe_ratio', 'N/A')}")
            lines.append(f"  Max Drawdown:    {self.performance.get('max_drawdown', 'N/A')}")
            lines.append(f"  Win Rate:        {self.performance.get('win_rate', 'N/A')}")
        return "\n".join(lines)


class Backtester:
    """Bar-by-bar backtesting engine.

    Processes historical data sequentially, calling the strategy at each
    decision point, executing simulated orders, and tracking P&L.
    """

    def __init__(
        self,
        strategy: Strategy,
        symbols: List[str],
        config: Optional[BacktestConfig] = None,
        decision_every_n_bars: int = 1,
    ) -> None:
        self._strategy = strategy
        self._symbols = symbols
        self._config = config or BacktestConfig()
        self._decision_every = max(1, decision_every_n_bars)

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """Run backtest on historical data.

        Parameters
        ----------
        data:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume.
            Must be sorted by timestamp.

        Returns
        -------
        BacktestResult with equity curve, trades, and performance metrics.
        """
        self._strategy.initialize(self._symbols, self._config.__dict__)

        balance = self._config.initial_capital
        positions: Dict[str, Dict[str, float]] = {}  # symbol -> {qty, entry_price}
        equity_curve: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []

        # Group data by timestamp for bar-by-bar processing
        if "timestamp" in data.columns:
            timestamps = sorted(data["timestamp"].unique())
        else:
            timestamps = sorted(data.index.unique())

        bar_count = 0
        lookback = self._config.lookback_bars

        for ts_idx, ts in enumerate(timestamps):
            bar_count += 1
            if isinstance(ts, str):
                ts = pd.Timestamp(ts)

            # Get current bar prices
            bar_data = data[data["timestamp"] == ts] if "timestamp" in data.columns else data.loc[ts:ts]
            current_prices: Dict[str, float] = {}
            for _, row in bar_data.iterrows():
                sym = row.get("symbol", "")
                if sym in self._symbols:
                    current_prices[sym] = float(row.get("close", 0))

            if not current_prices:
                continue

            # Mark to market
            unrealized_pnl = 0.0
            for sym, pos in positions.items():
                price = current_prices.get(sym, pos["entry_price"])
                unrealized_pnl += (price - pos["entry_price"]) * pos["qty"]

            equity = balance + unrealized_pnl
            equity_curve.append({
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "equity": equity,
                "balance": balance,
                "unrealized_pnl": unrealized_pnl,
            })

            # Decision point
            if bar_count % self._decision_every == 0 and ts_idx >= lookback:
                # Get lookback window
                start_idx = max(0, ts_idx - lookback)
                window_timestamps = timestamps[start_idx:ts_idx + 1]
                if "timestamp" in data.columns:
                    window_data = data[data["timestamp"].isin(window_timestamps)]
                else:
                    window_data = data.iloc[start_idx:ts_idx + 1]

                current_exposures = {}
                for sym in self._symbols:
                    pos = positions.get(sym)
                    if pos and equity > 0:
                        current_exposures[sym] = (pos["qty"] * current_prices.get(sym, 0)) / equity
                    else:
                        current_exposures[sym] = 0.0

                try:
                    tools = FinancialTools(window_data)
                    signal = self._strategy.generate_signal(
                        timestamp=pd.Timestamp(ts),
                        market_data=window_data,
                        tools=tools,
                        current_positions=current_exposures,
                    )
                except Exception as exc:
                    logger.warning("Strategy error at %s: %s", ts, exc)
                    signal = StrategySignal(
                        exposures={s: current_exposures.get(s, 0.0) for s in self._symbols},
                        action="HOLD",
                    )

                decisions.append({
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "action": signal.action,
                    "exposures": signal.exposures,
                    "reasoning": signal.reasoning,
                })

                if signal.action == "REBALANCE":
                    # Execute rebalancing
                    for sym in self._symbols:
                        target_exp = signal.exposures.get(sym, 0.0)
                        # Clip to limits
                        target_exp = max(-self._config.per_symbol_max_exposure,
                                        min(self._config.per_symbol_max_exposure, target_exp))

                        price = current_prices.get(sym)
                        if price is None or price <= 0 or equity <= 0:
                            continue

                        target_notional = target_exp * equity
                        target_qty = target_notional / price
                        current_pos = positions.get(sym)
                        current_qty = current_pos["qty"] if current_pos else 0.0

                        if abs(target_qty - current_qty) < 1e-8:
                            continue

                        # Close existing position
                        if current_pos and abs(current_qty) > 1e-8:
                            close_price = price * (1 - self._config.slippage if current_qty > 0 else 1 + self._config.slippage)
                            realized = (close_price - current_pos["entry_price"]) * current_qty
                            commission = abs(current_qty * close_price) * self._config.commission_rate
                            balance += realized - commission
                            trades.append({
                                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                                "symbol": sym,
                                "action": "close",
                                "quantity": current_qty,
                                "price": close_price,
                                "realized_pnl": realized - commission,
                                "commission": commission,
                            })
                            self._strategy.on_trade_result(
                                sym, realized - commission,
                                current_pos["entry_price"], close_price,
                            )
                            del positions[sym]

                        # Open new position
                        if abs(target_qty) > 1e-8:
                            open_price = price * (1 + self._config.slippage if target_qty > 0 else 1 - self._config.slippage)
                            commission = abs(target_qty * open_price) * self._config.commission_rate
                            balance -= commission
                            positions[sym] = {"qty": target_qty, "entry_price": open_price}
                            trades.append({
                                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                                "symbol": sym,
                                "action": "open",
                                "quantity": target_qty,
                                "price": open_price,
                                "realized_pnl": -commission,
                                "commission": commission,
                            })

        # Final mark-to-market
        final_unrealized = 0.0
        last_prices = {}
        for sym in self._symbols:
            if "timestamp" in data.columns:
                sym_data = data[data.get("symbol", pd.Series()) == sym]
            else:
                sym_data = data
            if len(sym_data) > 0:
                last_prices[sym] = float(sym_data.iloc[-1].get("close", 0))

        for sym, pos in positions.items():
            price = last_prices.get(sym, pos["entry_price"])
            final_unrealized += (price - pos["entry_price"]) * pos["qty"]

        final_equity = balance + final_unrealized
        total_return = (final_equity - self._config.initial_capital) / self._config.initial_capital

        # Calculate performance metrics
        performance = self._calculate_performance(equity_curve, trades)

        return BacktestResult(
            config=self._config,
            equity_curve=equity_curve,
            trades=trades,
            decisions=decisions,
            final_equity=final_equity,
            total_return=total_return,
            total_trades=len(trades),
            performance=performance,
            duration_bars=bar_count,
        )

    def _calculate_performance(
        self, equity_curve: List[Dict], trades: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate performance metrics from backtest results."""
        if len(equity_curve) < 2:
            return {}

        equities = [e["equity"] for e in equity_curve]

        # Returns
        returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

        if not returns:
            return {}

        import statistics
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0

        # Sharpe (annualized assuming minute bars, ~525600 bars/year)
        sharpe = (avg_return / std_return * (525600 ** 0.5)) if std_return > 0 else 0.0

        # Max drawdown
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Win rate from trades
        pnls = [t.get("realized_pnl", 0) for t in trades if t.get("action") == "close"]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "total_return": round((equities[-1] - equities[0]) / equities[0], 4) if equities[0] > 0 else 0,
            "avg_return_per_bar": round(avg_return, 8),
            "volatility": round(std_return, 8),
            "num_trades": len(pnls),
            "num_winning": wins,
            "num_losing": len(pnls) - wins,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        }
