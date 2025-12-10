"""Reporting for real-time leveraged trading."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from rich.console import Console
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.box import ROUNDED
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


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

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Log a trade execution."""
        ts = trade.get("timestamp", "")
        sym = trade.get("symbol")
        action = trade.get("action")
        qty = trade.get("quantity")
        price = trade.get("price")
        pnl = trade.get("realized_pnl", 0.0)
        fee = trade.get("fee", 0.0)
        print(f"[TRADE {ts}] {action} {sym} qty={qty:.4f} @ {price:.2f} pnl={pnl:.2f} fee={fee:.2f}")

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
        total_volume = sum(
            abs(float(entry.get("quantity", 0.0)) * float(entry.get("price", 0.0)))
            for entry in trade_log
        )
        total_unrealized = float(df["unrealized_pnl"].iloc[-1])
        num_trades = len(trade_log)

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
            "number_of_trades": num_trades,
            "total_volume": total_volume,
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
        print(f"Max drawdown: {max_drawdown * 100:.2f}%")
        print(f"Total Volume: {total_volume:,.2f} USDT")
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
    
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


@dataclass
class RichReporter(RealTimeReporter):
    """A rich-based UI reporter for real-time trading."""

    def __post_init__(self) -> None:
        self.console = Console()
        self.layout = Layout()
        self.live: Optional[Live] = None
        self._recent_logs: List[str] = []
        self._setup_layout()

    def _setup_layout(self) -> None:
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10),
        )
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right", ratio=2),
        )
        self.layout["left"].split(
            Layout(name="account", ratio=1),
            Layout(name="market", ratio=1),
        )
        self.layout["right"].update(Panel(Text("Waiting for data..."), title="Positions"))
        self.layout["footer"].update(Panel(Text("Initializing..."), title="Logs"))
        self.layout["header"].update(Panel(Text("MAPEval Real-Time Trading"), style="bold white on blue"))

    def start(self) -> None:
        if RICH_AVAILABLE:
            self.live = Live(
                self.layout,
                console=self.console,
                refresh_per_second=4,
                screen=True
            )
            self.live.start()

    def stop(self) -> None:
        if self.live:
            self.live.stop()

    def record_tick(self, timestamp: pd.Timestamp, account: Any, prices: Dict[str, float]) -> None:
        # Call super to store tick data (used for final report)
        super().record_tick(timestamp, account, prices)
        
        if not RICH_AVAILABLE:
            return

        # Update Header
        self.layout["header"].update(
            Panel(
                Text(f"MAPEval Real-Time Trading | {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}", justify="center"),
                style="bold white on blue",
                box=ROUNDED
            )
        )

        # Update Account Table
        account_table = Table(box=ROUNDED, expand=True)
        account_table.add_column("Metric", style="cyan")
        account_table.add_column("Value", justify="right", style="green")
        
        account_table.add_row("Balance", f"{account.balance:,.2f}")
        account_table.add_row("Equity", f"{account.equity:,.2f}")
        account_table.add_row("Unrealized PnL", f"{account.unrealized_pnl:,.2f}")
        account_table.add_row("Margin Used", f"{account.margin_used:,.2f}")
        account_table.add_row("Avail Margin", f"{account.available_margin:,.2f}")
        
        self.layout["account"].update(Panel(account_table, title="Account Summary", border_style="blue"))

        # Update Market Data
        market_table = Table(box=ROUNDED, expand=True)
        market_table.add_column("Symbol", style="yellow")
        market_table.add_column("Price", justify="right")
        
        sorted_symbols = sorted(prices.keys())
        for sym in sorted_symbols:
            price = prices[sym]
            market_table.add_row(sym, f"{price:,.4f}")
            
        self.layout["market"].update(Panel(market_table, title="Market Data", border_style="yellow"))

        # Update Positions
        pos_table = Table(box=ROUNDED, expand=True)
        pos_table.add_column("Symbol", style="yellow")
        pos_table.add_column("Size", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Mark", justify="right")
        pos_table.add_column("PnL", justify="right")
        pos_table.add_column("Lev", justify="right")

        positions = getattr(account, "positions", {})
        if not positions:
            pos_panel = Panel(Text("No open positions", justify="center"), title="Positions", border_style="green")
        else:
            for sym, pos in positions.items():
                mark_price = prices.get(sym, pos.entry_price)
                pnl = (mark_price - pos.entry_price) * pos.quantity
                pnl_style = "green" if pnl >= 0 else "red"
                pos_table.add_row(
                    sym,
                    f"{pos.quantity:.4f}",
                    f"{pos.entry_price:.4f}",
                    f"{mark_price:.4f}",
                    Text(f"{pnl:+.2f}", style=pnl_style),
                    f"{pos.leverage:.1f}x"
                )
            pos_panel = Panel(pos_table, title="Positions", border_style="green")
        
        self.layout["right"].update(pos_panel)

        # Update Logs
        log_text = Text()
        for log in self._recent_logs[-10:]:
            log_text.append(log + "\n")
        self.layout["footer"].update(Panel(log_text, title="Recent Logs", border_style="white"))

    def record_warning(self, timestamp: pd.Timestamp, message: str) -> None:
        # Store for final report
        warning = {"timestamp": timestamp, "message": message}
        self.warnings.append(warning)
        
        if RICH_AVAILABLE:
            log_msg = f"[WARN {timestamp.strftime('%H:%M:%S')}] {message}"
            self._recent_logs.append(log_msg)
            # Limit log history
            if len(self._recent_logs) > 20:
                self._recent_logs.pop(0)
        else:
            print(f"[WARN {timestamp}] {message}")

    def record_trade(self, trade: Dict[str, Any]) -> None:
        if not RICH_AVAILABLE:
            super().record_trade(trade)
            return

        # Add to logs
        try:
            ts = pd.Timestamp(trade["timestamp"]).strftime('%H:%M:%S')
        except Exception:
            ts = str(trade.get("timestamp", ""))

        sym = trade.get("symbol")
        action = str(trade.get("action", "")).upper()
        qty = float(trade.get("quantity", 0))
        price = float(trade.get("price", 0))
        pnl = float(trade.get("realized_pnl", 0))
        fee = float(trade.get("fee", 0))
        
        # Color coding
        if pnl > 0:
            pnl_str = f"[green]+{pnl:.2f}[/green]"
        elif pnl < 0:
            pnl_str = f"[red]{pnl:.2f}[/red]"
        else:
            pnl_str = f"{pnl:.2f}"
            
        if "OPEN" in action or "INCREASE" in action:
            action_color = "blue"
        elif "CLOSE" in action or "REDUCE" in action:
            action_color = "magenta"
        else:
            action_color = "cyan"

        msg = f"[{ts}] [{action_color}]{action}[/{action_color}] {sym} Q:{qty:.4f} @ {price:.2f} PnL:{pnl_str} Fee:{fee:.2f}"
        self._recent_logs.append(msg)
        if len(self._recent_logs) > 20:
            self._recent_logs.pop(0)

    def log_info(self, message: str) -> None:
        if RICH_AVAILABLE:
            ts = pd.Timestamp.utcnow().strftime('%H:%M:%S')
            self._recent_logs.append(f"[{ts}] {message}")
            if len(self._recent_logs) > 20:
                self._recent_logs.pop(0)


