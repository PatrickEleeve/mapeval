"""TUI Reporter using Rich for real-time dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

class TUIReporter:
    """Renders a real-time TUI dashboard for the trading session."""

    def __init__(self) -> None:
        self.console = Console()
        self.layout = self._make_layout()
        self.live = Live(self.layout, refresh_per_second=4, screen=True)
        self.logs: List[str] = []
        self.max_logs = 20
        self.warnings: List[Dict[str, Any]] = []
        self.live.start()

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=12),
        )
        layout["body"].split_row(
            Layout(name="account", ratio=1),
            Layout(name="positions", ratio=2),
        )
        return layout

    def record_tick(self, timestamp: pd.Timestamp, account: Any, prices: Dict[str, float]) -> None:
        self._update_header(timestamp)
        self._update_account(account)
        self._update_positions(account, prices)
        self._update_logs()
        self.live.refresh()

    def record_warning(self, timestamp: pd.Timestamp, message: str) -> None:
        log_entry = f"[{timestamp.strftime('%H:%M:%S')}] [yellow]WARN[/yellow]: {message}"
        self.logs.append(log_entry)
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)
        self.warnings.append({"timestamp": timestamp, "message": message})
        self._update_logs()

    def _update_header(self, timestamp: pd.Timestamp) -> None:
        title = f"MAPEval Real-Time Benchmark | {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        self.layout["header"].update(Panel(Text(title, justify="center", style="bold white on blue")))

    def _update_account(self, account: Any) -> None:
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Equity", f"{account.equity:,.2f}")
        table.add_row("Balance", f"{account.balance:,.2f}")
        table.add_row("Unrealized PnL", f"{account.unrealized_pnl:,.2f}")
        table.add_row("Realized PnL", f"{account.realized_pnl:,.2f}")
        table.add_row("Margin Used", f"{account.margin_used:,.2f}")
        table.add_row("Avail Margin", f"{account.available_margin:,.2f}")
        
        if getattr(account, "maintenance_margin_req", 0) > 0:
             table.add_row("Maint. Margin", f"{account.maintenance_margin_req:,.2f}")

        self.layout["account"].update(Panel(table, title="Account Summary"))

    def _update_positions(self, account: Any, prices: Dict[str, float]) -> None:
        table = Table(box=box.SIMPLE_HEAD, expand=True)
        table.add_column("Symbol", style="bold")
        table.add_column("Side")
        table.add_column("Size", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Mark", justify="right")
        table.add_column("PnL", justify="right")
        table.add_column("Lev", justify="right")

        positions = getattr(account, "positions", {})
        if not positions:
            table.add_row("-", "-", "-", "-", "-", "-", "-")
        else:
            for symbol, pos in positions.items():
                side = "LONG" if pos.quantity > 0 else "SHORT"
                side_style = "green" if pos.quantity > 0 else "red"
                mark_price = prices.get(symbol, pos.entry_price)
                pnl = (mark_price - pos.entry_price) * pos.quantity
                pnl_style = "green" if pnl >= 0 else "red"
                
                table.add_row(
                    symbol,
                    f"[{side_style}]{side}[/{side_style}]",
                    f"{abs(pos.quantity):.4f}",
                    f"{pos.entry_price:.4f}",
                    f"{mark_price:.4f}",
                    f"[{pnl_style}]{pnl:+.2f}[/{pnl_style}]",
                    f"{pos.leverage:.1f}x"
                )

        self.layout["positions"].update(Panel(table, title="Open Positions"))

    def _update_logs(self) -> None:
        log_text = "\n".join(self.logs)
        self.layout["footer"].update(Panel(log_text, title="Event Log"))

    def finalize(self, equity_history: List[Dict[str, Any]], trade_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.live.stop()
        # Return a dummy report or delegate to standard reporter logic if needed.
        # For now, we just return basic stats to satisfy the interface.
        # The main.py might print the summary again, which is fine.
        
        # We can reuse the logic from RealTimeReporter for the final print summary
        # by importing it or just letting main.py handle the return value.
        # But main.py expects this method to return the report dict.
        
        from reporter import RealTimeReporter
        # Create a temporary standard reporter to generate the final report
        std_reporter = RealTimeReporter()
        std_reporter.warnings = self.warnings
        return std_reporter.finalize(equity_history, trade_log)
