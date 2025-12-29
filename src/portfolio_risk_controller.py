"""Portfolio-level risk controller for P0/P1/P2 requirements.

P0: Margin consistency & liquidation audit logging
P1: Portfolio-level exposure caps (gross leverage, net exposure, max positions)
P2: Turnover constraints
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskLimits:
    gross_leverage_cap: float = 3.0
    net_exposure_cap: float = 1.0
    max_open_positions: int = 5
    max_turnover_per_step: float = 2.0
    turnover_penalty_rate: float = 0.001
    min_confidence_threshold: float = 0.3


@dataclass
class MarginMetrics:
    total_notional_long: float = 0.0
    total_notional_short: float = 0.0
    gross_notional: float = 0.0
    net_notional: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    margin_ratio: float = 0.0
    cumulative_fees: float = 0.0
    cumulative_slippage: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_notional_long": round(self.total_notional_long, 2),
            "total_notional_short": round(self.total_notional_short, 2),
            "gross_notional": round(self.gross_notional, 2),
            "net_notional": round(self.net_notional, 2),
            "gross_exposure": round(self.gross_exposure, 4),
            "net_exposure": round(self.net_exposure, 4),
            "initial_margin": round(self.initial_margin, 2),
            "maintenance_margin": round(self.maintenance_margin, 2),
            "margin_ratio": round(self.margin_ratio, 4),
            "cumulative_fees": round(self.cumulative_fees, 2),
            "cumulative_slippage": round(self.cumulative_slippage, 2),
        }


@dataclass
class PortfolioRiskController:
    limits: PortfolioRiskLimits = field(default_factory=PortfolioRiskLimits)
    initial_margin_rate: float = 0.1
    maintenance_margin_rate: float = 0.05
    _cumulative_fees: float = field(init=False, default=0.0)
    _cumulative_slippage: float = field(init=False, default=0.0)

    def accumulate_costs(self, fee: float, slippage_cost: float) -> None:
        self._cumulative_fees += fee
        self._cumulative_slippage += slippage_cost

    def reset_cumulative_costs(self) -> None:
        self._cumulative_fees = 0.0
        self._cumulative_slippage = 0.0

    def verify_equity_consistency(
        self,
        balance: float,
        unrealized_pnl: float,
        equity: float,
        tolerance: float = 1e-4,
    ) -> Tuple[bool, str]:
        expected = balance + unrealized_pnl
        diff = abs(equity - expected)
        if diff > tolerance:
            msg = (
                f"EQUITY MISMATCH: equity={equity:.6f} != "
                f"balance({balance:.6f}) + unrealized_pnl({unrealized_pnl:.6f}) = {expected:.6f}, "
                f"diff={diff:.6f}"
            )
            logger.error(msg)
            return False, msg
        return True, "OK"

    def compute_margin_metrics(
        self,
        positions: Dict[str, Any],
        prices: Dict[str, float],
        equity: float,
    ) -> MarginMetrics:
        total_long = 0.0
        total_short = 0.0

        for symbol, pos in positions.items():
            price = prices.get(symbol, 0.0)
            if price <= 0:
                continue
            qty = getattr(pos, "quantity", 0.0) if hasattr(pos, "quantity") else pos.get("quantity", 0.0)
            notional = price * qty
            if notional > 0:
                total_long += notional
            else:
                total_short += abs(notional)

        gross_notional = total_long + total_short
        net_notional = total_long - total_short

        safe_equity = max(equity, 1e-6)
        gross_exposure = gross_notional / safe_equity
        net_exposure = net_notional / safe_equity

        initial_margin = gross_notional * self.initial_margin_rate
        maintenance_margin = gross_notional * self.maintenance_margin_rate
        margin_ratio = maintenance_margin / safe_equity if safe_equity > 0 else float("inf")

        return MarginMetrics(
            total_notional_long=total_long,
            total_notional_short=total_short,
            gross_notional=gross_notional,
            net_notional=net_notional,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            margin_ratio=margin_ratio,
            cumulative_fees=self._cumulative_fees,
            cumulative_slippage=self._cumulative_slippage,
        )

    def create_liquidation_audit_log(
        self,
        timestamp: str,
        trigger_reason: str,
        balance: float,
        unrealized_pnl: float,
        equity: float,
        positions: Dict[str, Any],
        prices: Dict[str, float],
    ) -> Dict[str, Any]:
        margin_metrics = self.compute_margin_metrics(positions, prices, equity)
        
        positions_detail = []
        for symbol, pos in positions.items():
            price = prices.get(symbol, 0.0)
            qty = getattr(pos, "quantity", 0.0) if hasattr(pos, "quantity") else pos.get("quantity", 0.0)
            entry = getattr(pos, "entry_price", 0.0) if hasattr(pos, "entry_price") else pos.get("entry_price", 0.0)
            notional = price * qty
            unrealized = (price - entry) * qty if entry > 0 else 0.0
            positions_detail.append({
                "symbol": symbol,
                "quantity": round(qty, 6),
                "entry_price": round(entry, 6),
                "mark_price": round(price, 6),
                "notional": round(notional, 2),
                "unrealized_pnl": round(unrealized, 2),
                "side": "LONG" if qty > 0 else "SHORT",
            })

        audit = {
            "event": "LIQUIDATION",
            "timestamp": timestamp,
            "trigger_reason": trigger_reason,
            "account": {
                "balance": round(balance, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "equity": round(equity, 2),
                "equity_formula": "equity = balance + unrealized_pnl",
                "equity_check": f"{balance:.2f} + {unrealized_pnl:.2f} = {balance + unrealized_pnl:.2f}",
            },
            "margin": margin_metrics.to_dict(),
            "positions": positions_detail,
            "position_count": len(positions_detail),
        }

        log_line = (
            f"\n{'='*60}\n"
            f"🚨 LIQUIDATION EVENT @ {timestamp}\n"
            f"{'='*60}\n"
            f"Trigger: {trigger_reason}\n"
            f"\n[Account State]\n"
            f"  Balance:        {balance:>14.2f} USDT\n"
            f"  Unrealized PnL: {unrealized_pnl:>14.2f} USDT\n"
            f"  Equity:         {equity:>14.2f} USDT\n"
            f"  (equity = balance + unrealized_pnl)\n"
            f"\n[Margin Metrics]\n"
            f"  Notional Long:  {margin_metrics.total_notional_long:>14.2f} USDT\n"
            f"  Notional Short: {margin_metrics.total_notional_short:>14.2f} USDT\n"
            f"  Gross Notional: {margin_metrics.gross_notional:>14.2f} USDT\n"
            f"  Net Notional:   {margin_metrics.net_notional:>14.2f} USDT\n"
            f"  Gross Exposure: {margin_metrics.gross_exposure:>14.4f}x\n"
            f"  Net Exposure:   {margin_metrics.net_exposure:>14.4f}x\n"
            f"  Init Margin:    {margin_metrics.initial_margin:>14.2f} USDT\n"
            f"  Maint Margin:   {margin_metrics.maintenance_margin:>14.2f} USDT\n"
            f"  Margin Ratio:   {margin_metrics.margin_ratio:>14.4f}\n"
            f"\n[Cumulative Costs]\n"
            f"  Total Fees:     {margin_metrics.cumulative_fees:>14.2f} USDT\n"
            f"  Total Slippage: {margin_metrics.cumulative_slippage:>14.2f} USDT\n"
            f"\n[Positions at Liquidation] ({len(positions_detail)} open)\n"
        )
        for p in positions_detail:
            log_line += (
                f"  {p['symbol']:12s} {p['side']:5s} qty={p['quantity']:>12.4f} "
                f"entry={p['entry_price']:>10.4f} mark={p['mark_price']:>10.4f} "
                f"notional={p['notional']:>12.2f} pnl={p['unrealized_pnl']:>+10.2f}\n"
            )
        log_line += f"{'='*60}\n"

        logger.critical(log_line)
        print(log_line)

        return audit

    def validate_portfolio_constraints(
        self,
        proposed_exposures: Dict[str, float],
        current_exposures: Dict[str, float],
        allow_rescale: bool = True,
    ) -> Dict[str, Any]:
        notes: List[str] = []
        sanitized = dict(proposed_exposures)

        gross_leverage = sum(abs(v) for v in sanitized.values())
        if gross_leverage > self.limits.gross_leverage_cap + 1e-9:
            if allow_rescale and gross_leverage > 0:
                scale = self.limits.gross_leverage_cap / gross_leverage
                sanitized = {k: v * scale for k, v in sanitized.items()}
                notes.append(
                    f"GROSS_LEVERAGE: scaled {gross_leverage:.2f}x -> {self.limits.gross_leverage_cap:.2f}x "
                    f"(factor={scale:.4f})"
                )
                gross_leverage = self.limits.gross_leverage_cap
            else:
                return {
                    "valid": False,
                    "code": "GROSS_LEVERAGE_EXCEEDED",
                    "message": f"Gross leverage {gross_leverage:.2f}x > cap {self.limits.gross_leverage_cap:.2f}x",
                    "notes": notes,
                }

        net_exposure = sum(sanitized.values())
        if abs(net_exposure) > self.limits.net_exposure_cap + 1e-9:
            if allow_rescale and abs(net_exposure) > 0:
                scale = self.limits.net_exposure_cap / abs(net_exposure)
                sanitized = {k: v * scale for k, v in sanitized.items()}
                notes.append(
                    f"NET_EXPOSURE: scaled {net_exposure:+.2f}x -> {net_exposure * scale:+.2f}x "
                    f"to respect cap ±{self.limits.net_exposure_cap:.2f}x"
                )
            else:
                return {
                    "valid": False,
                    "code": "NET_EXPOSURE_EXCEEDED",
                    "message": f"Net exposure {net_exposure:+.2f}x > cap ±{self.limits.net_exposure_cap:.2f}x",
                    "notes": notes,
                }

        non_zero = [(k, v) for k, v in sanitized.items() if abs(v) > 1e-9]
        if len(non_zero) > self.limits.max_open_positions:
            if allow_rescale:
                sorted_by_size = sorted(non_zero, key=lambda x: abs(x[1]), reverse=True)
                kept_symbols = {s for s, _ in sorted_by_size[:self.limits.max_open_positions]}
                zeroed = [s for s, _ in sorted_by_size[self.limits.max_open_positions:]]
                sanitized = {k: (v if k in kept_symbols else 0.0) for k, v in sanitized.items()}
                notes.append(
                    f"MAX_POSITIONS: zeroed {len(zeroed)} smallest to keep {self.limits.max_open_positions}: "
                    f"{zeroed}"
                )
            else:
                return {
                    "valid": False,
                    "code": "MAX_POSITIONS_EXCEEDED",
                    "message": f"Open positions {len(non_zero)} > max {self.limits.max_open_positions}",
                    "notes": notes,
                }

        turnover = sum(
            abs(sanitized.get(k, 0.0) - current_exposures.get(k, 0.0))
            for k in set(sanitized) | set(current_exposures)
        )
        if turnover > self.limits.max_turnover_per_step + 1e-9:
            if allow_rescale and turnover > 0:
                scale = self.limits.max_turnover_per_step / turnover
                for k in sanitized:
                    current = current_exposures.get(k, 0.0)
                    delta = sanitized[k] - current
                    sanitized[k] = current + delta * scale
                notes.append(
                    f"TURNOVER: scaled deltas by {scale:.4f} to respect max_turnover={self.limits.max_turnover_per_step:.2f}x "
                    f"(was {turnover:.2f}x)"
                )
                turnover = self.limits.max_turnover_per_step
            else:
                return {
                    "valid": False,
                    "code": "TURNOVER_EXCEEDED",
                    "message": f"Turnover {turnover:.2f}x > max {self.limits.max_turnover_per_step:.2f}x",
                    "notes": notes,
                }

        final_gross = sum(abs(v) for v in sanitized.values())
        final_net = sum(sanitized.values())
        final_positions = sum(1 for v in sanitized.values() if abs(v) > 1e-9)

        return {
            "valid": True,
            "exposures": sanitized,
            "notes": notes,
            "metrics": {
                "gross_leverage": round(final_gross, 4),
                "net_exposure": round(final_net, 4),
                "open_positions": final_positions,
                "turnover": round(turnover, 4),
            },
        }

    def compute_turnover_penalty(
        self,
        proposed_exposures: Dict[str, float],
        current_exposures: Dict[str, float],
        equity: float,
    ) -> float:
        turnover = sum(
            abs(proposed_exposures.get(k, 0.0) - current_exposures.get(k, 0.0))
            for k in set(proposed_exposures) | set(current_exposures)
        )
        return turnover * equity * self.limits.turnover_penalty_rate

    def should_reject_low_confidence(
        self,
        confidence: Optional[float],
        reasoning: Optional[str],
    ) -> Tuple[bool, str]:
        if confidence is not None and confidence < self.limits.min_confidence_threshold:
            return True, f"Confidence {confidence:.2f} < threshold {self.limits.min_confidence_threshold:.2f}"
        if reasoning is not None and len(reasoning.strip()) < 20:
            return True, f"Reasoning too short ({len(reasoning.strip())} chars)"
        return False, "OK"

