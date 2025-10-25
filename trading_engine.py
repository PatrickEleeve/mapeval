"""Real-time trading engine with leveraged futures support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from tools import FinancialTools


@dataclass
class FuturesPosition:
    """Track an open futures position."""

    symbol: str
    quantity: float
    entry_price: float
    leverage: float
    opened_at: pd.Timestamp


@dataclass
class AccountState:
    """Maintain account balances, margin, and PnL."""

    balance: float
    positions: Dict[str, FuturesPosition] = field(default_factory=dict)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    available_margin: float = 0.0

    def __post_init__(self) -> None:
        self.equity = self.balance
        self.available_margin = self.balance

    def mark_to_market(self, prices: Dict[str, float], max_leverage: float) -> None:
        unrealized = 0.0
        total_notional = 0.0
        for symbol, position in self.positions.items():
            price = prices.get(symbol)
            if price is None:
                continue
            unrealized += (price - position.entry_price) * position.quantity
            total_notional += abs(price * position.quantity)
        self.unrealized_pnl = unrealized
        self.equity = self.balance + unrealized
        if max_leverage > 0.0:
            self.margin_used = total_notional / max_leverage
        else:
            self.margin_used = 0.0
        self.available_margin = self.equity - self.margin_used


class RealTimeTradingEngine:
    """Continuously poll prices, request signals, and manage leveraged positions."""

    def __init__(
        self,
        market_data,
        agent: Any,
        initial_capital: float,
        max_leverage: float,
        poll_interval_seconds: float,
        decision_interval_seconds: float,
        min_long_exposure: float = 0.0,
    ) -> None:
        self.market_data = market_data
        self.agent = agent
        self.account = AccountState(balance=initial_capital)
        self.max_leverage = max_leverage
        self.poll_interval_seconds = poll_interval_seconds
        self.decision_interval_seconds = decision_interval_seconds
        self.min_long_exposure = max(0.0, float(min_long_exposure))
        self.trade_log: List[Dict[str, Any]] = []
        self.decision_log: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = []

    def run(self, duration_seconds: float, reporter: Optional[Any] = None) -> Dict[str, Any]:
        end_time = time.time() + duration_seconds
        next_decision_ts = time.time()
        while time.time() < end_time:
            loop_ts = pd.Timestamp.utcnow().floor("s")
            try:
                prices = self.market_data.fetch_latest_prices()
            except RuntimeError as exc:
                if reporter is not None:
                    reporter.record_warning(loop_ts, str(exc))
                prices = self.market_data.latest_prices()
                if not prices:
                    time.sleep(self.poll_interval_seconds)
                    continue

            self.market_data.append_prices(prices, timestamp=loop_ts)
            self.account.mark_to_market(prices, self.max_leverage)
            self.equity_history.append(
                {
                    "timestamp": loop_ts,
                    "equity": self.account.equity,
                    "balance": self.account.balance,
                    "unrealized_pnl": self.account.unrealized_pnl,
                }
            )

            if reporter is not None:
                reporter.record_tick(loop_ts, self.account, prices)

            if time.time() >= next_decision_ts:
                history_slice = self.market_data.get_recent_window()
                funding = {}
                try:
                    funding = self.market_data.refresh_funding_rates(throttle_seconds=300)
                except Exception:
                    funding = {}
                tools = FinancialTools(history_slice, funding_rates=funding or None)
                try:
                    signal = self.agent.generate_trading_signal(loop_ts, history_slice, tools)
                except Exception as exc:
                    if reporter is not None:
                        reporter.record_warning(loop_ts, f"Agent error: {exc}")
                    signal = {}
                plan = {
                    "reasoning": getattr(self.agent, "last_reasoning", ""),
                    "actions": [
                        {"symbol": symbol, "target_exposure": signal.get(symbol, 0.0)}
                        for symbol in self.market_data.symbols
                    ],
                }
                response = self.execute_trading_plan(plan, source="llm_agent")
                if response.get("status") != "filled" and reporter is not None:
                    reason = response.get("reason")
                    message = reason.get("message") if isinstance(reason, dict) else "Unknown rejection."
                    reporter.record_warning(loop_ts, f"Trading plan rejected: {message}")
                next_decision_ts = time.time() + self.decision_interval_seconds

            if self.account.equity <= 0.0:
                if reporter is not None:
                    reporter.record_warning(loop_ts, "Equity depleted; stopping trading loop.")
                break

            time.sleep(self.poll_interval_seconds)

        summary = {
            "equity_history": self.equity_history,
            "trade_log": self.trade_log,
            "decision_log": self.decision_log,
            "final_account": {
                "balance": self.account.balance,
                "equity": self.account.equity,
                "realized_pnl": self.account.realized_pnl,
                "unrealized_pnl": self.account.unrealized_pnl,
                "margin_used": self.account.margin_used,
                "available_margin": self.account.available_margin,
            },
        }
        if reporter is not None:
            summary["reports"] = reporter.finalize(self.equity_history, self.trade_log)
        return summary

    def _apply_signal(
        self,
        exposures: Dict[str, float],
        prices: Dict[str, float],
        timestamp: pd.Timestamp,
    ) -> Dict[str, float]:
        validation = self._validate_exposures(exposures, allow_rescale=True)
        if not validation["valid"]:
            sanitized = {symbol: 0.0 for symbol in self.market_data.symbols}
        else:
            sanitized = validation["exposures"]
        if self.min_long_exposure > 0.0:
            for symbol, value in sanitized.items():
                if value > 0.0 and value < self.min_long_exposure:
                    sanitized[symbol] = 0.0
        equity = max(self.account.equity, 1e-6)
        for symbol in self.market_data.symbols:
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            target_exposure = sanitized.get(symbol, 0.0)
            target_notional = target_exposure * equity
            target_quantity = target_notional / price
            self._rebalance_position(symbol, target_quantity, price, timestamp)
        self.account.mark_to_market(prices, self.max_leverage)
        return sanitized

    def _validate_exposures(
        self,
        exposures: Dict[str, float],
        allow_rescale: bool,
    ) -> Dict[str, Any]:
        known_symbols = set(self.market_data.symbols)
        extra_symbols = sorted({symbol for symbol in exposures.keys()} - known_symbols)
        if extra_symbols:
            return {
                "valid": False,
                "code": "UNKNOWN_SYMBOL",
                "message": f"Unsupported symbols: {', '.join(extra_symbols)}",
            }

        sanitized: Dict[str, float] = {}
        for symbol in self.market_data.symbols:
            raw_value = exposures.get(symbol, 0.0)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return {
                    "valid": False,
                    "code": "INVALID_NUMBER",
                    "message": f"Exposure for {symbol} must be numeric.",
                }
            if not allow_rescale and abs(value) > self.max_leverage + 1e-9:
                return {
                    "valid": False,
                    "code": "PER_SYMBOL_LIMIT",
                    "message": f"{symbol} exposure {value:.4f} exceeds +/-{self.max_leverage}x.",
                }
            sanitized[symbol] = value

        if allow_rescale:
            sanitized = {
                symbol: max(-self.max_leverage, min(self.max_leverage, value))
                for symbol, value in sanitized.items()
            }

        total_abs = sum(abs(value) for value in sanitized.values())
        if total_abs > self.max_leverage + 1e-9:
            if allow_rescale and total_abs > 0.0:
                scale = self.max_leverage / total_abs
                sanitized = {symbol: value * scale for symbol, value in sanitized.items()}
            else:
                return {
                    "valid": False,
                    "code": "LEVERAGE_LIMIT",
                    "message": (
                        f"Aggregate exposure {total_abs:.4f} exceeds maximum leverage {self.max_leverage}."
                    ),
                }

        return {"valid": True, "exposures": sanitized}

    def _rebalance_position(
        self,
        symbol: str,
        target_quantity: float,
        price: float,
        timestamp: pd.Timestamp,
    ) -> None:
        position = self.account.positions.get(symbol)
        if position is None:
            if abs(target_quantity) < 1e-8:
                return
            leverage = self._compute_position_leverage(price, target_quantity)
            self.account.positions[symbol] = FuturesPosition(
                symbol=symbol,
                quantity=target_quantity,
                entry_price=price,
                leverage=leverage,
                opened_at=timestamp,
            )
            self.trade_log.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "action": "open",
                    "quantity": target_quantity,
                    "price": price,
                    "realized_pnl": 0.0,
                }
            )
            return

        existing_qty = position.quantity
        if abs(target_quantity) < 1e-8:
            realized = (price - position.entry_price) * existing_qty
            self.account.balance += realized
            self.account.realized_pnl += realized
            self.trade_log.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "action": "close",
                    "quantity": existing_qty,
                    "price": price,
                    "realized_pnl": realized,
                }
            )
            del self.account.positions[symbol]
            return

        if existing_qty * target_quantity > 0:
            if abs(target_quantity) > abs(existing_qty):
                delta_qty = target_quantity - existing_qty
                weighted_notional = position.entry_price * existing_qty + price * delta_qty
                position.quantity = target_quantity
                position.entry_price = weighted_notional / target_quantity
                position.leverage = self._compute_position_leverage(price, target_quantity)
                self.trade_log.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": symbol,
                        "action": "increase",
                        "quantity": delta_qty,
                        "price": price,
                        "realized_pnl": 0.0,
                    }
                )
            else:
                closed_qty = existing_qty - target_quantity
                realized = (price - position.entry_price) * closed_qty
                self.account.balance += realized
                self.account.realized_pnl += realized
                position.quantity = target_quantity
                self.trade_log.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": symbol,
                        "action": "reduce",
                        "quantity": closed_qty,
                        "price": price,
                        "realized_pnl": realized,
                    }
                )
                if abs(position.quantity) < 1e-8:
                    del self.account.positions[symbol]
            return

        # Direction flip: close prior position, open new one.
        realized = (price - position.entry_price) * existing_qty
        self.account.balance += realized
        self.account.realized_pnl += realized
        self.trade_log.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": "reverse_close",
                "quantity": existing_qty,
                "price": price,
                "realized_pnl": realized,
            }
        )
        leverage = self._compute_position_leverage(price, target_quantity)
        self.account.positions[symbol] = FuturesPosition(
            symbol=symbol,
            quantity=target_quantity,
            entry_price=price,
            leverage=leverage,
            opened_at=timestamp,
        )
        self.trade_log.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": "reverse_open",
                "quantity": target_quantity,
                "price": price,
                "realized_pnl": 0.0,
            }
        )

    def _compute_position_leverage(self, price: float, quantity: float) -> float:
        equity = max(self.account.equity, 1e-6)
        notional = abs(price * quantity)
        if equity <= 0:
            return self.max_leverage
        leverage = notional / equity
        return max(1.0, min(self.max_leverage, leverage))

    def execute_trading_plan(self, plan: Dict[str, Any], *, source: str = "external_command") -> Dict[str, Any]:
        timestamp = pd.Timestamp.utcnow().floor("s")
        try:
            prices = self.market_data.fetch_latest_prices()
        except RuntimeError as exc:
            prices = self.market_data.latest_prices()
            if not prices:
                reason = {
                    "code": "MARKET_DATA_UNAVAILABLE",
                    "message": str(exc),
                }
                response = {
                    "status": "rejected",
                    "timestamp": timestamp.isoformat(),
                    "reason": reason,
                    "positions": [],
                    "account": self._account_snapshot(),
                }
                self._log_decision(
                    timestamp,
                    source,
                    requested_exposure={},
                    applied_exposure={},
                    reasoning=plan.get("reasoning", ""),
                    status="rejected",
                    reason=reason,
                )
                return response
        self.market_data.append_prices(prices, timestamp=timestamp)
        self.account.mark_to_market(prices, self.max_leverage)

        current_exposures = self._current_exposures(prices)
        requested_exposures = current_exposures.copy()
        for action in plan.get("actions", []):
            symbol = str(action.get("symbol", "")).upper()
            if not symbol:
                continue
            value = action.get("target_exposure", 0.0)
            requested_exposures[symbol] = value

        validation = self._validate_exposures(requested_exposures, allow_rescale=False)
        if not validation["valid"]:
            reason = {
                "code": validation.get("code", "VALIDATION_ERROR"),
                "message": validation.get("message", "Exposure validation failed."),
            }
            response = {
                "status": "rejected",
                "timestamp": timestamp.isoformat(),
                "reason": reason,
                "positions": self._positions_snapshot(prices),
                "account": self._account_snapshot(),
            }
            self._log_decision(
                timestamp,
                source,
                requested_exposure=requested_exposures,
                applied_exposure=current_exposures,
                reasoning=plan.get("reasoning", ""),
                status="rejected",
                reason=reason,
            )
            return response

        exposures = validation["exposures"]
        if self.min_long_exposure > 0.0:
            violating = [
                symbol
                for symbol, target in exposures.items()
                if target > 0.0 and target < self.min_long_exposure
            ]
            if violating:
                reason = {
                    "code": "MIN_LONG_EXPOSURE",
                    "message": (
                        "Positive exposures must be at least "
                        f"{self.min_long_exposure:.4f}x equity; violating symbols: {', '.join(violating)}."
                    ),
                }
                response = {
                    "status": "rejected",
                    "timestamp": timestamp.isoformat(),
                    "reason": reason,
                    "positions": self._positions_snapshot(prices),
                    "account": self._account_snapshot(),
                }
                self._log_decision(
                    timestamp,
                    source,
                    requested_exposure=exposures,
                    applied_exposure=current_exposures,
                    reasoning=plan.get("reasoning", ""),
                    status="rejected",
                    reason=reason,
                )
                return response

        equity = self.account.equity
        if equity <= 0 and any(abs(value) > 1e-9 for value in exposures.values()):
            reason = {
                "code": "INSUFFICIENT_FUNDS",
                "message": "Account equity is non-positive; cannot open positions.",
            }
            response = {
                "status": "rejected",
                "timestamp": timestamp.isoformat(),
                "reason": reason,
                "positions": self._positions_snapshot(prices),
                "account": self._account_snapshot(),
            }
            self._log_decision(
                timestamp,
                source,
                requested_exposure=exposures,
                applied_exposure=current_exposures,
                reasoning=plan.get("reasoning", ""),
                status="rejected",
                reason=reason,
            )
            return response

        margin_required = self._margin_requirement(exposures, equity)
        if margin_required > equity + 1e-9:
            reason = {
                "code": "INSUFFICIENT_FUNDS",
                "message": f"Required margin {margin_required:.2f} exceeds available equity {equity:.2f}.",
            }
            response = {
                "status": "rejected",
                "timestamp": timestamp.isoformat(),
                "reason": reason,
                "positions": self._positions_snapshot(prices),
                "account": self._account_snapshot(),
            }
            self._log_decision(
                timestamp,
                source,
                requested_exposure=exposures,
                applied_exposure=current_exposures,
                reasoning=plan.get("reasoning", ""),
                status="rejected",
                reason=reason,
            )
            return response

        applied_exposures: Dict[str, float] = {}
        for symbol in self.market_data.symbols:
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            target_exposure = float(exposures.get(symbol, 0.0))
            target_notional = target_exposure * equity
            target_quantity = target_notional / price if price else 0.0
            self._rebalance_position(symbol, target_quantity, price, timestamp)
            applied_exposures[symbol] = target_exposure

        self.account.mark_to_market(prices, self.max_leverage)
        response = {
            "status": "filled",
            "timestamp": timestamp.isoformat(),
            "applied_exposures": applied_exposures,
            "positions": self._positions_snapshot(prices),
            "account": self._account_snapshot(),
            "reason": None,
        }

        reasoning = plan.get("reasoning", "")
        self._log_decision(
            timestamp,
            source,
            requested_exposure=exposures,
            applied_exposure=applied_exposures,
            reasoning=reasoning,
            status="filled",
            reason=None,
        )

        return response

    def _log_decision(
        self,
        timestamp: pd.Timestamp,
        source: str,
        *,
        requested_exposure: Dict[str, float],
        applied_exposure: Dict[str, float],
        reasoning: str,
        status: str,
        reason: Optional[Dict[str, Any]],
    ) -> None:
        entry: Dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "source": source,
            "requested_exposure": requested_exposure,
            "applied_exposure": applied_exposure,
            "reasoning": reasoning,
            "equity": self.account.equity,
            "status": status,
        }
        if reason is not None:
            entry["reason"] = reason
        self.decision_log.append(entry)

    def _current_exposures(self, prices: Dict[str, float]) -> Dict[str, float]:
        equity = self.account.equity if self.account.equity != 0 else 0.0
        exposures: Dict[str, float] = {}
        if abs(equity) < 1e-9:
            return {symbol: 0.0 for symbol in self.market_data.symbols}
        for symbol in self.market_data.symbols:
            position = self.account.positions.get(symbol)
            price = prices.get(symbol)
            if position is None or price is None:
                exposures[symbol] = 0.0
            else:
                exposures[symbol] = (position.quantity * price) / equity
        return exposures

    def _margin_requirement(self, exposures: Dict[str, float], equity: float) -> float:
        if self.max_leverage <= 0:
            return float("inf")
        total_notional = sum(abs(value) * max(equity, 0.0) for value in exposures.values())
        return total_notional / self.max_leverage

    def _positions_snapshot(self, prices: Dict[str, float]) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        for symbol in self.market_data.symbols:
            position = self.account.positions.get(symbol)
            price = prices.get(symbol)
            if position is None:
                snapshot.append(
                    {
                        "symbol": symbol,
                        "quantity": 0.0,
                        "entry_price": None,
                        "mark_price": price,
                        "leverage": 0.0,
                        "unrealized_pnl": 0.0,
                    }
                )
                continue
            mark_price = price if price is not None else position.entry_price
            unrealized = 0.0
            if mark_price is not None:
                unrealized = (mark_price - position.entry_price) * position.quantity
            snapshot.append(
                {
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "mark_price": mark_price,
                    "leverage": position.leverage,
                    "unrealized_pnl": unrealized,
                }
            )
        return snapshot

    def _account_snapshot(self) -> Dict[str, float]:
        return {
            "balance": float(self.account.balance),
            "equity": float(self.account.equity),
            "available_margin": float(self.account.available_margin),
            "margin_used": float(self.account.margin_used),
            "realized_pnl": float(self.account.realized_pnl),
            "unrealized_pnl": float(self.account.unrealized_pnl),
        }
