"""Real-time trading engine with leveraged futures support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from tools import FinancialTools
from portfolio_risk_controller import PortfolioRiskController, PortfolioRiskLimits


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
    maintenance_margin_req: float = 0.0

    def __post_init__(self) -> None:
        self.equity = self.balance
        self.available_margin = self.balance

    def mark_to_market(self, prices: Dict[str, float], max_leverage: float, maintenance_rate: float = 0.0) -> None:
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
        self.maintenance_margin_req = total_notional * maintenance_rate
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
        per_symbol_max_exposure: Optional[float] = None,
        max_exposure_delta: Optional[float] = None,
        commission_rate: float = 0.0,
        slippage: float = 0.0,
        liquidation_threshold: float = 0.0,
        gross_leverage_cap: Optional[float] = None,
        net_exposure_cap: Optional[float] = None,
        max_open_positions: Optional[int] = None,
        max_turnover_per_step: Optional[float] = None,
    ) -> None:
        self.market_data = market_data
        self.agent = agent
        self.account = AccountState(balance=initial_capital)
        self.max_leverage = max(0.0, float(max_leverage))
        self.poll_interval_seconds = poll_interval_seconds
        self.decision_interval_seconds = decision_interval_seconds
        self.min_long_exposure = max(0.0, float(min_long_exposure))
        self.commission_rate = max(0.0, float(commission_rate))
        self.slippage = max(0.0, float(slippage))
        self.liquidation_threshold = max(0.0, float(liquidation_threshold))
        per_symbol_cap = per_symbol_max_exposure if per_symbol_max_exposure is not None else self.max_leverage
        if self.max_leverage > 0.0 and per_symbol_cap is not None:
            per_symbol_cap = min(float(per_symbol_cap), self.max_leverage)
        self.per_symbol_max_exposure = max(0.0, float(per_symbol_cap if per_symbol_cap is not None else self.max_leverage))
        delta_cap = max_exposure_delta if max_exposure_delta is not None else self.per_symbol_max_exposure
        if delta_cap is None:
            delta_cap = self.per_symbol_max_exposure
        self.max_exposure_delta = max(0.0, min(float(delta_cap), self.per_symbol_max_exposure or float(delta_cap)))
        self.trade_log: List[Dict[str, Any]] = []
        self.decision_log: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = []
        self._last_applied_exposures: Dict[str, float] = {
            symbol: 0.0 for symbol in getattr(self.market_data, "symbols", [])
        }
        self._last_validation_notes: List[str] = []
        
        portfolio_limits = PortfolioRiskLimits(
            gross_leverage_cap=gross_leverage_cap if gross_leverage_cap is not None else self.max_leverage,
            net_exposure_cap=net_exposure_cap if net_exposure_cap is not None else self.max_leverage,
            max_open_positions=max_open_positions if max_open_positions is not None else len(getattr(self.market_data, "symbols", [])),
            max_turnover_per_step=max_turnover_per_step if max_turnover_per_step is not None else self.max_leverage * 2,
        )
        self.portfolio_risk = PortfolioRiskController(
            limits=portfolio_limits,
            initial_margin_rate=1.0 / self.max_leverage if self.max_leverage > 0 else 1.0,
            maintenance_margin_rate=self.liquidation_threshold,
        )
        self._liquidation_audit: Optional[Dict[str, Any]] = None

    def run(self, duration_seconds: float, reporter: Optional[Any] = None) -> Dict[str, Any]:
        end_time = time.time() + duration_seconds
        next_decision_ts = time.time()
        while time.time() < end_time:
            loop_ts = pd.Timestamp.utcnow().floor("s")
            try:
                prices = self.market_data.fetch_latest_prices()
            except StopIteration:
                if reporter is not None:
                    reporter.record_warning(loop_ts, "Backtest finished (StopIteration).")
                break
            except RuntimeError as exc:
                if reporter is not None:
                    reporter.record_warning(loop_ts, str(exc))
                prices = self.market_data.latest_prices()
                if not prices:
                    time.sleep(self.poll_interval_seconds)
                    continue

            self.market_data.append_prices(prices, timestamp=loop_ts)
            self.account.mark_to_market(prices, self.max_leverage, self.liquidation_threshold)
            
            if self._check_liquidation(prices, loop_ts):
                if reporter is not None:
                    reporter.record_warning(loop_ts, "Account liquidated due to insufficient margin.")
                break

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
                agent_notes = list(getattr(self.agent, "last_sanitization_notes", []) or [])
                plan = {
                    "reasoning": getattr(self.agent, "last_reasoning", ""),
                    "actions": [
                        {"symbol": symbol, "target_exposure": signal.get(symbol, 0.0)}
                        for symbol in self.market_data.symbols
                    ],
                    "agent_adjustments": agent_notes,
                }
                response = self.execute_trading_plan(plan, source="llm_agent")
                if response.get("status") != "filled" and reporter is not None:
                    reason = response.get("reason")
                    message = reason.get("message") if isinstance(reason, dict) else "Unknown rejection."
                    reporter.record_warning(loop_ts, f"Trading plan rejected: {message}")
                else:
                    adjustments = response.get("adjustments", {})
                    agent_notes = adjustments.get("agent") if isinstance(adjustments, dict) else None
                    engine_notes = adjustments.get("engine") if isinstance(adjustments, dict) else None
                    if reporter is not None and agent_notes:
                        for note in agent_notes:
                            reporter.record_warning(loop_ts, f"Agent clamp: {note}")
                    if reporter is not None and engine_notes:
                        for note in engine_notes:
                            reporter.record_warning(loop_ts, f"Constraint applied: {note}")
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
        validation = self._validate_exposures(
            exposures,
            allow_rescale=True,
            previous=self._last_applied_exposures,
        )
        notes = validation.get("notes", [])
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
        self.account.mark_to_market(prices, self.max_leverage, self.liquidation_threshold)
        self._last_applied_exposures = self._current_exposures(prices)
        self._last_validation_notes = notes
        return sanitized

    def _validate_exposures(
        self,
        exposures: Dict[str, float],
        allow_rescale: bool,
        previous: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        known_symbols = set(self.market_data.symbols)
        extra_symbols = sorted({symbol for symbol in exposures.keys()} - known_symbols)
        if extra_symbols:
            return {
                "valid": False,
                "code": "UNKNOWN_SYMBOL",
                "message": f"Unsupported symbols: {', '.join(extra_symbols)}",
                "notes": [],
            }

        per_symbol_cap = max(0.0, float(self.per_symbol_max_exposure))
        delta_cap = max(0.0, float(self.max_exposure_delta))
        previous_map = previous or self._last_applied_exposures or {}
        previous_exposures = {
            symbol: float(previous_map.get(symbol, 0.0)) for symbol in self.market_data.symbols
        }
        notes: List[str] = []
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
                    "notes": notes,
                }

            if per_symbol_cap > 0.0:
                if allow_rescale:
                    if abs(value) > per_symbol_cap + 1e-9:
                        clipped = max(-per_symbol_cap, min(per_symbol_cap, value))
                        notes.append(
                            f"{symbol}: clipped {value:.6f} -> {clipped:.6f} by per-symbol limit ±{per_symbol_cap:.6f}"
                        )
                        value = clipped
                else:
                    if abs(value) > per_symbol_cap + 1e-9:
                        return {
                            "valid": False,
                            "code": "PER_SYMBOL_LIMIT",
                            "message": f"{symbol} exposure {value:.4f} exceeds +/-{per_symbol_cap:.4f}x.",
                            "notes": notes,
                        }

            if delta_cap > 0.0:
                prior = previous_exposures.get(symbol, 0.0)
                lower = prior - delta_cap
                upper = prior + delta_cap
                if allow_rescale:
                    if value < lower - 1e-9 or value > upper + 1e-9:
                        clipped = max(lower, min(upper, value))
                        notes.append(
                            f"{symbol}: adjusted {value:.6f} -> {clipped:.6f} by delta limit ±{delta_cap:.6f} (prev {prior:.6f})"
                        )
                        value = clipped
                else:
                    if value < lower - 1e-9 or value > upper + 1e-9:
                        delta = value - prior
                        return {
                            "valid": False,
                            "code": "DELTA_LIMIT",
                            "message": (
                                f"{symbol} exposure change {delta:.4f} exceeds +/-{delta_cap:.4f}x "
                                f"from previous {prior:.4f}x."
                            ),
                            "notes": notes,
                        }

            if not allow_rescale and abs(value) > self.max_leverage + 1e-9:
                return {
                    "valid": False,
                    "code": "PER_SYMBOL_LIMIT",
                    "message": f"{symbol} exposure {value:.4f} exceeds +/-{self.max_leverage:.4f}x.",
                    "notes": notes,
                }
            sanitized[symbol] = value

        if self.max_leverage <= 0.0:
            if allow_rescale:
                if any(abs(value) > 1e-9 for value in sanitized.values()):
                    notes.append("Max leverage is 0; zeroing all exposures.")
                sanitized = {symbol: 0.0 for symbol in sanitized}
            else:
                if any(abs(value) > 1e-9 for value in sanitized.values()):
                    return {
                        "valid": False,
                        "code": "LEVERAGE_LIMIT",
                        "message": "Maximum leverage is 0; no positions may be opened.",
                        "notes": notes,
                    }
                sanitized = {symbol: 0.0 for symbol in sanitized}
        else:
            total_abs = sum(abs(value) for value in sanitized.values())
            if total_abs > self.max_leverage + 1e-9 and total_abs > 0.0:
                if allow_rescale:
                    scale = self.max_leverage / total_abs
                    sanitized = {symbol: value * scale for symbol, value in sanitized.items()}
                    notes.append(
                        f"Scaled exposures by {scale:.6f} to respect total leverage ±{self.max_leverage:.6f}"
                    )
                else:
                    return {
                        "valid": False,
                        "code": "LEVERAGE_LIMIT",
                        "message": (
                            f"Aggregate exposure {total_abs:.4f} exceeds maximum leverage {self.max_leverage:.4f}."
                        ),
                        "notes": notes,
                    }

        portfolio_result = self.portfolio_risk.validate_portfolio_constraints(
            proposed_exposures=sanitized,
            current_exposures=previous_exposures,
            allow_rescale=allow_rescale,
        )
        if not portfolio_result["valid"]:
            return {
                "valid": False,
                "code": portfolio_result.get("code", "PORTFOLIO_CONSTRAINT"),
                "message": portfolio_result.get("message", "Portfolio constraint violated"),
                "notes": notes + portfolio_result.get("notes", []),
            }
        sanitized = portfolio_result["exposures"]
        notes.extend(portfolio_result.get("notes", []))

        return {
            "valid": True,
            "exposures": sanitized,
            "notes": notes,
            "portfolio_metrics": portfolio_result.get("metrics", {}),
        }

    def _rebalance_position(
        self,
        symbol: str,
        target_quantity: float,
        price: float,
        timestamp: pd.Timestamp,
    ) -> None:
        position = self.account.positions.get(symbol)

        def calculate_execution(qty: float, base_price: float) -> tuple[float, float, float]:
            if qty > 0:
                exec_price = base_price * (1 + self.slippage)
            else:
                exec_price = base_price * (1 - self.slippage)
            commission = abs(qty * exec_price) * self.commission_rate
            slippage_cost = abs(qty * base_price * self.slippage)
            self.portfolio_risk.accumulate_costs(commission, slippage_cost)
            return exec_price, commission, slippage_cost

        if position is None:
            if abs(target_quantity) < 1e-8:
                return
            exec_price, comm, slip = calculate_execution(target_quantity, price)
            self.account.balance -= comm
            leverage = self._compute_position_leverage(exec_price, target_quantity)
            self.account.positions[symbol] = FuturesPosition(
                symbol=symbol,
                quantity=target_quantity,
                entry_price=exec_price,
                leverage=leverage,
                opened_at=timestamp,
            )
            self.trade_log.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "action": "open",
                    "quantity": target_quantity,
                    "price": exec_price,
                    "commission": comm,
                    "slippage_cost": slip,
                    "realized_pnl": -comm,
                }
            )
            return

        existing_qty = position.quantity
        if abs(target_quantity) < 1e-8:
            delta_qty = -existing_qty
            exec_price, comm, slip = calculate_execution(delta_qty, price)
            realized = (exec_price - position.entry_price) * existing_qty
            self.account.balance += realized - comm
            self.account.realized_pnl += realized - comm
            self.trade_log.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "action": "close",
                    "quantity": existing_qty,
                    "price": exec_price,
                    "commission": comm,
                    "slippage_cost": slip,
                    "realized_pnl": realized - comm,
                }
            )
            del self.account.positions[symbol]
            return

        if existing_qty * target_quantity > 0:
            if abs(target_quantity) > abs(existing_qty):
                delta_qty = target_quantity - existing_qty
                exec_price, comm, slip = calculate_execution(delta_qty, price)
                self.account.balance -= comm
                weighted_notional = position.entry_price * existing_qty + exec_price * delta_qty
                position.quantity = target_quantity
                position.entry_price = weighted_notional / target_quantity
                position.leverage = self._compute_position_leverage(price, target_quantity)
                self.trade_log.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": symbol,
                        "action": "increase",
                        "quantity": delta_qty,
                        "price": exec_price,
                        "commission": comm,
                        "slippage_cost": slip,
                        "realized_pnl": -comm,
                    }
                )
            else:
                closed_qty = existing_qty - target_quantity
                delta_qty = -closed_qty
                exec_price, comm, slip = calculate_execution(target_quantity - existing_qty, price)
                realized = (exec_price - position.entry_price) * closed_qty
                self.account.balance += realized - comm
                self.account.realized_pnl += realized - comm
                position.quantity = target_quantity
                self.trade_log.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": symbol,
                        "action": "reduce",
                        "quantity": closed_qty,
                        "price": exec_price,
                        "commission": comm,
                        "slippage_cost": slip,
                        "realized_pnl": realized - comm,
                    }
                )
                if abs(position.quantity) < 1e-8:
                    del self.account.positions[symbol]
            return

        delta_close = -existing_qty
        exec_close, comm_close, slip_close = calculate_execution(delta_close, price)
        realized = (exec_close - position.entry_price) * existing_qty
        self.account.balance += realized - comm_close
        self.account.realized_pnl += realized - comm_close
        self.trade_log.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": "reverse_close",
                "quantity": existing_qty,
                "price": exec_close,
                "commission": comm_close,
                "slippage_cost": slip_close,
                "realized_pnl": realized - comm_close,
            }
        )
        
        exec_open, comm_open, slip_open = calculate_execution(target_quantity, price)
        self.account.balance -= comm_open
        leverage = self._compute_position_leverage(exec_open, target_quantity)
        self.account.positions[symbol] = FuturesPosition(
            symbol=symbol,
            quantity=target_quantity,
            entry_price=exec_open,
            leverage=leverage,
            opened_at=timestamp,
        )
        self.trade_log.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": "reverse_open",
                "quantity": target_quantity,
                "price": exec_open,
                "commission": comm_open,
                "slippage_cost": slip_open,
                "realized_pnl": -comm_open,
            }
        )

    def _check_liquidation(self, prices: Dict[str, float], timestamp: pd.Timestamp) -> bool:
        self.portfolio_risk.verify_equity_consistency(
            self.account.balance,
            self.account.unrealized_pnl,
            self.account.equity,
        )
        
        is_liquidated = False
        trigger_reason = ""
        
        if self.account.equity <= 0:
            is_liquidated = True
            trigger_reason = f"Equity depleted: {self.account.equity:.2f} <= 0"
        elif self.account.maintenance_margin_req > 0 and self.account.equity < self.account.maintenance_margin_req:
            is_liquidated = True
            trigger_reason = (
                f"Margin call: equity {self.account.equity:.2f} < "
                f"maintenance_margin {self.account.maintenance_margin_req:.2f}"
            )
        
        if is_liquidated:
            self._liquidation_audit = self.portfolio_risk.create_liquidation_audit_log(
                timestamp=timestamp.isoformat(),
                trigger_reason=trigger_reason,
                balance=self.account.balance,
                unrealized_pnl=self.account.unrealized_pnl,
                equity=self.account.equity,
                positions=self.account.positions,
                prices=prices,
            )
            
            for symbol, position in list(self.account.positions.items()):
                qty = -position.quantity
                if qty > 0:
                    price = prices.get(symbol, position.entry_price) * (1 + self.slippage)
                else:
                    price = prices.get(symbol, position.entry_price) * (1 - self.slippage)
                
                commission = abs(qty * price) * self.commission_rate
                slippage_cost = abs(qty * prices.get(symbol, position.entry_price) * self.slippage)
                self.portfolio_risk.accumulate_costs(commission, slippage_cost)
                
                realized = (price - position.entry_price) * position.quantity
                self.account.balance += realized - commission
                self.account.realized_pnl += realized - commission
                
                self.trade_log.append({
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "action": "liquidation",
                    "quantity": position.quantity,
                    "price": price,
                    "commission": commission,
                    "slippage_cost": slippage_cost,
                    "realized_pnl": realized - commission,
                    "liquidation_audit": True,
                })
            self.account.positions.clear()
            self.account.equity = self.account.balance
            self.account.margin_used = 0
            self.account.available_margin = 0
            return True
        return False

    def _compute_position_leverage(self, price: float, quantity: float) -> float:
        equity = max(self.account.equity, 1e-6)
        notional = abs(price * quantity)
        if equity <= 0:
            return self.max_leverage
        leverage = notional / equity
        return max(1.0, min(self.max_leverage, leverage))

    def execute_trading_plan(self, plan: Dict[str, Any], *, source: str = "external_command") -> Dict[str, Any]:
        timestamp = pd.Timestamp.utcnow().floor("s")
        raw_agent_notes = plan.get("agent_adjustments") if isinstance(plan, dict) else []
        agent_notes = [str(note) for note in raw_agent_notes if note]
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
                    agent_notes=agent_notes,
                    engine_notes=None,
                )
                return response
        self.market_data.append_prices(prices, timestamp=timestamp)
        self.account.mark_to_market(prices, self.max_leverage, self.liquidation_threshold)

        current_exposures = self._current_exposures(prices)
        requested_exposures = current_exposures.copy()
        for action in plan.get("actions", []):
            symbol = str(action.get("symbol", "")).upper()
            if not symbol:
                continue
            value = action.get("target_exposure", 0.0)
            requested_exposures[symbol] = value

        allow_rescale = source == "llm_agent"
        validation = self._validate_exposures(
            requested_exposures,
            allow_rescale=allow_rescale,
            previous=current_exposures,
        )
        engine_notes = validation.get("notes", [])
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
                agent_notes=agent_notes,
                engine_notes=engine_notes,
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
                    agent_notes=agent_notes,
                    engine_notes=engine_notes,
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
                agent_notes=agent_notes,
                engine_notes=engine_notes,
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
                agent_notes=agent_notes,
                engine_notes=engine_notes,
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

        self.account.mark_to_market(prices, self.max_leverage, self.liquidation_threshold)
        self._last_applied_exposures = self._current_exposures(prices)
        self._last_validation_notes = engine_notes
        response = {
            "status": "filled",
            "timestamp": timestamp.isoformat(),
            "applied_exposures": applied_exposures,
            "positions": self._positions_snapshot(prices),
            "account": self._account_snapshot(),
            "reason": None,
            "adjustments": {
                "agent": agent_notes,
                "engine": engine_notes,
            },
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
            agent_notes=agent_notes,
            engine_notes=engine_notes,
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
        agent_notes: Optional[List[str]],
        engine_notes: Optional[List[str]],
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
        if agent_notes:
            entry["agent_notes"] = list(agent_notes)
        if engine_notes:
            entry["engine_notes"] = list(engine_notes)
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
