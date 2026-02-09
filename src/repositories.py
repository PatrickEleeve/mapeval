"""Repository pattern for database operations.

Provides high-level CRUD operations for each entity type,
abstracting away SQLAlchemy session details.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from sqlalchemy.orm import Session
    from db_models import (
        SQLALCHEMY_AVAILABLE,
        DecisionRecord,
        EquitySnapshot,
        OrderRecord,
        RiskEvent,
        SessionRecord,
        TradeRecord,
    )
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class SessionRepository:
    """CRUD operations for trading sessions."""

    def __init__(self, db_session: Session) -> None:
        self._session = db_session

    def create(
        self,
        initial_capital: float,
        execution_mode: str = "simulation",
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        record = SessionRecord(
            session_id=session_id,
            start_time=datetime.now(timezone.utc),
            status="running",
            execution_mode=execution_mode,
            config_json=json.dumps(config) if config else None,
            initial_capital=initial_capital,
        )
        self._session.add(record)
        self._session.flush()
        return session_id

    def complete(self, session_id: str, final_equity: float, total_pnl: float, total_trades: int) -> None:
        record = self._session.query(SessionRecord).filter_by(session_id=session_id).first()
        if record:
            record.end_time = datetime.now(timezone.utc)
            record.status = "completed"
            record.final_equity = final_equity
            record.total_pnl = total_pnl
            record.total_trades = total_trades

    def mark_crashed(self, session_id: str) -> None:
        record = self._session.query(SessionRecord).filter_by(session_id=session_id).first()
        if record:
            record.end_time = datetime.now(timezone.utc)
            record.status = "crashed"

    def get(self, session_id: str) -> Optional[SessionRecord]:
        return self._session.query(SessionRecord).filter_by(session_id=session_id).first()

    def get_incomplete(self) -> List[SessionRecord]:
        return self._session.query(SessionRecord).filter_by(status="running").all()

    def list_recent(self, limit: int = 20) -> List[SessionRecord]:
        return (
            self._session.query(SessionRecord)
            .order_by(SessionRecord.start_time.desc())
            .limit(limit)
            .all()
        )


class TradeRepository:
    """CRUD operations for trade records."""

    def __init__(self, db_session: Session) -> None:
        self._session = db_session

    def save(self, session_id: str, trade: Dict[str, Any]) -> int:
        record = TradeRecord(
            session_id=session_id,
            timestamp=datetime.fromisoformat(trade["timestamp"]) if isinstance(trade["timestamp"], str) else trade["timestamp"],
            symbol=trade["symbol"],
            action=trade["action"],
            quantity=trade["quantity"],
            price=trade["price"],
            commission=trade.get("commission", 0.0),
            slippage_cost=trade.get("slippage_cost", 0.0),
            realized_pnl=trade.get("realized_pnl", 0.0),
            order_id=trade.get("order_id"),
        )
        self._session.add(record)
        self._session.flush()
        return record.id

    def save_batch(self, session_id: str, trades: List[Dict[str, Any]]) -> int:
        count = 0
        for trade in trades:
            self.save(session_id, trade)
            count += 1
        return count

    def get_by_session(self, session_id: str, limit: int = 1000) -> List[TradeRecord]:
        return (
            self._session.query(TradeRecord)
            .filter_by(session_id=session_id)
            .order_by(TradeRecord.timestamp.desc())
            .limit(limit)
            .all()
        )

    def get_by_symbol(self, session_id: str, symbol: str) -> List[TradeRecord]:
        return (
            self._session.query(TradeRecord)
            .filter_by(session_id=session_id, symbol=symbol)
            .order_by(TradeRecord.timestamp)
            .all()
        )


class DecisionRepository:
    """CRUD operations for decision records."""

    def __init__(self, db_session: Session) -> None:
        self._session = db_session

    def save(self, session_id: str, decision: Dict[str, Any]) -> int:
        record = DecisionRecord(
            session_id=session_id,
            timestamp=datetime.fromisoformat(decision["timestamp"]) if isinstance(decision["timestamp"], str) else decision["timestamp"],
            source=decision.get("source", "unknown"),
            action=decision.get("action", "REBALANCE"),
            requested_exposures=json.dumps(decision.get("requested_exposure", {})),
            applied_exposures=json.dumps(decision.get("applied_exposure", {})),
            reasoning=decision.get("reasoning", ""),
            status=decision.get("status", "unknown"),
            reject_reason=json.dumps(decision.get("reason")) if decision.get("reason") else None,
            equity_at_decision=decision.get("equity", 0.0),
        )
        self._session.add(record)
        self._session.flush()
        return record.id

    def get_by_session(self, session_id: str, limit: int = 500) -> List[DecisionRecord]:
        return (
            self._session.query(DecisionRecord)
            .filter_by(session_id=session_id)
            .order_by(DecisionRecord.timestamp.desc())
            .limit(limit)
            .all()
        )


class EquityRepository:
    """CRUD operations for equity snapshots."""

    def __init__(self, db_session: Session) -> None:
        self._session = db_session

    def save(self, session_id: str, snapshot: Dict[str, Any]) -> int:
        ts = snapshot["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif not isinstance(ts, datetime):
            ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.now(timezone.utc)

        record = EquitySnapshot(
            session_id=session_id,
            timestamp=ts,
            equity=snapshot["equity"],
            balance=snapshot["balance"],
            unrealized_pnl=snapshot.get("unrealized_pnl", 0.0),
            margin_used=snapshot.get("margin_used", 0.0),
        )
        self._session.add(record)
        self._session.flush()
        return record.id

    def get_by_session(self, session_id: str) -> List[EquitySnapshot]:
        return (
            self._session.query(EquitySnapshot)
            .filter_by(session_id=session_id)
            .order_by(EquitySnapshot.timestamp)
            .all()
        )


class OrderRepository:
    """CRUD operations for order records."""

    def __init__(self, db_session: Session) -> None:
        self._session = db_session

    def save(self, session_id: str, order_data: Dict[str, Any]) -> int:
        record = OrderRecord(
            session_id=session_id,
            client_order_id=order_data.get("client_order_id", str(uuid.uuid4())[:16]),
            exchange_order_id=order_data.get("exchange_order_id"),
            timestamp=datetime.now(timezone.utc),
            symbol=order_data["symbol"],
            side=order_data["side"],
            order_type=order_data.get("order_type", "MARKET"),
            quantity=order_data["quantity"],
            price=order_data.get("price"),
            status=order_data.get("status", "FILLED"),
            filled_quantity=order_data.get("filled_quantity", 0.0),
            avg_fill_price=order_data.get("avg_fill_price", 0.0),
            commission=order_data.get("commission", 0.0),
            fills_json=json.dumps(order_data.get("fills")) if order_data.get("fills") else None,
        )
        self._session.add(record)
        self._session.flush()
        return record.id


class RiskEventRepository:
    """CRUD operations for risk events."""

    def __init__(self, db_session: Session) -> None:
        self._session = db_session

    def save(self, session_id: str, event_type: str, severity: str = "warning", details: Optional[Dict[str, Any]] = None) -> int:
        record = RiskEvent(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            details=json.dumps(details) if details else None,
        )
        self._session.add(record)
        self._session.flush()
        return record.id

    def get_by_session(self, session_id: str) -> List[RiskEvent]:
        return (
            self._session.query(RiskEvent)
            .filter_by(session_id=session_id)
            .order_by(RiskEvent.timestamp)
            .all()
        )
