"""SQLAlchemy database models for the trading system.

Defines tables for sessions, trades, decisions, equity snapshots, orders, and risk events.
Uses SQLAlchemy 2.0 declarative style with mapped_column.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

try:
    from sqlalchemy import (
        Column, DateTime, Float, Integer, String, Text, JSON, Boolean,
        ForeignKey, Index, create_engine,
    )
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


if SQLALCHEMY_AVAILABLE:
    class Base(DeclarativeBase):
        pass

    class SessionRecord(Base):
        """Trading session record."""
        __tablename__ = "sessions"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
        start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
        end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
        status: Mapped[str] = mapped_column(String(20), default="running")  # running, completed, crashed
        execution_mode: Mapped[str] = mapped_column(String(20), default="simulation")
        config_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        initial_capital: Mapped[float] = mapped_column(Float, default=100000.0)
        final_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        total_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        total_trades: Mapped[int] = mapped_column(Integer, default=0)

        trades = relationship("TradeRecord", back_populates="session", cascade="all, delete-orphan")
        decisions = relationship("DecisionRecord", back_populates="session", cascade="all, delete-orphan")
        equity_snapshots = relationship("EquitySnapshot", back_populates="session", cascade="all, delete-orphan")
        orders = relationship("OrderRecord", back_populates="session", cascade="all, delete-orphan")
        risk_events = relationship("RiskEvent", back_populates="session", cascade="all, delete-orphan")

    class TradeRecord(Base):
        """Individual trade execution record."""
        __tablename__ = "trades"
        __table_args__ = (
            Index("idx_trades_session_time", "session_id", "timestamp"),
            Index("idx_trades_symbol", "symbol"),
        )

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.session_id"), nullable=False)
        timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
        symbol: Mapped[str] = mapped_column(String(20), nullable=False)
        action: Mapped[str] = mapped_column(String(20), nullable=False)  # open, close, increase, reduce, etc.
        quantity: Mapped[float] = mapped_column(Float, nullable=False)
        price: Mapped[float] = mapped_column(Float, nullable=False)
        commission: Mapped[float] = mapped_column(Float, default=0.0)
        slippage_cost: Mapped[float] = mapped_column(Float, default=0.0)
        realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
        order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

        session = relationship("SessionRecord", back_populates="trades")

    class DecisionRecord(Base):
        """LLM trading decision record with reasoning."""
        __tablename__ = "decisions"
        __table_args__ = (
            Index("idx_decisions_session_time", "session_id", "timestamp"),
        )

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.session_id"), nullable=False)
        timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
        source: Mapped[str] = mapped_column(String(30), nullable=False)
        action: Mapped[str] = mapped_column(String(20), default="REBALANCE")
        requested_exposures: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
        applied_exposures: Mapped[Optional[str]] = mapped_column(Text, nullable=True)    # JSON
        reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        status: Mapped[str] = mapped_column(String(20), nullable=False)  # filled, rejected
        reject_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
        equity_at_decision: Mapped[float] = mapped_column(Float, default=0.0)

        session = relationship("SessionRecord", back_populates="decisions")

    class EquitySnapshot(Base):
        """Periodic equity/balance snapshot for charting."""
        __tablename__ = "equity_snapshots"
        __table_args__ = (
            Index("idx_equity_session_time", "session_id", "timestamp"),
        )

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.session_id"), nullable=False)
        timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
        equity: Mapped[float] = mapped_column(Float, nullable=False)
        balance: Mapped[float] = mapped_column(Float, nullable=False)
        unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
        margin_used: Mapped[float] = mapped_column(Float, default=0.0)

        session = relationship("SessionRecord", back_populates="equity_snapshots")

    class OrderRecord(Base):
        """Order lifecycle tracking."""
        __tablename__ = "orders"
        __table_args__ = (
            Index("idx_orders_session", "session_id"),
            Index("idx_orders_symbol_status", "symbol", "status"),
        )

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.session_id"), nullable=False)
        client_order_id: Mapped[str] = mapped_column(String(64), nullable=False)
        exchange_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
        timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
        symbol: Mapped[str] = mapped_column(String(20), nullable=False)
        side: Mapped[str] = mapped_column(String(10), nullable=False)
        order_type: Mapped[str] = mapped_column(String(20), nullable=False)
        quantity: Mapped[float] = mapped_column(Float, nullable=False)
        price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
        status: Mapped[str] = mapped_column(String(20), nullable=False)
        filled_quantity: Mapped[float] = mapped_column(Float, default=0.0)
        avg_fill_price: Mapped[float] = mapped_column(Float, default=0.0)
        commission: Mapped[float] = mapped_column(Float, default=0.0)
        fills_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

        session = relationship("SessionRecord", back_populates="orders")

    class RiskEvent(Base):
        """Risk management event log."""
        __tablename__ = "risk_events"
        __table_args__ = (
            Index("idx_risk_events_session", "session_id"),
        )

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.session_id"), nullable=False)
        timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
        event_type: Mapped[str] = mapped_column(String(30), nullable=False)
        severity: Mapped[str] = mapped_column(String(10), default="warning")  # info, warning, critical
        details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON

        session = relationship("SessionRecord", back_populates="risk_events")
