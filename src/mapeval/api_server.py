"""REST API server for the trading system.

Provides HTTP endpoints for monitoring, querying trade history, and
manual intervention. Uses FastAPI with optional WebSocket support for
live streaming.

Run standalone::

    uvicorn api_server:create_app --host 0.0.0.0 --port 8000

Or embed in the trading process via ``start_api_server()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from mapeval.security import constant_time_equals


# Global references set by the trading process
_engine = None
_event_bus = None
_db_manager = None
_api_token = None
_ws_clients: List = []


def set_engine(engine) -> None:
    global _engine
    _engine = engine


def set_event_bus(bus) -> None:
    global _event_bus
    _event_bus = bus


def set_db_manager(db) -> None:
    global _db_manager
    _db_manager = db


def set_api_token(token: Optional[str]) -> None:
    global _api_token
    _api_token = token


def _extract_api_token(request: "Request") -> Optional[str]:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return request.headers.get("x-api-key")


def _extract_websocket_token(websocket: "WebSocket") -> Optional[str]:
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return websocket.headers.get("x-api-key") or websocket.query_params.get("token")


def _require_api_token(request: "Request") -> None:
    if not _api_token:
        return
    provided = _extract_api_token(request)
    if not constant_time_equals(provided, _api_token):
        raise HTTPException(status_code=401, detail="Valid API token required")


async def _require_websocket_token(websocket: "WebSocket") -> bool:
    if not _api_token:
        return True
    provided = _extract_websocket_token(websocket)
    if constant_time_equals(provided, _api_token):
        return True
    await websocket.close(code=1008, reason="Valid API token required")
    return False


def _engine_controls() -> Dict[str, Any]:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")
    return {
        "read_only": bool(getattr(getattr(_engine, "read_only_guard", None), "is_read_only", False)),
        "kill_switch_active": bool(getattr(_engine, "_kill_switch_active", False)),
        "shutdown_requested": bool(getattr(_engine, "_shutdown_requested", False)),
    }


def create_app() -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is required. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="MAPEval Trading System API",
        version="1.0.0",
        description="Real-time trading system monitoring and control API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Status & Health ─────────────────────────────────────────────

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/status")
    async def get_status(request: Request):
        _require_api_token(request)
        if _engine is None:
            raise HTTPException(status_code=503, detail="Trading engine not initialized")
        account = _engine.account
        return {
            "execution_mode": getattr(_engine, "execution_mode", "unknown"),
            "account": {
                "balance": account.balance,
                "equity": account.equity,
                "realized_pnl": account.realized_pnl,
                "unrealized_pnl": account.unrealized_pnl,
                "margin_used": account.margin_used,
                "available_margin": account.available_margin,
            },
            "positions": [
                {
                    "symbol": sym,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "leverage": pos.leverage,
                }
                for sym, pos in account.positions.items()
            ],
            "open_position_count": len(account.positions),
            "total_trades": len(_engine.trade_log),
            "total_decisions": len(_engine.decision_log),
            **_engine_controls(),
        }

    # ── Trade History ───────────────────────────────────────────────

    @app.get("/api/trades")
    async def get_trades(request: Request, limit: int = 50, symbol: Optional[str] = None):
        _require_api_token(request)
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        trades = _engine.trade_log[-limit:]
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol.upper()]
        return {"trades": trades, "total": len(_engine.trade_log)}

    # ── Decision History ────────────────────────────────────────────

    @app.get("/api/decisions")
    async def get_decisions(request: Request, limit: int = 20):
        _require_api_token(request)
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        return {
            "decisions": _engine.decision_log[-limit:],
            "total": len(_engine.decision_log),
        }

    # ── Equity History ──────────────────────────────────────────────

    @app.get("/api/equity-history")
    async def get_equity_history(request: Request, limit: int = 500):
        _require_api_token(request)
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        history = _engine.equity_history[-limit:]
        return {
            "history": [
                {
                    "timestamp": str(h["timestamp"]),
                    "equity": h["equity"],
                    "balance": h["balance"],
                    "unrealized_pnl": h.get("unrealized_pnl", 0),
                }
                for h in history
            ],
            "total_points": len(_engine.equity_history),
        }

    # ── Risk Status ─────────────────────────────────────────────────

    @app.get("/api/risk")
    async def get_risk(request: Request):
        _require_api_token(request)
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        risk_report = {}
        if _engine.risk_manager is not None:
            risk_report = _engine.risk_manager.get_risk_report(_engine.account.equity)
        return {"risk": risk_report}

    # ── Configuration ───────────────────────────────────────────────

    @app.get("/api/config")
    async def get_config(request: Request):
        _require_api_token(request)
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        return {
            "max_leverage": _engine.max_leverage,
            "per_symbol_max_exposure": _engine.per_symbol_max_exposure,
            "max_exposure_delta": _engine.max_exposure_delta,
            "commission_rate": _engine.commission_rate,
            "slippage": _engine.slippage,
            "execution_mode": getattr(_engine, "execution_mode", "simulation"),
            "symbols": list(getattr(_engine.market_data, "symbols", [])),
        }

    # ── Manual Override ─────────────────────────────────────────────

    @app.post("/api/shutdown")
    async def request_shutdown(request: Request):
        _require_api_token(request)
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        _engine.shutdown()
        return {"status": "shutdown_requested"}

    @app.get("/api/controls")
    async def get_controls(request: Request):
        _require_api_token(request)
        return _engine_controls()

    @app.post("/api/read-only")
    async def set_read_only(request: Request, payload: Optional[Dict[str, Any]] = None):
        _require_api_token(request)
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        enabled = True if payload is None else bool(payload.get("enabled", True))
        if not hasattr(_engine, "set_read_only") or not _engine.set_read_only(enabled):
            raise HTTPException(status_code=400, detail="Read-only guard not configured")
        return {
            "status": "ok",
            "read_only": enabled,
            "kill_switch_active": bool(getattr(_engine, "_kill_switch_active", False)),
        }

    @app.post("/api/kill-switch")
    async def set_kill_switch(request: Request, payload: Optional[Dict[str, Any]] = None):
        _require_api_token(request)
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        payload = payload or {}
        enabled = bool(payload.get("enabled", True))
        reason = str(payload.get("reason", "api"))
        close_positions = bool(payload.get("close_positions", False))
        if enabled:
            return _engine.activate_kill_switch(reason=reason, close_positions=close_positions)
        return _engine.release_kill_switch()

    # ── WebSocket Live Stream ───────────────────────────────────────

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        if not await _require_websocket_token(websocket):
            return
        await websocket.accept()
        _ws_clients.append(websocket)
        try:
            while True:
                # Keep connection alive, send status every 5 seconds
                if _engine is not None:
                    data = {
                        "type": "status",
                        "equity": _engine.account.equity,
                        "balance": _engine.account.balance,
                        "positions": len(_engine.account.positions),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await websocket.send_json(data)
                await asyncio.sleep(5)
        except WebSocketDisconnect:
            _ws_clients.remove(websocket)

    # ── Event Bus Stats ─────────────────────────────────────────────

    @app.get("/api/events/stats")
    async def get_event_stats(request: Request):
        _require_api_token(request)
        if _event_bus is None:
            return {"status": "no event bus configured"}
        return _event_bus.get_stats()

    return app


def start_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    engine=None,
    event_bus=None,
    db_manager=None,
    api_token: Optional[str] = None,
) -> threading.Thread:
    """Start the API server in a background thread.

    Returns the thread for lifecycle management.
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, API server will not start")
        return None

    if engine is not None:
        set_engine(engine)
    if event_bus is not None:
        set_event_bus(event_bus)
    if db_manager is not None:
        set_db_manager(db_manager)
    set_api_token(api_token)

    app = create_app()

    def run():
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        server.run()

    thread = threading.Thread(target=run, daemon=True, name="api-server")
    thread.start()
    logger.info("API server started on %s:%d", host, port)
    return thread
