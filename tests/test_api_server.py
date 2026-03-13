"""Tests for API control endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from mapeval.api_server import FASTAPI_AVAILABLE

if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from mapeval.api_server import create_app, set_api_token, set_engine
    from mapeval.security import ReadOnlyGuard


    class DummyAccount:
        balance = 1000.0
        equity = 1000.0
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        margin_used = 0.0
        available_margin = 1000.0
        positions = {}


    class DummyEngine:
        def __init__(self) -> None:
            self.account = DummyAccount()
            self.execution_mode = "paper"
            self.trade_log = []
            self.decision_log = []
            self.equity_history = []
            self._shutdown_requested = False
            self._kill_switch_active = False
            self.read_only_guard = ReadOnlyGuard(enabled=False)
            self.market_data = type("MarketData", (), {"symbols": ["BTCUSDT"]})()
            self.max_leverage = 1.0
            self.per_symbol_max_exposure = 1.0
            self.max_exposure_delta = 1.0
            self.commission_rate = 0.0
            self.slippage = 0.0
            self.risk_manager = None

        def set_read_only(self, enabled: bool) -> bool:
            if enabled:
                self.read_only_guard.enable()
            else:
                self.read_only_guard.disable()
            return True

        def activate_kill_switch(self, reason: str = "api", close_positions: bool = False):
            self._kill_switch_active = True
            self.read_only_guard.enable()
            return {"kill_switch_active": True, "read_only": True, "positions_closed": close_positions}

        def release_kill_switch(self):
            self._kill_switch_active = False
            self.read_only_guard.disable()
            return {"kill_switch_active": False, "read_only": False}

        def shutdown(self):
            self._shutdown_requested = True


    def test_control_endpoints_toggle_state():
        set_engine(DummyEngine())
        set_api_token(None)
        client = TestClient(create_app())

        controls = client.get("/api/controls")
        assert controls.status_code == 200
        assert controls.json()["read_only"] is False

        read_only = client.post("/api/read-only", json={"enabled": True})
        assert read_only.status_code == 200
        assert read_only.json()["read_only"] is True

        kill_switch = client.post("/api/kill-switch", json={"enabled": True, "reason": "test"})
        assert kill_switch.status_code == 200
        assert kill_switch.json()["kill_switch_active"] is True

        release = client.post("/api/kill-switch", json={"enabled": False})
        assert release.status_code == 200
        assert release.json()["kill_switch_active"] is False

    def test_control_endpoints_require_token_when_configured():
        set_engine(DummyEngine())
        set_api_token("secret-token")
        client = TestClient(create_app())

        unauthorized = client.get("/api/controls")
        assert unauthorized.status_code == 401

        authorized = client.get("/api/controls", headers={"x-api-key": "secret-token"})
        assert authorized.status_code == 200
        assert authorized.json()["read_only"] is False

        bearer = client.post(
            "/api/read-only",
            json={"enabled": True},
            headers={"authorization": "Bearer secret-token"},
        )
        assert bearer.status_code == 200
        assert bearer.json()["read_only"] is True

        set_api_token(None)

    def test_websocket_requires_token_when_configured():
        set_engine(DummyEngine())
        set_api_token("secret-token")
        client = TestClient(create_app())

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/stream"):
                pass

        with client.websocket_connect("/ws/stream?token=secret-token") as websocket:
            message = websocket.receive_json()
            assert message["type"] == "status"

        set_api_token(None)


else:
    def test_fastapi_optional_dependency():
        pytest.skip("fastapi not installed")
