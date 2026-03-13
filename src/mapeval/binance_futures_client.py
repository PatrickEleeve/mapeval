"""Authenticated Binance USDT-M Futures REST API client.

Supports order placement, account queries, and position management with HMAC-SHA256
request signing. Designed for both mainnet and testnet.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from mapeval.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

MAINNET_BASE = "https://fapi.binance.com"
TESTNET_BASE = "https://testnet.binancefuture.com"


class BinanceFuturesClient:
    """Authenticated client for Binance USDT-M Futures API.

    Parameters
    ----------
    api_key:
        Binance API key.
    api_secret:
        Binance API secret.
    testnet:
        If True, use the testnet endpoint.
    base_url:
        Override the base URL directly (takes precedence over ``testnet``).
    rate_limiter:
        Optional shared rate limiter instance.
    recv_window:
        Timestamp tolerance in ms (default 5000).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        base_url: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        recv_window: int = 5000,
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("Binance API key and secret are required")
        self._api_key = api_key
        self._api_secret = api_secret.encode("utf-8")
        self._base_url = (base_url or (TESTNET_BASE if testnet else MAINNET_BASE)).rstrip("/")
        self._rate_limiter = rate_limiter
        self._recv_window = recv_window
        self._session = requests.Session() if requests is not None else None
        self._exchange_info_cache: Optional[Dict[str, Any]] = None
        self._symbol_rules_cache: Dict[str, Dict[str, Any]] = {}
        if self._session is not None:
            self._session.headers.update({
                "X-MBX-APIKEY": self._api_key,
                "Content-Type": "application/x-www-form-urlencoded",
            })

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add timestamp, recvWindow, and HMAC signature to params."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self._recv_window
        query_string = urlencode(params)
        signature = hmac.new(self._api_secret, query_string.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True,
        weight: int = 1,
    ) -> Any:
        """Send a request to the Binance Futures API."""
        if self._session is None:
            raise RuntimeError("requests library is required for BinanceFuturesClient")

        if self._rate_limiter is not None:
            self._rate_limiter.acquire(weight)

        url = self._base_url + path
        params = dict(params or {})
        if signed:
            params = self._sign(params)

        try:
            if method.upper() == "GET":
                resp = self._session.get(url, params=params, timeout=10)
            elif method.upper() == "POST":
                resp = self._session.post(url, params=params, timeout=10)
            elif method.upper() == "DELETE":
                resp = self._session.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if self._rate_limiter is not None:
                self._rate_limiter.update_from_headers(dict(resp.headers))

            if resp.status_code == 429:
                if self._rate_limiter is not None:
                    wait = self._rate_limiter.on_429()
                    time.sleep(wait)
                raise RuntimeError(f"Rate limited (429) from {path}")

            if self._rate_limiter is not None:
                self._rate_limiter.on_success()

            if resp.status_code >= 400:
                error_data = resp.json() if resp.content else {}
                raise RuntimeError(
                    f"Binance API error {resp.status_code}: "
                    f"code={error_data.get('code')}, msg={error_data.get('msg')}"
                )
            return resp.json()

        except requests.exceptions.RequestException as exc:
            logger.error("Request failed: %s %s - %s", method, path, exc)
            raise

    # ── Account & Position Queries ──────────────────────────────────────

    def get_account_info(self) -> Dict[str, Any]:
        """GET /fapi/v2/account - Account information including balances and positions."""
        return self._request("GET", "/fapi/v2/account", weight=5)

    def get_balance(self) -> List[Dict[str, Any]]:
        """GET /fapi/v2/balance - Account balance for all assets."""
        return self._request("GET", "/fapi/v2/balance", weight=5)

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """GET /fapi/v2/positionRisk - Current position information."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v2/positionRisk", params=params, weight=5)

    # ── Order Management ────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: Optional[str] = None,
        new_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /fapi/v1/order - Place a new order.

        Parameters
        ----------
        symbol: Trading pair (e.g. "BTCUSDT")
        side: "BUY" or "SELL"
        order_type: "LIMIT", "MARKET", "STOP", "STOP_MARKET", etc.
        quantity: Order quantity
        price: Limit price (required for LIMIT orders)
        reduce_only: If True, only reduce existing position
        time_in_force: "GTC", "IOC", "FOK" (required for LIMIT)
        new_client_order_id: Custom order ID
        """
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if quantity is not None:
            params["quantity"] = f"{quantity:.8f}".rstrip("0").rstrip(".")
        if price is not None:
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")
        if reduce_only:
            params["reduceOnly"] = "true"
        if time_in_force:
            params["timeInForce"] = time_in_force
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id
        return self._request("POST", "/fapi/v1/order", params=params, weight=1)

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Place a market order (convenience wrapper)."""
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=quantity,
            reduce_only=reduce_only,
        )

    def cancel_order(self, symbol: str, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """DELETE /fapi/v1/order - Cancel an active order."""
        params: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id is required")
        return self._request("DELETE", "/fapi/v1/order", params=params, weight=1)

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """DELETE /fapi/v1/allOpenOrders - Cancel all open orders for a symbol."""
        return self._request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol.upper()}, weight=1)

    def get_order(self, symbol: str, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """GET /fapi/v1/order - Query order status."""
        params: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        return self._request("GET", "/fapi/v1/order", params=params, weight=1)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """GET /fapi/v1/openOrders - All current open orders."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/fapi/v1/openOrders", params=params, weight=1 if symbol else 40)

    # ── Leverage & Margin ───────────────────────────────────────────────

    def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """POST /fapi/v1/leverage - Change initial leverage for a symbol."""
        return self._request("POST", "/fapi/v1/leverage", params={
            "symbol": symbol.upper(),
            "leverage": leverage,
        }, weight=1)

    def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """POST /fapi/v1/marginType - Change margin type (ISOLATED or CROSSED)."""
        return self._request("POST", "/fapi/v1/marginType", params={
            "symbol": symbol.upper(),
            "marginType": margin_type.upper(),
        }, weight=1)

    # ── Market Data (public, no signature needed) ───────────────────────

    def get_ticker_price(self, symbol: Optional[str] = None) -> Any:
        """GET /fapi/v1/ticker/price - Latest price."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/fapi/v1/ticker/price", params=params, signed=False, weight=1)

    def get_exchange_info(self) -> Dict[str, Any]:
        """GET /fapi/v1/exchangeInfo - Exchange trading rules and symbol information."""
        if self._exchange_info_cache is None:
            self._exchange_info_cache = self._request("GET", "/fapi/v1/exchangeInfo", signed=False, weight=1)
        return self._exchange_info_cache

    @staticmethod
    def _round_down_to_step(value: float, step: float) -> float:
        """Round a value down to the nearest valid exchange increment."""
        if step <= 0:
            return float(value)
        decimal_value = Decimal(str(value))
        decimal_step = Decimal(str(step))
        rounded = (decimal_value / decimal_step).to_integral_value(rounding=ROUND_DOWN) * decimal_step
        return float(rounded)

    def get_symbol_rules(self, symbol: str) -> Dict[str, Any]:
        """Return cached exchange rules for a specific symbol."""
        symbol = symbol.upper()
        cached = self._symbol_rules_cache.get(symbol)
        if cached is not None:
            return cached

        exchange_info = self.get_exchange_info()
        for symbol_info in exchange_info.get("symbols", []):
            if symbol_info.get("symbol") != symbol:
                continue
            filters = {item.get("filterType"): item for item in symbol_info.get("filters", [])}
            rules = {
                "quantity_precision": int(symbol_info.get("quantityPrecision", 8)),
                "price_precision": int(symbol_info.get("pricePrecision", 8)),
                "step_size": float(filters.get("LOT_SIZE", {}).get("stepSize", 0) or 0),
                "min_qty": float(filters.get("LOT_SIZE", {}).get("minQty", 0) or 0),
                "max_qty": float(filters.get("LOT_SIZE", {}).get("maxQty", 0) or 0),
                "tick_size": float(filters.get("PRICE_FILTER", {}).get("tickSize", 0) or 0),
                "min_price": float(filters.get("PRICE_FILTER", {}).get("minPrice", 0) or 0),
                "max_price": float(filters.get("PRICE_FILTER", {}).get("maxPrice", 0) or 0),
                "min_notional": float(filters.get("MIN_NOTIONAL", {}).get("notional", 0) or 0),
            }
            self._symbol_rules_cache[symbol] = rules
            return rules
        raise ValueError(f"Unknown Binance Futures symbol: {symbol}")

    def normalize_order(self, symbol: str, quantity: float, price: Optional[float] = None) -> Dict[str, Optional[float]]:
        """Normalize an order to exchange constraints and reject invalid sizes."""
        rules = self.get_symbol_rules(symbol)
        normalized_qty = abs(float(quantity))
        normalized_price = float(price) if price is not None else None

        if rules["step_size"] > 0:
            normalized_qty = self._round_down_to_step(normalized_qty, rules["step_size"])
        if rules["tick_size"] > 0 and normalized_price is not None:
            normalized_price = self._round_down_to_step(normalized_price, rules["tick_size"])

        if normalized_qty <= 0:
            raise ValueError(f"{symbol} quantity rounds down to 0")
        if rules["min_qty"] > 0 and normalized_qty < rules["min_qty"]:
            raise ValueError(f"{symbol} quantity {normalized_qty} below minQty {rules['min_qty']}")
        if rules["max_qty"] > 0 and normalized_qty > rules["max_qty"]:
            raise ValueError(f"{symbol} quantity {normalized_qty} exceeds maxQty {rules['max_qty']}")
        if normalized_price is not None:
            if rules["min_price"] > 0 and normalized_price < rules["min_price"]:
                raise ValueError(f"{symbol} price {normalized_price} below minPrice {rules['min_price']}")
            if rules["max_price"] > 0 and normalized_price > rules["max_price"]:
                raise ValueError(f"{symbol} price {normalized_price} exceeds maxPrice {rules['max_price']}")
            if rules["min_notional"] > 0 and normalized_qty * normalized_price < rules["min_notional"]:
                raise ValueError(
                    f"{symbol} notional {normalized_qty * normalized_price:.8f} below minNotional {rules['min_notional']}"
                )

        return {
            "quantity": normalized_qty,
            "price": normalized_price,
        }

    def ping(self) -> bool:
        """GET /fapi/v1/ping - Test connectivity."""
        try:
            self._request("GET", "/fapi/v1/ping", signed=False, weight=1)
            return True
        except Exception:
            return False

    def get_server_time(self) -> int:
        """GET /fapi/v1/time - Server time in milliseconds."""
        data = self._request("GET", "/fapi/v1/time", signed=False, weight=1)
        return data["serverTime"]
