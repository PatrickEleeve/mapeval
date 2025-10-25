"""Simple Binance REST API lightweight client for fetching klines and checking availability.

This module avoids extra dependencies and uses `requests` if available, otherwise falls back to
`urllib.request`. It exposes:
- check_api(base_url) -> dict with status and time
- fetch_klines(symbol, interval, start_str, end_str, limit, base_url) -> list of kline rows

Note: Binance rate limits apply. This is a minimal implementation intended for testing and
small backtests only.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover - requests may not be installed
    requests = None  # type: ignore

from urllib.parse import urlencode


DEFAULT_BASE = "https://api1.binance.com"


def _http_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Any:
    if requests is not None:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # fallback using urllib
    from urllib.request import urlopen, Request

    if params:
        url = url + "?" + urlencode(params)
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as f:
        body = f.read()
        return json.loads(body.decode())


def check_api(base_url: str = DEFAULT_BASE) -> Dict[str, Any]:
    """Quick health/time check against Binance endpoints.

    Returns a dict with keys: reachable (bool), server_time (int|None), ping_ms (float|None), error (str|None)
    """
    out = {"reachable": False, "server_time": None, "ping_ms": None, "error": None}
    ping_url = base_url.rstrip("/") + "/api/v3/ping"
    time_url = base_url.rstrip("/") + "/api/v3/time"
    try:
        t0 = time.perf_counter()
        _http_get(ping_url, params=None)
        t1 = time.perf_counter()
        out["ping_ms"] = (t1 - t0) * 1000.0
        time_json = _http_get(time_url)
        out["server_time"] = int(time_json.get("serverTime"))
        out["reachable"] = True
    except Exception as exc:  # pragma: no cover - network dependent
        out["error"] = str(exc)
    return out


def fetch_klines(
    symbol: str,
    interval: str = "1d",
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    limit: int = 1000,
    base_url: str = DEFAULT_BASE,
) -> List[List[Any]]:
    """Fetch klines (candles) from Binance. Returns raw kline arrays.

    Parameters follow Binance API naming. start_str/end_str can be ISO dates or timestamps in ms.
    """
    klines_url = base_url.rstrip("/") + "/api/v3/klines"
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_str is not None:
        params["startTime"] = start_str
    if end_str is not None:
        params["endTime"] = end_str
    data = _http_get(klines_url, params=params)
    return data


def get_ticker_price(
    symbol: Optional[str] = None, symbols: Optional[list[str]] = None, base_url: str = DEFAULT_BASE
) -> Any:
    """Call /api/v3/ticker/price to get recent price(s).

    - If `symbol` is provided, returns a single dict: {"symbol":..., "price":...}
    - If `symbols` is provided, sends the JSON-encoded list as the `symbols` parameter and
      returns a list of dicts.
    - If neither provided, returns all tickers (may be large).
    Note: `symbol` and `symbols` must not be used together as per Binance API.
    """
    url = base_url.rstrip("/") + "/api/v3/ticker/price"
    params: Dict[str, Any] = {}
    if symbol is not None and symbols is not None:
        raise ValueError("Provide only one of 'symbol' or 'symbols'")
    if symbol is not None:
        params["symbol"] = symbol
        return _http_get(url, params=params)
    if symbols is not None:
        # Binance expects a JSON array string for the `symbols` parameter.
        # Use compact separators (no spaces) to avoid `400 Bad Request` caused by
        # space characters after commas when the list is URL-encoded.
        params["symbols"] = json.dumps(symbols, separators=(",", ":"))
        return _http_get(url, params=params)
    # no parameters -> returns all
    return _http_get(url, params=None)


def get_futures_premium_index(
    symbol: Optional[str] = None,
    *,
    base_url: str = "https://fapi.binance.com",
) -> Any:
    """Fetch premium index info for USDT-margined futures.

    - If `symbol` is provided, returns a single dict for that symbol
    - If omitted, returns a list of dicts for all symbols
    """
    url = base_url.rstrip("/") + "/fapi/v1/premiumIndex"
    params: Dict[str, Any] = {}
    if symbol is not None:
        params["symbol"] = symbol
        return _http_get(url, params=params)
    return _http_get(url, params=None)
