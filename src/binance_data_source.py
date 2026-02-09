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
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import requests
except Exception:  # pragma: no cover - requests may not be installed
    requests = None  # type: ignore

from urllib.parse import urlencode

from rate_limiter import RateLimiter

DEFAULT_BASE = "https://api1.binance.com"
FALLBACK_BASES: Sequence[str] = (
    DEFAULT_BASE,
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api.binance.com",
)

# Module-level rate limiter instance, shared across all calls
_rate_limiter: Optional[RateLimiter] = None


def set_rate_limiter(limiter: RateLimiter) -> None:
    """Set a shared rate limiter for all Binance REST API calls."""
    global _rate_limiter
    _rate_limiter = limiter


def get_rate_limiter() -> Optional[RateLimiter]:
    """Return the current shared rate limiter, if set."""
    return _rate_limiter


def _normalize_base_urls(base_url: str | Iterable[str] | None) -> List[str]:
    if base_url is None:
        sources: Iterable[str] = FALLBACK_BASES
    elif isinstance(base_url, (list, tuple, set)):
        sources = base_url
    else:
        sources = [base_url]
    normalized: List[str] = []
    for raw in sources:
        text = str(raw).strip()
        if not text:
            continue
        normalized.append(text.rstrip("/"))
    return normalized or [DEFAULT_BASE.rstrip("/")]


def _http_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10, weight: int = 1) -> Any:
    limiter = _rate_limiter
    if limiter is not None:
        limiter.acquire(weight)

    if requests is not None:
        resp = requests.get(url, params=params, timeout=timeout)
        # Update rate limiter from response headers
        if limiter is not None:
            limiter.update_from_headers(dict(resp.headers))
        if resp.status_code == 429:
            if limiter is not None:
                wait = limiter.on_429()
                time.sleep(wait)
            raise RuntimeError(f"Rate limited (429) from {url}")
        if limiter is not None:
            limiter.on_success()
        resp.raise_for_status()
        return resp.json()

    # fallback using urllib
    from urllib.request import urlopen, Request

    if params:
        url = url + "?" + urlencode(params)
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as f:
        if limiter is not None:
            limiter.on_success()
        body = f.read()
        return json.loads(body.decode())


def check_api(base_url: str = DEFAULT_BASE) -> Dict[str, Any]:
    """Quick health/time check against Binance endpoints.

    Returns a dict with keys: reachable (bool), server_time (int|None), ping_ms (float|None), error (str|None)
    """
    url_candidates = _normalize_base_urls(base_url)
    last_error = None
    for root in url_candidates:
        ping_url = root + "/api/v3/ping"
        time_url = root + "/api/v3/time"
        try:
            t0 = time.perf_counter()
            _http_get(ping_url, params=None)
            t1 = time.perf_counter()
            ping_ms = (t1 - t0) * 1000.0
            time_json = _http_get(time_url)
            server_time = int(time_json.get("serverTime"))
            # Only return success after both ping and time checks pass
            return {
                "reachable": True,
                "server_time": server_time,
                "ping_ms": ping_ms,
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            continue
    # All endpoints failed - return failure state with last error
    return {
        "reachable": False,
        "server_time": None,
        "ping_ms": None,
        "error": str(last_error) if last_error is not None else None,
    }


def fetch_klines(
    symbol: str,
    interval: str = "1d",
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    limit: int = 1000,
    base_url: str | Iterable[str] = DEFAULT_BASE,
) -> List[List[Any]]:
    """Fetch klines (candles) from Binance. Returns raw kline arrays.

    Parameters follow Binance API naming. start_str/end_str can be ISO dates or timestamps in ms.
    """
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_str is not None:
        params["startTime"] = start_str
    if end_str is not None:
        params["endTime"] = end_str

    errors: List[Exception] = []
    for root in _normalize_base_urls(base_url):
        klines_url = root + "/api/v3/klines"
        try:
            return _http_get(klines_url, params=params)
        except Exception as exc:
            errors.append(exc)
            continue
    if errors:
        raise errors[-1]
    raise RuntimeError("No Binance REST endpoints available for klines request.")


def get_ticker_price(
    symbol: Optional[str] = None,
    symbols: Optional[list[str]] = None,
    base_url: str | Iterable[str] = DEFAULT_BASE,
) -> Any:
    """Call /api/v3/ticker/price to get recent price(s).

    - If `symbol` is provided, returns a single dict: {"symbol":..., "price":...}
    - If `symbols` is provided, sends the JSON-encoded list as the `symbols` parameter and
      returns a list of dicts.
    - If neither provided, returns all tickers (may be large).
    Note: `symbol` and `symbols` must not be used together as per Binance API.
    """
    if symbol is not None and symbols is not None:
        raise ValueError("Provide only one of 'symbol' or 'symbols'")

    params: Dict[str, Any] = {}
    if symbol is not None:
        params["symbol"] = symbol
    elif symbols is not None:
        params["symbols"] = json.dumps(symbols, separators=(",", ":"))

    errors: List[Exception] = []
    for root in _normalize_base_urls(base_url):
        url = root + "/api/v3/ticker/price"
        try:
            return _http_get(url, params=params or None)
        except Exception as exc:
            errors.append(exc)
            continue

    if errors:
        raise errors[-1]
    raise RuntimeError("No Binance REST endpoints available for ticker price request.")


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
