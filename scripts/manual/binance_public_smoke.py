"""
Simple tester for the public Binance REST API hosted at https://api1.binance.com.
The script probes a handful of unauthenticated endpoints and reports whether
they respond successfully alongside a short payload summary.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from urllib import error, request

BASE_URL = "https://api1.binance.com"

# Public, read-only endpoints that do not require an API key.
PUBLIC_ENDPOINTS: Sequence[Tuple[str, str]] = (
    ("Ping", "/api/v3/ping"),
    ("Server Time", "/api/v3/time"),
    ("Exchange Info (BTCUSDT)", "/api/v3/exchangeInfo?symbol=BTCUSDT"),
    ("Order Book Depth (BTCUSDT)", "/api/v3/depth?symbol=BTCUSDT&limit=5"),
    ("Ticker Price (BTCUSDT)", "/api/v3/ticker/price?symbol=BTCUSDT"),
)

USER_AGENT = "CodexBinanceAPITester/1.0"
DEFAULT_TIMEOUT = 10  # seconds


@dataclass
class EndpointResult:
    name: str
    url: str
    status: str
    http_status: int | None
    elapsed: float
    summary: str

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def summarize_payload(raw: bytes, content_type: str | None) -> str:
    """Produce a short, human-friendly description of the response body."""
    if not raw:
        return "empty body"

    if content_type and "application/json" in content_type:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # Fall back to showing the first bytes if parsing fails.
            return f"invalid JSON (first 120 bytes: {raw[:120]!r})"

        if isinstance(payload, dict):
            keys = ", ".join(list(payload.keys())[:5])
            if len(payload) > 5:
                keys += ", ..."
            return f"JSON object keys: {keys or '(none)'}"
        if isinstance(payload, list):
            preview = payload[:3]
            return f"JSON array length {len(payload)} preview {preview}"
        return f"JSON value type {type(payload).__name__}: {payload!r}"

    # For non-JSON content we only show a small snippet to avoid dumping too much.
    snippet = raw[:120].decode("utf-8", errors="replace")
    if len(raw) > 120:
        snippet += "..."
    return f"{content_type or 'unknown content-type'} payload: {snippet}"


def probe_endpoint(name: str, path: str, timeout: float = DEFAULT_TIMEOUT) -> EndpointResult:
    url = f"{BASE_URL}{path}"
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    began = time.perf_counter()

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - began
            body = resp.read()
            summary = summarize_payload(body, resp.headers.get("Content-Type"))
            return EndpointResult(
                name=name,
                url=url,
                status="OK",
                http_status=getattr(resp, "status", None),
                elapsed=elapsed,
                summary=summary,
            )
    except error.HTTPError as exc:
        elapsed = time.perf_counter() - began
        body = exc.read()
        summary = summarize_payload(body, exc.headers.get("Content-Type") if exc.headers else None)
        return EndpointResult(
            name=name,
            url=url,
            status=f"HTTP {exc.code}",
            http_status=exc.code,
            elapsed=elapsed,
            summary=summary,
        )
    except error.URLError as exc:
        elapsed = time.perf_counter() - began
        return EndpointResult(
            name=name,
            url=url,
            status="NETWORK ERROR",
            http_status=None,
            elapsed=elapsed,
            summary=str(exc.reason),
        )


def run_tests(endpoints: Sequence[Tuple[str, str]]) -> List[EndpointResult]:
    results: List[EndpointResult] = []
    for name, path in endpoints:
        results.append(probe_endpoint(name, path))
    return results


def print_report(results: Sequence[EndpointResult]) -> None:
    widest_name = max(len(result.name) for result in results)
    widest_status = max(len(result.status) for result in results)

    for result in results:
        line = (
            f"{result.name:<{widest_name}}  "
            f"{result.status:<{widest_status}}  "
            f"{result.elapsed:.3f}s  "
            f"{result.url}"
        )
        print(line)
        print(f"  ↳ {result.summary}")


def main() -> int:
    results = run_tests(PUBLIC_ENDPOINTS)
    print_report(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
