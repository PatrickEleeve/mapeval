from __future__ import annotations

import json
import threading
import time
from typing import Dict, List, Optional


class BinanceSpotWS:
    def __init__(self, symbols: List[str], base_ws: str = "wss://stream.binance.com:9443") -> None:
        self.symbols = [s.upper() for s in symbols]
        self.base_ws = base_ws.rstrip("/")
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._prices: Dict[str, float] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="binance-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_latest_prices(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._prices)

    def _run(self) -> None:
        try:
            import websocket  # type: ignore
        except Exception:
            return

        stream_names = [f"{s.lower()}@miniTicker" for s in self.symbols]
        url = f"{self.base_ws}/stream?streams={'/'.join(stream_names)}"

        def on_message(_ws, message: str) -> None:  # noqa: N803
            try:
                obj = json.loads(message)
            except Exception:
                return
            data = obj.get("data", obj)
            if not isinstance(data, dict):
                return
            s = data.get("s")
            c = data.get("c")
            if not s or c is None:
                return
            try:
                price = float(c)
            except Exception:
                return
            symbol = str(s).upper()
            if symbol not in self.symbols:
                return
            with self._lock:
                self._prices[symbol] = price

        def on_open(_ws) -> None:  # noqa: N803
            self._connected.set()

        def on_error(_ws, _err) -> None:  # noqa: N803
            pass

        def on_close(_ws, _code, _msg) -> None:  # noqa: N803
            self._connected.clear()

        while not self._stop.is_set():
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                )
                ws.run_forever(ping_interval=20, ping_timeout=10, origin=None)
            except Exception:
                pass
            if self._stop.is_set():
                break
            time.sleep(1.0)


