"""Exchange abstraction layer.

Defines a unified interface for interacting with cryptocurrency exchanges.
Currently implements Binance; extensible to Bybit, OKX, Hyperliquid, etc.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Ticker:
    symbol: str
    price: float
    timestamp: Optional[float] = None


@dataclass
class Kline:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class AccountInfo:
    total_balance: float
    available_balance: float
    total_unrealized_pnl: float
    total_margin: float


class Exchange(ABC):
    """Abstract interface for exchange operations."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get all available trading symbols."""
        ...

    @abstractmethod
    def get_ticker(self, symbol: str) -> Ticker:
        """Get latest ticker price for a symbol."""
        ...

    @abstractmethod
    def get_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        """Get latest ticker prices for multiple symbols."""
        ...

    @abstractmethod
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Kline]:
        """Get historical kline/candlestick data."""
        ...

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Get account balance and margin information."""
        ...

    @abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Place a new order."""
        ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an active order."""
        ...

    @abstractmethod
    def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        ...

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        ...

    @abstractmethod
    def change_leverage(self, symbol: str, leverage: int) -> None:
        """Change leverage for a symbol."""
        ...

    def start_websocket(self, symbols: List[str], on_price: Callable) -> None:
        """Start websocket streaming (optional, not all exchanges support)."""
        raise NotImplementedError

    def stop_websocket(self) -> None:
        """Stop websocket streaming."""
        raise NotImplementedError


class BinanceExchange(Exchange):
    """Binance USDT-M Futures exchange implementation.

    Wraps BinanceFuturesClient and binance_data_source for a unified interface.
    """

    def __init__(self, client=None, data_source=None, testnet: bool = True) -> None:
        self._client = client
        self._data_source = data_source
        self._testnet = testnet

    @property
    def name(self) -> str:
        return "binance_testnet" if self._testnet else "binance"

    def get_symbols(self) -> List[str]:
        if self._client is None:
            return []
        try:
            info = self._client.get_exchange_info()
            return [s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"]
        except Exception as exc:
            logger.error("Failed to get symbols: %s", exc)
            return []

    def get_ticker(self, symbol: str) -> Ticker:
        if self._client is not None:
            data = self._client.get_ticker_price(symbol)
            return Ticker(symbol=data["symbol"], price=float(data["price"]))
        if self._data_source is not None:
            from binance_data_source import get_ticker_price
            data = get_ticker_price(symbol)
            return Ticker(symbol=data["symbol"], price=float(data["price"]))
        raise RuntimeError("No client or data source configured")

    def get_tickers(self, symbols: List[str]) -> Dict[str, Ticker]:
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.get_ticker(sym)
            except Exception:
                pass
        return result

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Kline]:
        if self._data_source is not None:
            from binance_data_source import fetch_klines
            raw = fetch_klines(symbol, interval=interval, limit=limit)
            return [
                Kline(
                    timestamp=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                )
                for k in raw
            ]
        raise RuntimeError("No data source configured")

    def get_account(self) -> AccountInfo:
        if self._client is None:
            raise RuntimeError("Authenticated client required")
        info = self._client.get_account_info()
        return AccountInfo(
            total_balance=float(info.get("totalWalletBalance", 0)),
            available_balance=float(info.get("availableBalance", 0)),
            total_unrealized_pnl=float(info.get("totalUnrealizedProfit", 0)),
            total_margin=float(info.get("totalMarginBalance", 0)),
        )

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Authenticated client required")
        return self._client.place_order(
            symbol=symbol, side=side, order_type=order_type,
            quantity=quantity, price=price,
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        if self._client is None:
            return False
        try:
            self._client.cancel_order(symbol=symbol, order_id=int(order_id))
            return True
        except Exception:
            return False

    def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Authenticated client required")
        return self._client.get_order(symbol=symbol, order_id=int(order_id))

    def get_positions(self) -> List[Dict[str, Any]]:
        if self._client is None:
            return []
        return self._client.get_positions()

    def change_leverage(self, symbol: str, leverage: int) -> None:
        if self._client is None:
            raise RuntimeError("Authenticated client required")
        self._client.change_leverage(symbol, leverage)
