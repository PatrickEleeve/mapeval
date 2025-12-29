"""Unit tests for cache_manager.py."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

try:
    import pyarrow
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from cache_manager import KlineCache, SessionCache


class TestKlineCache:
    def test_cache_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = KlineCache(cache_dir=tmpdir)
            result = cache.get("BTCUSDT", "1m", 500)
            assert result is None

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = KlineCache(cache_dir=tmpdir)
            
            dates = pd.date_range("2024-01-01", periods=10, freq="1min")
            df = pd.DataFrame({
                "BTCUSDT_Close": [50000.0 + i * 100 for i in range(10)],
            }, index=dates)
            df.index.name = "Date"
            
            success = cache.put("BTCUSDT", "1m", 500, df)
            assert success is True
            
            result = cache.get("BTCUSDT", "1m", 500)
            assert result is not None
            assert len(result) == 10

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
    def test_different_params_different_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = KlineCache(cache_dir=tmpdir)
            
            dates1 = pd.date_range("2024-01-01", periods=5, freq="1min")
            df1 = pd.DataFrame({
                "BTCUSDT_Close": [50000.0] * 5,
            }, index=dates1)
            df1.index.name = "Date"
            
            dates2 = pd.date_range("2024-01-01", periods=10, freq="1min")
            df2 = pd.DataFrame({
                "BTCUSDT_Close": [60000.0] * 10,
            }, index=dates2)
            df2.index.name = "Date"
            
            cache.put("BTCUSDT", "1m", 500, df1)
            cache.put("BTCUSDT", "5m", 500, df2)
            
            result1 = cache.get("BTCUSDT", "1m", 500)
            result2 = cache.get("BTCUSDT", "5m", 500)
            
            assert len(result1) == 5
            assert len(result2) == 10

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
    def test_invalidate_single_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = KlineCache(cache_dir=tmpdir)
            
            dates = pd.date_range("2024-01-01", periods=5, freq="1min")
            df = pd.DataFrame({
                "Close": [50000.0] * 5,
            }, index=dates)
            df.index.name = "Date"
            
            cache.put("BTCUSDT", "1m", 500, df)
            cache.put("ETHUSDT", "1m", 500, df)
            
            removed = cache.invalidate("BTCUSDT")
            assert removed == 1
            
            assert cache.get("BTCUSDT", "1m", 500) is None
            assert cache.get("ETHUSDT", "1m", 500) is not None

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
    def test_invalidate_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = KlineCache(cache_dir=tmpdir)
            
            dates = pd.date_range("2024-01-01", periods=5, freq="1min")
            df = pd.DataFrame({
                "Close": [50000.0] * 5,
            }, index=dates)
            df.index.name = "Date"
            
            cache.put("BTCUSDT", "1m", 500, df)
            cache.put("ETHUSDT", "1m", 500, df)
            
            removed = cache.invalidate()
            assert removed == 2

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow not installed")
    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = KlineCache(cache_dir=tmpdir, max_age_hours=24)
            
            dates = pd.date_range("2024-01-01", periods=5, freq="1min")
            df = pd.DataFrame({
                "Close": [50000.0] * 5,
            }, index=dates)
            df.index.name = "Date"
            cache.put("BTCUSDT", "1m", 500, df)
            
            stats = cache.get_stats()
            
            assert stats["file_count"] == 1
            assert stats["entries"] == 1
            assert stats["max_age_hours"] == 24
            assert stats["total_size_mb"] >= 0


class TestSessionCache:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SessionCache(cache_dir=tmpdir)
            
            data = {
                "session_id": "test_123",
                "equity": 100000.0,
                "trades": [{"symbol": "BTCUSDT", "pnl": 500.0}],
            }
            
            path = cache.save_session("test_123", data)
            assert path.exists()
            
            loaded = cache.load_session("test_123")
            assert loaded is not None
            assert loaded["session_id"] == "test_123"
            assert loaded["equity"] == 100000.0

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SessionCache(cache_dir=tmpdir)
            result = cache.load_session("nonexistent")
            assert result is None

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SessionCache(cache_dir=tmpdir)
            
            cache.save_session("session_1", {"id": 1})
            cache.save_session("session_2", {"id": 2})
            cache.save_session("session_3", {"id": 3})
            
            sessions = cache.list_sessions()
            assert len(sessions) == 3
            assert "session_1" in sessions
            assert "session_2" in sessions
            assert "session_3" in sessions
