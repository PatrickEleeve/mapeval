"""Local data caching for historical market data."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


class KlineCache:
    """Cache historical kline data to disk for faster startup and offline usage."""
    
    def __init__(
        self,
        cache_dir: str | Path = ".cache/klines",
        max_age_hours: int = 24,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_hours * 3600
        self.metadata_file = self.cache_dir / "metadata.json"
        self._metadata: Dict[str, Any] = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception:
            pass
    
    def _cache_key(self, symbol: str, interval: str, lookback: int) -> str:
        key_str = f"{symbol}_{interval}_{lookback}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.parquet"
    
    def _is_fresh(self, cache_key: str) -> bool:
        meta = self._metadata.get(cache_key, {})
        cached_at = meta.get("cached_at")
        if not cached_at:
            return False
        
        try:
            cached_time = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age = (now - cached_time).total_seconds()
            return age < self.max_age_seconds
        except Exception:
            return False
    
    def get(
        self,
        symbol: str,
        interval: str,
        lookback: int,
    ) -> Optional[pd.DataFrame]:
        cache_key = self._cache_key(symbol, interval, lookback)
        cache_path = self._cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        if not self._is_fresh(cache_key):
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            if df.index.name == "Date" or "Date" in df.columns:
                if "Date" in df.columns:
                    df = df.set_index("Date")
            return df
        except Exception:
            return None
    
    def put(
        self,
        symbol: str,
        interval: str,
        lookback: int,
        data: pd.DataFrame,
    ) -> bool:
        cache_key = self._cache_key(symbol, interval, lookback)
        cache_path = self._cache_path(cache_key)
        
        try:
            df = data.copy()
            if df.index.name != "Date" and "Date" in df.columns:
                df = df.set_index("Date")
            
            df.to_parquet(cache_path, index=True)
            
            self._metadata[cache_key] = {
                "symbol": symbol,
                "interval": interval,
                "lookback": lookback,
                "rows": len(df),
                "cached_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            self._save_metadata()
            return True
        except Exception:
            return False
    
    def invalidate(self, symbol: Optional[str] = None) -> int:
        removed = 0
        keys_to_remove = []
        
        for key, meta in self._metadata.items():
            if symbol is None or meta.get("symbol") == symbol:
                cache_path = self._cache_path(key)
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        removed += 1
                    except Exception:
                        pass
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._metadata.pop(key, None)
        
        self._save_metadata()
        return removed
    
    def cleanup_stale(self) -> int:
        removed = 0
        keys_to_remove = []
        
        for key in list(self._metadata.keys()):
            if not self._is_fresh(key):
                cache_path = self._cache_path(key)
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                        removed += 1
                    except Exception:
                        pass
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._metadata.pop(key, None)
        
        self._save_metadata()
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        total_size = 0
        file_count = 0
        
        for cache_path in self.cache_dir.glob("*.parquet"):
            try:
                total_size += cache_path.stat().st_size
                file_count += 1
            except Exception:
                pass
        
        return {
            "cache_dir": str(self.cache_dir),
            "file_count": file_count,
            "total_size_mb": total_size / (1024 * 1024),
            "entries": len(self._metadata),
            "max_age_hours": self.max_age_seconds / 3600,
        }


class SessionCache:
    """Cache session results for analysis."""
    
    def __init__(self, cache_dir: str | Path = ".cache/sessions") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_session(self, session_id: str, data: Dict[str, Any]) -> Path:
        path = self.cache_dir / f"{session_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self.cache_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    
    def list_sessions(self) -> List[str]:
        return [p.stem for p in self.cache_dir.glob("*.json")]

