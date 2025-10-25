"""Session logging utilities for real-time trading experiments."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class SessionLogger:
    """Persist trading session artifacts to disk as JSON files."""

    log_dir: Path
    file_prefix: str = "session"
    history: list[Path] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save_session(
        self,
        *,
        run_args: Dict[str, Any],
        summary: Dict[str, Any],
        start_time: str,
        end_time: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Path:
        """Write a structured record for a completed trading session."""
        if end_time is None:
            end_time = _utcnow_iso()

        llm_provider = run_args.get("llm_provider") if isinstance(run_args, dict) else None

        summary_copy: Dict[str, Any] = copy.deepcopy(summary)
        decision_log = summary_copy.get("decision_log", [])
        llm_reasoning = [
            entry.get("reasoning")
            for entry in decision_log
            if isinstance(entry, dict) and isinstance(entry.get("reasoning"), str) and entry.get("reasoning")
        ]
        summary_copy["llm_reasoning"] = llm_reasoning

        record: Dict[str, Any] = {
            "metadata": {
                "session_id": self._generate_session_id(start_time, llm_provider),
                "started_at": start_time,
                "ended_at": end_time,
                "llm_provider": llm_provider,
                "notes": notes,
            },
            "parameters": run_args,
            "summary": summary_copy,
        }

        file_path = self.log_dir / f"{record['metadata']['session_id']}.json"
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False, default=self._json_default)
        self.history.append(file_path)
        return file_path

    def _generate_session_id(self, start_time: str, suffix: str | None = None) -> str:
        safe_ts = start_time.replace(":", "").replace("-", "")
        safe_ts = safe_ts.replace("T", "_").replace("Z", "")
        pid = os.getpid()
        if suffix:
            suffix_clean = suffix.lower().replace(" ", "-")
            return f"{self.file_prefix}_{suffix_clean}_{safe_ts}_{pid}"
        return f"{self.file_prefix}_{safe_ts}_{pid}"

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.isoformat().replace("+00:00", "Z")
        iso = getattr(obj, "isoformat", None)
        if callable(iso):
            try:
                return iso()
            except Exception:
                pass
        return str(obj)
