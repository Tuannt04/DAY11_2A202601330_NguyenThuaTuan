"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id (falls back to user_id)."""
        key = request_id or user_id
        self._open[key] = time.monotonic()
        self.logs.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "direction": "input",
                "text": text,
                "blocked": None,
                "layer": None,
                "latency_ms": None,
                "timestamp": utc_now_iso(),
            }
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision and latency; append to self.logs."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        latency_ms = (time.monotonic() - started) * 1000 if started is not None else None
        self.logs.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "direction": "output",
                "text": text,
                "blocked": blocked,
                "layer": layer,
                "latency_ms": latency_ms,
                "timestamp": utc_now_iso(),
            }
        )

    def find_by_request_id(self, request_id: str) -> list[dict]:
        """Return the logged snapshot (input + output rows) for one request_id —
        the starting point for an incident-response replay."""
        return [row for row in self.logs if row.get("request_id") == request_id]

    def export_json(self, filepath: str = "outputs/audit_log.json"):
        """Write logs to disk (JSON array)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
