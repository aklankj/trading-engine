"""
utils/heartbeat.py

Simple heartbeat logger that appends structured JSON lines to
logs/engine_heartbeat.jsonl after each job run.

Usage:
    from utils.heartbeat import write_heartbeat
    write_heartbeat("morning_scan", duration_ms=1234, success=True)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import cfg


HEARTBEAT_PATH: Path = cfg.LOG_DIR / "engine_heartbeat.jsonl"


def write_heartbeat(
    job_name: str,
    duration_ms: float,
    success: bool,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Append a single heartbeat JSON line to the heartbeat log file.

    Parameters
    ----------
    job_name : str
        Human-readable job identifier (e.g. "morning_scan", "evening_recap").
    duration_ms : float
        Wall-clock duration of the job in milliseconds.
    success : bool
        Whether the job completed without unhandled exceptions.
    extra : dict | None
        Optional extra fields to include (e.g. {"trades": 5, "symbols": 10}).
    """
    entry = {
        "ts": datetime.now().isoformat(),
        "job": job_name,
        "duration_ms": round(duration_ms, 1),
        "success": success,
        "mode": cfg.TRADING_MODE,
    }
    if extra:
        entry.update(extra)

    try:
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(str(HEARTBEAT_PATH), "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass  # Non-critical — don't crash the engine