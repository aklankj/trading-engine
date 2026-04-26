"""Utility helpers used across the engine."""

import json
import csv
from pathlib import Path
from datetime import datetime, date
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON file, return default if missing or corrupt."""
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return default if default is not None else {}


def save_json(path: Path, data: Any) -> None:
    """Atomically save JSON (write to tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.rename(path)


def append_csv(path: Path, row: dict) -> None:
    """Append a row to a CSV file, creating headers if new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def now_ist() -> datetime:
    """Current time in IST (UTC+5:30)."""
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist)


def today_str() -> str:
    """Today's date as YYYY-MM-DD string."""
    return now_ist().strftime("%Y-%m-%d")


def is_market_hours() -> bool:
    """Check if current IST time is within NSE trading hours (9:15 - 15:30)."""
    t = now_ist().time()
    from datetime import time as dt_time
    return dt_time(9, 15) <= t <= dt_time(15, 30)


def is_weekday() -> bool:
    """Check if today is a weekday (Mon-Fri)."""
    return now_ist().weekday() < 5


def fmt_inr(amount: float) -> str:
    """Format amount in Indian Rupee style."""
    if abs(amount) >= 1e7:
        return f"₹{amount/1e7:.2f}Cr"
    if abs(amount) >= 1e5:
        return f"₹{amount/1e5:.2f}L"
    if abs(amount) >= 1e3:
        return f"₹{amount/1e3:.1f}K"
    return f"₹{amount:.0f}"


def fmt_pct(value: float) -> str:
    """Format as percentage with sign."""
    return f"{'+' if value >= 0 else ''}{value * 100:.2f}%"
