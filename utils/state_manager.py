"""
utils/state_manager.py

Persistent JSON state validation and auto-repair.

Manages trade logs, portfolio state, regime state, and watchlist files.
On corruption: tries backup, falls back to defaults, logs all recovery actions.

Usage:
    from utils.state_manager import StateManager

    manager = StateManager(
        file_path=cfg.DATA_DIR / "paper_portfolio.json",
        version=1,
        required_keys=["positions", "cash", "version"],
        defaults={"positions": [], "cash": 100000, "version": 1},
    )
    data = manager.read()
    data["cash"] = 95000
    manager.write(data)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from utils.logger import log
from utils.exceptions import StateCorruptionError


class StateManager:
    """
    Manages a single JSON state file with validation, backup, and auto-repair.

    On read:
        1. Try reading the JSON file
        2. Validate version and required keys
        3. If corrupted → try .bak backup
        4. If backup also bad → log warning, return defaults
        5. If repaired from backup, mark so caller can re-save

    On write:
        1. Write to .tmp file first
        2. Rename existing file to .bak
        3. Rename .tmp to actual
    """

    def __init__(
        self,
        file_path: str | Path,
        version: int = 1,
        required_keys: list[str] | None = None,
        defaults: dict[str, Any] | None = None,
        backup: bool = True,
    ):
        self.file_path = Path(file_path)
        self.backup_path = self.file_path.with_name(self.file_path.name + ".bak")
        self.tmp_path = self.file_path.with_name(self.file_path.name + ".tmp")
        self.version = version
        self.required_keys = required_keys or []
        self.defaults = defaults or {}
        self.backup_enabled = backup
        self.repaired = False  # Set True if auto-repair occurred

    def read(self) -> dict[str, Any]:
        """Read and validate the state file. Auto-repairs if needed."""
        self.repaired = False
        data = self._try_read(self.file_path)
        if data is not None:
            if self._validate(data):
                return data
            log.warning(f"Validation failed for {self.file_path}, trying backup...")

        # Try backup
        if self.backup_enabled and self.backup_path.exists():
            data = self._try_read(self.backup_path)
            if data is not None and self._validate(data):
                log.warning(f"Restored {self.file_path} from backup {self.backup_path}")
                self.repaired = True
                # Re-write to restore original
                self.write(data)
                return data

        # Fallback to defaults — write them so next read succeeds
        msg = f"Corrupted state file: {self.file_path}. Using defaults."
        log.warning(msg)
        self.repaired = True
        was_repaired = self.repaired
        self.write(dict(self.defaults))
        self.repaired = was_repaired  # write() resets repaired flag
        return dict(self.defaults)

    def write(self, data: dict[str, Any]) -> bool:
        """
        Write data to file atomically via .tmp.

        Returns True on success, False on failure.
        """
        # Inject version
        data["version"] = self.version

        try:
            # Write to temp
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(self.tmp_path), "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
            # Backup old file
            if self.backup_enabled and self.file_path.exists():
                shutil.copy2(str(self.file_path), str(self.backup_path))

            # Atomic rename
            self.tmp_path.rename(self.file_path)
            self.repaired = False
            return True

        except (OSError, json.JSONDecodeError) as e:
            log.error(f"Failed to write state file {self.file_path}: {e}")
            # Clean up temp file
            try:
                self.tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def _try_read(self, path: Path) -> dict[str, Any] | None:
        """Try to read and parse a JSON file. Returns None on failure."""
        if not path.exists():
            return None
        try:
            with open(str(path)) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _validate(self, data: Any) -> bool:
        """Check that data is a dict with required keys and correct version."""
        if not isinstance(data, dict):
            return False
        for key in self.required_keys:
            if key not in data:
                return False
        if self.version > 0 and data.get("version", 0) > self.version:
            # Future version — accept but log
            log.debug(f"State file {self.file_path} has newer version {data.get('version')}")
        return True