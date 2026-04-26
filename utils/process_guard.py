"""
utils/process_guard.py

Prevents duplicate instances of the trading engine via PID file + flock locking.
"""

from __future__ import annotations

import os
import sys
import atexit
import fcntl
import signal
from pathlib import Path
from typing import Optional

from config.settings import cfg
from utils.logger import log


class ProcessGuardError(Exception):
    """Raised when a duplicate instance is detected."""


class ProcessGuard:
    """
    Context manager that prevents multiple running instances.

    Uses fcntl.flock for atomic exclusion on Unix. Falls back to
    PID existence check. Cleans up PID file on normal exit.

    Usage:
        with ProcessGuard(cfg.DATA_DIR / "engine.pid"):
            # ... run main loop ...
    """

    def __init__(
        self,
        pid_path: str | Path,
        label: str = "engine",
        force: bool = False,
    ):
        self.pid_path = Path(pid_path)
        self.label = label
        self.force = force
        self._fp = None  # File handle for the PID file
        self._locked = False

    def __enter__(self) -> "ProcessGuard":
        self._acquire()
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def _acquire(self) -> None:
        """Try to acquire the exclusive lock."""
        # Ensure parent directory exists
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)

        # Open (or create) the PID file
        try:
            self._fp = open(self.pid_path, "a+")  # noqa: SIM115
        except OSError as e:
            log.warning(f"Could not open PID file {self.pid_path}: {e}")
            # Non-fatal — proceed without locking (e.g., read-only filesystem)
            return

        # Try flock
        try:
            fcntl.flock(self._fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._locked = True
        except (IOError, OSError):
            # Lock failed — check if PID still alive
            self._fp.seek(0)
            try:
                old_pid_str = self._fp.read().strip()
                if old_pid_str:
                    old_pid = int(old_pid_str)
                    try:
                        os.kill(old_pid, 0)  # Check if process exists
                        pidfile_path = str(self.pid_path)
                        msg = (
                            f"{self.label} already running (PID {old_pid}). "
                            f"Lock file: {pidfile_path}"
                        )
                        if self.force:
                            log.warning(f"{msg} --force: removing and re-acquiring")
                            # Force kill old process
                            os.kill(old_pid, signal.SIGTERM)
                            self._fp.close()
                            self.pid_path.unlink(missing_ok=True)
                            self._fp = open(self.pid_path, "a+")  # noqa: SIM115
                            fcntl.flock(self._fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            self._locked = True
                            os.getpid()
                            return
                        raise ProcessGuardError(msg)
                    except ProcessLookupError:
                        # Process dead, we can take over
                        pass
            except (ValueError, OSError):
                pass

            # Re-truncate and lock
            self._fp.seek(0)
            self._fp.truncate()
            try:
                fcntl.flock(self._fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._locked = True
            except (IOError, OSError):
                log.warning("flock unavailable — proceeding without exclusive lock")
                self._locked = False

        # Write our PID
        self._fp.seek(0)
        self._fp.truncate()
        self._fp.write(str(os.getpid()))
        self._fp.flush()

        # Register cleanup
        atexit.register(self.release)

        log.debug(f"ProcessGuard acquired PID {os.getpid()} → {self.pid_path}")

    def release(self) -> None:
        """Release the lock and remove PID file."""
        if self._fp is not None:
            try:
                fcntl.flock(self._fp, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None

        try:
            if self.pid_path.exists() and self.pid_path.read_text().strip() == str(os.getpid()):
                self.pid_path.unlink()
        except Exception:
            pass

        self._locked = False