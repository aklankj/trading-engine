"""
Structured logging with loguru.
Usage: from utils.logger import log
"""

import sys
from loguru import logger as log
from config.settings import cfg

# Remove default handler
log.remove()

# Console: colored, concise
log.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# File: full detail, daily rotation
log.add(
    cfg.LOG_DIR / "engine_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    compression="gz",
)

# Trade-specific log
log.add(
    cfg.LOG_DIR / "trades_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    level="INFO",
    filter=lambda record: "trade" in record["extra"],
    rotation="00:00",
    retention="90 days",
)

__all__ = ["log"]
