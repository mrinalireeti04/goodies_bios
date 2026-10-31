"""Logging configuration for FORGE TRACE."""

import logging
import sys

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for FORGE TRACE.

    Safe to call multiple times; subsequent calls are no-ops unless *level*
    changes.
    """
    global _configured
    if _configured:
        logger = logging.getLogger("forge_trace")
        if logger.level != level:
            logger.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    logger = logging.getLogger("forge_trace")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``forge_trace`` namespace."""
    setup_logging()
    return logging.getLogger(f"forge_trace.{name}")
