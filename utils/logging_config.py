"""Centralized logging configuration for the fatigue driving detection project.

Usage::

    from utils.logging_config import setup_logging

    setup_logging()           # INFO level, console output
    setup_logging("DEBUG")    # DEBUG level for verbose output
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a standard format.

    Sets up a ``StreamHandler`` writing to *stderr* with a consistent
    timestamp / module / level format.  Calling this function more than
    once is safe -- it will not add duplicate handlers.

    Args:
        level: Logging level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"``).  Defaults to ``"INFO"``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()

    if root.handlers:
        # Already configured (e.g. by a prior call) -- just update level.
        root.setLevel(numeric_level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.setLevel(numeric_level)
    root.addHandler(handler)
