"""
utils.py — Shared utility functions for the NareshIT RAG Assistant.

Includes logging setup, path helpers, and common text utilities.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from config import settings


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(name: str = "nareshit_rag") -> logging.Logger:
    """Create and return a configured logger instance."""

    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: Path, extension: str = "*.pdf") -> list[Path]:
    """Return sorted list of files matching *extension* inside *directory*."""
    return sorted(directory.glob(extension))


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def safe_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested-dict safe accessor.  Returns *default* if any key is missing."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current