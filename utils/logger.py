"""Logging configuration for AI Story Generator Pro.

Sets up two file handlers (``generation.log`` at DEBUG level and
``errors.log`` at ERROR level) with rotation, plus a console handler
at INFO level.  A ``QueueHandler`` is available for safe cross-thread
log forwarding to the GUI.

Typical usage::

    from utils.logger import setup_logging
    log_queue = setup_logging(log_dir="logs", level="DEBUG")
    # Pass log_queue to the GUI for real-time log display.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
from pathlib import Path


def setup_logging(
    log_dir: str | Path = "logs",
    level: str = "DEBUG",
    max_files: int = 10,
    max_file_size_mb: int = 10,
) -> queue.Queue[logging.LogRecord]:
    """Configure application-wide logging.

    Creates:
    - ``generation.log`` — verbose (level from parameter, default DEBUG)
    - ``errors.log``     — ERROR and above only
    - Console handler    — INFO and above
    - QueueHandler       — all records, for GUI consumption

    Args:
        log_dir: Directory for log files (created if missing).
        level: Root logger level (DEBUG, INFO, WARNING, ERROR).
        max_files: Maximum rotated backup files per handler.
        max_file_size_mb: Maximum size of a single log file in megabytes.

    Returns:
        A ``queue.Queue`` of ``LogRecord`` objects that the GUI can read
        from to display log messages in real-time.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    # Clear any existing handlers to allow re-initialisation
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    max_bytes = max_file_size_mb * 1024 * 1024
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── generation.log (verbose) ────────────────────────────────────
    gen_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "generation.log",
        maxBytes=max_bytes,
        backupCount=max_files,
        encoding="utf-8",
    )
    gen_handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    gen_handler.setFormatter(formatter)
    root_logger.addHandler(gen_handler)

    # ── errors.log (errors only) ────────────────────────────────────
    err_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "errors.log",
        maxBytes=max_bytes,
        backupCount=max_files,
        encoding="utf-8",
    )
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(formatter)
    root_logger.addHandler(err_handler)

    # ── Console (INFO) ──────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # ── Queue handler for GUI integration ───────────────────────────
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue()
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)

    root_logger.info("Logging initialised — dir=%s, level=%s", log_path, level)
    return log_queue
