"""File I/O utilities for AI Story Generator Pro.

Provides safe, UTF-8-only helpers for reading and writing text and JSON
files.  Writes are **atomic** (write to a temp file in the same
directory, then ``os.rename`` to the target) to prevent corruption on
crashes.

All functions in this module should be used instead of raw ``open()``
calls throughout the project.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path to ensure.

    Returns:
        The resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    logger.debug("Directory ensured: %s", p)
    return p


def read_file(path: str | Path) -> str:
    """Read an entire text file as a UTF-8 string.

    Args:
        path: Path to the file.

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: On other I/O errors.
    """
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    logger.debug("Read file: %s (%d chars)", p, len(content))
    return content


def write_file(path: str | Path, content: str) -> Path:
    """Write a UTF-8 text file atomically.

    Writes to a temporary file in the same directory, then renames it
    to the target path.  This prevents partial/corrupt files if the
    process is killed mid-write.

    Args:
        path: Target file path.
        content: Text content to write.

    Returns:
        The resolved ``Path`` of the written file.

    Raises:
        OSError: On I/O errors.
    """
    p = Path(path)
    ensure_dir(p.parent)

    fd, tmp_path = tempfile.mkstemp(
        dir=p.parent, prefix=".tmp_", suffix=p.suffix
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, p)
        logger.debug("Wrote file atomically: %s (%d chars)", p, len(content))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return p


def read_json(path: str | Path) -> dict[str, Any]:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    text = read_file(path)
    data = json.loads(text)
    logger.debug("Read JSON: %s", path)
    return data


def write_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    """Write data as a JSON file atomically.

    Args:
        path: Target file path.
        data: Data to serialise (must be JSON-serializable).
        indent: JSON indentation level.

    Returns:
        The resolved ``Path`` of the written file.

    Raises:
        TypeError: If *data* is not JSON-serializable.
        OSError: On I/O errors.
    """
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    return write_file(path, content)


def list_files(directory: str | Path, extension: str = "") -> list[Path]:
    """List files in a directory, optionally filtered by extension.

    Args:
        directory: Directory to scan.
        extension: File extension filter including the dot (e.g., ``".txt"``).
            Empty string means no filter (return all files).

    Returns:
        Sorted list of matching ``Path`` objects. Returns an empty list
        if the directory does not exist.
    """
    d = Path(directory)
    if not d.is_dir():
        logger.debug("Directory does not exist: %s", d)
        return []

    if extension:
        files = sorted(d.glob(f"*{extension}"))
    else:
        files = sorted(f for f in d.iterdir() if f.is_file())

    logger.debug("Listed %d file(s) in %s (ext=%r)", len(files), d, extension)
    return files
