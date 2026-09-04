"""Input validation for AI Story Generator Pro.

``InputValidator`` validates user-provided input files before
processing begins.  It handles:

- **Topics files**: reads, strips whitespace, removes empty lines,
  deduplicates (with warnings), and validates UTF-8 encoding.
  Lines starting with ``OK `` are treated as already-completed
  topics and are skipped automatically.
- **Adaptation folders**: lists ``.txt`` files, detects the language
  of each, and validates readability.

Results are returned as pydantic models with a clean list of topics
or files plus a warnings list.

Typical usage::

    iv = InputValidator()
    validated = iv.validate_topics_file(Path("themes.txt"))
    # validated.topics   -> clean, deduplicated list
    # validated.warnings -> any warnings generated

    folder = iv.validate_adaptation_folder(Path("existing_texts/"))
    for fi in folder.files:
        # fi.path, fi.detected_language  -> per-file results
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field

from core.exceptions import ConfigError
from core.language_detector import LanguageDetector, LanguageDetectionResult
from utils.file_handler import list_files, read_file

logger = logging.getLogger(__name__)

# Maximum number of topics allowed in a single file.
_MAX_TOPICS: int = 500

# Minimum topic length (characters) after stripping.
_MIN_TOPIC_LENGTH: int = 3

# Prefix that marks a topic as already completed in the topics file.
DONE_PREFIX: str = "OK "


# ── Result models ────────────────────────────────────────────────────


class ValidatedTopics(BaseModel):
    """Result of validating a topics file.

    Attributes:
        topics: Clean, deduplicated list of topic strings.
        original_count: Number of lines read before cleaning.
        removed_empty: Number of empty lines removed.
        removed_duplicates: Number of duplicate lines removed.
        skipped_done: Number of already-completed topics skipped.
        warnings: Human-readable warning messages.
    """

    topics: list[str] = Field(default_factory=list, description="Clean topic list.")
    original_count: int = Field(default=0, description="Raw line count before cleaning.")
    removed_empty: int = Field(default=0, description="Empty lines removed.")
    removed_duplicates: int = Field(default=0, description="Duplicate lines removed.")
    skipped_done: int = Field(default=0, description="Already-completed topics skipped.")
    warnings: list[str] = Field(default_factory=list, description="Warnings generated.")


class FileInfo(BaseModel):
    """Information about a single validated file.

    Attributes:
        path: Absolute path to the file.
        filename: File name (stem + extension).
        detected_language: Two-letter language code detected.
        confidence: Detection confidence (0.0–1.0).
        is_supported: Whether the detected language is supported.
        char_count: Number of characters in the file.
        error: Error message if the file could not be read.
    """

    path: str = Field(description="Absolute file path.")
    filename: str = Field(description="File name.")
    detected_language: str = Field(default="", description="Detected language code.")
    confidence: float = Field(default=0.0, description="Detection confidence.")
    is_supported: bool = Field(default=False, description="Language is supported.")
    char_count: int = Field(default=0, ge=0, description="Character count.")
    error: str = Field(default="", description="Error if file unreadable.")


class ValidatedFolder(BaseModel):
    """Result of validating an adaptation source folder.

    Attributes:
        files: List of validated file information objects.
        total_files: Total ``.txt`` files found.
        readable: Number of files successfully read.
        unreadable: Number of files that could not be read.
        warnings: Human-readable warning messages.
    """

    files: list[FileInfo] = Field(default_factory=list, description="Validated files.")
    total_files: int = Field(default=0, description="Total .txt files found.")
    readable: int = Field(default=0, description="Files that could be read.")
    unreadable: int = Field(default=0, description="Files that could not be read.")
    warnings: list[str] = Field(default_factory=list, description="Warnings generated.")


# ── Validator ────────────────────────────────────────────────────────


class InputValidator:
    """Validates topics files and adaptation folders.

    Args:
        language_detector: Optional ``LanguageDetector`` instance.
            If not provided, one is created with default settings.
    """

    def __init__(
        self, language_detector: LanguageDetector | None = None
    ) -> None:
        self._detector = language_detector or LanguageDetector()
        logger.debug("InputValidator initialised")

    def validate_topics_file(self, path: str | Path) -> ValidatedTopics:
        """Validate a topics file.

        Reads the file, strips each line, removes empty lines and
        duplicates, skips lines starting with ``OK `` (already
        completed), and validates UTF-8 encoding.

        Args:
            path: Path to the topics file (one topic per line).

        Returns:
            A ``ValidatedTopics`` with the clean topic list and any
            warnings.

        Raises:
            ConfigError: If the file does not exist, is not a file, or
                cannot be decoded as UTF-8.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise ConfigError(f"Topics file not found: {file_path}")
        if not file_path.is_file():
            raise ConfigError(f"Topics path is not a file: {file_path}")

        # Read with explicit UTF-8 — raise on encoding errors.
        try:
            content = read_file(file_path)
        except UnicodeDecodeError as exc:
            raise ConfigError(
                f"Topics file is not valid UTF-8: {file_path} — {exc}"
            ) from exc
        except OSError as exc:
            raise ConfigError(
                f"Cannot read topics file: {file_path} — {exc}"
            ) from exc

        raw_lines = content.splitlines()
        original_count = len(raw_lines)

        warnings: list[str] = []

        # Strip whitespace.
        stripped = [line.strip() for line in raw_lines]

        # Remove empty lines.
        non_empty = [line for line in stripped if line]
        removed_empty = original_count - len(non_empty)
        if removed_empty > 0:
            warnings.append(
                f"Removed {removed_empty} empty line(s) from topics file."
            )

        # Skip already-completed topics (lines starting with "OK ").
        pending: list[str] = []
        skipped_done = 0
        for line in non_empty:
            if line.startswith(DONE_PREFIX):
                skipped_done += 1
            else:
                pending.append(line)

        if skipped_done > 0:
            warnings.append(
                f"Skipped {skipped_done} already-completed topic(s) "
                f"(marked with '{DONE_PREFIX}' prefix)."
            )
            logger.info(
                "InputValidator: skipped %d completed topics in %s",
                skipped_done,
                file_path,
            )

        # Filter out topics that are too short.
        valid_length: list[str] = []
        short_count = 0
        for line in pending:
            if len(line) < _MIN_TOPIC_LENGTH:
                short_count += 1
            else:
                valid_length.append(line)

        if short_count > 0:
            warnings.append(
                f"Removed {short_count} topic(s) shorter than "
                f"{_MIN_TOPIC_LENGTH} characters."
            )

        # Remove duplicates (preserve order).
        seen: set[str] = set()
        unique: list[str] = []
        duplicate_count = 0
        for topic in valid_length:
            normalised = topic.lower()
            if normalised in seen:
                duplicate_count += 1
            else:
                seen.add(normalised)
                unique.append(topic)

        if duplicate_count > 0:
            warnings.append(
                f"Removed {duplicate_count} duplicate topic(s)."
            )
            logger.warning(
                "InputValidator: removed %d duplicate topics from %s",
                duplicate_count,
                file_path,
            )

        # Check maximum count.
        if len(unique) > _MAX_TOPICS:
            warnings.append(
                f"Topics file contains {len(unique)} topics — "
                f"only the first {_MAX_TOPICS} will be used."
            )
            unique = unique[:_MAX_TOPICS]

        if not unique:
            if skipped_done > 0:
                warnings.append(
                    "All topics in the file are already completed."
                )
            else:
                warnings.append("No valid topics found in the file.")

        result = ValidatedTopics(
            topics=unique,
            original_count=original_count,
            removed_empty=removed_empty,
            removed_duplicates=duplicate_count,
            skipped_done=skipped_done,
            warnings=warnings,
        )

        logger.info(
            "InputValidator: validated topics file %s — "
            "%d raw lines → %d valid topics, %d done, %d warning(s)",
            file_path,
            original_count,
            len(unique),
            skipped_done,
            len(warnings),
        )

        return result

    def validate_adaptation_folder(self, path: str | Path) -> ValidatedFolder:
        """Validate an adaptation source folder.

        Lists all ``.txt`` files, reads each one, detects its language,
        and reports any issues.

        Args:
            path: Path to the folder containing ``.txt`` files.

        Returns:
            A ``ValidatedFolder`` with file information and warnings.

        Raises:
            ConfigError: If the path does not exist or is not a
                directory.
        """
        folder_path = Path(path)

        if not folder_path.exists():
            raise ConfigError(f"Adaptation folder not found: {folder_path}")
        if not folder_path.is_dir():
            raise ConfigError(
                f"Adaptation path is not a directory: {folder_path}"
            )

        txt_files = list_files(folder_path, extension=".txt")

        if not txt_files:
            raise ConfigError(
                f"No .txt files found in adaptation folder: {folder_path}"
            )

        warnings: list[str] = []
        file_infos: list[FileInfo] = []
        readable_count = 0
        unreadable_count = 0

        for file_path in txt_files:
            file_info = self._validate_single_file(file_path)
            file_infos.append(file_info)

            if file_info.error:
                unreadable_count += 1
                warnings.append(
                    f"Cannot read file '{file_info.filename}': {file_info.error}"
                )
            else:
                readable_count += 1

                if not file_info.is_supported and file_info.detected_language:
                    warnings.append(
                        f"File '{file_info.filename}' detected as "
                        f"'{file_info.detected_language}' which is not a "
                        f"supported language."
                    )

                if file_info.confidence < 0.5 and file_info.detected_language:
                    warnings.append(
                        f"Low confidence ({file_info.confidence:.2f}) for "
                        f"language detection of '{file_info.filename}'."
                    )

        result = ValidatedFolder(
            files=file_infos,
            total_files=len(txt_files),
            readable=readable_count,
            unreadable=unreadable_count,
            warnings=warnings,
        )

        logger.info(
            "InputValidator: validated adaptation folder %s — "
            "%d files, %d readable, %d unreadable, %d warning(s)",
            folder_path,
            len(txt_files),
            readable_count,
            unreadable_count,
            len(warnings),
        )

        return result

    # ── Private helpers ───────────────────────────────────────────────

    def _validate_single_file(self, file_path: Path) -> FileInfo:
        """Read and validate a single adaptation source file.

        Args:
            file_path: Path to the ``.txt`` file.

        Returns:
            A ``FileInfo`` with content stats and language detection.
        """
        try:
            content = read_file(file_path)
        except UnicodeDecodeError as exc:
            logger.error(
                "InputValidator: encoding error for %s: %s",
                file_path,
                exc,
            )
            return FileInfo(
                path=str(file_path.resolve()),
                filename=file_path.name,
                error=f"Not valid UTF-8: {exc}",
            )
        except OSError as exc:
            logger.error(
                "InputValidator: I/O error for %s: %s",
                file_path,
                exc,
            )
            return FileInfo(
                path=str(file_path.resolve()),
                filename=file_path.name,
                error=f"I/O error: {exc}",
            )

        # Detect language.
        detection = self._detector.detect_file(file_path)

        return FileInfo(
            path=str(file_path.resolve()),
            filename=file_path.name,
            detected_language=detection.lang_code,
            confidence=detection.confidence,
            is_supported=detection.is_supported,
            char_count=len(content),
        )
