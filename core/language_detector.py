"""Language detection for AI Story Generator Pro.

Wraps the ``langdetect`` library with error handling, confidence
thresholds, and support for the project's 11 supported languages.
Provides single-text, file-based, and batch detection methods.

Typical usage::

    ld = LanguageDetector()
    result = ld.detect("Im dunklen Wald stand ein altes Haus.")
    # result.lang_code == "de", result.confidence ≈ 0.99, result.is_supported == True

    result = ld.detect_file(Path("story.txt"))
    results = ld.detect_batch([Path("a.txt"), Path("b.txt")])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from models.config import SUPPORTED_LANGUAGE_CODES
from utils.file_handler import read_file

logger = logging.getLogger(__name__)

# Number of leading characters to read from a file for detection.
_FILE_SAMPLE_CHARS: int = 1000

# Default minimum confidence threshold below which detection is
# considered unreliable.
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class LanguageDetectionResult:
    """Result of a language detection attempt.

    Attributes:
        lang_code: Two-letter ISO 639-1 code detected (e.g. ``"de"``).
            Empty string if detection failed.
        confidence: Confidence score (0.0–1.0).  ``0.0`` if detection
            failed.
        is_supported: Whether the detected language is one of the 11
            supported languages.
        source: Description of the input source (e.g. file path or
            ``"text"``).
        error: Error message if detection failed, empty otherwise.
    """

    lang_code: str = ""
    confidence: float = 0.0
    is_supported: bool = False
    source: str = "text"
    error: str = ""


class LanguageDetector:
    """Detects the language of text, files, or batches of files.

    Wraps ``langdetect`` with graceful error handling, confidence
    thresholds, and awareness of the project's 11 supported languages.

    Args:
        confidence_threshold: Minimum confidence below which the
            detection result is flagged as unreliable.  Defaults to
            ``0.5``.
    """

    def __init__(
        self, confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD
    ) -> None:
        self._threshold = confidence_threshold
        logger.debug(
            "LanguageDetector initialised (confidence_threshold=%.2f)",
            self._threshold,
        )

    def detect(self, text: str, source: str = "text") -> LanguageDetectionResult:
        """Detect the language of a text string.

        Args:
            text: Input text to analyse.  An empty or whitespace-only
                string returns a failed result.
            source: Description of the input origin (for logging).

        Returns:
            A ``LanguageDetectionResult`` with the detected language,
            confidence, and whether it is a supported language.
        """
        stripped = text.strip()
        if not stripped:
            logger.warning("LanguageDetector: empty text provided (source=%s)", source)
            return LanguageDetectionResult(
                source=source,
                error="Empty or whitespace-only text",
            )

        try:
            from langdetect import detect_langs
        except ImportError as exc:
            logger.error("LanguageDetector: langdetect not installed: %s", exc)
            return LanguageDetectionResult(
                source=source,
                error="langdetect library not available",
            )

        try:
            detected = detect_langs(stripped)
        except Exception as exc:
            logger.error(
                "LanguageDetector: detection failed (source=%s): %s",
                source,
                exc,
            )
            return LanguageDetectionResult(
                source=source,
                error=f"Detection failed: {exc}",
            )

        if not detected:
            logger.warning(
                "LanguageDetector: no languages detected (source=%s)", source
            )
            return LanguageDetectionResult(
                source=source,
                error="No languages detected",
            )

        # Take the top result.
        top = detected[0]
        lang_code = str(top.lang)
        confidence = float(top.prob)
        is_supported = lang_code in SUPPORTED_LANGUAGE_CODES

        if confidence < self._threshold:
            logger.warning(
                "LanguageDetector: low confidence %.2f for '%s' (source=%s, "
                "threshold=%.2f)",
                confidence,
                lang_code,
                source,
                self._threshold,
            )

        if not is_supported:
            logger.info(
                "LanguageDetector: detected '%s' (%.2f) which is not in "
                "supported languages (source=%s)",
                lang_code,
                confidence,
                source,
            )

        logger.debug(
            "LanguageDetector: detected '%s' with confidence %.2f "
            "(supported=%s, source=%s)",
            lang_code,
            confidence,
            is_supported,
            source,
        )

        return LanguageDetectionResult(
            lang_code=lang_code,
            confidence=confidence,
            is_supported=is_supported,
            source=source,
        )

    def detect_file(
        self, path: str | Path, sample_chars: int = _FILE_SAMPLE_CHARS
    ) -> LanguageDetectionResult:
        """Detect the language of a text file.

        Reads the first *sample_chars* characters from the file and
        passes them to ``detect()``.

        Args:
            path: Path to the text file.
            sample_chars: Number of leading characters to read.

        Returns:
            A ``LanguageDetectionResult`` with the file path as source.
        """
        file_path = Path(path)
        source = str(file_path)

        if not file_path.exists():
            logger.error(
                "LanguageDetector: file not found: %s", file_path
            )
            return LanguageDetectionResult(
                source=source,
                error=f"File not found: {file_path}",
            )

        if not file_path.is_file():
            logger.error(
                "LanguageDetector: path is not a file: %s", file_path
            )
            return LanguageDetectionResult(
                source=source,
                error=f"Not a file: {file_path}",
            )

        try:
            content = read_file(file_path)
        except UnicodeDecodeError as exc:
            logger.error(
                "LanguageDetector: encoding error reading %s: %s",
                file_path,
                exc,
            )
            return LanguageDetectionResult(
                source=source,
                error=f"Encoding error: {exc}",
            )
        except OSError as exc:
            logger.error(
                "LanguageDetector: I/O error reading %s: %s",
                file_path,
                exc,
            )
            return LanguageDetectionResult(
                source=source,
                error=f"I/O error: {exc}",
            )

        sample = content[:sample_chars]
        return self.detect(sample, source=source)

    def detect_batch(
        self, paths: Sequence[str | Path]
    ) -> list[LanguageDetectionResult]:
        """Detect the language of multiple files.

        Args:
            paths: Sequence of file paths to analyse.

        Returns:
            A list of ``LanguageDetectionResult`` objects in the same
            order as the input paths.
        """
        results: list[LanguageDetectionResult] = []
        for path in paths:
            result = self.detect_file(path)
            results.append(result)

        detected_langs = [
            r.lang_code for r in results if r.lang_code and not r.error
        ]
        logger.info(
            "LanguageDetector: batch detection of %d files — "
            "detected languages: %s",
            len(paths),
            ", ".join(detected_langs) if detected_langs else "(none)",
        )

        return results
