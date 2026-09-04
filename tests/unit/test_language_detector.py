"""Unit tests for ``core.language_detector``.

Tests cover: detecting English, Russian, German text correctly,
confidence threshold handling, unsupported language detection,
empty text, detect_file with a real temp file, and detect_batch
with multiple files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from core.language_detector import LanguageDetector, LanguageDetectionResult


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def detector() -> LanguageDetector:
    """Return a LanguageDetector with default threshold."""
    return LanguageDetector(confidence_threshold=0.5)


@pytest.fixture()
def strict_detector() -> LanguageDetector:
    """Return a LanguageDetector with a very high threshold."""
    return LanguageDetector(confidence_threshold=0.99)


# Sample texts — long enough for langdetect to work reliably.

ENGLISH_TEXT = (
    "The ancient temple stood at the edge of the forest, its stone walls "
    "covered in moss and ivy. For centuries, no one had dared to enter. "
    "But on this particular morning, a young archaeologist named Elena "
    "decided that today would be different. She packed her equipment "
    "carefully and set out before dawn."
)

RUSSIAN_TEXT = (
    "Древний храм стоял на краю леса, его каменные стены были покрыты "
    "мхом и плющом. На протяжении веков никто не осмеливался войти. "
    "Но в это особенное утро молодой археолог по имени Елена решила, "
    "что сегодня всё будет иначе. Она тщательно собрала своё снаряжение "
    "и отправилась в путь ещё до рассвета."
)

GERMAN_TEXT = (
    "Der alte Tempel stand am Rande des Waldes, seine Steinmauern waren "
    "mit Moos und Efeu bedeckt. Jahrhundertelang hatte sich niemand "
    "hineingewagt. Doch an diesem besonderen Morgen beschloss eine junge "
    "Archäologin namens Elena, dass heute alles anders sein würde. "
    "Sie packte ihre Ausrüstung sorgfältig ein und machte sich noch "
    "vor Morgengrauen auf den Weg."
)

FRENCH_TEXT = (
    "Le vieux temple se dressait à la lisière de la forêt, ses murs de "
    "pierre recouverts de mousse et de lierre. Pendant des siècles, "
    "personne n'avait osé y entrer. Mais ce matin-là, une jeune "
    "archéologue nommée Elena décida que tout serait différent. "
    "Elle prépara soigneusement son équipement et se mit en route."
)


# ── Tests: detect individual languages ────────────────────────────────


class TestDetectLanguages:
    """Tests for detecting supported languages from text."""

    def test_detect_english(self, detector: LanguageDetector) -> None:
        """Should detect English text correctly."""
        result = detector.detect(ENGLISH_TEXT)
        assert result.lang_code == "en"
        assert result.confidence > 0.5
        assert result.is_supported is True
        assert result.error == ""

    def test_detect_russian(self, detector: LanguageDetector) -> None:
        """Should detect Russian text correctly."""
        result = detector.detect(RUSSIAN_TEXT)
        assert result.lang_code == "ru"
        assert result.confidence > 0.5
        assert result.is_supported is True
        assert result.error == ""

    def test_detect_german(self, detector: LanguageDetector) -> None:
        """Should detect German text correctly."""
        result = detector.detect(GERMAN_TEXT)
        assert result.lang_code == "de"
        assert result.confidence > 0.5
        assert result.is_supported is True
        assert result.error == ""

    def test_detect_french(self, detector: LanguageDetector) -> None:
        """Should detect French text correctly."""
        result = detector.detect(FRENCH_TEXT)
        assert result.lang_code == "fr"
        assert result.confidence > 0.5
        assert result.is_supported is True
        assert result.error == ""


# ── Tests: edge cases ─────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_text_returns_error(self, detector: LanguageDetector) -> None:
        """Empty text should return an error result."""
        result = detector.detect("")
        assert result.lang_code == ""
        assert result.confidence == 0.0
        assert result.is_supported is False
        assert result.error != ""

    def test_whitespace_only_returns_error(self, detector: LanguageDetector) -> None:
        """Whitespace-only text should return an error result."""
        result = detector.detect("   \n\t  ")
        assert result.lang_code == ""
        assert result.error != ""

    def test_unsupported_language(self, detector: LanguageDetector) -> None:
        """Text in an unsupported language should set is_supported=False."""
        # Japanese is not one of the 11 supported languages.
        japanese_text = (
            "古代の寺院は森の端に立っていました。何世紀もの間、"
            "誰もあえて入ろうとしませんでした。しかし、この特別な朝、"
            "若い考古学者のエレナは、今日はすべてが違うと決めました。"
        )
        result = detector.detect(japanese_text)
        assert result.lang_code == "ja"
        assert result.is_supported is False

    def test_source_field_propagates(self, detector: LanguageDetector) -> None:
        """The source field should be set to the provided value."""
        result = detector.detect(ENGLISH_TEXT, source="my_test")
        assert result.source == "my_test"


# ── Tests: detect_file ────────────────────────────────────────────────


class TestDetectFile:
    """Tests for file-based language detection."""

    def test_detect_file_english(
        self, detector: LanguageDetector, tmp_dir: Path
    ) -> None:
        """Should detect language from a real temp file."""
        file_path = tmp_dir / "english_story.txt"
        file_path.write_text(ENGLISH_TEXT, encoding="utf-8")

        result = detector.detect_file(file_path)
        assert result.lang_code == "en"
        assert result.confidence > 0.5
        assert result.is_supported is True
        assert result.error == ""
        assert str(file_path) in result.source

    def test_detect_file_german(
        self, detector: LanguageDetector, tmp_dir: Path
    ) -> None:
        """Should detect German from a temp file."""
        file_path = tmp_dir / "german_story.txt"
        file_path.write_text(GERMAN_TEXT, encoding="utf-8")

        result = detector.detect_file(file_path)
        assert result.lang_code == "de"
        assert result.is_supported is True

    def test_detect_file_not_found(
        self, detector: LanguageDetector, tmp_dir: Path
    ) -> None:
        """Missing file should return an error result."""
        result = detector.detect_file(tmp_dir / "nonexistent.txt")
        assert result.error != ""
        assert result.lang_code == ""

    def test_detect_file_not_a_file(
        self, detector: LanguageDetector, tmp_dir: Path
    ) -> None:
        """Passing a directory path should return an error result."""
        result = detector.detect_file(tmp_dir)
        assert result.error != ""
        assert result.lang_code == ""

    def test_detect_file_empty(
        self, detector: LanguageDetector, tmp_dir: Path
    ) -> None:
        """Empty file should return an error result."""
        file_path = tmp_dir / "empty.txt"
        file_path.write_text("", encoding="utf-8")

        result = detector.detect_file(file_path)
        assert result.error != ""


# ── Tests: detect_batch ───────────────────────────────────────────────


class TestDetectBatch:
    """Tests for batch file detection."""

    def test_detect_batch_multiple_languages(
        self, detector: LanguageDetector, tmp_dir: Path
    ) -> None:
        """Batch detection should handle multiple languages."""
        en_file = tmp_dir / "en.txt"
        de_file = tmp_dir / "de.txt"
        ru_file = tmp_dir / "ru.txt"

        en_file.write_text(ENGLISH_TEXT, encoding="utf-8")
        de_file.write_text(GERMAN_TEXT, encoding="utf-8")
        ru_file.write_text(RUSSIAN_TEXT, encoding="utf-8")

        results = detector.detect_batch([en_file, de_file, ru_file])

        assert len(results) == 3
        assert results[0].lang_code == "en"
        assert results[1].lang_code == "de"
        assert results[2].lang_code == "ru"

    def test_detect_batch_empty_list(self, detector: LanguageDetector) -> None:
        """Empty batch should return empty results."""
        results = detector.detect_batch([])
        assert results == []

    def test_detect_batch_with_missing_file(
        self, detector: LanguageDetector, tmp_dir: Path
    ) -> None:
        """Batch should handle missing files gracefully."""
        en_file = tmp_dir / "en.txt"
        en_file.write_text(ENGLISH_TEXT, encoding="utf-8")

        missing_file = tmp_dir / "missing.txt"

        results = detector.detect_batch([en_file, missing_file])

        assert len(results) == 2
        assert results[0].lang_code == "en"
        assert results[0].error == ""
        assert results[1].error != ""
        assert results[1].lang_code == ""
