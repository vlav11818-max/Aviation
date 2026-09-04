"""Unit tests for ``core.input_validator``.

Tests cover: valid topics file, file with empty lines (stripped),
file with duplicates (removed with warning), non-UTF-8 file handling,
folder with valid .txt files, folder with no .txt files, empty folder,
and mixed-language files in a folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.exceptions import ConfigError
from core.input_validator import InputValidator, ValidatedFolder, ValidatedTopics


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def validator() -> InputValidator:
    """Return an InputValidator with default settings."""
    return InputValidator()


# ── Helper: write temp topics file ────────────────────────────────────


def _write_topics(tmp_dir: Path, filename: str, lines: list[str]) -> Path:
    """Write a topics file and return its path."""
    path = tmp_dir / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Tests: validate_topics_file ───────────────────────────────────────


class TestValidateTopicsFile:
    """Tests for InputValidator.validate_topics_file."""

    def test_valid_topics_file(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """A clean topics file should return all topics."""
        topics = ["Ancient Temple", "Lost City", "Haunted Lighthouse"]
        path = _write_topics(tmp_dir, "topics.txt", topics)

        result = validator.validate_topics_file(path)

        assert isinstance(result, ValidatedTopics)
        assert result.topics == topics
        assert result.original_count == 3
        assert result.removed_empty == 0
        assert result.removed_duplicates == 0
        assert len(result.warnings) == 0

    def test_strips_whitespace(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Lines with leading/trailing whitespace should be stripped."""
        lines = ["  Ancient Temple  ", "Lost City\t", " Haunted Lighthouse"]
        path = _write_topics(tmp_dir, "ws.txt", lines)

        result = validator.validate_topics_file(path)

        assert result.topics == ["Ancient Temple", "Lost City", "Haunted Lighthouse"]

    def test_removes_empty_lines(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Empty lines should be removed with a warning."""
        lines = ["Ancient Temple", "", "Lost City", "", "", "Haunted Lighthouse"]
        path = _write_topics(tmp_dir, "empty.txt", lines)

        result = validator.validate_topics_file(path)

        assert result.topics == ["Ancient Temple", "Lost City", "Haunted Lighthouse"]
        assert result.removed_empty == 3
        assert any("empty" in w.lower() for w in result.warnings)

    def test_removes_duplicates_with_warning(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Duplicate topics should be removed with a warning."""
        lines = [
            "Ancient Temple",
            "Lost City",
            "ancient temple",  # case-insensitive duplicate
            "Lost City",       # exact duplicate
            "Haunted Lighthouse",
        ]
        path = _write_topics(tmp_dir, "dupes.txt", lines)

        result = validator.validate_topics_file(path)

        assert len(result.topics) == 3
        assert result.topics[0] == "Ancient Temple"
        assert result.removed_duplicates == 2
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_removes_short_topics(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Topics shorter than the minimum should be removed."""
        lines = ["AB", "Ancient Temple", "XY", "Lost City"]
        path = _write_topics(tmp_dir, "short.txt", lines)

        result = validator.validate_topics_file(path)

        assert len(result.topics) == 2
        assert "Ancient Temple" in result.topics
        assert "Lost City" in result.topics

    def test_file_not_found_raises(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Missing file should raise ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            validator.validate_topics_file(tmp_dir / "nonexistent.txt")

    def test_not_a_file_raises(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Passing a directory should raise ConfigError."""
        with pytest.raises(ConfigError, match="not a file"):
            validator.validate_topics_file(tmp_dir)

    def test_non_utf8_file_raises(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """A non-UTF-8 file should raise ConfigError."""
        path = tmp_dir / "latin1.txt"
        # Write bytes that are invalid UTF-8.
        path.write_bytes(b"Caf\xe9\nNa\xefve\n")

        with pytest.raises(ConfigError, match="UTF-8"):
            validator.validate_topics_file(path)

    def test_empty_file_produces_warning(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """An empty file should return no topics and a warning."""
        path = _write_topics(tmp_dir, "empty_file.txt", [])

        result = validator.validate_topics_file(path)

        assert result.topics == []
        assert any("no valid" in w.lower() for w in result.warnings)

    def test_preserves_order(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Topics should preserve original file order."""
        lines = ["Zebra Story", "Apple Adventure", "Mountain Quest"]
        path = _write_topics(tmp_dir, "order.txt", lines)

        result = validator.validate_topics_file(path)

        assert result.topics == lines


# ── Tests: validate_adaptation_folder ─────────────────────────────────


class TestValidateAdaptationFolder:
    """Tests for InputValidator.validate_adaptation_folder."""

    def test_valid_folder(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Folder with valid .txt files should return file info."""
        folder = tmp_dir / "texts"
        folder.mkdir()

        en_text = (
            "The ancient temple stood at the edge of the forest, "
            "its stone walls covered in moss and ivy. For centuries, "
            "no one had dared to enter the sacred halls."
        )
        de_text = (
            "Der alte Tempel stand am Rande des Waldes, seine "
            "Steinmauern waren mit Moos und Efeu bedeckt. "
            "Jahrhundertelang hatte sich niemand hineingewagt."
        )

        (folder / "story_en.txt").write_text(en_text, encoding="utf-8")
        (folder / "story_de.txt").write_text(de_text, encoding="utf-8")

        result = validator.validate_adaptation_folder(folder)

        assert isinstance(result, ValidatedFolder)
        assert result.total_files == 2
        assert result.readable == 2
        assert result.unreadable == 0
        assert len(result.files) == 2

        # Check that files have detected languages.
        lang_codes = {fi.detected_language for fi in result.files}
        assert "en" in lang_codes
        assert "de" in lang_codes

    def test_folder_not_found_raises(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Missing folder should raise ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            validator.validate_adaptation_folder(tmp_dir / "nonexistent")

    def test_not_a_directory_raises(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Passing a file as folder should raise ConfigError."""
        file_path = tmp_dir / "file.txt"
        file_path.write_text("hello", encoding="utf-8")

        with pytest.raises(ConfigError, match="not a directory"):
            validator.validate_adaptation_folder(file_path)

    def test_no_txt_files_raises(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Folder with no .txt files should raise ConfigError."""
        folder = tmp_dir / "no_txt"
        folder.mkdir()
        (folder / "readme.md").write_text("# Readme", encoding="utf-8")

        with pytest.raises(ConfigError, match="No .txt files"):
            validator.validate_adaptation_folder(folder)

    def test_mixed_language_files(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Folder with mixed languages should report each correctly."""
        folder = tmp_dir / "mixed"
        folder.mkdir()

        en_text = (
            "The young explorer ventured deep into the forgotten cave. "
            "Water dripped from the ceiling, echoing through the darkness. "
            "She knew she was close to the ancient treasure."
        )
        ru_text = (
            "Молодой исследователь углубился в забытую пещеру. "
            "Вода капала с потолка, эхом разносясь по темноте. "
            "Она знала, что древнее сокровище совсем близко."
        )

        (folder / "story_en.txt").write_text(en_text, encoding="utf-8")
        (folder / "story_ru.txt").write_text(ru_text, encoding="utf-8")

        result = validator.validate_adaptation_folder(folder)

        assert result.total_files == 2
        assert result.readable == 2

        lang_map = {fi.filename: fi.detected_language for fi in result.files}
        assert lang_map["story_en.txt"] == "en"
        assert lang_map["story_ru.txt"] == "ru"

    def test_folder_with_unreadable_file(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """Folder containing an unreadable file should warn but not crash."""
        folder = tmp_dir / "mixed_health"
        folder.mkdir()

        good_text = (
            "The ancient temple stood at the edge of the forest. "
            "Its stone walls were covered in moss and ivy. "
            "For centuries, no one had dared to enter."
        )
        (folder / "good.txt").write_text(good_text, encoding="utf-8")

        # Create a file with invalid UTF-8 bytes.
        (folder / "bad.txt").write_bytes(b"Caf\xe9 au lait \x80\x81 invalid bytes")

        result = validator.validate_adaptation_folder(folder)

        assert result.total_files == 2
        assert result.readable == 1
        assert result.unreadable == 1
        assert any("bad.txt" in w for w in result.warnings)

    def test_file_info_has_char_count(
        self, validator: InputValidator, tmp_dir: Path
    ) -> None:
        """FileInfo should include the character count of the file."""
        folder = tmp_dir / "charcount"
        folder.mkdir()

        content = "Hello world! " * 50  # 650 chars
        (folder / "story.txt").write_text(content, encoding="utf-8")

        result = validator.validate_adaptation_folder(folder)

        assert result.files[0].char_count == len(content)
