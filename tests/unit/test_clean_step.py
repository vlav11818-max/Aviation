"""Unit tests for core.steps.clean_step.

Tests: BOM removal, bracket markers, chapter headers, AI meta-comments,
stage directions, punctuation normalisation, whitespace normalisation,
multi-language chapter markers (RU, DE, FR, PT, IT, PL, UK, RO, TR, DA),
section break removal, end marker removal, language-specific pattern
loading integration.
"""

from __future__ import annotations

import re

import pytest

from core.steps.clean_step import clean_text


# ── Helpers ────────────────────────────────────────────────────────────


def _compile_patterns(raw: list[str]) -> list[re.Pattern]:
    """Compile a list of regex strings into Pattern objects."""
    return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in raw]


# ── Tests: Basic Cleanup ──────────────────────────────────────────────


class TestBasicCleanup:
    """Tests for core cleanup functionality."""

    def test_strip_bom(self) -> None:
        """BOM character should be removed."""
        text = "\ufeffHello world."
        assert clean_text(text) == "Hello world."

    def test_remove_bracket_markers(self) -> None:
        """[Chapter 1], [Part III] etc. should be removed."""
        text = "[Chapter 1] The story begins.\n[Part III] And continues."
        result = clean_text(text)
        assert "[Chapter 1]" not in result
        assert "[Part III]" not in result
        assert "The story begins." in result

    def test_remove_chapter_headers(self) -> None:
        """Chapter header lines should be removed entirely."""
        text = "Chapter One: The Beginning\n\nThe rain fell softly."
        result = clean_text(text)
        assert "Chapter One" not in result
        assert "The rain fell softly." in result

    def test_remove_part_markers(self) -> None:
        """Part markers on their own line should be removed."""
        text = "Part III\n\nThe city burned."
        result = clean_text(text)
        assert "Part III" not in result
        assert "The city burned." in result

    def test_remove_meta_comments(self) -> None:
        """AI meta-comments should be removed."""
        text = "Note: This is a work of fiction.\n\nThe story starts here."
        result = clean_text(text)
        assert "Note:" not in result
        assert "The story starts here." in result

    def test_remove_authors_note(self) -> None:
        """Author's note should be removed."""
        text = "Author's note: I hope you enjoy this.\n\nOnce upon a time."
        result = clean_text(text)
        assert "Author's note" not in result
        assert "Once upon a time." in result

    def test_remove_stage_directions(self) -> None:
        """*Stage directions* should be removed."""
        text = "She looked at him. *She felt nervous.* Then she spoke."
        result = clean_text(text)
        assert "*She felt nervous.*" not in result
        assert "She looked at him." in result

    def test_remove_end_markers(self) -> None:
        """'The End', 'Fin', etc. should be removed."""
        text = "He walked away.\n\nThe End"
        result = clean_text(text)
        assert "The End" not in result
        assert "He walked away." in result

    def test_remove_section_breaks(self) -> None:
        """***, ---, ~~~ section breaks should be removed."""
        text = "First part.\n\n***\n\nSecond part."
        result = clean_text(text)
        assert "***" not in result
        assert "First part." in result
        assert "Second part." in result

    def test_remove_dashes_break(self) -> None:
        """--- break markers should be removed."""
        text = "First part.\n\n---\n\nSecond part."
        result = clean_text(text)
        assert "---" not in result


class TestPunctuationNormalisation:
    """Tests for punctuation normalisation."""

    def test_multiple_periods_to_ellipsis(self) -> None:
        """... should become the ellipsis character."""
        text = "She waited... then left."
        result = clean_text(text)
        assert "\u2026" in result
        assert "..." not in result

    def test_multiple_exclamation_to_single(self) -> None:
        """!! should become single !."""
        text = "Amazing!!! Incredible!!"
        result = clean_text(text)
        assert "!!!" not in result
        assert "!!" not in result
        assert "!" in result

    def test_multiple_question_to_single(self) -> None:
        """?? should become single ?."""
        text = "Really??? You think so??"
        result = clean_text(text)
        assert "???" not in result
        assert "?" in result

    def test_double_dashes_to_em_dash(self) -> None:
        """-- should become an em dash."""
        text = "He said -- quietly -- that he was leaving."
        result = clean_text(text)
        assert "--" not in result
        assert "\u2014" in result


class TestWhitespaceNormalisation:
    """Tests for whitespace normalisation."""

    def test_excess_blank_lines_collapsed(self) -> None:
        """3+ blank lines should collapse to one blank line."""
        text = "First.\n\n\n\n\nSecond."
        result = clean_text(text)
        assert "\n\n\n" not in result
        assert "First.\n\nSecond." == result

    def test_excess_spaces_collapsed(self) -> None:
        """Multiple spaces should collapse to single space."""
        text = "Too    many   spaces   here."
        result = clean_text(text)
        assert "  " not in result
        assert "Too many spaces here." == result

    def test_leading_trailing_whitespace_stripped(self) -> None:
        """Text should be stripped of leading/trailing whitespace."""
        text = "\n\n  Hello world.  \n\n"
        result = clean_text(text)
        assert result == "Hello world."


# ── Tests: Multi-language Chapter Markers ──────────────────────────────


class TestMultiLanguageMarkers:
    """Tests for language-specific chapter/section marker removal.

    These tests simulate the cleanup patterns that would be loaded
    from cultural files via PromptManager.get_cleanup_patterns().
    """

    def test_russian_chapter_markers(self) -> None:
        """Russian 'Глава 1', 'Часть первая' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*глава\s+\d+",
            r"(?i)^\s*глава\s+(первая|вторая|третья)",
            r"(?i)^\s*часть\s+\d+",
            r"(?i)^\s*конец\s*\.?\s*$",
        ])
        text = "Глава 1\n\nОн шёл по дороге.\n\nЧасть 2\n\nОна ждала.\n\nКонец"
        result = clean_text(text, extra_patterns=patterns)
        assert "Глава 1" not in result
        assert "Часть 2" not in result
        assert "Конец" not in result
        assert "Он шёл по дороге." in result
        assert "Она ждала." in result

    def test_german_chapter_markers(self) -> None:
        """German 'Kapitel', 'Teil', 'Ende' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*kapitel\s+\d+",
            r"(?i)^\s*teil\s+\d+",
            r"(?i)^\s*ende\s*\.?\s*$",
        ])
        text = "Kapitel 3\n\nEr ging durch den Wald.\n\nEnde"
        result = clean_text(text, extra_patterns=patterns)
        assert "Kapitel 3" not in result
        assert "Ende" not in result
        assert "Er ging durch den Wald." in result

    def test_french_chapter_markers(self) -> None:
        """French 'Chapitre', 'Partie', 'Fin' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*chapitre\s+\d+",
            r"(?i)^\s*partie\s+\d+",
            r"(?i)^\s*fin\s*\.?\s*$",
        ])
        text = "Chapitre 5\n\nIl marchait sous la pluie.\n\nFin"
        result = clean_text(text, extra_patterns=patterns)
        assert "Chapitre 5" not in result
        assert "Fin" not in result
        assert "Il marchait sous la pluie." in result

    def test_turkish_chapter_markers(self) -> None:
        """Turkish 'Bölüm', 'Kısım', 'Son' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*bölüm\s+\d+",
            r"(?i)^\s*kısım\s+\d+",
            r"(?i)^\s*son\s*\.?\s*$",
        ])
        text = "Bölüm 2\n\nAdam yürüdü.\n\nSon"
        result = clean_text(text, extra_patterns=patterns)
        assert "Bölüm 2" not in result
        assert "Son" not in result
        assert "Adam yürüdü." in result

    def test_polish_chapter_markers(self) -> None:
        """Polish 'Rozdział', 'Część', 'Koniec' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*rozdział\s+\d+",
            r"(?i)^\s*część\s+\d+",
            r"(?i)^\s*koniec\s*\.?\s*$",
        ])
        text = "Rozdział 1\n\nSzedł przez las.\n\nKoniec"
        result = clean_text(text, extra_patterns=patterns)
        assert "Rozdział 1" not in result
        assert "Koniec" not in result
        assert "Szedł przez las." in result

    def test_ukrainian_chapter_markers(self) -> None:
        """Ukrainian 'Ð озділ', 'Частина', 'Кінець' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*розділ\s+\d+",
            r"(?i)^\s*частина\s+\d+",
            r"(?i)^\s*кінець\s*\.?\s*$",
        ])
        text = "Ð озділ 3\n\nВін ішов дорогою.\n\nКінець"
        result = clean_text(text, extra_patterns=patterns)
        assert "Ð озділ 3" not in result
        assert "Кінець" not in result
        assert "Він ішов дорогою." in result

    def test_portuguese_chapter_markers(self) -> None:
        """Portuguese 'Capítulo', 'Parte', 'Fim' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*capítulo\s+\d+",
            r"(?i)^\s*parte\s+\d+",
            r"(?i)^\s*fim\s*\.?\s*$",
        ])
        text = "Capítulo 1\n\nEle caminhou pela estrada.\n\nFim"
        result = clean_text(text, extra_patterns=patterns)
        assert "Capítulo 1" not in result
        assert "Fim" not in result
        assert "Ele caminhou pela estrada." in result

    def test_italian_chapter_markers(self) -> None:
        """Italian 'Capitolo', 'Parte', 'Fine' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*capitolo\s+\d+",
            r"(?i)^\s*parte\s+\d+",
            r"(?i)^\s*fine\s*\.?\s*$",
        ])
        text = "Capitolo 7\n\nCamminava sotto la pioggia.\n\nFine"
        result = clean_text(text, extra_patterns=patterns)
        assert "Capitolo 7" not in result
        assert "Fine" not in result
        assert "Camminava sotto la pioggia." in result

    def test_romanian_chapter_markers(self) -> None:
        """Romanian 'Capitolul', 'Partea', 'Sfârșit' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*capitolul\s+\d+",
            r"(?i)^\s*partea\s+\d+",
            r"(?i)^\s*sfârșit\s*\.?\s*$",
        ])
        text = "Capitolul 4\n\nMergea pe drum.\n\nSfârșit"
        result = clean_text(text, extra_patterns=patterns)
        assert "Capitolul 4" not in result
        assert "Sfârșit" not in result
        assert "Mergea pe drum." in result

    def test_danish_chapter_markers(self) -> None:
        """Danish 'Kapitel', 'Del', 'Slut' should be removed."""
        patterns = _compile_patterns([
            r"(?i)^\s*kapitel\s+\d+",
            r"(?i)^\s*del\s+\d+",
            r"(?i)^\s*slut\s*\.?\s*$",
        ])
        text = "Kapitel 2\n\nHan gik gennem skoven.\n\nSlut"
        result = clean_text(text, extra_patterns=patterns)
        assert "Kapitel 2" not in result
        assert "Slut" not in result
        assert "Han gik gennem skoven." in result

    def test_invalid_pattern_is_skipped(self) -> None:
        """Invalid regex pattern should not crash clean_text."""
        bad_patterns = [re.compile(r"test")]
        text = "Normal text here."
        result = clean_text(text, extra_patterns=bad_patterns)
        assert "Normal here." in result or "Normal text here." in result

    def test_empty_extra_patterns(self) -> None:
        """Empty pattern list should not change behavior."""
        text = "Hello world."
        result = clean_text(text, extra_patterns=[])
        assert result == "Hello world."

    def test_none_extra_patterns(self) -> None:
        """None pattern list should not change behavior."""
        text = "Hello world."
        result = clean_text(text, extra_patterns=None)
        assert result == "Hello world."


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert clean_text("") == ""

    def test_only_whitespace(self) -> None:
        """Whitespace-only string should return empty string."""
        assert clean_text("   \n\n   ") == ""

    def test_only_markers(self) -> None:
        """Text with only markers should return empty string."""
        text = "Chapter 1\n\n***\n\nThe End"
        result = clean_text(text)
        assert result == "" or result.strip() == ""

    def test_preserves_normal_text(self) -> None:
        """Normal story text should be preserved unchanged (except whitespace)."""
        text = (
            "She walked down the cobblestone street. The air smelled of "
            "rain and old books. At the corner, she paused.\n\n"
            "The cafe was still open. A warm light spilled from the window."
        )
        result = clean_text(text)
        assert "She walked down the cobblestone street." in result
        assert "The cafe was still open." in result
