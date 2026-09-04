"""Unit tests for ``exporters.ssml_exporter``.

Tests cover: single paragraph export, multi-paragraph with breaks,
dialog detection, dramatic pause detection, scene break detection,
prosody tag for dramatic content, valid XML output (parseable by
lxml), correct lang tag mapping for all 11 languages, empty text
handling, and text with special XML characters (& < > properly
escaped).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from lxml import etree

from core.exceptions import ExportError
from exporters.ssml_exporter import (
    LANG_TAG_MAP,
    SSMLExporter,
    _DefaultSSMLSettings,
)


# ── Constants ──────────────────────────────────────────────────────────

_SSML_NS = "http://www.w3.org/2001/10/synthesis"
_NS = {"s": _SSML_NS}

# All 11 supported languages.
_ALL_LANGUAGES = ["en", "ru", "de", "fr", "pt", "it", "pl", "uk", "ro", "tr", "da"]


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def exporter() -> SSMLExporter:
    """Return a fresh SSMLExporter instance."""
    return SSMLExporter()


@pytest.fixture()
def default_settings() -> _DefaultSSMLSettings:
    """Return default SSML settings."""
    return _DefaultSSMLSettings()


@pytest.fixture()
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory cleaned up after the test."""
    with tempfile.TemporaryDirectory(prefix="ssml_test_") as d:
        yield Path(d)


# ── Helpers ────────────────────────────────────────────────────────────


def _parse_ssml(ssml_str: str) -> etree._Element:
    """Parse an SSML string into an lxml Element, validating well-formedness.

    Args:
        ssml_str: SSML XML string.

    Returns:
        Root ``<speak>`` element.

    Raises:
        etree.XMLSyntaxError: If the XML is malformed.
    """
    return etree.fromstring(ssml_str.encode("utf-8"))


def _export_and_parse(
    exporter: SSMLExporter,
    text: str,
    tmp_dir: Path,
    language: str = "en",
    ssml_settings: "_DefaultSSMLSettings | None" = None,
) -> etree._Element:
    """Export text to SSML file and parse it back.

    Args:
        exporter: SSMLExporter instance.
        text: Input text.
        tmp_dir: Temporary directory for output.
        language: Language code.
        ssml_settings: Optional settings override.

    Returns:
        Parsed root element.
    """
    output_path = tmp_dir / f"story_{language}.ssml"
    result_path = exporter.export(
        text=text,
        output_path=output_path,
        language=language,
        ssml_settings=ssml_settings,
    )
    content = result_path.read_text(encoding="utf-8")
    return _parse_ssml(content)


def _get_all_text(element: etree._Element) -> str:
    """Extract all text content from an element and its descendants.

    Args:
        element: lxml element.

    Returns:
        Concatenated text content.
    """
    texts: list[str] = []
    if element.text:
        texts.append(element.text)
    for child in element:
        texts.append(_get_all_text(child))
        if child.tail:
            texts.append(child.tail)
    return "".join(texts)


# ── Tests: single paragraph export ────────────────────────────────────


class TestSingleParagraph:
    """Tests for exporting a single paragraph."""

    def test_single_paragraph_produces_one_p(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """A single paragraph should produce one <p> element."""
        text = "The old house stood at the edge of the forest."
        root = _export_and_parse(exporter, text, tmp_dir)

        p_elements = root.findall("s:p", _NS)
        assert len(p_elements) == 1

    def test_single_sentence_produces_one_s(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """A single sentence should produce one <s> element inside <p>."""
        text = "The old house stood at the edge of the forest."
        root = _export_and_parse(exporter, text, tmp_dir)

        p_elem = root.find("s:p", _NS)
        assert p_elem is not None
        s_elements = p_elem.findall("s:s", _NS)
        assert len(s_elements) == 1

    def test_multiple_sentences_in_paragraph(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Multiple sentences in one paragraph should produce multiple <s> elements."""
        text = (
            "The old house stood at the edge of the forest. "
            "Nobody had entered it for decades. "
            "Its windows were dark and broken."
        )
        root = _export_and_parse(exporter, text, tmp_dir)

        p_elem = root.find("s:p", _NS)
        assert p_elem is not None
        s_elements = p_elem.findall("s:s", _NS)
        assert len(s_elements) == 3

    def test_exported_file_exists(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """The export method should create the output file."""
        text = "A simple test sentence."
        output_path = tmp_dir / "test_output.ssml"

        result_path = exporter.export(
            text=text,
            output_path=output_path,
            language="en",
        )

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_returned_path_matches_output(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """The returned path should resolve to the requested output path."""
        text = "A simple test sentence."
        output_path = tmp_dir / "check_path.ssml"

        result_path = exporter.export(
            text=text,
            output_path=output_path,
            language="en",
        )

        assert result_path.resolve() == output_path.resolve()


# ── Tests: multi-paragraph with breaks ────────────────────────────────


class TestMultiParagraph:
    """Tests for multi-paragraph text with break elements."""

    def test_single_newline_paragraphs(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Single-newline-separated lines should produce multiple <p> elements."""
        text = (
            "The morning sun rose over the village.\n"
            "Children were already playing in the streets.\n"
            "The baker opened his shop early."
        )
        root = _export_and_parse(exporter, text, tmp_dir)

        p_elements = root.findall("s:p", _NS)
        assert len(p_elements) == 3

    def test_paragraph_breaks_between_sub_paragraphs(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Sub-paragraphs within a scene should have <break> elements between them."""
        text = (
            "The morning sun rose over the village.\n"
            "Children were already playing in the streets."
        )
        root = _export_and_parse(exporter, text, tmp_dir)

        # There should be a <break> element between the two <p> elements.
        break_elements = root.findall("s:break", _NS)
        assert len(break_elements) >= 1


# ── Tests: scene break detection ──────────────────────────────────────


class TestSceneBreak:
    """Tests for scene break (double newline) detection."""

    def test_double_newline_creates_scene_break(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Double newlines should produce a <break> with scene_break duration."""
        text = (
            "The hero entered the cave.\n\n"
            "Meanwhile, in the village, panic was spreading."
        )
        root = _export_and_parse(exporter, text, tmp_dir)

        break_elements = root.findall("s:break", _NS)
        scene_breaks = [
            b for b in break_elements
            if b.get("time") == "1000ms"
        ]
        assert len(scene_breaks) >= 1

    def test_triple_newline_treated_as_scene_break(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Triple (or more) newlines should also produce a scene break."""
        text = (
            "Part one of the story.\n\n\n"
            "Part two of the story."
        )
        root = _export_and_parse(exporter, text, tmp_dir)

        break_elements = root.findall("s:break", _NS)
        scene_breaks = [
            b for b in break_elements
            if b.get("time") == "1000ms"
        ]
        assert len(scene_breaks) >= 1

    def test_multiple_scene_breaks(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Multiple scene breaks should produce multiple <break> elements."""
        text = (
            "Scene one.\n\n"
            "Scene two.\n\n"
            "Scene three."
        )
        root = _export_and_parse(exporter, text, tmp_dir)

        break_elements = root.findall("s:break", _NS)
        scene_breaks = [
            b for b in break_elements
            if b.get("time") == "1000ms"
        ]
        assert len(scene_breaks) == 2


# ── Tests: dialog detection ───────────────────────────────────────────


class TestDialogDetection:
    """Tests for dialog line detection and dialog pause insertion."""

    def test_dialog_with_left_double_quote(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Lines starting with \u201c should be detected as dialog."""
        text = "\u201cWho goes there?\u201d asked the guard."
        root = _export_and_parse(exporter, text, tmp_dir)

        # Dialog should produce a <break> with dialog_pause duration after <p>.
        break_elements = root.findall("s:break", _NS)
        dialog_breaks = [
            b for b in break_elements
            if b.get("time") == "400ms"
        ]
        assert len(dialog_breaks) >= 1

    def test_dialog_with_guillemet(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Lines starting with \u00ab (guillemet) should be detected as dialog."""
        text = "\u00ab Qui va l\u00e0 ? \u00bb demanda le garde."
        root = _export_and_parse(exporter, text, tmp_dir)

        break_elements = root.findall("s:break", _NS)
        dialog_breaks = [
            b for b in break_elements
            if b.get("time") == "400ms"
        ]
        assert len(dialog_breaks) >= 1

    def test_dialog_with_german_low_quote(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Lines starting with \u201e (German low quote) should be dialog."""
        text = "\u201eWer bist du?\u201c fragte eine Stimme aus der Dunkelheit."
        root = _export_and_parse(exporter, text, tmp_dir)

        break_elements = root.findall("s:break", _NS)
        dialog_breaks = [
            b for b in break_elements
            if b.get("time") == "400ms"
        ]
        assert len(dialog_breaks) >= 1

    def test_non_dialog_has_no_dialog_break(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Non-dialog text should not produce a dialog_pause break."""
        text = "The old man walked slowly down the path."
        root = _export_and_parse(exporter, text, tmp_dir)

        break_elements = root.findall("s:break", _NS)
        dialog_breaks = [
            b for b in break_elements
            if b.get("time") == "400ms"
        ]
        assert len(dialog_breaks) == 0


# ── Tests: dramatic pause detection ───────────────────────────────────


class TestDramaticPause:
    """Tests for dramatic pause (...) detection."""

    def test_ellipsis_three_dots(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Three dots (...) should produce a <break> with dramatic_pause duration."""
        text = "The door creaked open... and there was nothing inside."
        root = _export_and_parse(exporter, text, tmp_dir)

        break_elements = root.findall(".//s:break", _NS)
        dramatic_breaks = [
            b for b in break_elements
            if b.get("time") == "800ms"
        ]
        assert len(dramatic_breaks) >= 1

    def test_unicode_ellipsis(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Unicode ellipsis (\u2026) should produce a dramatic pause break."""
        text = "She looked into the darkness\u2026 something moved."
        root = _export_and_parse(exporter, text, tmp_dir)

        break_elements = root.findall(".//s:break", _NS)
        dramatic_breaks = [
            b for b in break_elements
            if b.get("time") == "800ms"
        ]
        assert len(dramatic_breaks) >= 1

    def test_text_split_around_dramatic_pause(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Text around ellipsis should be split with a break between parts."""
        text = "He waited... then ran."
        root = _export_and_parse(exporter, text, tmp_dir)

        # The <s> element should contain text parts and a <break>.
        s_elem = root.find(".//s:s", _NS)
        assert s_elem is not None

        # The full text should be preserved (minus the ellipsis).
        full_text = _get_all_text(s_elem)
        assert "waited" in full_text
        assert "then ran" in full_text


# ── Tests: prosody for dramatic content ───────────────────────────────


class TestProsodyDramatic:
    """Tests for <prosody rate='slow'> wrapping on dramatic content."""

    def test_slow_prosody_for_dramatic_non_dialog(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Non-dialog sentences with ellipsis should get <prosody rate='slow'>."""
        text = "The shadows grew longer... the air turned cold."
        root = _export_and_parse(exporter, text, tmp_dir)

        prosody_elements = root.findall(".//s:prosody", _NS)
        slow_elements = [
            p for p in prosody_elements
            if p.get("rate") == "slow"
        ]
        assert len(slow_elements) >= 1

    def test_no_slow_prosody_for_dialog(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Dialog lines with ellipsis should NOT get slow prosody."""
        text = "\u201cI don\u2019t know... maybe we should leave,\u201d she whispered."
        root = _export_and_parse(exporter, text, tmp_dir)

        prosody_elements = root.findall(".//s:prosody", _NS)
        slow_elements = [
            p for p in prosody_elements
            if p.get("rate") == "slow"
        ]
        assert len(slow_elements) == 0

    def test_slow_prosody_disabled(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """When slow_for_dramatic is False, no <prosody> should appear."""
        settings = _DefaultSSMLSettings()
        settings.slow_for_dramatic = False

        text = "The shadows grew longer... the air turned cold."
        root = _export_and_parse(exporter, text, tmp_dir, ssml_settings=settings)

        prosody_elements = root.findall(".//s:prosody", _NS)
        slow_elements = [
            p for p in prosody_elements
            if p.get("rate") == "slow"
        ]
        assert len(slow_elements) == 0


# ── Tests: valid XML output ───────────────────────────────────────────


class TestValidXML:
    """Tests for valid, well-formed XML output."""

    def test_output_is_parseable_xml(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """SSML output must be parseable by lxml without errors."""
        text = (
            "The old house stood at the edge of the forest. "
            "Nobody had entered it for decades.\n\n"
            "One day, a young girl decided to explore."
        )
        output_path = tmp_dir / "valid.ssml"
        exporter.export(text=text, output_path=output_path, language="en")

        content = output_path.read_text(encoding="utf-8")
        # This should not raise.
        root = etree.fromstring(content.encode("utf-8"))
        assert root.tag == f"{{{_SSML_NS}}}speak"

    def test_xml_declaration_present(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Output should include an XML declaration."""
        text = "A test sentence."
        output_path = tmp_dir / "declaration.ssml"
        exporter.export(text=text, output_path=output_path, language="en")

        content = output_path.read_text(encoding="utf-8")
        assert content.startswith("<?xml")

    def test_speak_element_has_version(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """The <speak> root element should have version='1.1'."""
        text = "A test sentence."
        root = _export_and_parse(exporter, text, tmp_dir)
        assert root.get("version") == "1.1"

    def test_speak_element_has_xmlns(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """The <speak> root element should have the SSML namespace."""
        text = "A test sentence."
        root = _export_and_parse(exporter, text, tmp_dir)
        assert root.tag == f"{{{_SSML_NS}}}speak"


# ── Tests: language tag mapping ───────────────────────────────────────


class TestLanguageTagMapping:
    """Tests for correct xml:lang attribute for all 11 languages."""

    @pytest.mark.parametrize(
        "lang_code,expected_tag",
        [
            ("en", "en-US"),
            ("ru", "ru-RU"),
            ("de", "de-DE"),
            ("fr", "fr-FR"),
            ("pt", "pt-BR"),
            ("it", "it-IT"),
            ("pl", "pl-PL"),
            ("uk", "uk-UA"),
            ("ro", "ro-RO"),
            ("tr", "tr-TR"),
            ("da", "da-DK"),
        ],
    )
    def test_lang_tag_mapping(
        self,
        exporter: SSMLExporter,
        tmp_dir: Path,
        lang_code: str,
        expected_tag: str,
    ) -> None:
        """Each supported language should map to the correct BCP-47 tag."""
        text = "A simple test sentence for language mapping."
        root = _export_and_parse(exporter, text, tmp_dir, language=lang_code)

        xml_lang = root.get(f"{{{etree.QName('http://www.w3.org/XML/1998/namespace', 'lang').namespace}}}lang")
        # Alternative: the attribute may be stored as xml:lang.
        if xml_lang is None:
            xml_lang = root.get("{http://www.w3.org/XML/1998/namespace}lang")
        assert xml_lang == expected_tag

    def test_all_eleven_languages_in_map(self) -> None:
        """LANG_TAG_MAP should contain entries for all 11 languages."""
        for lang in _ALL_LANGUAGES:
            assert lang in LANG_TAG_MAP, f"Missing mapping for '{lang}'"

    def test_unknown_language_generates_fallback_tag(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """An unmapped language code should generate a fallback tag like 'xx-XX'."""
        text = "A test sentence for an unknown language."
        root = _export_and_parse(exporter, text, tmp_dir, language="xx")

        xml_lang = root.get("{http://www.w3.org/XML/1998/namespace}lang")
        assert xml_lang == "xx-XX"


# ── Tests: empty text handling ────────────────────────────────────────


class TestEmptyText:
    """Tests for empty and whitespace-only text handling."""

    def test_empty_string_raises(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Empty string should raise ExportError."""
        with pytest.raises(ExportError, match="empty"):
            exporter.export(
                text="",
                output_path=tmp_dir / "empty.ssml",
                language="en",
            )

    def test_whitespace_only_raises(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Whitespace-only string should raise ExportError."""
        with pytest.raises(ExportError, match="empty"):
            exporter.export(
                text="   \n\n  \t  ",
                output_path=tmp_dir / "whitespace.ssml",
                language="en",
            )

    def test_none_text_raises(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """None text should raise ExportError."""
        with pytest.raises(ExportError):
            exporter.export(
                text=None,  # type: ignore[arg-type]
                output_path=tmp_dir / "none.ssml",
                language="en",
            )


# ── Tests: special XML characters ─────────────────────────────────────


class TestSpecialXMLCharacters:
    """Tests for proper escaping of special XML characters."""

    def test_ampersand_escaped(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Ampersand (&) in text should be properly escaped in output."""
        text = "Tom & Jerry went to the store."
        output_path = tmp_dir / "ampersand.ssml"
        exporter.export(text=text, output_path=output_path, language="en")

        content = output_path.read_text(encoding="utf-8")
        # The raw file should have &amp; not bare &.
        assert "&amp;" in content
        # It should still be valid XML.
        _parse_ssml(content)

    def test_less_than_escaped(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Less-than (<) in text should be properly escaped."""
        text = "The temperature was < 0 degrees."
        output_path = tmp_dir / "less_than.ssml"
        exporter.export(text=text, output_path=output_path, language="en")

        content = output_path.read_text(encoding="utf-8")
        # Should have &lt; in the text content area.
        assert "&lt;" in content
        _parse_ssml(content)

    def test_greater_than_escaped(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Greater-than (>) in text should be properly escaped."""
        text = "The score was > 100 points."
        output_path = tmp_dir / "greater_than.ssml"
        exporter.export(text=text, output_path=output_path, language="en")

        content = output_path.read_text(encoding="utf-8")
        _parse_ssml(content)

    def test_mixed_special_chars(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Text with multiple special characters should be fully escaped."""
        text = "A < B & B > C said the professor."
        output_path = tmp_dir / "mixed.ssml"
        exporter.export(text=text, output_path=output_path, language="en")

        content = output_path.read_text(encoding="utf-8")
        root = _parse_ssml(content)

        # Verify the text content round-trips correctly.
        full_text = _get_all_text(root)
        assert "<" in full_text or "&lt;" not in full_text
        assert "professor" in full_text

    def test_quotes_in_text(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Quotes in text should not break XML structure."""
        text = 'He said "hello" and she replied \'goodbye\'.'
        output_path = tmp_dir / "quotes.ssml"
        exporter.export(text=text, output_path=output_path, language="en")

        content = output_path.read_text(encoding="utf-8")
        _parse_ssml(content)


# ── Tests: default settings ───────────────────────────────────────────


class TestDefaultSettings:
    """Tests for _DefaultSSMLSettings values."""

    def test_default_paragraph_break(
        self, default_settings: _DefaultSSMLSettings
    ) -> None:
        """Default paragraph break should be 600ms."""
        assert default_settings.paragraph_break == "600ms"

    def test_default_scene_break(
        self, default_settings: _DefaultSSMLSettings
    ) -> None:
        """Default scene break should be 1000ms."""
        assert default_settings.scene_break == "1000ms"

    def test_default_dialog_pause(
        self, default_settings: _DefaultSSMLSettings
    ) -> None:
        """Default dialog pause should be 400ms."""
        assert default_settings.dialog_pause == "400ms"

    def test_default_dramatic_pause(
        self, default_settings: _DefaultSSMLSettings
    ) -> None:
        """Default dramatic pause should be 800ms."""
        assert default_settings.dramatic_pause == "800ms"

    def test_default_slow_for_dramatic(
        self, default_settings: _DefaultSSMLSettings
    ) -> None:
        """Default slow_for_dramatic should be True."""
        assert default_settings.slow_for_dramatic is True

    def test_default_emphasis_for_key_words(
        self, default_settings: _DefaultSSMLSettings
    ) -> None:
        """Default emphasis_for_key_words should be False."""
        assert default_settings.emphasis_for_key_words is False


# ── Tests: comprehensive integration ──────────────────────────────────


class TestComprehensiveExport:
    """Integration-style tests combining multiple features."""

    def test_full_story_segment(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """A realistic story segment should produce valid, structured SSML."""
        text = (
            "Im dunklen Wald stand ein altes Haus. "
            "Niemand wusste, wer darin wohnte.\n\n"
            "Eines Tages beschloss ein kleines M\u00e4dchen, "
            "die Wahrheit herauszufinden.\n\n"
            "Die T\u00fcr \u00f6ffnete sich langsam... von selbst.\n\n"
            "\u201eWer bist du?\u201c fragte eine Stimme aus der Dunkelheit."
        )
        root = _export_and_parse(exporter, text, tmp_dir, language="de")

        # Should have the correct language.
        xml_lang = root.get("{http://www.w3.org/XML/1998/namespace}lang")
        assert xml_lang == "de-DE"

        # Should have paragraphs.
        p_elements = root.findall("s:p", _NS)
        assert len(p_elements) >= 4

        # Should have scene breaks.
        break_elements = root.findall("s:break", _NS)
        scene_breaks = [
            b for b in break_elements
            if b.get("time") == "1000ms"
        ]
        assert len(scene_breaks) >= 1

        # Should have a dramatic pause for "...".
        dramatic_breaks = [
            b for b in root.findall(".//s:break", _NS)
            if b.get("time") == "800ms"
        ]
        assert len(dramatic_breaks) >= 1

        # Should have dialog pause for the German quote.
        dialog_breaks = [
            b for b in break_elements
            if b.get("time") == "400ms"
        ]
        assert len(dialog_breaks) >= 1

    def test_custom_settings_applied(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Custom settings should change break durations in output."""
        settings = _DefaultSSMLSettings()
        settings.scene_break = "2000ms"
        settings.dialog_pause = "700ms"
        settings.dramatic_pause = "1500ms"

        text = (
            "Part one.\n\n"
            "Part two... dramatic.\n\n"
            "\u201cHello!\u201d said someone."
        )
        root = _export_and_parse(
            exporter, text, tmp_dir, ssml_settings=settings
        )

        break_elements = root.findall("s:break", _NS)
        all_break_elements = root.findall(".//s:break", _NS)

        # Custom scene break duration.
        scene_breaks = [
            b for b in break_elements
            if b.get("time") == "2000ms"
        ]
        assert len(scene_breaks) >= 1

        # Custom dramatic pause.
        dramatic_breaks = [
            b for b in all_break_elements
            if b.get("time") == "1500ms"
        ]
        assert len(dramatic_breaks) >= 1

        # Custom dialog pause.
        dialog_breaks = [
            b for b in break_elements
            if b.get("time") == "700ms"
        ]
        assert len(dialog_breaks) >= 1

    def test_cyrillic_text_preserved(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Cyrillic text should be preserved in SSML output."""
        text = (
            "\u0412 \u0442\u0451\u043c\u043d\u043e\u043c \u043b\u0435\u0441\u0443 "
            "\u0441\u0442\u043e\u044f\u043b \u0441\u0442\u0430\u0440\u044b\u0439 "
            "\u0434\u043e\u043c. "
            "\u041d\u0438\u043a\u0442\u043e \u043d\u0435 \u0437\u043d\u0430\u043b, "
            "\u043a\u0442\u043e \u0432 \u043d\u0451\u043c \u0436\u0438\u0432\u0451\u0442."
        )
        root = _export_and_parse(exporter, text, tmp_dir, language="ru")

        full_text = _get_all_text(root)
        assert "\u043b\u0435\u0441\u0443" in full_text
        assert "\u0434\u043e\u043c" in full_text

    def test_turkish_text_preserved(
        self, exporter: SSMLExporter, tmp_dir: Path
    ) -> None:
        """Turkish special characters should be preserved."""
        text = (
            "Karanl\u0131k ormanda eski bir ev vard\u0131. "
            "Hi\u00e7 kimse i\u00e7inde kimin ya\u015fad\u0131\u011f\u0131n\u0131 bilmiyordu."
        )
        root = _export_and_parse(exporter, text, tmp_dir, language="tr")

        xml_lang = root.get("{http://www.w3.org/XML/1998/namespace}lang")
        assert xml_lang == "tr-TR"

        full_text = _get_all_text(root)
        assert "ormanda" in full_text
