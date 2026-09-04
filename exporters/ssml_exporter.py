"""SSML exporter for AI Story Generator Pro.

``SSMLExporter`` converts plain story text into valid SSML 1.1 markup
suitable for TTS (text-to-speech) engines.  Features:

- Paragraph splitting → ``<p>`` elements.
- Sentence splitting → ``<s>`` elements.
- Scene breaks (double newline) → ``<break>`` with configurable duration.
- Dialog detection → dialog pause.
- Dramatic pause detection (``...``) → ``<break>`` with dramatic duration.
- Optional ``<prosody rate="slow">`` wrapping for dramatic content.
- Language tag mapping (e.g. ``"de"`` → ``"de-DE"``).
- **Language-specific pause durations** — Romance languages get shorter
  pauses (faster syllable rate), Slavic and agglutinative languages get
  longer pauses (more breathing room).
- Valid XML output via ``lxml``.

Typical usage::

    exporter = SSMLExporter()
    path = exporter.export(
        text="Im dunklen Wald stand ein altes Haus. ...",
        output_path=Path("output/de/story.ssml"),
        language="de",
        ssml_settings=settings.ssml,
    )
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

from lxml import etree

from core.exceptions import ExportError
from utils.file_handler import write_file

if TYPE_CHECKING:
    from core.settings import SSMLSettings

logger = logging.getLogger(__name__)

# ── Language tag mapping ──────────────────────────────────────────────

LANG_TAG_MAP: dict[str, str] = {
    "en": "en-US",
    "ru": "ru-RU",
    "de": "de-DE",
    "fr": "fr-FR",
    "pt": "pt-BR",
    "it": "it-IT",
    "pl": "pl-PL",
    "uk": "uk-UA",
    "ro": "ro-RO",
    "tr": "tr-TR",
    "da": "da-DK",
}

# ── Language-specific pause durations ──────────────────────────────────
# Different languages have different speech rhythms.  These maps provide
# per-language overrides for default SSML pause durations.
# Values in milliseconds.  Languages not listed use the defaults.
#
# Rationale:
# - Romance languages (FR, IT, PT, RO): slightly shorter pauses due to
#   faster syllable rate and liaison/elision patterns.
# - Slavic languages (RU, PL, UK): moderate pauses; longer words need
#   more breathing room.
# - Turkish: SOV structure means listeners wait for the verb at the end
#   of the sentence; slightly longer sentence pauses aid comprehension.
# - Danish: stød (glottal stop) and soft d create unique rhythm;
#   slightly longer pauses between sentences.
# - German: compound words and Nebensatz create dense information;
#   standard pauses work well.

LANGUAGE_PAUSE_OVERRIDES: dict[str, dict[str, str]] = {
    "fr": {
        "paragraph_break": "500ms",
        "scene_break": "900ms",
        "dialog_pause": "350ms",
        "dramatic_pause": "750ms",
        "sentence_pause": "180ms",
    },
    "it": {
        "paragraph_break": "500ms",
        "scene_break": "900ms",
        "dialog_pause": "350ms",
        "dramatic_pause": "750ms",
        "sentence_pause": "180ms",
    },
    "pt": {
        "paragraph_break": "550ms",
        "scene_break": "950ms",
        "dialog_pause": "350ms",
        "dramatic_pause": "750ms",
        "sentence_pause": "180ms",
    },
    "ro": {
        "paragraph_break": "550ms",
        "scene_break": "950ms",
        "dialog_pause": "350ms",
        "dramatic_pause": "750ms",
        "sentence_pause": "190ms",
    },
    "ru": {
        "paragraph_break": "650ms",
        "scene_break": "1100ms",
        "dialog_pause": "450ms",
        "dramatic_pause": "900ms",
        "sentence_pause": "220ms",
    },
    "uk": {
        "paragraph_break": "650ms",
        "scene_break": "1100ms",
        "dialog_pause": "450ms",
        "dramatic_pause": "900ms",
        "sentence_pause": "220ms",
    },
    "pl": {
        "paragraph_break": "650ms",
        "scene_break": "1050ms",
        "dialog_pause": "400ms",
        "dramatic_pause": "850ms",
        "sentence_pause": "210ms",
    },
    "tr": {
        "paragraph_break": "650ms",
        "scene_break": "1050ms",
        "dialog_pause": "400ms",
        "dramatic_pause": "850ms",
        "sentence_pause": "230ms",
    },
    "da": {
        "paragraph_break": "650ms",
        "scene_break": "1050ms",
        "dialog_pause": "400ms",
        "dramatic_pause": "850ms",
        "sentence_pause": "220ms",
    },
    # en and de use default values (no override needed).
}


# ── Regex patterns ────────────────────────────────────────────────────

# Sentence boundary: period, exclamation, or question mark followed by
# whitespace or end of string.  Handles multiple punctuation marks like
# "?!" and "..." gracefully.
_PAT_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZА-ЯÀ-ÖÙ-Ýa-zà-öù-ý\u0400-\u04FF\u0100-\u024F"„«\'\u201C\u201E\u00AB])'
)

# Dialog detection: lines starting with common quote characters.
_PAT_DIALOG = re.compile(
    r'^[\u201C\u201E\u00AB\u2018\u201A"\u2039\u00BB—–-]',
)

# Dramatic pause: ellipsis (… or ...) possibly with surrounding
# whitespace.
_PAT_DRAMATIC_PAUSE = re.compile(r"\s*\.{3,}\s*|\s*…\s*")

# Scene break: two or more newlines (normalised as \n\n).
_PAT_SCENE_BREAK = re.compile(r"\n{2,}")

# SSML namespace.
_SSML_NS = "http://www.w3.org/2001/10/synthesis"


# ── Pause override helper ─────────────────────────────────────────────


def _apply_language_overrides(
    settings: Any,
    language: str,
) -> "_DefaultSSMLSettings":
    """Create a new settings object with language-specific pause overrides.

    If the language has entries in ``LANGUAGE_PAUSE_OVERRIDES``, a new
    ``_DefaultSSMLSettings`` is created with the base values from
    *settings* and the language-specific values applied on top.

    If there are no overrides for the language, the original *settings*
    object is returned unchanged.

    Args:
        settings: Original SSML settings object (``SSMLSettings`` or
            ``_DefaultSSMLSettings``).
        language: Two-letter language code.

    Returns:
        Settings object with overrides applied (or the original if no
        overrides exist for the language).
    """
    overrides = LANGUAGE_PAUSE_OVERRIDES.get(language)
    if not overrides:
        return settings

    result = _DefaultSSMLSettings()

    # Copy existing values from original settings.
    for attr in (
        "paragraph_break",
        "scene_break",
        "dialog_pause",
        "dramatic_pause",
        "sentence_pause",
        "slow_for_dramatic",
        "emphasis_for_key_words",
    ):
        if hasattr(settings, attr):
            setattr(result, attr, getattr(settings, attr))

    # Apply language-specific overrides.
    for key, value in overrides.items():
        if hasattr(result, key):
            setattr(result, key, value)

    return result


# ── Main exporter class ───────────────────────────────────────────────


class SSMLExporter:
    """Exports story text as valid SSML 1.1 markup.

    Converts plain text into structured SSML with paragraphs, sentences,
    pauses, and optional prosody.  Uses ``lxml`` for XML generation to
    guarantee well-formed output.

    Language-specific pause durations are applied automatically based
    on the ``language`` parameter.
    """

    def export(
        self,
        text: str,
        output_path: str | Path,
        language: str,
        ssml_settings: "SSMLSettings | None" = None,
    ) -> Path:
        """Convert text to SSML and write to a file.

        Args:
            text: The story text to convert.
            output_path: Target ``.ssml`` file path.
            language: Two-letter language code.
            ssml_settings: SSML configuration (break durations, prosody
                flags).  If ``None``, default settings are used.

        Returns:
            The ``Path`` of the written file.

        Raises:
            ExportError: If the text is empty, conversion fails, or the
                write fails.
        """
        if not text or not text.strip():
            raise ExportError(
                "Cannot export empty text to SSML",
                export_format="ssml",
            )

        # Resolve base settings.
        settings = ssml_settings if ssml_settings is not None else _DefaultSSMLSettings()

        # Apply language-specific pause overrides.
        settings = _apply_language_overrides(settings, language)

        overrides = LANGUAGE_PAUSE_OVERRIDES.get(language, {})
        if overrides:
            logger.debug(
                "SSMLExporter: applied %d pause overrides for language '%s'",
                len(overrides),
                language,
            )

        lang_tag = LANG_TAG_MAP.get(language, f"{language}-{language.upper()}")

        logger.info(
            "SSMLExporter: exporting %d chars (%s / %s) to %s",
            len(text),
            language,
            lang_tag,
            output_path,
        )

        try:
            ssml_str = self._build_ssml(text, lang_tag, settings)
        except Exception as exc:
            raise ExportError(
                f"Failed to build SSML for language '{language}': {exc}",
                export_format="ssml",
            ) from exc

        try:
            result_path = write_file(output_path, ssml_str)
        except OSError as exc:
            raise ExportError(
                f"Failed to write SSML file {output_path}: {exc}",
                export_format="ssml",
            ) from exc

        logger.info(
            "SSMLExporter: export complete: %s (%d chars SSML)",
            result_path,
            len(ssml_str),
        )
        return result_path

    # ── SSML construction ──────────────────────────────────────────

    def _build_ssml(
        self,
        text: str,
        lang_tag: str,
        settings: "SSMLSettings | _DefaultSSMLSettings",
    ) -> str:
        """Build a complete SSML document from plain text.

        Args:
            text: Story text.
            lang_tag: Full BCP-47 language tag (e.g. ``"de-DE"``).
            settings: SSML settings (with language overrides already
                applied).

        Returns:
            String of the complete SSML XML document.
        """
        nsmap = {None: _SSML_NS}
        speak = etree.Element(
            "speak",
            nsmap=nsmap,
            attrib={
                "version": "1.1",
                "{http://www.w3.org/XML/1998/namespace}lang": lang_tag,
            },
        )

        # Normalise line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on scene breaks (double newline).
        raw_paragraphs = _PAT_SCENE_BREAK.split(text)

        for para_idx, raw_para in enumerate(raw_paragraphs):
            raw_para = raw_para.strip()
            if not raw_para:
                continue

            # Insert scene break between paragraphs (not before the first).
            if para_idx > 0:
                scene_break = etree.SubElement(speak, "break")
                scene_break.set("time", settings.scene_break)

            # Each raw paragraph may contain single-newline-separated lines
            # that should be treated as logical paragraphs.
            sub_paragraphs = raw_para.split("\n")
            sub_paragraphs = [sp.strip() for sp in sub_paragraphs if sp.strip()]

            for sub_idx, sub_para in enumerate(sub_paragraphs):
                # Insert paragraph break between sub-paragraphs.
                if sub_idx > 0:
                    para_break = etree.SubElement(speak, "break")
                    para_break.set("time", settings.paragraph_break)

                is_dialog = bool(_PAT_DIALOG.match(sub_para))

                # Build <p> element.
                p_elem = etree.SubElement(speak, "p")

                # Split into sentences.
                sentences = _PAT_SENTENCE_SPLIT.split(sub_para)
                sentences = [s.strip() for s in sentences if s.strip()]

                if not sentences:
                    sentences = [sub_para]

                for sent in sentences:
                    self._add_sentence(
                        parent=p_elem,
                        sentence_text=sent,
                        is_dialog=is_dialog,
                        settings=settings,
                    )

                # Insert dialog pause after dialog paragraphs.
                if is_dialog:
                    dialog_break = etree.SubElement(speak, "break")
                    dialog_break.set("time", settings.dialog_pause)

        # Serialize to string.
        xml_bytes = etree.tostring(
            speak,
            xml_declaration=True,
            encoding="UTF-8",
            pretty_print=True,
        )
        ssml_str = xml_bytes.decode("utf-8")

        return ssml_str

    def _add_sentence(
        self,
        parent: etree._Element,
        sentence_text: str,
        is_dialog: bool,
        settings: "SSMLSettings | _DefaultSSMLSettings",
    ) -> None:
        """Add a sentence element to the parent, handling dramatic pauses.

        If the sentence contains an ellipsis (``...`` or ``…``), it is
        split around the pause, with a ``<break>`` element inserted.

        If ``slow_for_dramatic`` is enabled and the sentence looks
        dramatic, it is wrapped in ``<prosody rate="slow">``.

        Args:
            parent: Parent ``<p>`` element.
            sentence_text: The sentence text.
            is_dialog: Whether this sentence is part of dialog.
            settings: SSML settings.
        """
        # Check for dramatic pauses.
        has_dramatic_pause = bool(_PAT_DRAMATIC_PAUSE.search(sentence_text))

        # Determine if the sentence should use slow prosody.
        use_slow_prosody = (
            settings.slow_for_dramatic
            and has_dramatic_pause
            and not is_dialog
        )

        s_elem = etree.SubElement(parent, "s")

        if has_dramatic_pause:
            # Split sentence around dramatic pauses.
            parts = _PAT_DRAMATIC_PAUSE.split(sentence_text)
            parts = [p.strip() for p in parts if p.strip()]

            if use_slow_prosody and parts:
                prosody = etree.SubElement(s_elem, "prosody")
                prosody.set("rate", "slow")
                self._add_text_with_pauses(
                    prosody, parts, settings.dramatic_pause
                )
            elif parts:
                self._add_text_with_pauses(
                    s_elem, parts, settings.dramatic_pause
                )
            else:
                # Entire sentence is just ellipsis — add a break.
                br = etree.SubElement(s_elem, "break")
                br.set("time", settings.dramatic_pause)
        elif use_slow_prosody:
            prosody = etree.SubElement(s_elem, "prosody")
            prosody.set("rate", "slow")
            prosody.text = sentence_text
        else:
            s_elem.text = sentence_text

    @staticmethod
    def _add_text_with_pauses(
        parent: etree._Element,
        parts: list[str],
        pause_duration: str,
    ) -> None:
        """Add text parts with ``<break>`` elements between them.

        Args:
            parent: Parent XML element.
            parts: Text segments between dramatic pauses.
            pause_duration: Duration string for breaks.
        """
        for idx, part in enumerate(parts):
            if idx == 0:
                parent.text = part
            else:
                br = etree.SubElement(parent, "break")
                br.set("time", pause_duration)
                br.tail = part


class _DefaultSSMLSettings:
    """Default SSML settings used when no settings object is provided.

    Mirrors the default values from ``core.settings.SSMLSettings``
    without requiring the full settings infrastructure.
    """

    paragraph_break: str = "600ms"
    scene_break: str = "1000ms"
    dialog_pause: str = "400ms"
    dramatic_pause: str = "800ms"
    sentence_pause: str = "200ms"
    slow_for_dramatic: bool = True
    emphasis_for_key_words: bool = False
