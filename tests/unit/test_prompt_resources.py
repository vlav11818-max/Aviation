"""Unit tests for Phase 4 prompt resources.

Verifies:
- All 9 template files exist and are non-empty (including summary.txt).
- voiceover_rules.txt resource exists and is non-empty.
- retention_techniques.txt resource exists and is non-empty.
- All 11 cultural files exist, are non-empty, and contain CLEANUP_PATTERNS.
- All 5 structure JSON files exist, parse as valid JSON, and contain
  expected keys.
- No template has {variables} that are not in the documented variable list.

UPDATED: Added retention_techniques.txt resource tests, updated
voiceover_rules content checks for new EMOTIONAL EXPRESSION section.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# ── Resource root ──────────────────────────────────────────────────────

RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources" / "prompts"
TEMPLATES_DIR = RESOURCES_DIR / "templates"
CULTURAL_DIR = RESOURCES_DIR / "cultural"
STRUCTURES_DIR = RESOURCES_DIR / "structures"


# ── Expected inventory ─────────────────────────────────────────────────

EXPECTED_TEMPLATES: list[str] = [
    "concept",
    "outline",
    "section",
    "single_shot",
    "stitching",
    "evaluation",
    "revision",
    "adaptation",
    "summary",
]

EXPECTED_CULTURAL_LANGUAGES: list[str] = [
    "en", "ru", "de", "fr", "pt", "it", "pl", "uk", "ro", "tr", "da",
]

EXPECTED_STRUCTURES: list[str] = [
    "three_act", "hero_journey", "in_medias_res", "episodic", "circular",
]

# Documented variables per template.
# Note: concept.txt asks the LLM to output hook_type, hook_description,
# and retention_plan in the JSON response — but these are NOT template
# variables (they do not appear as {hook_type} in the template).
TEMPLATE_VARIABLES: dict[str, set[str]] = {
    "concept": {
        "topic", "language", "native_language_name", "tone", "perspective",
        "register", "pacing", "audience", "genres", "dialog_density",
        "target_words", "cultural_instructions", "retention_techniques",
    },
    "outline": {
        "concept_json", "structure_template", "structure_name",
        "target_words", "num_sections", "language", "cultural_instructions",
    },
    "section": {
        "story_bible", "full_outline", "section_plan", "section_index",
        "total_sections", "previous_summary", "last_500_words",
        "target_words_for_section", "language", "cultural_instructions",
        "voiceover_rules", "retention_techniques",
    },
    "single_shot": {
        "topic", "language", "target_words", "tone", "perspective",
        "register", "pacing", "audience", "genres", "dialog_density",
        "cultural_instructions", "voiceover_rules", "retention_techniques",
    },
    "stitching": {
        "end_of_previous_section", "start_of_next_section",
        "story_bible_summary", "language",
    },
    "evaluation": {
        "draft_text", "language", "target_words", "original_topic",
        "tone", "perspective", "voiceover_requirements",
    },
    "revision": {
        "draft_text", "evaluation_json", "critical_issues",
        "story_bible", "language", "voiceover_rules",
    },
    "adaptation": {
        "source_text", "source_language", "target_language",
        "adaptation_mode", "cultural_instructions", "adapt_names",
        "adapt_references", "adapt_units", "adapt_setting",
        "preserve_length", "voiceover_optimize", "voiceover_rules",
    },
    "summary": {
        "section_text", "story_bible_brief", "section_index",
        "total_sections", "language",
    },
}

# Required top-level keys for each structure JSON.
STRUCTURE_REQUIRED_KEYS: dict[str, set[str]] = {
    "three_act": {"name", "description", "acts"},
    "hero_journey": {"name", "description", "stages"},
    "in_medias_res": {"name", "description", "phases", "timelines"},
    "episodic": {"name", "description", "phases", "framework"},
    "circular": {"name", "description", "phases", "framework"},
}

_VARIABLE_RE = re.compile(r"(?<!\{)\{([a-z_]+)\}(?!\})")


# ── Helpers ────────────────────────────────────────────────────────────


def _read_text(path: Path) -> str:
    """Read a text file and return its content."""
    return path.read_text(encoding="utf-8")


def _extract_variables(template_text: str) -> set[str]:
    """Extract all {variable} placeholders from a template."""
    return set(_VARIABLE_RE.findall(template_text))


# ═══════════════════════════════════════════════════════════════════════
# Tests: Template Files
# ═══════════════════════════════════════════════════════════════════════


class TestTemplateFiles:
    """Verify all 9 prompt template files exist and are non-empty."""

    @pytest.mark.parametrize("template_name", EXPECTED_TEMPLATES)
    def test_template_exists(self, template_name: str) -> None:
        """Each expected template file must exist."""
        path = TEMPLATES_DIR / f"{template_name}.txt"
        assert path.exists(), f"Template file missing: {path}"

    @pytest.mark.parametrize("template_name", EXPECTED_TEMPLATES)
    def test_template_non_empty(self, template_name: str) -> None:
        """Each template file must contain content."""
        path = TEMPLATES_DIR / f"{template_name}.txt"
        if not path.exists():
            pytest.skip(f"Template file missing: {path}")
        content = _read_text(path)
        assert len(content.strip()) > 0, f"Template is empty: {path}"

    @pytest.mark.parametrize("template_name", EXPECTED_TEMPLATES)
    def test_template_variables_documented(self, template_name: str) -> None:
        """All {variables} in a template must be in the documented set."""
        path = TEMPLATES_DIR / f"{template_name}.txt"
        if not path.exists():
            pytest.skip(f"Template file missing: {path}")

        content = _read_text(path)
        found_vars = _extract_variables(content)
        documented_vars = TEMPLATE_VARIABLES.get(template_name, set())

        undocumented = found_vars - documented_vars
        assert not undocumented, (
            f"Template '{template_name}' has undocumented variables: "
            f"{undocumented}. "
            f"Found: {found_vars}. "
            f"Documented: {documented_vars}"
        )

    def test_template_count(self) -> None:
        """There should be exactly 9 template files."""
        assert len(EXPECTED_TEMPLATES) == 9


# ═══════════════════════════════════════════════════════════════════════
# Tests: Voiceover Rules Resource
# ═══════════════════════════════════════════════════════════════════════


class TestVoiceoverRulesFile:
    """Verify voiceover_rules.txt exists and is substantive."""

    def test_voiceover_rules_exists(self) -> None:
        """voiceover_rules.txt must exist in resources/prompts/."""
        path = RESOURCES_DIR / "voiceover_rules.txt"
        assert path.exists(), f"Voiceover rules file missing: {path}"

    def test_voiceover_rules_non_empty(self) -> None:
        """voiceover_rules.txt must have substantial content."""
        path = RESOURCES_DIR / "voiceover_rules.txt"
        if not path.exists():
            pytest.skip("voiceover_rules.txt missing")
        content = _read_text(path)
        assert len(content) >= 1000, (
            f"voiceover_rules.txt is too short ({len(content)} chars)"
        )

    def test_voiceover_rules_has_key_sections(self) -> None:
        """voiceover_rules.txt should cover sentence structure and punctuation."""
        path = RESOURCES_DIR / "voiceover_rules.txt"
        if not path.exists():
            pytest.skip("voiceover_rules.txt missing")
        content = _read_text(path).lower()
        assert "sentence" in content, "Missing sentence structure guidance"
        assert "punctuation" in content or "pause" in content, (
            "Missing punctuation/pause guidance"
        )

    def test_voiceover_rules_has_emotional_expression(self) -> None:
        """voiceover_rules.txt should cover emotional expression guidance."""
        path = RESOURCES_DIR / "voiceover_rules.txt"
        if not path.exists():
            pytest.skip("voiceover_rules.txt missing")
        content = _read_text(path).lower()
        assert "somatic" in content or "physical" in content, (
            "Missing somatic/physical reaction guidance"
        )
        assert "my heart sank" in content or "heart sank" in content, (
            "Missing stale phrase blacklist"
        )


# ═══════════════════════════════════════════════════════════════════════
# Tests: Retention Techniques Resource
# ═══════════════════════════════════════════════════════════════════════


class TestRetentionTechniquesFile:
    """Verify retention_techniques.txt exists and is substantive."""

    def test_retention_techniques_exists(self) -> None:
        """retention_techniques.txt must exist in resources/prompts/."""
        path = RESOURCES_DIR / "retention_techniques.txt"
        assert path.exists(), f"Retention techniques file missing: {path}"

    def test_retention_techniques_non_empty(self) -> None:
        """retention_techniques.txt must have substantial content."""
        path = RESOURCES_DIR / "retention_techniques.txt"
        if not path.exists():
            pytest.skip("retention_techniques.txt missing")
        content = _read_text(path)
        assert len(content) >= 3000, (
            f"retention_techniques.txt is too short ({len(content)} chars)"
        )

    def test_retention_techniques_has_key_sections(self) -> None:
        """retention_techniques.txt should cover all three technique groups."""
        path = RESOURCES_DIR / "retention_techniques.txt"
        if not path.exists():
            pytest.skip("retention_techniques.txt missing")
        content = _read_text(path)
        assert "INFORMATION MANAGEMENT" in content, (
            "Missing Group A — Information Management"
        )
        assert "EMOTIONAL ANCHORS" in content, (
            "Missing Group B — Emotional Anchors"
        )
        assert "STRUCTURAL DEVICES" in content, (
            "Missing Group C — Structural Devices"
        )

    def test_retention_techniques_has_hook_typology(self) -> None:
        """retention_techniques.txt should contain hook typology section."""
        path = RESOURCES_DIR / "retention_techniques.txt"
        if not path.exists():
            pytest.skip("retention_techniques.txt missing")
        content = _read_text(path)
        assert "HOOK TYPOLOGY" in content, "Missing hook typology section"

    def test_retention_techniques_has_blacklist(self) -> None:
        """retention_techniques.txt should contain storytelling blacklist."""
        path = RESOURCES_DIR / "retention_techniques.txt"
        if not path.exists():
            pytest.skip("retention_techniques.txt missing")
        content = _read_text(path)
        assert "BLACKLIST" in content, "Missing storytelling blacklist section"

    def test_retention_techniques_has_minimum_techniques(self) -> None:
        """retention_techniques.txt should define at least 15 techniques."""
        path = RESOURCES_DIR / "retention_techniques.txt"
        if not path.exists():
            pytest.skip("retention_techniques.txt missing")
        content = _read_text(path)
        # Count numbered technique headers (e.g., "1. OPEN LOOP", "18. MIRROR JUSTICE")
        technique_count = len(
            re.findall(r"^\d+\.\s+[A-Z][A-Z\s]+\(", content, re.MULTILINE)
        )
        assert technique_count >= 15, (
            f"Expected at least 15 retention techniques, found {technique_count}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Tests: Cultural Files
# ═══════════════════════════════════════════════════════════════════════


class TestCulturalFiles:
    """Verify all 11 cultural instruction files."""

    @pytest.mark.parametrize("language", EXPECTED_CULTURAL_LANGUAGES)
    def test_cultural_file_exists(self, language: str) -> None:
        """Each expected cultural file must exist."""
        path = CULTURAL_DIR / f"{language}.txt"
        assert path.exists(), f"Cultural file missing: {path}"

    @pytest.mark.parametrize("language", EXPECTED_CULTURAL_LANGUAGES)
    def test_cultural_file_non_empty(self, language: str) -> None:
        """Each cultural file must contain content."""
        path = CULTURAL_DIR / f"{language}.txt"
        if not path.exists():
            pytest.skip(f"Cultural file missing: {path}")
        content = _read_text(path)
        assert len(content.strip()) > 0, f"Cultural file is empty: {path}"

    @pytest.mark.parametrize("language", EXPECTED_CULTURAL_LANGUAGES)
    def test_cultural_file_has_minimum_length(self, language: str) -> None:
        """Each cultural file should have substantial content (at least 4000 chars)."""
        path = CULTURAL_DIR / f"{language}.txt"
        if not path.exists():
            pytest.skip(f"Cultural file missing: {path}")
        content = _read_text(path)
        assert len(content) >= 4000, (
            f"Cultural file is too short ({len(content)} chars): {path}"
        )

    @pytest.mark.parametrize("language", EXPECTED_CULTURAL_LANGUAGES)
    def test_cultural_file_has_cleanup_patterns(self, language: str) -> None:
        """Each cultural file should contain a CLEANUP_PATTERNS section."""
        path = CULTURAL_DIR / f"{language}.txt"
        if not path.exists():
            pytest.skip(f"Cultural file missing: {path}")
        content = _read_text(path)
        assert "CLEANUP_PATTERNS" in content, (
            f"Cultural file missing CLEANUP_PATTERNS section: {path}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Tests: Structure Files
# ═══════════════════════════════════════════════════════════════════════


class TestStructureFiles:
    """Verify all 5 structure JSON template files."""

    @pytest.mark.parametrize("structure_name", EXPECTED_STRUCTURES)
    def test_structure_exists(self, structure_name: str) -> None:
        """Each expected structure file must exist."""
        path = STRUCTURES_DIR / f"{structure_name}.json"
        assert path.exists(), f"Structure file missing: {path}"

    @pytest.mark.parametrize("structure_name", EXPECTED_STRUCTURES)
    def test_structure_valid_json(self, structure_name: str) -> None:
        """Each structure file must be valid JSON."""
        path = STRUCTURES_DIR / f"{structure_name}.json"
        if not path.exists():
            pytest.skip(f"Structure file missing: {path}")
        content = _read_text(path)
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Structure file is not valid JSON: {path}: {exc}")

    @pytest.mark.parametrize("structure_name", EXPECTED_STRUCTURES)
    def test_structure_has_required_keys(self, structure_name: str) -> None:
        """Each structure file must contain its required top-level keys."""
        path = STRUCTURES_DIR / f"{structure_name}.json"
        if not path.exists():
            pytest.skip(f"Structure file missing: {path}")
        content = _read_text(path)
        data = json.loads(content)
        required = STRUCTURE_REQUIRED_KEYS[structure_name]
        missing = required - set(data.keys())
        assert not missing, (
            f"Structure '{structure_name}' missing keys: {missing}"
        )

    @pytest.mark.parametrize("structure_name", EXPECTED_STRUCTURES)
    def test_structure_has_name(self, structure_name: str) -> None:
        """Each structure must have a non-empty 'name' field."""
        path = STRUCTURES_DIR / f"{structure_name}.json"
        if not path.exists():
            pytest.skip(f"Structure file missing: {path}")
        data = json.loads(_read_text(path))
        assert data.get("name"), (
            f"Structure '{structure_name}' has empty or missing 'name'"
        )

    @pytest.mark.parametrize("structure_name", EXPECTED_STRUCTURES)
    def test_structure_has_description(self, structure_name: str) -> None:
        """Each structure must have a non-empty 'description' field."""
        path = STRUCTURES_DIR / f"{structure_name}.json"
        if not path.exists():
            pytest.skip(f"Structure file missing: {path}")
        data = json.loads(_read_text(path))
        assert data.get("description"), (
            f"Structure '{structure_name}' has empty or missing 'description'"
        )
