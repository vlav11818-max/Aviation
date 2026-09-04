"""Unit tests for core.prompt_manager.PromptManager.

Tests: render template with valid variables, missing template raises
PromptTemplateError, cultural instructions loading for all 11
languages (file existence check), structure template loading, variable
substitution correctness, template caching, and curly-brace injection
safety (topics containing literal { } characters must not crash).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.exceptions import PromptTemplateError
from core.prompt_manager import PromptManager


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def resources_dir(tmp_path: Path) -> Path:
    """Create a minimal resources directory with test templates."""
    templates_dir = tmp_path / "prompts" / "templates"
    templates_dir.mkdir(parents=True)
    cultural_dir = tmp_path / "prompts" / "cultural"
    cultural_dir.mkdir(parents=True)
    structures_dir = tmp_path / "prompts" / "structures"
    structures_dir.mkdir(parents=True)

    # Primary test template — uses topic, tone, target_words, language,
    # cultural_instructions placeholders.
    (templates_dir / "concept.txt").write_text(
        "Generate a story about {topic} in {language}.\n"
        "Tone: {tone}. Target: {target_words} words.\n"
        "{cultural_instructions}",
        encoding="utf-8",
    )

    # Minimal single_shot template for injection tests.
    (templates_dir / "single_shot.txt").write_text(
        "Write a story.\nTopic: {topic}\nOutline: {outline_context}",
        encoding="utf-8",
    )

    # Cultural files.
    (cultural_dir / "en.txt").write_text(
        "Write in standard English.",
        encoding="utf-8",
    )
    (cultural_dir / "de.txt").write_text(
        "Schreibe auf Deutsch.",
        encoding="utf-8",
    )

    # Structure JSON.
    (structures_dir / "three_act.json").write_text(
        '{"name": "Three Act", "acts": ["setup", "confrontation", "resolution"]}',
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def pm(resources_dir: Path) -> PromptManager:
    """PromptManager pointed at the test resources directory."""
    return PromptManager(resources_dir=resources_dir)


# ── Tests: render ─────────────────────────────────────────────────────────────


class TestRender:
    """Tests for PromptManager.render."""

    def test_basic_render(self, pm: PromptManager) -> None:
        """Render should substitute variables correctly."""
        result = pm.render(
            "concept",
            language="en",
            topic="Ancient Temple",
            tone="dramatic",
            target_words="3000",
        )
        assert "Ancient Temple" in result
        assert "dramatic" in result
        assert "3000" in result

    def test_cultural_instructions_injected(self, pm: PromptManager) -> None:
        """Render should inject cultural instructions for the language."""
        result = pm.render(
            "concept",
            language="en",
            topic="Test",
            tone="calm",
            target_words="1000",
        )
        assert "standard English" in result

    def test_german_cultural_instructions(self, pm: PromptManager) -> None:
        """Render should inject German cultural instructions."""
        result = pm.render(
            "concept",
            language="de",
            topic="Test",
            tone="calm",
            target_words="1000",
        )
        assert "Deutsch" in result

    def test_unmatched_variables_preserved(self, pm: PromptManager) -> None:
        """Variables not provided should remain as {placeholder}."""
        result = pm.render(
            "concept",
            language="en",
            topic="Test",
            # tone is not provided — should remain as {tone}
        )
        assert "{tone}" in result

    def test_missing_template_raises(self, pm: PromptManager) -> None:
        """Rendering a non-existent template should raise PromptTemplateError."""
        with pytest.raises(PromptTemplateError):
            pm.render("nonexistent_template", language="en")

    def test_language_injected_automatically(self, pm: PromptManager) -> None:
        """Language should be injected automatically if not passed explicitly."""
        result = pm.render(
            "concept",
            language="en",
            topic="Test",
            tone="dramatic",
            target_words="1000",
        )
        assert "en" in result


# ── Tests: curly-brace injection safety ───────────────────────────────────────


class TestCurlyBraceInjectionSafety:
    """Tests that user-supplied values containing { } do not crash render.

    This validates Fix #2: topics (and other user input) that contain
    literal curly braces must be safely escaped before format_map
    processes the template.
    """

    def test_topic_with_curly_braces_does_not_crash(
        self, pm: PromptManager
    ) -> None:
        """A topic containing {braces} must not raise any exception."""
        result = pm.render(
            "concept",
            language="en",
            topic="{The Secret} of the ancient world",
            tone="dramatic",
            target_words="3000",
        )
        # The topic text must appear verbatim in the output.
        assert "{The Secret} of the ancient world" in result

    def test_topic_with_format_specifier_does_not_crash(
        self, pm: PromptManager
    ) -> None:
        """A topic containing {0} (positional format) must not crash."""
        result = pm.render(
            "concept",
            language="en",
            topic="The story of {0} heroes",
            tone="epic",
            target_words="2000",
        )
        assert "The story of {0} heroes" in result

    def test_topic_with_repr_specifier_does_not_crash(
        self, pm: PromptManager
    ) -> None:
        """A topic containing {!r} (conversion flag) must not crash."""
        result = pm.render(
            "concept",
            language="en",
            topic="Mystery {!r} solved",
            tone="suspense",
            target_words="1500",
        )
        assert "Mystery {!r} solved" in result

    def test_topic_with_nested_braces_does_not_crash(
        self, pm: PromptManager
    ) -> None:
        """A topic with nested braces {{like this}} must not crash."""
        result = pm.render(
            "concept",
            language="en",
            topic="AI & {Machine Learning} revolution",
            tone="educational",
            target_words="2000",
        )
        assert "AI & {Machine Learning} revolution" in result

    def test_multiple_variables_with_braces_do_not_crash(
        self, pm: PromptManager
    ) -> None:
        """Multiple user-supplied values with braces must all be safe."""
        result = pm.render(
            "single_shot",
            language="en",
            topic="{The Secret}",
            outline_context="1. {Setup}\n2. {Conflict}\n3. {Resolution}",
        )
        assert "{The Secret}" in result
        assert "{Setup}" in result
        assert "{Conflict}" in result
        assert "{Resolution}" in result

    def test_topic_with_only_braces_does_not_crash(
        self, pm: PromptManager
    ) -> None:
        """A topic that is only braces like {} must not crash."""
        # Empty braces would cause IndexError in format_map without the fix.
        result = pm.render(
            "concept",
            language="en",
            topic="{}",
            tone="dramatic",
            target_words="1000",
        )
        assert "{}" in result

    def test_integer_variable_not_affected(self, pm: PromptManager) -> None:
        """Non-string variables (int, float) should pass through unchanged."""
        result = pm.render(
            "concept",
            language="en",
            topic="Normal topic",
            tone="dramatic",
            target_words=3000,  # integer, not string
        )
        assert "3000" in result

    def test_cultural_instructions_not_escaped(
        self, pm: PromptManager
    ) -> None:
        """Cultural instructions (trusted) must NOT have braces escaped."""
        # Inject a cultural instructions value that contains a brace pattern.
        # It should appear in the output as-is (not double-escaped).
        result = pm.render(
            "concept",
            language="en",
            topic="Normal topic",
            tone="dramatic",
            target_words="2000",
            cultural_instructions="Use natural language (no {brackets}).",
        )
        assert "no {brackets}" in result


# ── Tests: get_cultural_instructions ─────────────────────────────────────────


class TestCulturalInstructions:
    """Tests for PromptManager.get_cultural_instructions."""

    def test_english_instructions(self, pm: PromptManager) -> None:
        """Should return English cultural instructions."""
        result = pm.get_cultural_instructions("en")
        assert "standard English" in result

    def test_german_instructions(self, pm: PromptManager) -> None:
        """Should return German cultural instructions."""
        result = pm.get_cultural_instructions("de")
        assert "Deutsch" in result

    def test_missing_language_returns_empty(self, pm: PromptManager) -> None:
        """Missing cultural file should return empty string, not raise."""
        result = pm.get_cultural_instructions("zz")
        assert result == ""


# ── Tests: get_structure_template ────────────────────────────────────────────


class TestStructureTemplate:
    """Tests for PromptManager.get_structure_template."""

    def test_loads_three_act(self, pm: PromptManager) -> None:
        """Should load and parse the three_act structure."""
        result = pm.get_structure_template("three_act")
        assert isinstance(result, dict)
        assert result["name"] == "Three Act"
        assert "acts" in result

    def test_missing_structure_raises(self, pm: PromptManager) -> None:
        """Missing structure should raise PromptTemplateError."""
        with pytest.raises(PromptTemplateError):
            pm.get_structure_template("nonexistent_structure")

    def test_structure_returns_dict(self, pm: PromptManager) -> None:
        """Structure template should always be a dict."""
        result = pm.get_structure_template("three_act")
        assert isinstance(result, dict)


# ── Tests: caching ────────────────────────────────────────────────────────────


class TestCaching:
    """Tests for template caching behaviour."""

    def test_template_cached_returns_same_result(
        self, pm: PromptManager
    ) -> None:
        """Second render of same template should return consistent results."""
        r1 = pm.render(
            "concept", language="en", topic="A", tone="x", target_words="1"
        )
        r2 = pm.render(
            "concept", language="en", topic="A", tone="x", target_words="1"
        )
        assert r1 == r2

    def test_template_cached_different_variables(
        self, pm: PromptManager
    ) -> None:
        """Different variables with the same cached template produce different output."""
        r1 = pm.render(
            "concept", language="en", topic="Alpha", tone="x", target_words="1"
        )
        r2 = pm.render(
            "concept", language="en", topic="Beta", tone="x", target_words="1"
        )
        assert "Alpha" in r1
        assert "Beta" in r2
        assert r1 != r2

    def test_cultural_cached(self, pm: PromptManager) -> None:
        """Cultural instructions should be identical on repeated calls."""
        r1 = pm.get_cultural_instructions("en")
        r2 = pm.get_cultural_instructions("en")
        assert r1 == r2

    def test_structure_cached(self, pm: PromptManager) -> None:
        """Structure template should be identical on repeated calls."""
        r1 = pm.get_structure_template("three_act")
        r2 = pm.get_structure_template("three_act")
        assert r1 == r2

    def test_clear_cache_works(self, pm: PromptManager) -> None:
        """clear_cache() should reset all caches without error."""
        pm.render("concept", language="en", topic="A", tone="x", target_words="1")
        pm.get_cultural_instructions("en")
        pm.get_structure_template("three_act")
        pm.clear_cache()
        # After clearing, templates should still be loadable.
        result = pm.render(
            "concept", language="en", topic="A", tone="x", target_words="1"
        )
        assert "A" in result


# ── Tests: cultural files for all 11 languages (real resources) ───────────────


class TestAllLanguagesCulturalFiles:
    """Check that real cultural files exist for all 11 supported languages.

    These tests only run if the resources directory exists (i.e. Phase 4
    has been completed).  They are skipped gracefully otherwise.
    """

    _SUPPORTED_LANGUAGES = [
        "en", "ru", "de", "fr", "pt", "it", "pl", "uk", "ro", "tr", "da"
    ]

    @pytest.mark.parametrize("lang", _SUPPORTED_LANGUAGES)
    def test_cultural_file_exists(self, lang: str) -> None:
        """Real cultural file must exist for each supported language."""
        resources = Path("resources")
        if not resources.exists():
            pytest.skip("resources/ directory not present (Phase 4 not complete)")

        cultural_path = resources / "prompts" / "cultural" / f"{lang}.txt"
        assert cultural_path.exists(), (
            f"Cultural file missing for language '{lang}': {cultural_path}"
        )

    @pytest.mark.parametrize("lang", _SUPPORTED_LANGUAGES)
    def test_cultural_file_non_empty(self, lang: str) -> None:
        """Real cultural file must contain content."""
        resources = Path("resources")
        if not resources.exists():
            pytest.skip("resources/ directory not present (Phase 4 not complete)")

        cultural_path = resources / "prompts" / "cultural" / f"{lang}.txt"
        if not cultural_path.exists():
            pytest.skip(f"Cultural file missing: {cultural_path}")

        content = cultural_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, (
            f"Cultural file is empty for language '{lang}'"
        )
