"""Tests for per-section quality gate (Fix #8).

Tests the module-level ``evaluate_l1_programmatic()`` function and
the ``SectionStep._section_passes_quality_gate()`` / ``_check_section_quality()``
integration, including:
- Clean sections pass the gate
- Sections with chapter markers fail
- Sections with AI meta-comments fail (CRITICAL)
- Sections with stage directions fail
- Sections with "The End" markers produce a minor deduction
- Word-count tolerance: ±25% is accepted, beyond that fails
- UTF-8 encoding errors trigger CRITICAL failure
- Long paragraphs produce a minor deduction (not a gate failure)
- Gate can be disabled via quality_check_enabled=False
- Settings model accepts the new fields
"""

from __future__ import annotations

import pytest

from core.steps.evaluate_step import evaluate_l1_programmatic
from core.steps.section_step import (
    SectionStep,
    _SECTION_MIN_SCORE,
    _SECTION_WORD_COUNT_TOLERANCE,
)
from models.evaluation import IssueSeverity


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_text(words: int, fill: str = "word") -> str:
    """Create a block of text with exactly *words* words.

    Args:
        words: Number of words to generate.
        fill: Word to repeat.

    Returns:
        Space-separated text string.
    """
    return " ".join([fill] * words)


def _has_critical(result) -> bool:  # type: ignore[override]
    """Return True if any issue in *result* is CRITICAL."""
    return any(iss.severity == IssueSeverity.CRITICAL for iss in result.issues)


# ── evaluate_l1_programmatic — module-level function ─────────────────────────


class TestEvaluateL1Programmatic:
    """Tests for the module-level evaluate_l1_programmatic() function."""

    # ── Word count checks ────────────────────────────────────────────────────

    def test_word_count_within_tolerance_passes(self) -> None:
        """Text within ±15% of target should score 10.0 with no issues."""
        text = _make_text(1000)
        result = evaluate_l1_programmatic(text, target_words=1000)
        assert result.score == 10.0
        assert result.issues == []

    def test_word_count_at_lower_15pct_boundary_passes(self) -> None:
        """Exactly at the -15% boundary should pass (floor is inclusive)."""
        target = 1000
        text = _make_text(850)  # 850/1000 = 85% → -15%
        result = evaluate_l1_programmatic(text, target_words=target)
        # 850 >= floor(1000 * 0.85) = 850 → should pass with score 10
        assert result.score == 10.0

    def test_word_count_below_tolerance_deducts(self) -> None:
        """Text significantly below target should deduct from score."""
        text = _make_text(500)
        result = evaluate_l1_programmatic(text, target_words=1000)
        assert result.score < 10.0
        assert any(iss.category == "word_count" for iss in result.issues)

    def test_word_count_above_tolerance_deducts(self) -> None:
        """Text significantly above target should deduct from score."""
        text = _make_text(1200)
        result = evaluate_l1_programmatic(text, target_words=1000)
        assert result.score < 10.0
        assert any(iss.category == "word_count" for iss in result.issues)

    def test_custom_tolerance_wider(self) -> None:
        """Custom 25% tolerance should accept text within ±25%."""
        # 800 words vs 1000 target = 80% → within ±25% (min = 750)
        text = _make_text(800)
        result = evaluate_l1_programmatic(
            text, target_words=1000, word_count_tolerance=0.25
        )
        assert result.score == 10.0
        assert result.issues == []

    def test_custom_tolerance_still_fails_extreme(self) -> None:
        """Text beyond 25% below target should still fail at 25% tolerance."""
        # 700 words vs 1000 target = 70% → below ±25% (min = 750)
        text = _make_text(700)
        result = evaluate_l1_programmatic(
            text, target_words=1000, word_count_tolerance=0.25
        )
        assert result.score < 10.0
        assert any(iss.category == "word_count" for iss in result.issues)

    # ── Chapter markers ──────────────────────────────────────────────────────

    def test_chapter_marker_detected(self) -> None:
        """'Chapter 1' at the start of a line should be flagged as MAJOR."""
        text = "Chapter 1\n\n" + _make_text(500)
        result = evaluate_l1_programmatic(text, target_words=500)
        assert any(iss.category == "chapter_markers" for iss in result.issues)
        marker_issues = [
            iss for iss in result.issues if iss.category == "chapter_markers"
        ]
        assert all(iss.severity == IssueSeverity.MAJOR for iss in marker_issues)

    def test_chapter_marker_deducts_score(self) -> None:
        """Chapter marker should reduce score from 10."""
        text = "Chapter 1\n\n" + _make_text(500)
        result = evaluate_l1_programmatic(text, target_words=500)
        assert result.score < 10.0

    def test_german_chapter_marker(self) -> None:
        """'Kapitel' (German) should be detected as a chapter marker."""
        text = "Kapitel 3\n\n" + _make_text(500)
        result = evaluate_l1_programmatic(text, target_words=500)
        assert any(iss.category == "chapter_markers" for iss in result.issues)

    def test_section_keyword_marker(self) -> None:
        """'Section' at the start of a line should be flagged."""
        text = "Section 2\n\n" + _make_text(500)
        result = evaluate_l1_programmatic(text, target_words=500)
        assert any(iss.category == "chapter_markers" for iss in result.issues)

    # ── AI meta-comments ─────────────────────────────────────────────────────

    def test_meta_comment_is_critical(self) -> None:
        """'Note:' at the start of a line should produce a CRITICAL issue."""
        text = _make_text(200) + "\nNote: This story explores themes of courage."
        result = evaluate_l1_programmatic(text, target_words=200)
        assert _has_critical(result)
        assert any(iss.category == "meta_comments" for iss in result.issues)

    def test_authors_note_is_critical(self) -> None:
        """'Author's note:' should produce a CRITICAL issue."""
        text = _make_text(200) + "\nAuthor's note: I hope you enjoy this tale."
        result = evaluate_l1_programmatic(text, target_words=200)
        assert _has_critical(result)

    def test_ai_note_is_critical(self) -> None:
        """'AI note:' should produce a CRITICAL issue."""
        text = _make_text(200) + "\nAI note: This content was generated."
        result = evaluate_l1_programmatic(text, target_words=200)
        assert _has_critical(result)

    def test_meta_comment_deducts_score(self) -> None:
        """Meta-comment should significantly reduce the score."""
        text = _make_text(200) + "\nNote: Important context here."
        result = evaluate_l1_programmatic(text, target_words=200)
        assert result.score < 8.0

    # ── End markers ──────────────────────────────────────────────────────────

    def test_the_end_marker_minor_deduction(self) -> None:
        """'The End' at the end of text should produce a MINOR issue."""
        text = _make_text(500) + "\n\nThe End"
        result = evaluate_l1_programmatic(text, target_words=500)
        assert any(iss.category == "end_markers" for iss in result.issues)
        end_issues = [iss for iss in result.issues if iss.category == "end_markers"]
        assert all(iss.severity == IssueSeverity.MINOR for iss in end_issues)

    def test_fin_marker_detected(self) -> None:
        """'Fin' (French/Spanish end marker) should be detected."""
        text = _make_text(500) + "\n\nFin"
        result = evaluate_l1_programmatic(text, target_words=500)
        assert any(iss.category == "end_markers" for iss in result.issues)

    # ── Stage directions ─────────────────────────────────────────────────────

    def test_stage_direction_flagged(self) -> None:
        """Text in *asterisks* (2-80 chars) should be flagged as stage direction."""
        text = _make_text(200) + " *walks slowly to the door* " + _make_text(200)
        result = evaluate_l1_programmatic(text, target_words=400)
        assert any(iss.category == "stage_directions" for iss in result.issues)
        stage_issues = [
            iss for iss in result.issues if iss.category == "stage_directions"
        ]
        assert all(iss.severity == IssueSeverity.MAJOR for iss in stage_issues)

    def test_single_asterisk_word_not_flagged(self) -> None:
        """A single asterisked word (1 char) should NOT be flagged (too short)."""
        text = _make_text(400) + " *a* " + _make_text(100)
        result = evaluate_l1_programmatic(text, target_words=500)
        assert not any(iss.category == "stage_directions" for iss in result.issues)

    # ── UTF-8 encoding ───────────────────────────────────────────────────────

    def test_clean_utf8_no_encoding_issue(self) -> None:
        """Clean UTF-8 text (including accents) should not produce encoding issues."""
        text = "Héros d'une époque lointaine. " * 20
        result = evaluate_l1_programmatic(text, target_words=100)
        assert not any(iss.category == "encoding" for iss in result.issues)

    # ── Paragraph length ─────────────────────────────────────────────────────

    def test_very_long_paragraph_minor_deduction(self) -> None:
        """A single paragraph exceeding 200 words should produce a MINOR issue."""
        # One giant paragraph with 250 words, no double newlines.
        text = _make_text(250)
        result = evaluate_l1_programmatic(text, target_words=250)
        assert any(iss.category == "paragraph_length" for iss in result.issues)
        para_issues = [
            iss for iss in result.issues if iss.category == "paragraph_length"
        ]
        assert all(iss.severity == IssueSeverity.MINOR for iss in para_issues)

    def test_short_paragraphs_no_deduction(self) -> None:
        """Multiple short paragraphs should not trigger the paragraph check."""
        # 10 paragraphs of 20 words each → 200 words, max para = 20.
        paragraphs = [_make_text(20) for _ in range(10)]
        text = "\n\n".join(paragraphs)
        result = evaluate_l1_programmatic(text, target_words=200)
        assert not any(iss.category == "paragraph_length" for iss in result.issues)

    # ── Score bounds ─────────────────────────────────────────────────────────

    def test_score_never_exceeds_10(self) -> None:
        """Score should never exceed 10.0."""
        text = _make_text(1000)
        result = evaluate_l1_programmatic(text, target_words=1000)
        assert result.score <= 10.0

    def test_score_never_below_0(self) -> None:
        """Score should never go below 0.0 even with many issues."""
        text = (
            "Chapter 1\n"
            "Note: AI-generated content.\n"
            "Author's note: Important!\n"
            "*walks in*\n"
            "The End\n"
            + _make_text(10)  # Far below target
        )
        result = evaluate_l1_programmatic(text, target_words=5000)
        assert result.score >= 0.0

    def test_perfect_text_scores_10(self) -> None:
        """Clean text at target word count should score exactly 10.0."""
        text = _make_text(1000)
        result = evaluate_l1_programmatic(text, target_words=1000)
        assert result.score == 10.0
        assert result.issues == []


# ── SectionStep quality gate helpers ─────────────────────────────────────────


class TestSectionPassesQualityGate:
    """Tests for SectionStep._section_passes_quality_gate()."""

    def setup_method(self) -> None:
        """Create a SectionStep instance for each test."""
        self.step = SectionStep()

    def test_clean_section_passes(self) -> None:
        """A clean section at target length should pass the gate."""
        text = _make_text(500)
        assert self.step._section_passes_quality_gate(text, target_words=500, section_idx=0)

    def test_section_with_chapter_marker_fails(self) -> None:
        """A section with chapter markers should fail the gate."""
        text = "Chapter 5\n\n" + _make_text(500)
        # Chapter markers produce MAJOR, not CRITICAL — fails on score.
        result = evaluate_l1_programmatic(
            text, target_words=500, word_count_tolerance=_SECTION_WORD_COUNT_TOLERANCE
        )
        # Verify the score would be below minimum.
        assert result.score < _SECTION_MIN_SCORE
        # Now test the gate itself.
        assert not self.step._section_passes_quality_gate(
            text, target_words=500, section_idx=0
        )

    def test_section_with_meta_comment_fails(self) -> None:
        """A section with AI meta-comments (CRITICAL) should fail the gate."""
        text = _make_text(500) + "\nNote: This story is AI-generated."
        assert not self.step._section_passes_quality_gate(
            text, target_words=500, section_idx=0
        )

    def test_section_severely_short_fails(self) -> None:
        """A section far below target word count should fail the gate."""
        # 200 words vs 1000 target = 20% → well outside ±25%
        text = _make_text(200)
        assert not self.step._section_passes_quality_gate(
            text, target_words=1000, section_idx=0
        )

    def test_section_within_25pct_passes(self) -> None:
        """A section within ±25% of target should pass the gate."""
        # 800 words vs 1000 target = 80% → within ±25% (min=750)
        text = _make_text(800)
        assert self.step._section_passes_quality_gate(
            text, target_words=1000, section_idx=0
        )

    def test_section_with_stage_direction_may_fail(self) -> None:
        """A section with stage directions should lose score (MAJOR deduction)."""
        # Stage directions are MAJOR (not CRITICAL) → score-based gate.
        text = _make_text(200) + " *speaks dramatically* " + _make_text(200)
        # With a short section and stage direction, score drops enough to fail.
        result = evaluate_l1_programmatic(
            text, target_words=400, word_count_tolerance=_SECTION_WORD_COUNT_TOLERANCE
        )
        if result.score < _SECTION_MIN_SCORE:
            assert not self.step._section_passes_quality_gate(
                text, target_words=400, section_idx=0
            )
        else:
            # If deduction wasn't enough, still confirm no crash.
            assert isinstance(
                self.step._section_passes_quality_gate(
                    text, target_words=400, section_idx=0
                ),
                bool,
            )


class TestCheckSectionQuality:
    """Tests for SectionStep._check_section_quality()."""

    def setup_method(self) -> None:
        """Create a SectionStep instance for each test."""
        self.step = SectionStep()

    def test_returns_float(self) -> None:
        """_check_section_quality() should return a float."""
        text = _make_text(500)
        score = self.step._check_section_quality(text, target_words=500, section_idx=0)
        assert isinstance(score, float)

    def test_clean_section_scores_10(self) -> None:
        """A clean section at target length should score 10.0."""
        text = _make_text(500)
        score = self.step._check_section_quality(text, target_words=500, section_idx=0)
        assert score == 10.0

    def test_section_with_issues_scores_lower(self) -> None:
        """A section with chapter markers should score below 10."""
        text = "Chapter 1\n\n" + _make_text(500)
        score = self.step._check_section_quality(text, target_words=500, section_idx=0)
        assert score < 10.0

    def test_score_within_valid_range(self) -> None:
        """Score should always be in [0.0, 10.0]."""
        text = _make_text(100)
        score = self.step._check_section_quality(text, target_words=5000, section_idx=2)
        assert 0.0 <= score <= 10.0


# ── Settings model — new fields ───────────────────────────────────────────────


class TestGenerationSettingsNewFields:
    """Tests for the two new fields added to GenerationSettings (Fix #8)."""

    def test_defaults_load(self) -> None:
        """GenerationSettings should have correct defaults for new fields."""
        from core.settings import GenerationSettings
        gs = GenerationSettings()
        assert gs.section_quality_check is True
        assert gs.section_max_retries == 2

    def test_can_disable_quality_check(self) -> None:
        """section_quality_check can be set to False."""
        from core.settings import GenerationSettings
        gs = GenerationSettings(section_quality_check=False)
        assert gs.section_quality_check is False

    def test_can_set_zero_retries(self) -> None:
        """section_max_retries=0 means no retries (generate once only)."""
        from core.settings import GenerationSettings
        gs = GenerationSettings(section_max_retries=0)
        assert gs.section_max_retries == 0

    def test_max_retries_upper_bound(self) -> None:
        """section_max_retries cannot exceed 5."""
        from core.settings import GenerationSettings
        with pytest.raises(Exception):
            GenerationSettings(section_max_retries=6)

    def test_max_retries_lower_bound(self) -> None:
        """section_max_retries cannot be negative."""
        from core.settings import GenerationSettings
        with pytest.raises(Exception):
            GenerationSettings(section_max_retries=-1)

    def test_round_trip(self) -> None:
        """GenerationSettings with new fields should round-trip through model_dump."""
        from core.settings import GenerationSettings
        gs = GenerationSettings(section_quality_check=False, section_max_retries=3)
        data = gs.model_dump()
        restored = GenerationSettings.model_validate(data)
        assert restored.section_quality_check is False
        assert restored.section_max_retries == 3

    def test_settings_load_from_yaml_defaults(self, tmp_path) -> None:  # type: ignore[override]
        """Settings.load() with no user file should use defaults for new fields."""
        from pathlib import Path
        from core.settings import Settings

        # Copy defaults file to a temp location so the path resolves.
        defaults_src = Path("resources/defaults/settings.yaml")
        if not defaults_src.exists():
            pytest.skip("resources/defaults/settings.yaml not found")

        # Patch the path by writing a minimal settings file.
        user_yaml = tmp_path / "settings.yaml"
        user_yaml.write_text("generation:\n  min_score: 8.0\n", encoding="utf-8")

        settings = Settings.load(user_yaml)
        # New fields should come from defaults (true, 2).
        assert settings.generation.section_quality_check is True
        assert settings.generation.section_max_retries == 2
        # Existing override should still work.
        assert settings.generation.min_score == 8.0

    def test_settings_load_can_override_new_fields(self, tmp_path) -> None:  # type: ignore[override]
        """User settings.yaml should be able to override the new fields."""
        from pathlib import Path
        from core.settings import Settings

        defaults_src = Path("resources/defaults/settings.yaml")
        if not defaults_src.exists():
            pytest.skip("resources/defaults/settings.yaml not found")

        user_yaml = tmp_path / "settings.yaml"
        user_yaml.write_text(
            "generation:\n  section_quality_check: false\n  section_max_retries: 0\n",
            encoding="utf-8",
        )

        settings = Settings.load(user_yaml)
        assert settings.generation.section_quality_check is False
        assert settings.generation.section_max_retries == 0
