"""Unit tests for all data models.

Tests creation, validation, serialization to dict/JSON, deserialization,
default values, required field enforcement, and nested model handling.
"""

from __future__ import annotations

import json

import pytest

from models.config import (
    APIConfig,
    APIProvider,
    Audience,
    DialogDensity,
    GenerationConfig,
    LANGUAGES,
    Pacing,
    Perspective,
    PROVIDER_CONFIG,
    Register,
    StructureType,
    SUPPORTED_LANGUAGE_CODES,
    Tone,
)
from models.evaluation import (
    EvaluationIssue,
    EvaluationLevel,
    EvaluationResult,
    IssueSeverity,
    LevelResult,
)
from models.metadata import StoryMetadata
from models.outline import Outline, OutlineSection
from models.section import Section
from models.state import PipelineState, PipelineStatus
from models.story import Story, StoryStatus
from models.story_bible import Character, Setting, StoryBible


# ════════════════════════════════════════════════════════════════════════
# GenerationConfig
# ════════════════════════════════════════════════════════════════════════


class TestGenerationConfig:
    """Tests for the GenerationConfig model."""

    def test_defaults(self) -> None:
        """All defaults should populate without arguments."""
        gc = GenerationConfig()
        assert gc.tone == Tone.DRAMATIC_CINEMATIC
        assert gc.perspective == Perspective.THIRD_PERSON
        assert gc.target_words == 3000
        assert gc.min_score == 9.0
        assert gc.voiceover_optimized is True

    def test_round_trip(self, sample_generation_config: GenerationConfig) -> None:
        """model_dump → model_validate round-trip should be lossless."""
        data = sample_generation_config.model_dump()
        restored = GenerationConfig.model_validate(data)
        assert restored == sample_generation_config

    def test_json_round_trip(self, sample_generation_config: GenerationConfig) -> None:
        """JSON serialization round-trip should be lossless."""
        json_str = sample_generation_config.model_dump_json()
        restored = GenerationConfig.model_validate_json(json_str)
        assert restored == sample_generation_config

    def test_target_words_validation(self) -> None:
        """target_words must be between 500 and 10000."""
        with pytest.raises(Exception):
            GenerationConfig(target_words=100)
        with pytest.raises(Exception):
            GenerationConfig(target_words=20000)

    def test_min_score_validation(self) -> None:
        """min_score must be between 0 and 10."""
        with pytest.raises(Exception):
            GenerationConfig(min_score=-1.0)
        with pytest.raises(Exception):
            GenerationConfig(min_score=11.0)

    def test_enum_values(self) -> None:
        """Enum fields should accept string values."""
        gc = GenerationConfig(tone="suspenseful", perspective="first_person")
        assert gc.tone == Tone.SUSPENSEFUL
        assert gc.perspective == Perspective.FIRST_PERSON


# ════════════════════════════════════════════════════════════════════════
# APIConfig
# ════════════════════════════════════════════════════════════════════════


class TestAPIConfig:
    """Tests for the APIConfig model."""

    def test_defaults(self) -> None:
        """Defaults should produce a valid config."""
        ac = APIConfig()
        assert ac.provider == APIProvider.OPENROUTER
        assert ac.max_retries == 3
        assert ac.timeout == 120

    def test_round_trip(self, sample_api_config: APIConfig) -> None:
        """Round-trip should be lossless."""
        data = sample_api_config.model_dump()
        restored = APIConfig.model_validate(data)
        assert restored == sample_api_config

    def test_fallback_none(self) -> None:
        """Fallback provider can be None."""
        ac = APIConfig(fallback_provider=None)
        assert ac.fallback_provider is None

    def test_timeout_validation(self) -> None:
        """Timeout must be between 10 and 600."""
        with pytest.raises(Exception):
            APIConfig(timeout=5)


# ════════════════════════════════════════════════════════════════════════
# Section
# ════════════════════════════════════════════════════════════════════════


class TestSection:
    """Tests for the Section model."""

    def test_creation(self, sample_section: Section) -> None:
        """Section should be created with all fields."""
        assert sample_section.index == 0
        assert sample_section.title == "The Discovery"
        assert sample_section.word_count == 780

    def test_round_trip(self, sample_section: Section) -> None:
        """Round-trip should be lossless."""
        data = sample_section.model_dump()
        restored = Section.model_validate(data)
        assert restored == sample_section

    def test_defaults(self) -> None:
        """Required field is only index."""
        s = Section(index=0)
        assert s.title == ""
        assert s.text == ""
        assert s.key_events == []


# ════════════════════════════════════════════════════════════════════════
# StoryBible
# ════════════════════════════════════════════════════════════════════════


class TestStoryBible:
    """Tests for the StoryBible model."""

    def test_creation(self, sample_story_bible: StoryBible) -> None:
        """StoryBible should contain characters and themes."""
        assert len(sample_story_bible.characters) == 2
        assert sample_story_bible.characters[0].name == "Elena"
        assert len(sample_story_bible.themes) == 3

    def test_round_trip(self, sample_story_bible: StoryBible) -> None:
        """Round-trip should preserve nested models."""
        data = sample_story_bible.model_dump()
        restored = StoryBible.model_validate(data)
        assert restored == sample_story_bible
        assert restored.characters[0].traits == ["brave", "curious", "compassionate"]

    def test_empty_defaults(self) -> None:
        """Empty StoryBible should have sane defaults."""
        sb = StoryBible()
        assert sb.premise == ""
        assert sb.characters == []
        assert sb.setting.location == ""


# ════════════════════════════════════════════════════════════════════════
# Outline
# ════════════════════════════════════════════════════════════════════════


class TestOutline:
    """Tests for the Outline model."""

    def test_creation(self, sample_outline: Outline) -> None:
        """Outline should have the correct number of sections."""
        assert len(sample_outline.sections) == 3
        assert sample_outline.total_target_words == 3000

    def test_round_trip(self, sample_outline: Outline) -> None:
        """Round-trip should preserve nested OutlineSections."""
        data = sample_outline.model_dump()
        restored = Outline.model_validate(data)
        assert restored == sample_outline

    def test_section_fields(self, sample_outline: Outline) -> None:
        """Each section should have act labels and events."""
        act1 = sample_outline.sections[0]
        assert act1.act_label == "Act I — Setup"
        assert "Elena finds the hidden entrance" in act1.key_events


# ════════════════════════════════════════════════════════════════════════
# EvaluationResult
# ════════════════════════════════════════════════════════════════════════


class TestEvaluationResult:
    """Tests for the EvaluationResult model."""

    def test_passing(self, sample_evaluation_passing: EvaluationResult) -> None:
        """Passing evaluation should have score >= 9."""
        assert sample_evaluation_passing.passed is True
        assert sample_evaluation_passing.overall_score >= 9.0
        assert sample_evaluation_passing.issue_count == 0

    def test_failing(self, sample_evaluation_failing: EvaluationResult) -> None:
        """Failing evaluation should have issues."""
        assert sample_evaluation_failing.passed is False
        assert sample_evaluation_failing.issue_count > 0
        assert len(sample_evaluation_failing.critical_issues) > 0

    def test_round_trip(self, sample_evaluation_passing: EvaluationResult) -> None:
        """Round-trip should be lossless."""
        data = sample_evaluation_passing.model_dump()
        restored = EvaluationResult.model_validate(data)
        assert restored == sample_evaluation_passing

    def test_all_issues_flat_list(
        self, sample_evaluation_failing: EvaluationResult
    ) -> None:
        """all_issues should aggregate across all levels."""
        issues = sample_evaluation_failing.all_issues
        assert isinstance(issues, list)
        assert all(isinstance(i, EvaluationIssue) for i in issues)

    def test_evaluation_issue_fields(
        self, sample_evaluation_issue: EvaluationIssue
    ) -> None:
        """EvaluationIssue should have all fields populated."""
        assert sample_evaluation_issue.level == EvaluationLevel.L1_TECHNICAL
        assert sample_evaluation_issue.severity == IssueSeverity.MAJOR
        assert sample_evaluation_issue.line_reference == "line 42"


# ════════════════════════════════════════════════════════════════════════
# Story
# ════════════════════════════════════════════════════════════════════════


class TestStory:
    """Tests for the Story model."""

    def test_creation(self, sample_story: Story) -> None:
        """Story should have correct id and status."""
        assert sample_story.id == "ancient_temple_de_20250115"
        assert sample_story.status == StoryStatus.PENDING

    def test_round_trip(self, sample_story: Story) -> None:
        """Round-trip should be lossless."""
        data = sample_story.model_dump()
        restored = Story.model_validate(data)
        assert restored.id == sample_story.id
        assert restored.topic == sample_story.topic

    def test_touch_updates_timestamp(self, sample_story: Story) -> None:
        """touch() should update updated_at."""
        old = sample_story.updated_at
        sample_story.touch()
        assert sample_story.updated_at >= old

    def test_defaults(self) -> None:
        """Defaults should produce a valid story."""
        s = Story()
        assert s.status == StoryStatus.PENDING
        assert s.sections == []
        assert s.final_text == ""


# ════════════════════════════════════════════════════════════════════════
# PipelineState
# ════════════════════════════════════════════════════════════════════════


class TestPipelineState:
    """Tests for the PipelineState model."""

    def test_creation(self, sample_pipeline_state: PipelineState) -> None:
        """State should have correct initial values."""
        assert sample_pipeline_state.story_id == "ancient_temple_de_20250115"
        assert sample_pipeline_state.language == "de"
        assert sample_pipeline_state.status == PipelineStatus.PENDING

    def test_round_trip(self, sample_pipeline_state: PipelineState) -> None:
        """Full serialization round-trip including nested models."""
        data = sample_pipeline_state.model_dump()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = PipelineState.model_validate(restored_data)
        assert restored.story_id == sample_pipeline_state.story_id
        assert restored.generation_config == sample_pipeline_state.generation_config

    def test_latest_draft_empty(self, sample_pipeline_state: PipelineState) -> None:
        """latest_draft should be empty when no drafts exist."""
        assert sample_pipeline_state.latest_draft == ""

    def test_latest_draft_returns_last(
        self, sample_pipeline_state: PipelineState
    ) -> None:
        """latest_draft should return the most recent draft."""
        sample_pipeline_state.drafts = ["draft_v1", "draft_v2", "draft_v3"]
        assert sample_pipeline_state.latest_draft == "draft_v3"

    def test_latest_evaluation_none(
        self, sample_pipeline_state: PipelineState
    ) -> None:
        """latest_evaluation should be None when no evaluations exist."""
        assert sample_pipeline_state.latest_evaluation is None

    def test_word_count_zero_when_empty(
        self, sample_pipeline_state: PipelineState
    ) -> None:
        """word_count should be 0 when no draft exists."""
        assert sample_pipeline_state.word_count == 0

    def test_word_count_with_draft(
        self, sample_pipeline_state: PipelineState
    ) -> None:
        """word_count should approximate words in latest draft."""
        sample_pipeline_state.drafts = ["one two three four five"]
        assert sample_pipeline_state.word_count == 5

    def test_touch(self, sample_pipeline_state: PipelineState) -> None:
        """touch() should update updated_at."""
        old = sample_pipeline_state.updated_at
        sample_pipeline_state.touch()
        assert sample_pipeline_state.updated_at >= old

    def test_with_story_bible(
        self,
        sample_pipeline_state: PipelineState,
        sample_story_bible: StoryBible,
    ) -> None:
        """State should accept and serialize a StoryBible."""
        sample_pipeline_state.story_bible = sample_story_bible
        data = sample_pipeline_state.model_dump()
        restored = PipelineState.model_validate(data)
        assert restored.story_bible is not None
        assert restored.story_bible.premise == sample_story_bible.premise

    def test_with_outline(
        self,
        sample_pipeline_state: PipelineState,
        sample_outline: Outline,
    ) -> None:
        """State should accept and serialize an Outline."""
        sample_pipeline_state.outline = sample_outline
        data = sample_pipeline_state.model_dump()
        restored = PipelineState.model_validate(data)
        assert restored.outline is not None
        assert len(restored.outline.sections) == 3


# ════════════════════════════════════════════════════════════════════════
# StoryMetadata
# ════════════════════════════════════════════════════════════════════════


class TestStoryMetadata:
    """Tests for the StoryMetadata model."""

    def test_creation(self, sample_metadata: StoryMetadata) -> None:
        """Metadata should have correct fields."""
        assert sample_metadata.story_id == "ancient_temple_de_20250115"
        assert sample_metadata.word_count == 3012
        assert sample_metadata.estimated_cost_usd == 0.45

    def test_round_trip(self, sample_metadata: StoryMetadata) -> None:
        """Round-trip should be lossless."""
        data = sample_metadata.model_dump()
        restored = StoryMetadata.model_validate(data)
        assert restored == sample_metadata

    def test_defaults(self) -> None:
        """Defaults should produce a valid metadata object."""
        m = StoryMetadata()
        assert m.story_id == ""
        assert m.attempts == 1
        assert m.output_files == []


# ════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Tests for module-level constants."""

    def test_languages_count(self) -> None:
        """LANGUAGES should have exactly 11 entries."""
        assert len(LANGUAGES) == 11

    def test_supported_language_codes(self) -> None:
        """SUPPORTED_LANGUAGE_CODES should match LANGUAGES keys."""
        assert set(SUPPORTED_LANGUAGE_CODES) == set(LANGUAGES.keys())

    def test_language_entries_have_required_keys(self) -> None:
        """Each language entry should have name, flag, and native."""
        for code, info in LANGUAGES.items():
            assert "name" in info, f"Missing 'name' for {code}"
            assert "flag" in info, f"Missing 'flag' for {code}"
            assert "native" in info, f"Missing 'native' for {code}"

    def test_provider_config_count(self) -> None:
        """PROVIDER_CONFIG should have exactly 6 providers."""
        assert len(PROVIDER_CONFIG) == 6

    def test_provider_config_entries(self) -> None:
        """Each provider should have base_url, format, and models."""
        for name, cfg in PROVIDER_CONFIG.items():
            assert "base_url" in cfg, f"Missing 'base_url' for {name}"
            assert "format" in cfg, f"Missing 'format' for {name}"
            assert "models" in cfg, f"Missing 'models' for {name}"
            assert len(cfg["models"]) > 0, f"No models for {name}"
