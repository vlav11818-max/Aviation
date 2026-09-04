"""Unit tests for core.state_manager.StateManager.

Tests: create_new state, save/load round-trip, add_section, add_draft,
add_evaluation, get_context_for_section (verifies Story Bible + outline
+ prev summary + last 500 words), mark_completed/failed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.exceptions import StateError
from core.state_manager import StateManager
from models.config import APIConfig, GenerationConfig
from models.evaluation import EvaluationResult, LevelResult
from models.outline import Outline, OutlineSection
from models.section import Section
from models.state import PipelineState, PipelineStatus
from models.story_bible import StoryBible


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def state_mgr() -> StateManager:
    """Fresh StateManager instance."""
    return StateManager()


@pytest.fixture
def gen_config() -> GenerationConfig:
    """Default GenerationConfig."""
    return GenerationConfig()


@pytest.fixture
def api_config() -> APIConfig:
    """Default APIConfig."""
    return APIConfig()


@pytest.fixture
def base_state(state_mgr: StateManager, gen_config: GenerationConfig, api_config: APIConfig) -> PipelineState:
    """A fresh PipelineState with defaults."""
    return state_mgr.create_new(
        topic="Ancient Temple",
        language="en",
        gen_config=gen_config,
        api_config=api_config,
        strategy_name="full",
    )


@pytest.fixture
def sample_story_bible() -> StoryBible:
    """A minimal Story Bible for testing."""
    return StoryBible(
        premise="A traveler discovers an ancient temple in the jungle.",
        tone_description="Mysterious and atmospheric.",
        themes=["discovery", "courage"],
    )


@pytest.fixture
def sample_outline() -> Outline:
    """A minimal Outline with 3 sections."""
    return Outline(
        structure_type="three_act",
        total_target_words=3000,
        sections=[
            OutlineSection(
                index=0,
                title="The Discovery",
                act_label="Act I",
                target_words=1000,
                key_events=["finds map"],
            ),
            OutlineSection(
                index=1,
                title="The Journey",
                act_label="Act II",
                target_words=1200,
                key_events=["enters jungle"],
            ),
            OutlineSection(
                index=2,
                title="The Temple",
                act_label="Act III",
                target_words=800,
                key_events=["opens door"],
            ),
        ],
    )


@pytest.fixture
def sample_section() -> Section:
    """A completed Section for testing."""
    return Section(
        index=0,
        title="The Discovery",
        text="The old map crinkled under his fingers. " * 100,
        summary="A traveler finds an ancient map leading to a hidden temple.",
        word_count=800,
        target_words=1000,
        key_events=["finds map"],
    )


@pytest.fixture
def sample_evaluation() -> EvaluationResult:
    """An evaluation result for testing."""
    return EvaluationResult(
        l1_technical=LevelResult(score=9.5),
        l2_linguistic=LevelResult(score=9.0),
        l3_content=LevelResult(score=9.2),
        l4_voiceover=LevelResult(score=9.3),
        overall_score=9.25,
        passed=True,
        summary="Excellent story.",
        attempt_number=1,
    )


# ── Tests: create_new ─────────────────────────────────────────────────


class TestCreateNew:
    """Tests for StateManager.create_new."""

    def test_creates_state_with_topic(self, base_state: PipelineState) -> None:
        """State should contain the topic."""
        assert base_state.topic == "Ancient Temple"

    def test_creates_state_with_language(self, base_state: PipelineState) -> None:
        """State should contain the language."""
        assert base_state.language == "en"

    def test_creates_state_with_strategy(self, base_state: PipelineState) -> None:
        """State should contain the strategy name."""
        assert base_state.strategy_name == "full"

    def test_creates_state_with_pending_status(self, base_state: PipelineState) -> None:
        """State should start as PENDING."""
        assert base_state.status == PipelineStatus.PENDING

    def test_creates_unique_story_id(
        self,
        state_mgr: StateManager,
        gen_config: GenerationConfig,
        api_config: APIConfig,
    ) -> None:
        """Two states for the same topic should have different IDs."""
        s1 = state_mgr.create_new("Test", "en", gen_config, api_config, "full")
        s2 = state_mgr.create_new("Test", "en", gen_config, api_config, "full")
        assert s1.story_id != s2.story_id

    def test_story_id_contains_topic_slug(self, base_state: PipelineState) -> None:
        """Story ID should contain a slugified version of the topic."""
        assert "ancient_temple" in base_state.story_id

    def test_initial_attempt_is_one(self, base_state: PipelineState) -> None:
        """Current attempt should start at 1."""
        assert base_state.current_attempt == 1

    def test_initial_lists_empty(self, base_state: PipelineState) -> None:
        """All list fields should be empty initially."""
        assert base_state.drafts == []
        assert base_state.evaluations == []
        assert base_state.sections_completed == []
        assert base_state.section_summaries == []


# ── Tests: save / load ────────────────────────────────────────────────


class TestSaveLoad:
    """Tests for StateManager.save and .load round-trip."""

    def test_save_creates_file(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        tmp_path: Path,
    ) -> None:
        """save() should create a state.json file."""
        result = state_mgr.save(base_state, tmp_path)
        assert result.exists()
        assert result.name == "state.json"

    def test_round_trip(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        tmp_path: Path,
    ) -> None:
        """Load after save should produce an identical state."""
        state_mgr.save(base_state, tmp_path)
        loaded = state_mgr.load(tmp_path)
        assert loaded.story_id == base_state.story_id
        assert loaded.topic == base_state.topic
        assert loaded.language == base_state.language
        assert loaded.strategy_name == base_state.strategy_name

    def test_round_trip_with_data(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_story_bible: StoryBible,
        sample_evaluation: EvaluationResult,
        tmp_path: Path,
    ) -> None:
        """Round-trip should preserve story_bible, drafts, evaluations."""
        state = state_mgr.update_story_bible(base_state, sample_story_bible)
        state = state_mgr.add_draft(state, "Once upon a time...", version=1)
        state = state_mgr.add_evaluation(state, sample_evaluation)

        state_mgr.save(state, tmp_path)
        loaded = state_mgr.load(tmp_path)

        assert loaded.story_bible is not None
        assert loaded.story_bible.premise == sample_story_bible.premise
        assert loaded.drafts == ["Once upon a time..."]
        assert len(loaded.evaluations) == 1
        assert loaded.evaluations[0].overall_score == 9.25

    def test_save_no_output_dir_raises(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """save() without output_dir should raise StateError."""
        base_state.output_dir = ""
        with pytest.raises(StateError):
            state_mgr.save(base_state)

    def test_load_missing_file_raises(
        self,
        state_mgr: StateManager,
        tmp_path: Path,
    ) -> None:
        """load() on a directory without state.json should raise StateError."""
        with pytest.raises(StateError, match="not found"):
            state_mgr.load(tmp_path)


# ── Tests: mutations ──────────────────────────────────────────────────


class TestMutations:
    """Tests for individual mutation methods."""

    def test_update_story_bible(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_story_bible: StoryBible,
    ) -> None:
        """update_story_bible should set the story_bible field."""
        state = state_mgr.update_story_bible(base_state, sample_story_bible)
        assert state.story_bible is not None
        assert state.story_bible.premise == sample_story_bible.premise

    def test_update_outline(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_outline: Outline,
    ) -> None:
        """update_outline should set the outline field."""
        state = state_mgr.update_outline(base_state, sample_outline)
        assert state.outline is not None
        assert len(state.outline.sections) == 3

    def test_add_section(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_section: Section,
    ) -> None:
        """add_section should append section and summary."""
        state = state_mgr.add_section(base_state, sample_section)
        assert len(state.sections_completed) == 1
        assert state.sections_completed[0].index == 0
        assert len(state.section_summaries) == 1

    def test_add_draft_version_one(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """add_draft with version=1 should create the first draft."""
        state = state_mgr.add_draft(base_state, "Draft text.", version=1)
        assert state.drafts == ["Draft text."]

    def test_add_draft_replaces(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """add_draft replacing an existing version should overwrite."""
        state = state_mgr.add_draft(base_state, "v1", version=1)
        state = state_mgr.add_draft(state, "v1-clean", version=1)
        assert state.drafts == ["v1-clean"]

    def test_add_draft_version_two(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """add_draft with version=2 after version=1 should append."""
        state = state_mgr.add_draft(base_state, "v1", version=1)
        state = state_mgr.add_draft(state, "v2", version=2)
        assert state.drafts == ["v1", "v2"]

    def test_add_draft_invalid_version_raises(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """add_draft with version=0 should raise StateError."""
        with pytest.raises(StateError):
            state_mgr.add_draft(base_state, "text", version=0)

    def test_add_evaluation(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_evaluation: EvaluationResult,
    ) -> None:
        """add_evaluation should append to evaluations list."""
        state = state_mgr.add_evaluation(base_state, sample_evaluation)
        assert len(state.evaluations) == 1
        assert state.evaluations[0].overall_score == 9.25

    def test_increment_attempt(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """increment_attempt should increase current_attempt by 1."""
        assert base_state.current_attempt == 1
        state = state_mgr.increment_attempt(base_state)
        assert state.current_attempt == 2

    def test_update_cost(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """update_cost should accumulate tokens and cost."""
        state = state_mgr.update_cost(base_state, 100, 200, 0.05)
        state = state_mgr.update_cost(state, 50, 100, 0.03)
        assert state.tokens_used_in == 150
        assert state.tokens_used_out == 300
        assert abs(state.cost_accumulated - 0.08) < 1e-6

    def test_mark_in_progress(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """mark_in_progress should set status to IN_PROGRESS."""
        state = state_mgr.mark_in_progress(base_state)
        assert state.status == PipelineStatus.IN_PROGRESS

    def test_mark_completed(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """mark_completed should set status to COMPLETED."""
        state = state_mgr.mark_completed(base_state)
        assert state.status == PipelineStatus.COMPLETED

    def test_mark_failed(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
    ) -> None:
        """mark_failed should set status and error message."""
        state = state_mgr.mark_failed(base_state, "something broke")
        assert state.status == PipelineStatus.FAILED
        assert state.error_message == "something broke"


# ── Tests: get_context_for_section ────────────────────────────────────


class TestGetContext:
    """Tests for StateManager.get_context_for_section."""

    def test_context_requires_story_bible(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_outline: Outline,
    ) -> None:
        """Missing story_bible should raise StateError."""
        state = state_mgr.update_outline(base_state, sample_outline)
        with pytest.raises(StateError, match="story_bible"):
            state_mgr.get_context_for_section(state, 0)

    def test_context_requires_outline(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_story_bible: StoryBible,
    ) -> None:
        """Missing outline should raise StateError."""
        state = state_mgr.update_story_bible(base_state, sample_story_bible)
        with pytest.raises(StateError, match="outline"):
            state_mgr.get_context_for_section(state, 0)

    def test_context_includes_story_bible(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_story_bible: StoryBible,
        sample_outline: Outline,
    ) -> None:
        """Context for section 0 should include STORY BIBLE marker."""
        state = state_mgr.update_story_bible(base_state, sample_story_bible)
        state = state_mgr.update_outline(state, sample_outline)
        context = state_mgr.get_context_for_section(state, 0)
        assert "STORY BIBLE" in context

    def test_context_includes_outline(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_story_bible: StoryBible,
        sample_outline: Outline,
    ) -> None:
        """Context should include OUTLINE marker."""
        state = state_mgr.update_story_bible(base_state, sample_story_bible)
        state = state_mgr.update_outline(state, sample_outline)
        context = state_mgr.get_context_for_section(state, 0)
        assert "OUTLINE" in context

    def test_context_first_section_no_previous(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_story_bible: StoryBible,
        sample_outline: Outline,
    ) -> None:
        """Context for section 0 should NOT include previous summary."""
        state = state_mgr.update_story_bible(base_state, sample_story_bible)
        state = state_mgr.update_outline(state, sample_outline)
        context = state_mgr.get_context_for_section(state, 0)
        assert "PREVIOUS SECTION SUMMARY" not in context

    def test_context_later_section_has_summary(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_story_bible: StoryBible,
        sample_outline: Outline,
        sample_section: Section,
    ) -> None:
        """Context for section 1 should include the previous section summary."""
        state = state_mgr.update_story_bible(base_state, sample_story_bible)
        state = state_mgr.update_outline(state, sample_outline)
        state = state_mgr.add_section(state, sample_section)

        context = state_mgr.get_context_for_section(state, 1)
        assert "PREVIOUS SECTION SUMMARY" in context
        assert "ancient map" in context

    def test_context_later_section_has_last_words(
        self,
        state_mgr: StateManager,
        base_state: PipelineState,
        sample_story_bible: StoryBible,
        sample_outline: Outline,
        sample_section: Section,
    ) -> None:
        """Context for section 1 should include last N words."""
        state = state_mgr.update_story_bible(base_state, sample_story_bible)
        state = state_mgr.update_outline(state, sample_outline)
        state = state_mgr.add_section(state, sample_section)

        context = state_mgr.get_context_for_section(state, 1)
        assert "LAST 500 WORDS" in context
