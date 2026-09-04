"""Shared pytest fixtures for AI Story Generator Pro.

Provides pre-built instances of all data models, temporary directories,
sample settings, and mock objects for the API client.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest

from models.config import (
    APIConfig,
    APIProvider,
    Audience,
    DialogDensity,
    GenerationConfig,
    Pacing,
    Perspective,
    Register,
    StructureType,
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


# ── Temporary directory ─────────────────────────────────────────────────


@pytest.fixture()
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory(prefix="story_gen_test_") as d:
        yield Path(d)


# ── Settings fixtures ───────────────────────────────────────────────────


@pytest.fixture()
def sample_settings_yaml(tmp_dir: Path) -> Path:
    """Write a minimal valid settings.yaml to a temp dir and return path."""
    content = {
        "generation": {
            "tone": "dramatic_cinematic",
            "perspective": "third_person",
            "register": "conversational",
            "pacing": "medium",
            "audience": "all_ages",
            "dialog_density": "medium",
            "target_words": 3000,
            "structure": "three_act",
            "genres": ["fantasy"],
            "min_score": 9.0,
            "max_attempts": 5,
            "voiceover_optimized": True,
            "avoid_complex_sentences": True,
            "pause_markers": True,
            "no_headers": True,
            "no_meta_comments": True,
        },
        "api": {
            "primary_provider": "openrouter",
            "primary_model": "anthropic/claude-3.5-sonnet",
            "fallback_provider": "openai",
            "fallback_model": "gpt-4o",
            "auto_fallback": True,
            "max_retries": 3,
            "timeout_seconds": 120,
            "rate_limits": {
                "openrouter": 5,
                "openai": 3,
                "anthropic": 2,
                "google": 5,
                "deepseek": 3,
                "qwen": 3,
            },
            "pricing": {},
        },
        "retry": {
            "initial_delay_seconds": 1.0,
            "max_delay_seconds": 16.0,
            "exponential_base": 2,
            "max_retries": 3,
        },
        "parallelism": {
            "max_workers": 3,
            "auto_throttle": True,
            "separate_queues": True,
        },
        "ssml": {
            "paragraph_break": "600ms",
            "scene_break": "1000ms",
            "dialog_pause": "400ms",
            "dramatic_pause": "800ms",
            "sentence_pause": "200ms",
            "slow_for_dramatic": True,
            "emphasis_for_key_words": False,
        },
        "logging": {
            "level": "DEBUG",
            "max_files": 10,
            "max_file_size_mb": 10,
            "log_dir": str(tmp_dir / "logs"),
        },
        "paths": {
            "output_dir": str(tmp_dir / "output"),
            "data_dir": str(tmp_dir / "data"),
            "resources_dir": "resources",
            "recovery_dir": str(tmp_dir / "data" / "recovery"),
            "cache_dir": str(tmp_dir / "data" / "cache"),
            "analytics_dir": str(tmp_dir / "data" / "analytics"),
        },
        "cache": {
            "enabled": True,
            "skip_processed": True,
        },
        "strategy": {
            "single_shot_max": 2000,
            "two_pass_max": 4000,
        },
    }
    settings_path = tmp_dir / "settings.yaml"
    import yaml

    settings_path.write_text(
        yaml.dump(content, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return settings_path


# ── Model fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def sample_generation_config() -> GenerationConfig:
    """Return a standard GenerationConfig instance."""
    return GenerationConfig(
        tone=Tone.DRAMATIC_CINEMATIC,
        perspective=Perspective.THIRD_PERSON,
        register=Register.CONVERSATIONAL,
        pacing=Pacing.MEDIUM,
        audience=Audience.ALL_AGES,
        dialog_density=DialogDensity.MEDIUM,
        target_words=3000,
        structure=StructureType.THREE_ACT,
        genres=["fantasy", "mystery"],
        min_score=9.0,
        max_attempts=5,
    )


@pytest.fixture()
def sample_api_config() -> APIConfig:
    """Return a standard APIConfig instance."""
    return APIConfig(
        primary_provider=APIProvider.OPENROUTER,
        primary_model="anthropic/claude-3.5-sonnet",
        api_key="test-key-123",
        base_url="https://openrouter.ai/api/v1",
        auto_fallback=True,
        fallback_provider=APIProvider.OPENAI,
        fallback_model="gpt-4o",
        fallback_api_key="test-fallback-key",
        max_retries=3,
        timeout=120,
    )


@pytest.fixture()
def sample_character() -> Character:
    """Return a sample Character."""
    return Character(
        name="Elena",
        role="protagonist",
        traits=["brave", "curious", "compassionate"],
        arc="From cautious observer to courageous leader",
    )


@pytest.fixture()
def sample_story_bible(sample_character: Character) -> StoryBible:
    """Return a sample StoryBible."""
    return StoryBible(
        premise="A young archaeologist discovers an ancient temple that holds the key to a forgotten civilization.",
        setting=Setting(
            location="Dense jungle in Central America",
            time_period="Present day",
            atmosphere="Mysterious and awe-inspiring",
        ),
        characters=[
            sample_character,
            Character(
                name="Marcus",
                role="mentor",
                traits=["wise", "secretive"],
                arc="Reveals his true identity as a guardian",
            ),
        ],
        themes=["discovery", "sacrifice", "legacy"],
        tone_description="Dramatic and cinematic with moments of wonder",
        narrative_voice="Third-person omniscient with a cinematic quality",
        key_rules=[
            "No graphic violence",
            "Keep language accessible",
            "End on a hopeful note",
        ],
    )


@pytest.fixture()
def sample_section() -> Section:
    """Return a sample completed Section."""
    return Section(
        index=0,
        title="The Discovery",
        target_words=800,
        key_events=["Elena finds the hidden entrance", "Strange symbols glow"],
        characters_present=["Elena", "Marcus"],
        transition_from="",
        transition_to="Elena decides to enter despite warnings",
        text="The jungle canopy parted just enough to let a single beam of light through...",
        summary="Elena discovers a hidden temple entrance covered in glowing symbols.",
        word_count=780,
    )


@pytest.fixture()
def sample_outline() -> Outline:
    """Return a sample Outline with three sections."""
    return Outline(
        structure_type="three_act",
        total_target_words=3000,
        sections=[
            OutlineSection(
                index=0,
                title="The Discovery",
                act_label="Act I — Setup",
                target_words=750,
                key_events=["Elena finds the hidden entrance"],
                characters_present=["Elena", "Marcus"],
                transition_from="",
                transition_to="Elena decides to enter",
            ),
            OutlineSection(
                index=1,
                title="The Descent",
                act_label="Act II — Confrontation",
                target_words=1500,
                key_events=["Navigate traps", "Discover murals"],
                characters_present=["Elena", "Marcus"],
                transition_from="Elena enters the temple",
                transition_to="They reach the central chamber",
            ),
            OutlineSection(
                index=2,
                title="The Revelation",
                act_label="Act III — Resolution",
                target_words=750,
                key_events=["Ancient secret revealed", "Elena makes her choice"],
                characters_present=["Elena", "Marcus"],
                transition_from="Central chamber opens",
                transition_to="",
            ),
        ],
    )


@pytest.fixture()
def sample_evaluation_issue() -> EvaluationIssue:
    """Return a sample EvaluationIssue."""
    return EvaluationIssue(
        level=EvaluationLevel.L1_TECHNICAL,
        category="marker",
        description="Chapter header found: [Chapter 1]",
        severity=IssueSeverity.MAJOR,
        line_reference="line 42",
    )


@pytest.fixture()
def sample_evaluation_passing() -> EvaluationResult:
    """Return a passing EvaluationResult (score >= 9.0)."""
    return EvaluationResult(
        l1_technical=LevelResult(score=9.5, issues=[]),
        l2_linguistic=LevelResult(score=9.2, issues=[]),
        l3_content=LevelResult(score=9.4, issues=[]),
        l4_voiceover=LevelResult(score=9.3, issues=[]),
        overall_score=9.35,
        passed=True,
        summary="Excellent story with minor polish needed.",
        critical_issues=[],
        attempt_number=1,
        timestamp="2025-01-15T14:30:00Z",
    )


@pytest.fixture()
def sample_evaluation_failing(
    sample_evaluation_issue: EvaluationIssue,
) -> EvaluationResult:
    """Return a failing EvaluationResult (score < 9.0)."""
    return EvaluationResult(
        l1_technical=LevelResult(
            score=7.0,
            issues=[sample_evaluation_issue],
        ),
        l2_linguistic=LevelResult(score=8.0, issues=[]),
        l3_content=LevelResult(score=7.5, issues=[]),
        l4_voiceover=LevelResult(score=7.0, issues=[]),
        overall_score=7.4,
        passed=False,
        summary="Multiple issues need addressing.",
        critical_issues=[sample_evaluation_issue],
        attempt_number=1,
        timestamp="2025-01-15T14:30:00Z",
    )


@pytest.fixture()
def sample_pipeline_state(
    sample_generation_config: GenerationConfig,
    sample_api_config: APIConfig,
) -> PipelineState:
    """Return a sample PipelineState in PENDING status."""
    return PipelineState(
        story_id="ancient_temple_de_20250115",
        topic="Ancient Temple",
        language="de",
        generation_config=sample_generation_config,
        api_config=sample_api_config,
        current_step_index=0,
        strategy_name="full_pipeline",
        status=PipelineStatus.PENDING,
    )


@pytest.fixture()
def sample_story(
    sample_generation_config: GenerationConfig,
) -> Story:
    """Return a sample Story in PENDING status."""
    return Story(
        id="ancient_temple_de_20250115",
        topic="Ancient Temple",
        language="de",
        config=sample_generation_config,
        status=StoryStatus.PENDING,
    )


@pytest.fixture()
def sample_metadata() -> StoryMetadata:
    """Return a sample StoryMetadata for a completed story."""
    return StoryMetadata(
        story_id="ancient_temple_de_20250115",
        topic="Ancient Temple",
        language="de",
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        strategy_used="full_pipeline",
        final_score=9.35,
        attempts=1,
        started_at="2025-01-15T14:00:00Z",
        completed_at="2025-01-15T14:05:32Z",
        duration_seconds=332.0,
        total_tokens_in=15000,
        total_tokens_out=4500,
        estimated_cost_usd=0.45,
        word_count=3012,
        section_count=3,
        output_files=["final.txt", "final.ssml", "metadata.json"],
    )


# ── Mock API client ─────────────────────────────────────────────────────


class MockAPIResponse:
    """Lightweight mock for an API response."""

    def __init__(self, text: str, tokens_in: int = 100, tokens_out: int = 50) -> None:
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out


class MockAPIClient:
    """Mock API client that returns canned responses.

    Configure via ``responses`` dict mapping step/prompt keywords to
    response text.  Falls back to a default response for unknown prompts.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = '{"status": "ok"}',
    ) -> None:
        self.responses = responses or {}
        self.default_response = default_response
        self.call_log: list[dict[str, Any]] = []

    async def send(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Return a canned response based on message content.

        Args:
            messages: Chat messages list.
            temperature: Sampling temperature (ignored in mock).
            max_tokens: Max tokens (ignored in mock).

        Returns:
            Canned response string.
        """
        self.call_log.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        for keyword, response in self.responses.items():
            if keyword.lower() in user_content.lower():
                return response

        return self.default_response
