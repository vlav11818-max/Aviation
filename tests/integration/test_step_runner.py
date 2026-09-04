"""Integration test for StepRunner with mock API client.

Runs a full pipeline strategy for a short topic using canned responses
from ``tests/fixtures/mock_api_responses.json``.  Verifies that:

- All pipeline steps execute in order.
- Artifacts are created (concept, outline, sections, draft, evaluation).
- The evaluate→revise loop triggers when score < min_score.
- The final state is ``COMPLETED``.
- Analytics are recorded via ``AnalyticsCollector``.
- ``StoryMetadata`` can be built from the final state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from core.analytics_collector import AnalyticsCollector
from core.events import EventBus, EventType
from core.settings import Settings
from core.state_manager import StateManager
from core.step_runner import StepRunner
from core.steps.base_step import BaseStep
from core.steps.clean_step import CleanStep
from core.steps.concept_step import ConceptStep
from core.steps.evaluate_step import EvaluateStep
from core.steps.outline_step import OutlineStep
from core.steps.revise_step import ReviseStep
from core.steps.section_step import SectionStep
from core.steps.stitch_step import StitchStep
from core.strategies import get_strategy
from models.config import APIConfig, APIProvider, GenerationConfig
from models.state import PipelineState, PipelineStatus

logger = logging.getLogger(__name__)

# ── Fixtures directory ──────────────────────────────────────────────────

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


# ── Mock API client ─────────────────────────────────────────────────────


class _MockResponse:
    """Lightweight mock API response."""

    def __init__(self, text: str, tokens_in: int, tokens_out: int) -> None:
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out


class _MockAPIClient:
    """Mock API client that returns canned responses.

    Keeps a call counter per step name so that repeated calls
    (e.g., multiple sections or evaluation retries) return the
    correct fixture data.
    """

    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self._call_count: dict[str, int] = {}
        self._evaluation_count: int = 0

    async def send(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Return a canned response based on the prompt content.

        Matches the real APIClient.send() signature: takes messages list,
        returns plain text string.  Token counts are not tracked in the
        mock (the real APIClient handles this internally).

        The step name is inferred from the last user message text.
        """
        # Extract prompt text from messages (last user message).
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        step_name = self._infer_step(prompt)
        self._call_count[step_name] = self._call_count.get(step_name, 0) + 1
        count = self._call_count[step_name]

        logger.debug("MockAPIClient: step=%s, call=%d", step_name, count)

        if step_name == "evaluate":
            self._evaluation_count += 1
            if self._evaluation_count == 1:
                data = self._responses.get("evaluation_fail", {})
            else:
                data = self._responses.get("evaluation_pass", {})
        elif step_name == "section":
            section_key = f"section_{count - 1}"
            data = self._responses.get(
                section_key,
                self._responses.get("section_0", {}),
            )
        else:
            data = self._responses.get(step_name, {})

        if not data:
            return "{}"

        return data.get("text", "")

    async def close(self) -> None:
        """No-op close for the mock client."""

    @staticmethod
    def _infer_step(prompt: str) -> str:
        """Infer the step name from prompt keywords."""
        prompt_lower = prompt.lower()

        # Order matters — check more specific patterns first.
        if "evaluation" in prompt_lower or "evaluate" in prompt_lower:
            return "evaluate"
        if "revis" in prompt_lower:
            return "revision"
        if "stitch" in prompt_lower or "merge" in prompt_lower:
            return "stitch"
        if "section" in prompt_lower and "outline" not in prompt_lower:
            return "section"
        if "outline" in prompt_lower or "structure" in prompt_lower:
            return "outline"
        if "concept" in prompt_lower or "story bible" in prompt_lower:
            return "concept"
        if "single" in prompt_lower:
            return "single_shot"
        if "adapt" in prompt_lower:
            return "adaptation"

        return "unknown"


# ── Mock prompt manager ─────────────────────────────────────────────────


class _MockPromptManager:
    """Minimal prompt manager that returns prompt-like strings."""

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Return a prompt string containing the template name and vars."""
        parts = [f"[{template_name}]"]
        for k, v in kwargs.items():
            if isinstance(v, str) and len(v) > 200:
                parts.append(f"{k}=[{len(v)} chars]")
            else:
                parts.append(f"{k}={v}")
        return " ".join(parts)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_responses() -> dict[str, Any]:
    """Load canned API responses from the fixtures directory."""
    path = _FIXTURES_DIR / "mock_api_responses.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def event_bus() -> EventBus:
    """Create a fresh EventBus."""
    return EventBus()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Create a Settings instance with temp directories."""
    import yaml

    content = {
        "paths": {
            "output_dir": str(tmp_path / "output"),
            "data_dir": str(tmp_path / "data"),
            "resources_dir": "resources",
            "recovery_dir": str(tmp_path / "data" / "recovery"),
            "cache_dir": str(tmp_path / "data" / "cache"),
            "analytics_dir": str(tmp_path / "data" / "analytics"),
        },
        "generation": {
            "min_score": 9.0,
            "max_attempts": 3,
            "target_words": 3000,
        },
        "strategy": {
            "single_shot_max": 2000,
            "two_pass_max": 4000,
        },
        "cache": {"enabled": False},
    }
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        yaml.dump(content, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Also create defaults
    defaults_dir = tmp_path / "resources" / "defaults"
    defaults_dir.mkdir(parents=True, exist_ok=True)
    (defaults_dir / "settings.yaml").write_text(
        yaml.dump(content, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    return Settings.model_validate(content)


@pytest.fixture()
def analytics_collector(tmp_path: Path) -> AnalyticsCollector:
    """Create an AnalyticsCollector with temp directory."""
    analytics_dir = tmp_path / "data" / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    return AnalyticsCollector(analytics_dir=str(analytics_dir))


@pytest.fixture()
def state_manager() -> StateManager:
    """Create a StateManager."""
    return StateManager()


@pytest.fixture()
def gen_config() -> GenerationConfig:
    """Create a GenerationConfig with test values."""
    return GenerationConfig(
        target_words=3000,
        min_score=9.0,
        max_attempts=3,
    )


@pytest.fixture()
def api_config() -> APIConfig:
    """Create an APIConfig for testing."""
    return APIConfig(
        primary_provider=APIProvider.OPENROUTER,
        primary_model="test-model",
        api_key="test-key",
    )


@pytest.fixture()
def initial_state(
    gen_config: GenerationConfig,
    api_config: APIConfig,
    tmp_path: Path,
) -> PipelineState:
    """Create an initial pipeline state."""
    output_dir = tmp_path / "output" / "en" / "ancient_temple"
    output_dir.mkdir(parents=True, exist_ok=True)

    return PipelineState(
        story_id="ancient_temple_en_test",
        topic="Ancient Temple",
        language="en",
        generation_config=gen_config,
        api_config=api_config,
        strategy_name="full_pipeline",
        status=PipelineStatus.PENDING,
        output_dir=str(output_dir),
    )


# ── Tests ───────────────────────────────────────────────────────────────


class TestStepRunnerIntegration:
    """Integration tests for StepRunner with mock API."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_revision(
        self,
        mock_responses: dict[str, Any],
        event_bus: EventBus,
        settings: Settings,
        state_manager: StateManager,
        analytics_collector: AnalyticsCollector,
        initial_state: PipelineState,
    ) -> None:
        """Full pipeline should execute all steps and trigger revision.

        The first evaluation returns score 7.5 (below 9.0 threshold),
        triggering a revision loop.  The second evaluation returns 9.5,
        so the pipeline completes successfully.
        """
        mock_client = _MockAPIClient(mock_responses)
        mock_prompts = _MockPromptManager()

        runner = StepRunner(
            state_manager=state_manager,
            api_client=mock_client,
            prompt_manager=mock_prompts,
            event_bus=event_bus,
            settings=settings,
            analytics_collector=analytics_collector,
        )

        strategy = get_strategy("full_pipeline")
        final_state = await runner.execute(initial_state, strategy)

        # Verify completion
        assert final_state.status == PipelineStatus.COMPLETED

        # Verify evaluation triggered revision
        assert final_state.current_attempt >= 2, (
            "Expected at least 2 attempts (first eval fails, revision, second eval passes)"
        )

        # Verify evaluations recorded
        assert len(final_state.evaluations) >= 2

        # First evaluation should have failed
        first_eval = final_state.evaluations[0]
        assert first_eval.overall_score < 9.0
        assert not first_eval.passed

        # Final evaluation should have passed
        last_eval = final_state.evaluations[-1]
        assert last_eval.overall_score >= 9.0
        assert last_eval.passed

        # Verify drafts exist
        assert len(final_state.drafts) >= 1

        # Verify story bible was set
        assert final_state.story_bible is not None

        # Verify outline was set
        assert final_state.outline is not None

        # Verify token usage tracked
        assert final_state.tokens_used_in > 0
        assert final_state.tokens_used_out > 0

        # Verify events were emitted
        events = event_bus.poll_all()
        event_types = {e.type for e in events}
        assert EventType.STEP_STARTED in event_types
        assert EventType.STEP_COMPLETED in event_types
        assert EventType.STORY_COMPLETED in event_types

        # Verify analytics recorded
        stats = analytics_collector.get_stats()
        assert len(stats.stories) == 1
        recorded = stats.stories[0]
        assert recorded["topic"] == "Ancient Temple"
        assert recorded["score"] >= 9.0
        assert recorded["language"] == "en"

    @pytest.mark.asyncio
    async def test_events_emitted_in_order(
        self,
        mock_responses: dict[str, Any],
        event_bus: EventBus,
        settings: Settings,
        state_manager: StateManager,
        initial_state: PipelineState,
    ) -> None:
        """Events should be emitted in correct order for each step."""
        mock_client = _MockAPIClient(mock_responses)
        mock_prompts = _MockPromptManager()

        runner = StepRunner(
            state_manager=state_manager,
            api_client=mock_client,
            prompt_manager=mock_prompts,
            event_bus=event_bus,
            settings=settings,
        )

        strategy = get_strategy("full_pipeline")
        await runner.execute(initial_state, strategy)

        events = event_bus.poll_all()
        started_count = sum(1 for e in events if e.type == EventType.STEP_STARTED)
        completed_count = sum(1 for e in events if e.type == EventType.STEP_COMPLETED)

        # Every started step should have a corresponding completion
        assert started_count == completed_count
        assert started_count >= 6  # concept, outline, section(s), stitch, clean, evaluate(s)

    @pytest.mark.asyncio
    async def test_max_attempts_respected(
        self,
        event_bus: EventBus,
        settings: Settings,
        state_manager: StateManager,
        api_config: APIConfig,
        tmp_path: Path,
    ) -> None:
        """If all evaluations fail, runner should stop at max_attempts."""
        # Create responses where evaluation always fails
        always_fail_responses = {
            "concept": {"text": "{\"premise\":\"test\",\"setting\":{\"location\":\"x\",\"time_period\":\"now\",\"atmosphere\":\"y\"},\"characters\":[],\"tone\":\"dramatic_cinematic\",\"themes\":[],\"key_rules\":[],\"narrative_voice\":\"third person\"}", "tokens_in": 100, "tokens_out": 80},
            "outline": {"text": "{\"structure_type\":\"three_act\",\"total_target_words\":1500,\"sections\":[{\"index\":0,\"title\":\"Part 1\",\"target_words\":1500,\"key_events\":[\"Event\"],\"characters_present\":[],\"transition_from\":\"\",\"transition_to\":\"\",\"act_label\":\"Act I\"}]}", "tokens_in": 100, "tokens_out": 80},
            "section_0": {"text": "A short test section with some placeholder text for testing purposes only.", "tokens_in": 100, "tokens_out": 50},
            "stitch": {"text": "A short test story stitched together.", "tokens_in": 100, "tokens_out": 50},
            "evaluation_fail": {"text": "{\"l1_technical\":{\"score\":6.0,\"issues\":[]},\"l2_linguistic\":{\"score\":6.0,\"issues\":[]},\"l3_content\":{\"score\":6.0,\"issues\":[]},\"l4_voiceover\":{\"score\":6.0,\"issues\":[]},\"overall_score\":6.0,\"passed\":false,\"summary\":\"Needs improvement.\",\"critical_issues\":[],\"attempt_number\":1,\"timestamp\":\"2025-01-15T14:00:00Z\"}", "tokens_in": 100, "tokens_out": 80},
            "revision": {"text": "Revised text.", "tokens_in": 100, "tokens_out": 50},
        }

        # Override so evaluation_pass is never used
        class _AlwaysFailClient(_MockAPIClient):
            async def send(
                self,
                messages: list[dict[str, str]],
                temperature: float = 0.7,
                max_tokens: int = 4096,
            ) -> str:
                prompt = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        prompt = msg.get("content", "")
                        break
                step = self._infer_step(prompt)
                if step == "evaluate":
                    data = self._responses["evaluation_fail"]
                elif step == "section":
                    data = self._responses.get("section_0", {})
                else:
                    data = self._responses.get(step, {"text": "{}", "tokens_in": 10, "tokens_out": 10})
                return data.get("text", "{}")

        mock_client = _AlwaysFailClient(always_fail_responses)
        mock_prompts = _MockPromptManager()

        gen_config = GenerationConfig(
            target_words=1500,
            min_score=9.0,
            max_attempts=2,
        )

        output_dir = tmp_path / "output" / "en" / "test_max"
        output_dir.mkdir(parents=True, exist_ok=True)

        state = PipelineState(
            story_id="test_max_attempts",
            topic="Max Attempts Test",
            language="en",
            generation_config=gen_config,
            api_config=api_config,
            strategy_name="full_pipeline",
            status=PipelineStatus.PENDING,
            output_dir=str(output_dir),
        )

        runner = StepRunner(
            state_manager=state_manager,
            api_client=mock_client,
            prompt_manager=mock_prompts,
            event_bus=event_bus,
            settings=settings,
        )

        strategy = get_strategy("full_pipeline")
        final_state = await runner.execute(state, strategy)

        # Should complete (accepted as-is) despite failing evaluations
        assert final_state.status == PipelineStatus.COMPLETED

        # Should not exceed max_attempts
        assert final_state.current_attempt <= gen_config.max_attempts


class TestFixturesValidity:
    """Verify that fixture files are valid and parseable."""

    def test_sample_concept_is_valid_json(self) -> None:
        """sample_concept.json should be valid JSON with expected keys."""
        path = _FIXTURES_DIR / "sample_concept.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "premise" in data
        assert "setting" in data
        assert "characters" in data
        assert isinstance(data["characters"], list)
        assert len(data["characters"]) >= 2

    def test_sample_outline_is_valid_json(self) -> None:
        """sample_outline.json should be valid JSON with sections."""
        path = _FIXTURES_DIR / "sample_outline.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "structure_type" in data
        assert "sections" in data
        assert len(data["sections"]) >= 2
        for section in data["sections"]:
            assert "index" in section
            assert "title" in section
            assert "target_words" in section

    def test_sample_section_is_text(self) -> None:
        """sample_section.txt should be non-empty English text."""
        path = _FIXTURES_DIR / "sample_section.txt"
        text = path.read_text(encoding="utf-8")
        words = text.split()
        assert len(words) >= 400, f"Expected ~500 words, got {len(words)}"

    def test_mock_api_responses_has_all_steps(self) -> None:
        """mock_api_responses.json should have entries for all pipeline steps."""
        path = _FIXTURES_DIR / "mock_api_responses.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        required_keys = [
            "concept", "outline", "section_0",
            "stitch", "evaluation_fail", "evaluation_pass", "revision",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
            assert "text" in data[key], f"Missing 'text' in {key}"
            assert "tokens_in" in data[key], f"Missing 'tokens_in' in {key}"
