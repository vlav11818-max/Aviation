"""Pipeline state data model.

``PipelineState`` is the single source of truth carried through every
pipeline step.  It holds the Story Bible, outline, section data,
versioned drafts, evaluations, cost/token accumulators, and progress
metadata.  Fully serializable to/from JSON for crash recovery.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from models.config import APIConfig, GenerationConfig
from models.evaluation import EvaluationResult
from models.outline import Outline
from models.section import Section
from models.story_bible import StoryBible


class PipelineStatus(str, Enum):
    """Execution status of the pipeline for a single story."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineState(BaseModel):
    """Unified state object for a single story's pipeline execution.

    One ``PipelineState`` instance is created per topic and passed
    through every step.  The ``StateManager`` is the exclusive API
    for mutating this object.

    Attributes:
        story_id: Unique identifier for this run.
        topic: Original topic/theme string.
        language: Two-letter language code.
        generation_config: Snapshot of creative parameters.
        api_config: Snapshot of API settings.
        current_step_index: Index of the step currently executing.
        strategy_name: Name of the selected strategy.
        story_bible: Story Bible produced by the concept step.
        outline: Structural outline produced by the outline step.
        sections_completed: Completed ``Section`` objects.
        section_summaries: Summary strings for context propagation.
        drafts: Versioned draft texts (index 0 = v1).
        evaluations: Evaluation results per attempt.
        current_attempt: Current evaluation/revision attempt (1-based).
        status: Pipeline execution status.
        cost_accumulated: Total USD cost so far.
        tokens_used_in: Total input tokens consumed.
        tokens_used_out: Total output tokens consumed.
        started_at: Run start timestamp (ISO-8601).
        updated_at: Last state mutation timestamp (ISO-8601).
        output_dir: Path to the output directory for this story.
        error_message: Last error message if status is FAILED.
    """

    story_id: str = Field(
        default="",
        description="Unique run identifier.",
    )
    topic: str = Field(
        default="",
        description="Original topic/theme.",
    )
    language: str = Field(
        default="en",
        description="Two-letter language code.",
    )
    generation_config: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Creative parameter snapshot.",
    )
    api_config: APIConfig = Field(
        default_factory=APIConfig,
        description="API settings snapshot.",
    )
    current_step_index: int = Field(
        default=0,
        ge=0,
        description="Index of the current step in the strategy.",
    )
    strategy_name: str = Field(
        default="",
        description="Name of the selected strategy.",
    )

    # ── Produced data ───────────────────────────────────────────────

    story_bible: StoryBible | None = Field(
        default=None,
        description="Story Bible (set after concept step).",
    )
    outline: Outline | None = Field(
        default=None,
        description="Structural outline (set after outline step).",
    )
    sections_completed: list[Section] = Field(
        default_factory=list,
        description="Completed sections.",
    )
    section_summaries: list[str] = Field(
        default_factory=list,
        description="Summaries for context propagation.",
    )
    drafts: list[str] = Field(
        default_factory=list,
        description="Versioned draft texts.",
    )
    evaluations: list[EvaluationResult] = Field(
        default_factory=list,
        description="Evaluation results per attempt.",
    )

    # ── Progress ────────────────────────────────────────────────────

    current_attempt: int = Field(
        default=1,
        ge=1,
        description="Current evaluation/revision attempt (1-based).",
    )
    status: PipelineStatus = Field(
        default=PipelineStatus.PENDING,
        description="Pipeline execution status.",
    )

    # ── Cost / tokens ───────────────────────────────────────────────

    cost_accumulated: float = Field(
        default=0.0,
        ge=0.0,
        description="Total USD cost so far.",
    )
    tokens_used_in: int = Field(
        default=0,
        ge=0,
        description="Total input tokens consumed.",
    )
    tokens_used_out: int = Field(
        default=0,
        ge=0,
        description="Total output tokens consumed.",
    )

    # ── Timestamps ──────────────────────────────────────────────────

    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Run start timestamp (ISO-8601).",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last mutation timestamp (ISO-8601).",
    )

    # ── Output ──────────────────────────────────────────────────────

    output_dir: str = Field(
        default="",
        description="Path to the output directory for this story.",
    )
    error_message: str = Field(
        default="",
        description="Last error message (set on failure).",
    )

    # ── Helpers ──────────────────────────────────────────────────────

    def touch(self) -> None:
        """Update ``updated_at`` to the current time."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @property
    def latest_draft(self) -> str:
        """Return the most recent draft text, or empty string if none."""
        return self.drafts[-1] if self.drafts else ""

    @property
    def latest_evaluation(self) -> EvaluationResult | None:
        """Return the most recent evaluation, or ``None`` if none."""
        return self.evaluations[-1] if self.evaluations else None

    @property
    def word_count(self) -> int:
        """Approximate word count of the latest draft."""
        return len(self.latest_draft.split()) if self.latest_draft else 0
