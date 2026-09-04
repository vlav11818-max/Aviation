"""Story metadata data model.

``StoryMetadata`` captures all information about a completed story run:
configuration, timing, cost, quality scores, and output file paths.
Written as ``metadata.json`` alongside each story's output files.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.config import GenerationConfig
from models.evaluation import EvaluationResult


class StoryMetadata(BaseModel):
    """Full metadata record for a completed story.

    Written to ``metadata.json`` in the story output directory and
    appended to the analytics store.

    Attributes:
        story_id: Unique identifier for this run.
        topic: Original topic/theme string.
        language: Two-letter language code.
        provider: API provider used for the final generation.
        model: Model identifier used for the final generation.
        generation_config: Snapshot of creative parameters.
        strategy_used: Name of the strategy that was executed.
        final_score: Final evaluation score (0.0–10.0).
        evaluation_history: All evaluation results across attempts.
        attempts: Total number of evaluation/revision attempts.
        started_at: Run start timestamp (ISO-8601).
        completed_at: Run completion timestamp (ISO-8601).
        duration_seconds: Wall-clock duration of the run in seconds.
        total_tokens_in: Total input tokens consumed.
        total_tokens_out: Total output tokens consumed.
        estimated_cost_usd: Estimated total USD cost.
        word_count: Final story word count.
        section_count: Number of sections (0 for single-shot).
        output_files: List of relative paths to output files.
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
    provider: str = Field(
        default="",
        description="API provider used.",
    )
    model: str = Field(
        default="",
        description="Model identifier used.",
    )
    generation_config: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Creative parameter snapshot.",
    )
    strategy_used: str = Field(
        default="",
        description="Name of the executed strategy.",
    )
    final_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Final evaluation score.",
    )
    evaluation_history: list[EvaluationResult] = Field(
        default_factory=list,
        description="All evaluations across attempts.",
    )
    attempts: int = Field(
        default=1,
        ge=1,
        description="Total evaluation/revision attempts.",
    )
    started_at: str = Field(
        default="",
        description="Run start timestamp (ISO-8601).",
    )
    completed_at: str = Field(
        default="",
        description="Run completion timestamp (ISO-8601).",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration in seconds.",
    )
    total_tokens_in: int = Field(
        default=0,
        ge=0,
        description="Total input tokens consumed.",
    )
    total_tokens_out: int = Field(
        default=0,
        ge=0,
        description="Total output tokens consumed.",
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated total cost in USD.",
    )
    word_count: int = Field(
        default=0,
        ge=0,
        description="Final story word count.",
    )
    section_count: int = Field(
        default=0,
        ge=0,
        description="Number of sections (0 for single-shot).",
    )
    output_files: list[str] = Field(
        default_factory=list,
        description="Relative paths to output files.",
    )
