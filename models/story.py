"""Story data model.

Top-level model representing a single story throughout its lifecycle,
from creation to completion.  Holds references to sections, drafts,
evaluations, and final output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from models.config import GenerationConfig
from models.evaluation import EvaluationResult
from models.section import Section


class StoryStatus(str, Enum):
    """Lifecycle status of a story."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Story(BaseModel):
    """Represents a single story throughout the generation lifecycle.

    Attributes:
        id: Unique story identifier (typically ``{topic}_{lang}_{timestamp}``).
        topic: The original topic or theme string.
        language: Two-letter language code (e.g., "de").
        config: Generation configuration snapshot.
        status: Current lifecycle status.
        sections: Generated sections (populated during section step).
        drafts: Versioned draft texts (draft_v1, draft_v2, …).
        evaluations: Evaluation results for each attempt.
        final_text: Accepted final text (set on completion).
        created_at: Creation timestamp (ISO-8601).
        updated_at: Last modification timestamp (ISO-8601).
    """

    id: str = Field(
        default="",
        description="Unique story identifier.",
    )
    topic: str = Field(
        default="",
        description="Original topic or theme.",
    )
    language: str = Field(
        default="en",
        description="Two-letter language code.",
    )
    config: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Generation config snapshot.",
    )
    status: StoryStatus = Field(
        default=StoryStatus.PENDING,
        description="Current lifecycle status.",
    )
    sections: list[Section] = Field(
        default_factory=list,
        description="Generated sections.",
    )
    drafts: list[str] = Field(
        default_factory=list,
        description="Versioned draft texts.",
    )
    evaluations: list[EvaluationResult] = Field(
        default_factory=list,
        description="Evaluation results per attempt.",
    )
    final_text: str = Field(
        default="",
        description="Accepted final text.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp (ISO-8601).",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last modification timestamp (ISO-8601).",
    )

    def touch(self) -> None:
        """Update the ``updated_at`` timestamp to now."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
