"""Section data model.

Represents a single section within a story, produced during the
section-by-section generation stage of the full pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A single section of a generated story.

    Attributes:
        index: Zero-based position within the story.
        title: Short descriptive title for this section.
        target_words: Intended word count for this section.
        key_events: Plot events that should occur in this section.
        characters_present: Character names active in this section.
        transition_from: Brief note on how this section begins
            (connecting to the previous section).
        transition_to: Brief note on how this section ends
            (leading into the next section).
        text: The generated prose for this section (empty until written).
        summary: A short summary used as context for subsequent sections.
        word_count: Actual word count after generation.
    """

    index: int = Field(
        description="Zero-based section index within the story.",
    )
    title: str = Field(
        default="",
        description="Short descriptive title.",
    )
    target_words: int = Field(
        default=0,
        ge=0,
        description="Target word count for this section.",
    )
    key_events: list[str] = Field(
        default_factory=list,
        description="Plot events occurring in this section.",
    )
    characters_present: list[str] = Field(
        default_factory=list,
        description="Characters active in this section.",
    )
    transition_from: str = Field(
        default="",
        description="How this section connects from the previous one.",
    )
    transition_to: str = Field(
        default="",
        description="How this section leads into the next one.",
    )
    text: str = Field(
        default="",
        description="Generated prose text (empty until written).",
    )
    summary: str = Field(
        default="",
        description="Short summary for context propagation.",
    )
    word_count: int = Field(
        default=0,
        ge=0,
        description="Actual word count after generation.",
    )
