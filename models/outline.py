"""Outline data model.

Represents the structural outline produced in the outline stage.
Contains the overall structure type, target word count, and a
per-section plan with act labels, events, and transitions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutlineSection(BaseModel):
    """Plan for a single section within the outline.

    Attributes:
        index: Zero-based section position.
        title: Short descriptive title.
        act_label: Act or phase label (e.g., "Act I — Setup").
        target_words: Planned word count for this section.
        key_events: Major plot events in this section.
        characters_present: Characters active in this section.
        transition_from: How this section connects from the previous.
        transition_to: How this section leads into the next.
    """

    index: int = Field(description="Zero-based section position.")
    title: str = Field(default="", description="Short descriptive title.")
    act_label: str = Field(default="", description="Act or phase label.")
    target_words: int = Field(
        default=0,
        ge=0,
        description="Planned word count.",
    )
    key_events: list[str] = Field(
        default_factory=list,
        description="Major plot events.",
    )
    characters_present: list[str] = Field(
        default_factory=list,
        description="Characters active in this section.",
    )
    transition_from: str = Field(
        default="",
        description="Connection from previous section.",
    )
    transition_to: str = Field(
        default="",
        description="Lead-in to next section.",
    )


class Outline(BaseModel):
    """Structural outline for a story.

    Produced by the outline step and consumed by the section-generation
    and stitching steps.

    Attributes:
        structure_type: Name of the structure template used.
        total_target_words: Combined target word count across all sections.
        sections: Ordered list of section plans.
    """

    structure_type: str = Field(
        default="three_act",
        description="Structure template name.",
    )
    total_target_words: int = Field(
        default=0,
        ge=0,
        description="Combined target word count.",
    )
    sections: list[OutlineSection] = Field(
        default_factory=list,
        description="Ordered section plans.",
    )
