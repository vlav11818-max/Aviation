"""Story Bible data model.

The Story Bible is a compact reference document produced in the concept
stage and carried through the entire pipeline as the single source of
truth for setting, characters, tone, and narrative rules.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Character(BaseModel):
    """A character defined in the Story Bible.

    Attributes:
        name: Character's display name.
        role: Narrative role (protagonist, antagonist, mentor, etc.).
        traits: Key personality traits.
        arc: Brief description of the character's arc over the story.
    """

    name: str = Field(description="Character display name.")
    role: str = Field(default="", description="Narrative role.")
    traits: list[str] = Field(
        default_factory=list,
        description="Key personality traits.",
    )
    arc: str = Field(
        default="",
        description="Brief character arc description.",
    )


class Setting(BaseModel):
    """Story setting details.

    Attributes:
        location: Primary location or world description.
        time_period: Temporal setting (era, year, season, etc.).
        atmosphere: Overall mood or atmosphere of the setting.
    """

    location: str = Field(default="", description="Primary location or world.")
    time_period: str = Field(default="", description="Temporal setting.")
    atmosphere: str = Field(default="", description="Mood/atmosphere.")


class StoryBible(BaseModel):
    """Compact reference document for a story generation run.

    Contains all foundational information that every pipeline step
    needs: premise, setting, characters, themes, tone description,
    narrative voice, and key rules that must be respected.

    Attributes:
        premise: One-paragraph story premise.
        setting: Location, time, and atmosphere details.
        characters: List of characters with roles and arcs.
        themes: Thematic threads running through the story.
        tone_description: Prose description of the target tone.
        narrative_voice: Description of the narrator's voice.
        key_rules: Hard constraints (e.g., no deaths, keep PG).
    """

    premise: str = Field(
        default="",
        description="One-paragraph story premise.",
    )
    setting: Setting = Field(
        default_factory=Setting,
        description="Setting details (location, time, atmosphere).",
    )
    characters: list[Character] = Field(
        default_factory=list,
        description="Characters with roles, traits, and arcs.",
    )
    themes: list[str] = Field(
        default_factory=list,
        description="Thematic threads.",
    )
    tone_description: str = Field(
        default="",
        description="Prose description of the target tone.",
    )
    narrative_voice: str = Field(
        default="",
        description="Description of the narrator's voice.",
    )
    key_rules: list[str] = Field(
        default_factory=list,
        description="Hard narrative constraints.",
    )
