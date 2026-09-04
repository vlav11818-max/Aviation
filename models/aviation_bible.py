"""Aviation-flavoured Story Bible and RAG extraction schemas.

An :class:`AviationStoryBible` is the aviation-domain version of the
generic ``StoryBible``. It carries everything the writing agents need
to stay technically correct and narratively consistent across a
15,000-word manuscript:

* the aircraft, operator, route, crew, and cabin/cargo load;
* the technical timeline of the incident (UTC + local + altitude +
  airspeed + system state at each beat);
* the causal chain and contributing factors (regulator-style);
* known CVR excerpts, notable quotes, glossary;
* the narrative structure being used and the mode
  (:class:`Mode.REAL` for RAG-grounded, :class:`Mode.FICTIONAL` for
  invented).

:class:`ExtractedFacts` is the JSON schema the ingest step fills in
from an uploaded accident-report PDF. The Fact-Checker step compares
each generated chapter against these facts.

These models coexist with the generic :mod:`models.story_bible` — the
existing steps still work with a generic ``StoryBible``, and the new
aviation-specific steps use ``AviationStoryBible``. A helper
:meth:`AviationStoryBible.to_generic` collapses the aviation model
into a generic one so the generic critic/evaluate/revise steps can
receive it as a ``story_bible`` field on ``PipelineState`` without
schema changes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from models.story_bible import Character, Setting, StoryBible


class Mode(str, Enum):
    """Whether the incident is grounded in a real report or invented."""

    REAL = "real"
    FICTIONAL = "fictional"


class NarrativeStructure(str, Enum):
    """Aviation-specific narrative structures.

    The Global History Manager rotates through these in order so
    successive batch stories don't all sound the same.
    """

    IN_MEDIA_RES = "in_media_res"
    THREE_ACT = "three_act"
    RASHOMON = "rashomon"
    REVERSE_CHRONOLOGICAL = "reverse_chronological"
    FRAME_STORY = "frame_story"


ROTATION_ORDER: list[NarrativeStructure] = [
    NarrativeStructure.IN_MEDIA_RES,
    NarrativeStructure.THREE_ACT,
    NarrativeStructure.RASHOMON,
    NarrativeStructure.REVERSE_CHRONOLOGICAL,
    NarrativeStructure.FRAME_STORY,
]


# ── Aircraft, route, crew ─────────────────────────────────────────────


class Aircraft(BaseModel):
    """Aircraft identification and configuration."""

    type: str = Field(default="", description="Airframe type, e.g. 'Airbus A320-232'.")
    operator: str = Field(default="", description="Operating airline / carrier / operator.")
    registration: str = Field(default="", description="Tail number, e.g. 'N123AB' or 'D-AIPT'.")
    flight_number: str = Field(default="", description="Flight number, e.g. 'AB1234'.")
    msn: str = Field(default="", description="Manufacturer serial number, if known.")
    engines: str = Field(default="", description="Powerplant, e.g. '2 × CFM56-5B4/P'.")
    seat_capacity: Optional[int] = Field(default=None, description="Configured seat capacity.")


class Route(BaseModel):
    """Filed and actual route information."""

    origin: str = Field(default="", description="Departure airport (city, ICAO/IATA).")
    destination: str = Field(default="", description="Filed destination.")
    alternate: str = Field(default="", description="Filed alternate, if any.")
    filed_altitude: str = Field(default="", description="Filed cruise altitude (e.g. 'FL370').")
    actual_diversion: str = Field(default="", description="Actual airport used, if different.")


class CrewMember(BaseModel):
    """A named human involved in the flight."""

    name: str = Field(description="Personal name.")
    role: str = Field(default="", description="Role: captain, first officer, cabin crew, ATC, jump seat…")
    seat: str = Field(default="", description="Physical seat, e.g. 'left', 'right', 'observer', 'ATC-KEF-APP'.")
    hours_total: Optional[int] = Field(default=None, description="Total flight hours, if reported.")
    hours_on_type: Optional[int] = Field(default=None, description="Hours on this aircraft type.")
    traits: list[str] = Field(default_factory=list, description="Personality traits used by the writer.")
    arc: str = Field(default="", description="Narrative arc, if any.")


# ── Timeline & facts ──────────────────────────────────────────────────


class TimelineEvent(BaseModel):
    """One entry on the incident timeline."""

    time_utc: str = Field(default="", description="UTC time, e.g. '02:14:33Z'.")
    time_local: str = Field(default="", description="Local time, e.g. '21:14:33 EST'.")
    altitude: str = Field(default="", description="Altitude at event, e.g. 'FL370' or '3,200 ft AGL'.")
    airspeed: str = Field(default="", description="Indicated airspeed, e.g. '280 KIAS' or '.82 M'.")
    phase: str = Field(default="", description="Phase of flight (taxi, takeoff, climb, cruise, descent, approach, landing).")
    description: str = Field(description="What happened at this beat.")
    source_reference: str = Field(default="", description="Citation into the source doc (page/section).")


class CausalLink(BaseModel):
    """One link in the causal chain."""

    factor: str = Field(description="Factor name (e.g. 'crew fatigue', 'sensor freeze').")
    role: str = Field(default="contributing", description="'primary' | 'contributing' | 'latent'.")
    description: str = Field(default="", description="One-sentence explanation.")


# ── Extracted facts (RAG output) ──────────────────────────────────────


class ExtractedFacts(BaseModel):
    """Structured facts extracted from an accident-report PDF.

    Filled by the IngestStep; consumed by the Planner (to seed the
    aviation StoryBible) and by the FactCheckerStep (to verify each
    chapter's technical claims).
    """

    incident_name: str = Field(default="", description="Common name of the incident.")
    date: str = Field(default="", description="ISO date of the incident.")
    location: str = Field(default="", description="Impact / diversion / event location.")
    operator_and_flight: str = Field(default="", description="e.g. 'Air France 447'.")
    aircraft: Aircraft = Field(default_factory=Aircraft)
    crew: list[str] = Field(default_factory=list, description="Names / callsigns from the report.")
    sequence_of_events: list[str] = Field(
        default_factory=list, description="Ordered, terse beat list from the report."
    )
    probable_cause: str = Field(default="", description="Probable cause finding.")
    contributing_factors: list[str] = Field(default_factory=list)
    technical_findings: list[str] = Field(default_factory=list)
    cvr_highlights: list[str] = Field(default_factory=list, description="Notable CVR excerpts, verbatim.")
    casualties: str = Field(default="", description="Casualty summary line, e.g. '228 fatalities of 228 souls onboard'.")
    recommendations: list[str] = Field(default_factory=list)
    notable_quotes: list[str] = Field(default_factory=list)


# ── Aviation Story Bible ──────────────────────────────────────────────


class AviationStoryBible(BaseModel):
    """Compact reference document for one aviation-story generation run.

    Everything downstream steps need to stay technically correct and
    narratively consistent.
    """

    mode: Mode = Field(default=Mode.FICTIONAL, description="Real (RAG) vs fictional.")
    working_title: str = Field(default="", description="Working title of the finished narrative.")
    premise: str = Field(default="", description="One-paragraph story premise.")
    logline: str = Field(default="", description="One-sentence logline used for the YouTube hook.")

    aircraft: Aircraft = Field(default_factory=Aircraft)
    route: Route = Field(default_factory=Route)
    crew: list[CrewMember] = Field(default_factory=list, description="Cockpit + ATC + notable cabin crew.")
    other_characters: list[Character] = Field(
        default_factory=list, description="Passengers, engineers, investigators, family members named on-page."
    )

    timeline: list[TimelineEvent] = Field(default_factory=list)
    technical_facts: list[str] = Field(
        default_factory=list,
        description="Immutable technical claims the writer must never contradict.",
    )
    causal_chain: list[CausalLink] = Field(default_factory=list)
    cvr_excerpts: list[str] = Field(default_factory=list, description="Verbatim CVR quotes (source-truth).")
    glossary: dict[str, str] = Field(
        default_factory=dict,
        description="Term → gloss so the narrator can define jargon on first use.",
    )

    narrative_structure: NarrativeStructure = Field(
        default=NarrativeStructure.THREE_ACT,
        description="Chosen narrative structure. Set by the Global History Manager for a fresh incident.",
    )
    tone_description: str = Field(default="", description="Voice / register.")
    narrative_voice: str = Field(default="", description="Who is narrating and how.")
    key_rules: list[str] = Field(
        default_factory=list,
        description="Hard constraints (no graphic injury, real-ATC phraseology, UTC timestamps, etc.).",
    )
    retention_plan: list[str] = Field(
        default_factory=list,
        description="Retention techniques the writer will apply.",
    )

    fictionalization_notice: str = Field(
        default="",
        description="Mode-aware disclaimer text for the YouTube description.",
    )

    def to_generic(self) -> StoryBible:
        """Collapse to the generic :class:`StoryBible` so the existing
        critic/evaluate/revise steps (which know the generic schema
        only) can consume us as their ``story_bible`` field.

        Kept intentionally lossy — the generic view is only used by
        steps that don't inspect aviation-specific fields.
        """
        chars: list[Character] = []
        for c in self.crew:
            chars.append(Character(name=c.name, role=c.role, traits=c.traits, arc=c.arc))
        chars.extend(self.other_characters)

        setting = Setting(
            location=f"{self.route.origin} → {self.route.destination}"
            + (f" (diverted to {self.route.actual_diversion})" if self.route.actual_diversion else ""),
            time_period=self.timeline[0].time_utc if self.timeline else "",
            atmosphere=self.tone_description,
        )
        return StoryBible(
            premise=self.premise,
            setting=setting,
            characters=chars,
            themes=[c.factor for c in self.causal_chain if c.factor][:5],
            tone_description=self.tone_description,
            narrative_voice=self.narrative_voice,
            key_rules=self.key_rules,
        )
