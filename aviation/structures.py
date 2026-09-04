"""Six documented narrative structures with per-structure LLM prompts.

Adapted from the ``06_NARRATIVE_STRUCTURES`` design brief. Each
structure has:

* a short brand-facing name and a one-line description;
* a canonical 13-scene beat sheet (the outline LLM produces one JSON
  chapter per beat);
* an aviation-flavoured LLM prompt template with the structure's hard
  rules (things it must / must not do);
* ``when_to_use`` / ``when_not_to_use`` guidance the decision-tree
  helper draws from;
* a target quarterly quota (see ``STRUCTURE_QUARTERLY_QUOTAS`` in
  :mod:`aviation.axes`).

The chapter-planner step calls :func:`prompt_for` to load the outline
prompt for the chosen structure; the writer / editor prompts stay
common (they just receive the outline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from aviation.axes import NarrativeStructureV2 as Structure
from aviation.axes import STRUCTURE_QUARTERLY_QUOTAS


@dataclass(frozen=True)
class StructureSpec:
    key: Structure
    display_name: str
    tagline: str
    quarterly_quota: int
    scene_count: int
    beat_summary: str
    when_to_use: list[str] = field(default_factory=list)
    when_not_to_use: list[str] = field(default_factory=list)


# ── beat summaries + rules ────────────────────────────────────────────

SPECS: dict[Structure, StructureSpec] = {
    Structure.THREE_ACT: StructureSpec(
        key=Structure.THREE_ACT,
        display_name="Three-Act Classic",
        tagline="Hook → Setup → Escalation → Twist → Climax → Resolution.",
        quarterly_quota=STRUCTURE_QUARTERLY_QUOTAS[Structure.THREE_ACT],
        scene_count=13,
        beat_summary=(
            "Sc1 cold-open hook (0-3 min). Sc2 protagonist setup (3-6). "
            "Sc3-4 flight setup, foreshadowing (6-13). Sc5 FIRST TURN: "
            "incident begins (13-17). Sc6-7 escalation (17-25). Sc8 MIDPOINT "
            "twist begins to reveal (25-28). Sc9-10 maximum pressure (28-35). "
            "Sc11 CLIMAX (35-40). Sc12 resolution (40-43). Sc13 CODA — "
            "editorial voice on what changed after this."
        ),
        when_to_use=[
            "Standard single-protagonist incidents with a clear failure mode.",
            "Miracle emergency landings, cabin crisis, in-flight medical.",
            "You want a safe high-retention default.",
        ],
        when_not_to_use=[
            "Mystery / investigative narratives.",
            "Stories with multiple equally-important POVs.",
        ],
    ),
    Structure.KISHOTENKETSU: StructureSpec(
        key=Structure.KISHOTENKETSU,
        display_name="Kishōtenketsu",
        tagline="Japanese four-part with no conflict in the first half; the twist recontextualises everything.",
        quarterly_quota=STRUCTURE_QUARTERLY_QUOTAS[Structure.KISHOTENKETSU],
        scene_count=13,
        beat_summary=(
            "KI (sc1-4, 0-12 min): introduce hero + routine — NO conflict, "
            "no foreshadowing. Plant details that will matter. "
            "SHŌ (sc5-8, 12-25 min): deepen the world, competence beats. "
            "Only the last scene of SHŌ carries the first unexplained signal. "
            "TEN (sc9-10, 25-32 min): SHARP turn out of nowhere — the truth "
            "that recontextualises everything in Ki/Shō. "
            "KETSU (sc11-13, 32-45 min): action in the new reality; coda returns "
            "to Ki details with fresh meaning."
        ),
        when_to_use=[
            "ATC hero, In-flight medical (nurse POV), General aviation (bush pilot routine).",
            "Sleep-friendly, background-listening content.",
            "Emotional beat: Redemption or Unlikely hero recognised.",
        ],
        when_not_to_use=[
            "Miracle emergency landing (needs tension from second one).",
            "Seed incident that is a sudden event with no preamble.",
        ],
    ),
    Structure.IN_MEDIA_RES: StructureSpec(
        key=Structure.IN_MEDIA_RES,
        display_name="In Media Res + Nested Flashback",
        tagline="Open at the moment of maximum danger, cut to 24 hours earlier, then walk back to and past it.",
        quarterly_quota=STRUCTURE_QUARTERLY_QUOTAS[Structure.IN_MEDIA_RES],
        scene_count=13,
        beat_summary=(
            "PART 1 OPEN AT CLIMAX (sc1-2, 0-8 min): drop the viewer into "
            "the critical moment, no context. Sc2 ends on decision-in-motion; "
            "title card 'Twenty-four hours earlier'. "
            "PART 2 NESTED FLASHBACK (sc3-11, 8-38 min): walk chronologically "
            "from 24 h before back to the open. At least 3 details echo the "
            "open (a jacket, a phrase, a gesture). "
            "PART 3 BEYOND THE OPEN (sc12-13, 38-45 min): sc12 catches up to "
            "and continues past the open — resolution. Sc13 coda (months / years later)."
        ),
        when_to_use=[
            "Miracle emergency landing, Weather disaster, Mechanical failure.",
            "Incidents with an iconic critical moment usable as cold open.",
            "Protagonist has one recognisable image to anchor.",
        ],
        when_not_to_use=[
            "Mystery stories — kills the 'what happened' hook.",
            "Investigation-heavy narratives.",
        ],
    ),
    Structure.RASHOMON: StructureSpec(
        key=Structure.RASHOMON,
        display_name="Rashomon-Style Multi-POV",
        tagline="One incident, 3-4 POVs; each adds information the others lacked.",
        quarterly_quota=STRUCTURE_QUARTERLY_QUOTAS[Structure.RASHOMON],
        scene_count=13,
        beat_summary=(
            "PROLOGUE (sc1, 0-3 min): framing device (investigator asks a "
            "question). "
            "POV 1 CAPTAIN (sc2-4, 3-15 min): full cockpit-side experience, "
            "ends on certainty that will be undermined. "
            "POV 2 CABIN (sc5-7, 15-25 min): re-runs the flight with human "
            "detail cockpit could not see. Cliffhanger: a hint captain didn't know. "
            "POV 3 PASSENGER / OUTSIDER (sc8-10, 25-35 min): a third angle. "
            "POV 4 SYNTHESIS (sc11-12, 35-42 min): investigator assembles all "
            "three; reveals what really happened. "
            "CODA (sc13, 42-45 min): how each POV lives with the day."
        ),
        when_to_use=[
            "Incidents with multiple parties who each saw a different piece.",
            "Cabin crisis, Mechanical failure, Aviation mystery.",
            "Emotional beat: 'Unlikely hero recognised'.",
        ],
        when_not_to_use=[
            "Solo pilot incidents (no natural extra POVs).",
            "Short incidents (< 5 min real-time — not enough material for 4 POVs).",
            "One POV radically stronger than the others — go Three-Act instead.",
        ],
    ),
    Structure.INVESTIGATION: StructureSpec(
        key=Structure.INVESTIGATION,
        display_name="Investigation-First (Reverse Chronology)",
        tagline="NTSB investigator's POV; the incident unfolds through discovery-driven flashbacks.",
        quarterly_quota=STRUCTURE_QUARTERLY_QUOTAS[Structure.INVESTIGATION],
        scene_count=13,
        beat_summary=(
            "PART 1 PROLOGUE AT SCENE (sc1-2, 0-8 min): investigator arrives "
            "days/weeks later. Three questions she seeks answers to. "
            "PART 2 DISCOVERY-DRIVEN FLASHBACKS (sc3-11, 8-40 min): each scene = "
            "one discovery + one 2-4-min flashback (not chronological). Around "
            "sc8-9 hypothesis forms. Sc10-11 obstacles (corporate pushback, "
            "missing evidence) then the final piece breaks the block. "
            "PART 3 RECONSTRUCTION + CODA (sc12-13, 40-45 min): sc12 is the "
            "ONLY time we see the whole incident linearly. Sc13 industry impact."
        ),
        when_to_use=[
            "Aviation mystery, Mechanical failure with unclear cause.",
            "Emotional beats: 'Bureaucratic injustice overcome', 'Investigation reveals hero action posthumously'.",
            "45-minute format is your friend here.",
        ],
        when_not_to_use=[
            "Ordinary miracle landings where the cause is obvious in minute one.",
            "Incidents whose investigation was trivial or uninteresting.",
        ],
    ),
    Structure.BRAID: StructureSpec(
        key=Structure.BRAID,
        display_name="Documentary Braid (Parallel Stories)",
        tagline="Three story lines braided: main incident (present) + historical parallel + meta industry context.",
        quarterly_quota=STRUCTURE_QUARTERLY_QUOTAS[Structure.BRAID],
        scene_count=13,
        beat_summary=(
            "OPENING TRIPLE (sc1-3, 0-8 min): snapshot of the main incident, "
            "snapshot of the historical parallel, meta-context (3 min each). "
            "BRAID (sc4-11, 8-40 min): alternate Main → Historical → Meta → "
            "Main → Historical → Meta → Main → Historical. Each scene 3-5 min. "
            "Split roughly 45% Main / 35% Historical / 20% Meta. "
            "CONVERGENCE (sc12-13, 40-45 min): all three braids converge — "
            "investigation of main incident shows what happened / didn't happen "
            "to historical lessons. Coda: where the industry stands now."
        ),
        when_to_use=[
            "Incidents with a known historical parallel (metal fatigue, icing, CFIT).",
            "Mechanical failure recreation, Aviation mystery.",
            "Educated / enthusiast audience.",
        ],
        when_not_to_use=[
            "Unique incidents without a natural parallel.",
            "Miracle landings / ATC hero — braid dilutes the individual focus.",
        ],
    ),
}


# ── outline prompt per structure ───────────────────────────────────────

_COMMON_HEADER = (
    "__STEP__=outline\n"
    "You are building the chapter-by-chapter beat sheet for a long-form "
    "aviation script for YouTube narration (~45 minutes at 150 wpm ≈ "
    "8,500-9,500 words).\n\n"
    "STRUCTURE: {structure_name} — {tagline}\n"
    "REQUIRED SCENE COUNT: 13\n\n"
    "STORY BIBLE (do not contradict):\n{story_bible}\n\n"
)


_COMMON_OUTPUT = (
    "\n\nReturn ONLY a JSON array — one object per chapter, in order:\n"
    "[\n"
    '  {"index": 0, "title": "…", "act_label": "…", '
    '"outline_bullets": ["beat","beat"], "target_words": 650, '
    '"opens_with": "…", "ends_with": "cliffhanger phrase"}\n'
    "]\n"
    "Every chapter MUST end with a cliffhanger, an open question, or a "
    "clear scene shift. No flat endings.\n"
)


THREE_ACT_RULES = (
    "SCENE MAP (adhere strictly):\n"
    " • Sc1 (0-3 min): COLD OPEN. Danger or mystery, no explanation. Ends: "
    "'But to understand how we got here, we need to go back.'\n"
    " • Sc2 (3-6): protagonist setup. Who they are, gentle hint that not "
    "all is well.\n"
    " • Sc3-4 (6-13): flight setup. Small details that will matter. First "
    "quiet warning signal.\n"
    " • Sc5 (13-17): FIRST TURN. The incident begins.\n"
    " • Sc6-7 (17-25): escalation. First attempts to solve it fail. Bring "
    "in a secondary character.\n"
    " • Sc8 (25-28): MIDPOINT. Key revelation. The twist begins to open up.\n"
    " • Sc9-10 (28-35): maximum pressure. Slow the pace only for the "
    "decisive decision.\n"
    " • Sc11 (35-40): CLIMAX. The decisive act. Resolution begins.\n"
    " • Sc12 (40-43): resolution — the aftermath in the seat, on the ground.\n"
    " • Sc13 (43-45): CODA. The editorial voice — how this case changed "
    "an industry rule or a personal practice.\n"
)


KISHOTENKETSU_RULES = (
    "SCENE MAP (adhere strictly):\n"
    "KI — Introduction (sc1-4, 25% ≈ 12 min): the world and the routine.\n"
    " • NO conflict. NO foreshadowing.\n"
    " • Plant 2-3 details that will matter in Ten but read as pure "
    "background here.\n"
    "SHŌ — Development (sc5-8, 30% ≈ 13 min): deepen the world.\n"
    " • Introduce a secondary character.\n"
    " • Show competence in routine work.\n"
    " • Only the LAST scene may carry the first unexplained signal — still "
    "not alarming.\n"
    "TEN — Turn (sc9-10, 15% ≈ 7 min): SHARP change out of nowhere.\n"
    " • Not an escalation of Shō — a new angle that recontextualises Ki/Shō.\n"
    " • The audience must feel an 'oh shit' moment.\n"
    "KETSU — Conclusion (sc11-13, 30% ≈ 13 min):\n"
    " • Action in the new reality. Resolution. Coda returning to a Ki detail "
    "with new meaning.\n\n"
    "HARD BANS:\n"
    " • Sc1 may NOT be a hook-teaser. No cold open. Start with routine.\n"
    " • In sc1-8 you may NOT use the words: 'danger', 'wrong', 'unusual', "
    "'sensed', 'little did she know', 'ominous'.\n"
    " • Twist in Ten must be built from a Ki/Shō detail but may not be "
    "predictable from it.\n"
)


IN_MEDIA_RES_RULES = (
    "SCENE MAP (adhere strictly):\n"
    "PART 1 — OPEN AT CLIMAX (sc1-2, 0-8 min):\n"
    " • Sc1: straight into the critical moment. No setup. Technical detail, "
    "sensory overload. Ends on maximum uncertainty.\n"
    " • Sc2: continue the same beat 3-4 min. Key decision. Cut on the moment "
    "of execution. Title card: 'Twenty-four hours earlier.'\n"
    "PART 2 — NESTED FLASHBACK (sc3-11, 8-38 min):\n"
    " • Sc3: quiet contrast beat 24 h earlier.\n"
    " • Sc4-9: full setup of the incident chronologically. Every scene must "
    "carry at least ONE detail that later echoes the open (an object, a "
    "phrase, a gesture, a location).\n"
    " • Sc10-11: approach the moment of the open. A sense of inevitability.\n"
    "PART 3 — BEYOND THE OPEN (sc12-13, 38-45 min):\n"
    " • Sc12: catch up to sc2 and continue past it. Resolution.\n"
    " • Sc13: coda much later (months / years).\n\n"
    "HARD BANS:\n"
    " • The open (sc1-2) must work as a self-contained micro-narrative.\n"
    " • DO NOT spoil the outcome in the open — show the critical moment, "
    "not the resolution.\n"
    " • Between sc2 and sc12 there must be NO temporal gap: sc12 picks up "
    "exactly from the cut in sc2.\n"
)


RASHOMON_RULES = (
    "SCENE MAP (adhere strictly):\n"
    "PROLOGUE (sc1, 0-3 min): framing device (investigator or narrator "
    "asks 'what happened on Flight X?').\n"
    "POV 1 — CAPTAIN (sc2-4, 3-15 min): full cockpit-side POV. Technical. "
    "Ends on certainty that will be undermined.\n"
    "POV 2 — CABIN (sc5-7, 15-25 min): re-run the flight from cabin POV. "
    "Adds information cockpit could not see. Cliffhanger: hint that the "
    "captain doesn't know something.\n"
    "POV 3 — PASSENGER / OUTSIDER (sc8-10, 25-35 min): a third angle "
    "(off-duty pilot, engineer, radio ham, ATC). Cliffhanger: overall "
    "picture is more complex than either POV suggested.\n"
    "POV 4 — SYNTHESIS (sc11-12, 35-42 min): investigator assembles the "
    "three. Reveals what really happened.\n"
    "CODA (sc13, 42-45 min): how each POV now lives with it.\n\n"
    "HARD BANS:\n"
    " • Each POV must ADD information, never repeat.\n"
    " • Different POVs may contradict — that's a feature.\n"
    " • Each POV has its own voice: captain (technical, sure), cabin "
    "(emotional, observant), passenger (human, one unique advantage), "
    "investigator (analytical).\n"
    " • POV shifts must be signalled clearly (time marker OR explicit "
    "'now from X's perspective…').\n"
)


INVESTIGATION_RULES = (
    "SCENE MAP (adhere strictly):\n"
    "PART 1 — PROLOGUE AT SCENE (sc1-2, 0-8 min):\n"
    " • Investigator arrives days/weeks after. Give them a name, age, "
    "personal stake (career, family, prior similar case).\n"
    " • Establish three questions they seek answers to.\n"
    " • NO flashbacks yet.\n"
    "PART 2 — DISCOVERY-DRIVEN FLASHBACKS (sc3-11, 8-40 min):\n"
    " • Each scene = one discovery + one 2-4-min flashback fragment (NOT "
    "in the incident's chronological order).\n"
    " • Order of discoveries: physical evidence → recorder data → witness "
    "testimony → cross-reference.\n"
    " • By sc8-9 the investigator and the audience share a hypothesis.\n"
    " • Sc10-11: obstacles (corporate pushback, insufficient evidence) then "
    "the final piece breaks the block.\n"
    "PART 3 — RECONSTRUCTION + CODA (sc12-13, 40-45 min):\n"
    " • Sc12: full linear reconstruction of the incident (the ONLY time we "
    "see it linearly).\n"
    " • Sc13: coda much later, industry impact.\n\n"
    "HARD BANS:\n"
    " • Sc3-11 flashbacks must be FRAGMENTS, not the incident in order.\n"
    " • The investigator is an ACTIVE hero, not a passive narrator.\n"
    " • Aviation mysteries work best when the cause is a combination "
    "of factors (Swiss cheese), not a single cause.\n"
)


BRAID_RULES = (
    "SCENE MAP (adhere strictly):\n"
    "OPENING TRIPLE (sc1-3, 0-8 min): snapshot Main incident (3 min); "
    "snapshot Historical parallel (3 min); Meta-context — what this class "
    "of problem is (2 min).\n"
    "BRAID (sc4-11, 8-40 min): alternate strands M → H → X → M → H → X "
    "→ M → H. Each scene 3-5 min, each scene ADDS information to its own "
    "strand.\n"
    " • M strand develops chronologically.\n"
    " • H strand develops chronologically.\n"
    " • X strand develops thematically (not chronologically).\n"
    "CONVERGENCE (sc12-13, 40-45 min):\n"
    " • Sc12: three strands converge — main-incident investigation reveals "
    "whether historical lessons were applied; meta strand explains why or "
    "why not.\n"
    " • Sc13: coda — where the industry stands now.\n\n"
    "HARD RULES:\n"
    " • Historical parallel must be a REAL known incident (Aloha 243, "
    "Tenerife, Sioux City, Hudson — keep its real name; the historical "
    "citation is part of authenticity).\n"
    " • Meta strand must be FACTUAL — real regulations, real training "
    "programmes, real engineering evolution.\n"
    " • Transitions between strands must be seamless via thematic echoes "
    "('Like the captain 35 years earlier, he reached for the rudder pedal…').\n"
    " • Not 33% each — Main ≈ 45%, Historical ≈ 35%, Meta ≈ 20%.\n"
)


_RULES: dict[Structure, str] = {
    Structure.THREE_ACT: THREE_ACT_RULES,
    Structure.KISHOTENKETSU: KISHOTENKETSU_RULES,
    Structure.IN_MEDIA_RES: IN_MEDIA_RES_RULES,
    Structure.RASHOMON: RASHOMON_RULES,
    Structure.INVESTIGATION: INVESTIGATION_RULES,
    Structure.BRAID: BRAID_RULES,
}


def prompt_for(structure: Structure, story_bible_json: str) -> str:
    """Assemble the outline-generation prompt for a chosen structure."""
    spec = SPECS[structure]
    header = _COMMON_HEADER.format(
        structure_name=spec.display_name,
        tagline=spec.tagline,
        story_bible=story_bible_json,
    )
    return header + _RULES[structure] + _COMMON_OUTPUT


# ── decision-tree helper ────────────────────────────────────────────

def suggest_structure(
    *,
    has_historical_parallel: bool = False,
    multiple_povs_available: bool = False,
    cause_is_mystery: bool = False,
    has_iconic_critical_moment: bool = True,
    routine_pro_with_surprise: bool = False,
) -> Structure:
    """Walk the decision tree from the design brief."""
    if has_historical_parallel:
        return Structure.BRAID if not cause_is_mystery else Structure.INVESTIGATION
    if multiple_povs_available:
        return Structure.RASHOMON
    if cause_is_mystery:
        return Structure.INVESTIGATION
    if has_iconic_critical_moment:
        return Structure.IN_MEDIA_RES
    if routine_pro_with_surprise:
        return Structure.KISHOTENKETSU
    return Structure.THREE_ACT


# ── rotation ────────────────────────────────────────────────────────

_ROTATION_ORDER_V2: list[Structure] = [
    Structure.THREE_ACT,
    Structure.IN_MEDIA_RES,
    Structure.KISHOTENKETSU,
    Structure.THREE_ACT,
    Structure.INVESTIGATION,
    Structure.BRAID,
    Structure.INVESTIGATION,
    Structure.THREE_ACT,
    Structure.RASHOMON,
    Structure.IN_MEDIA_RES,
]


def next_in_rotation(previous: Structure | None) -> Structure:
    """Pick the next structure in the recommended-startup rotation
    (from Layer 6 assignment table). Called only when the caller does not
    ask :func:`suggest_structure` for a data-driven pick.
    """
    if previous is None:
        return _ROTATION_ORDER_V2[0]
    try:
        idx = _ROTATION_ORDER_V2.index(previous)
    except ValueError:
        return _ROTATION_ORDER_V2[0]
    return _ROTATION_ORDER_V2[(idx + 1) % len(_ROTATION_ORDER_V2)]


def all_structures() -> Iterable[StructureSpec]:
    return SPECS.values()
