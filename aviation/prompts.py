"""All prompts used by the aviation content factory.

Kept inline (Python strings) rather than in separate template files
because the aviation flow is a fixed short list of prompts and the
in-line form is easier to keep in sync with the schemas they produce.

Every prompt injects a ``__STEP__=<name>`` marker at the top so the
offline mock provider knows which schema to satisfy (see
:mod:`core.llm.mock_provider`).
"""

from __future__ import annotations

from typing import Iterable

from models.aviation_bible import (
    AviationStoryBible,
    ExtractedFacts,
    Mode,
    NarrativeStructure,
)


# ── shared style block ────────────────────────────────────────────────

AVIATION_STYLE_RULES = """\
AVIATION VOICE & TECHNICAL RULES:
- Use standard ATC phraseology for radio calls: callsign first, then
  clearance / request. Example: 'Speedbird one-two, cleared to land
  runway two seven left.'
- Timestamps in UTC (Zulu) unless a specific local time is dramatic.
- Altitudes in flight levels (FL370) above 18,000 ft, in feet MSL
  below. Airspeeds in KIAS (knots indicated) below FL280, .82 M
  and similar above.
- Aircraft systems: name them like line pilots do (ECAM, EICAS,
  autothrust, FADEC, FMS, PFD, ND, HYD Y, GEN 2, PACK 1, RAT).
- Radio callsigns are voiced as the operator's telephony name, not
  the marketing brand ('Speedbird' for British Airways, 'Ryanair'
  for Ryanair — most match, some don't).
- Never invent or falsify what the CVR / DFDR said. In real mode,
  only quote CVR excerpts that appear in the source. In fictional
  mode, mark invented dialogue clearly as internal narration rather
  than as verbatim CVR.
- Do NOT dramatise passenger injury on-screen. Keep casualty
  reporting in the narrator's voice, sober and specific.
- Weather: forecast + actual + significance. Not just 'stormy'.
- Do not use present-tense narration to fabricate a real crew's
  private thoughts. In real mode, keep interiority to what the
  report can support (checklists said, decisions made, radio calls).
- Numbers rule for TTS: write dramatic numbers as words ('thirty-seven
  thousand feet'), keep technical designators as digits ('Flight 447').
- Dialogue attribution is mandatory — TTS without attribution loses
  who is speaking. Every quoted line gets a 'said X' / 'the captain
  answered' clause.
"""

# Evergreen bans from Layer 5. Always injected into planner + writer prompts.
BLACKLIST_RULES = """\
HARD BLACKLIST (never violate):
- Never use real names of pilots / ATC / passengers from the last
  25 years. Real names of historical figures (1900-2000, e.g. Chuck
  Yeager, Kelly Johnson) are fine if the story does not portray
  them negatively.
- Never use real airline names for the incidents. Use fictional
  airline names from the pool the Planner is given.
- Never use active real-world flight numbers. Avoid 447, 232, 800 —
  they are all associated with real catastrophes.
- Never use registrations of real aircraft.
- Never use real ATC controller names.
- Never dramatise passenger injury or death in graphic sensory
  detail. Casualty statements stay in the narrator's voice, sober
  and specific.
- Never sexualise a rescue scene or post-crash situation.
- Never glorify pilot-suicide-by-plane (Germanwings, MH370-theory).
  Investigation-frame is acceptable; dramatisation is not.
- Never dramatise conspiracy theories about active investigations.
- Never carry deliberately anti-airline messaging. 'The system
  failed' is fine; 'Boeing is evil' is not.
- Never explain hijacking / security-bypass techniques in operational
  detail, even for historical incidents.
"""


RETENTION_TIPS = """\
RETENTION TECHNIQUES to apply throughout (pick at least three per
chapter):
1. Open loop — foreshadow a specific decision or reading, deliver it
   later.
2. Delayed revelation — hint that a system is quietly failing before
   the crew sees it.
3. Somatic detail — a hand hovering over the thrust levers; the low
   moan of a pressurisation change; the particular click of a
   selected transponder code.
4. Sensory anchor — a recurring detail that gains meaning (a
   passenger's silent baby, the yellow-limit tape on a gauge).
5. Echo — return to a phrase from the pre-flight briefing at the
   moment it becomes ironic or prophetic.
6. Breather scene — a beat of quiet between crises so the next
   crisis lands harder.
"""


# ── ingest ────────────────────────────────────────────────────────────

def ingest_prompt(chunk: str, filename: str) -> str:
    return (
        "__STEP__=ingest\n"
        "You are an aviation-incident analyst. Extract structured facts "
        "from the following excerpt of an official accident report. "
        "Return ONLY a JSON object matching the schema below, no prose "
        "before or after.\n\n"
        f"SOURCE FILE: {filename}\n\n"
        "EXCERPT (verbatim, may be truncated):\n"
        "---\n"
        f"{chunk}\n"
        "---\n\n"
        "SCHEMA (fill in only what is supported by this excerpt; leave "
        "fields empty when unsure — do not guess):\n"
        "{\n"
        '  "incident_name": "…",\n'
        '  "date": "YYYY-MM-DD",\n'
        '  "location": "…",\n'
        '  "operator_and_flight": "…",\n'
        '  "aircraft": {"type":"…","operator":"…","registration":"…","flight_number":"…"},\n'
        '  "crew": ["Captain X (…hrs)", "First Officer Y (…hrs)"],\n'
        '  "sequence_of_events": ["HH:MM:SSZ — beat", …],\n'
        '  "probable_cause": "…",\n'
        '  "contributing_factors": ["…"],\n'
        '  "technical_findings": ["…"],\n'
        '  "cvr_highlights": ["\\"verbatim quote\\" — role"],\n'
        '  "casualties": "…",\n'
        '  "recommendations": ["…"],\n'
        '  "notable_quotes": ["…"]\n'
        "}\n"
    )


# ── planner ───────────────────────────────────────────────────────────

def _crew_hint(mode: Mode, facts: ExtractedFacts | None) -> str:
    if mode == Mode.REAL and facts is not None:
        return (
            "REAL MODE — use the following extracted facts as the "
            "immutable source of truth:\n"
            f"{facts.model_dump_json(indent=2)}\n"
        )
    return (
        "FICTIONAL MODE — invent a realistic incident. All names, "
        "airlines, registrations, cities, and flight numbers must be "
        "invented and must not match any real airline or aircraft."
    )


STRUCTURE_GUIDANCE: dict[NarrativeStructure, str] = {
    NarrativeStructure.IN_MEDIA_RES: (
        "Cold-open at the moment of highest tension (the first ECAM "
        "warning, the sudden yaw, the smoke). Then fold back to the "
        "pre-flight briefing and walk forward chronologically until "
        "the timeline catches up with the cold open, then continue "
        "past it to the resolution."
    ),
    NarrativeStructure.THREE_ACT: (
        "Act I — routine dispatch, crew, aircraft, latent conditions. "
        "Act II — the trigger, the cascade, escalating decisions under "
        "compressing time. Act III — the outcome, the investigation, "
        "the systemic lesson."
    ),
    NarrativeStructure.RASHOMON: (
        "Rotate through 3-5 points of view of the same incident "
        "(captain, first officer, ATC, engineer on the ground, senior "
        "cabin crew). Each POV knows different things and reveals a "
        "different piece of the causal chain. The final chapter "
        "reconciles them into the full sequence."
    ),
    NarrativeStructure.REVERSE_CHRONOLOGICAL: (
        "Open on the aftermath — the investigator standing at the "
        "wreckage, or the family notification. Then walk backwards "
        "one beat per chapter, ending on the small deferred "
        "maintenance advisory or scheduling decision that quietly "
        "started the chain."
    ),
    NarrativeStructure.FRAME_STORY: (
        "A framing character (a retired investigator giving a talk, a "
        "widow reading the report, a young cadet in a simulator) "
        "recounts the incident. Each chapter alternates between the "
        "frame and the recounted flight."
    ),
}


def planner_prompt(
    *,
    topic: str,
    mode: Mode,
    structure: NarrativeStructure,
    target_words: int,
    chapter_target: int,
    facts: ExtractedFacts | None,
    forbidden_elements: dict[str, list[str]],
    fictionalization_notice: str,
    axis_selection: dict[str, str] | None = None,
    seed_summary: str = "",
    airline_pool_hint: list[str] | None = None,
    name_pool_hint: list[str] | None = None,
) -> str:
    total_chapters = max(6, min(18, target_words // max(600, chapter_target)))
    forbidden_str = _format_forbidden(forbidden_elements) if mode == Mode.FICTIONAL else ""
    axes_block = ""
    if axis_selection:
        rows = "\n".join(f"- {k}: {v}" for k, v in axis_selection.items() if v)
        axes_block = (
            "\nAXIS SELECTION (already picked by the Global History Manager — "
            "the story MUST embody these):\n" + rows + "\n"
        )
    seed_block = f"\nSEED INCIDENT (grounding, use its dramatic details):\n{seed_summary}\n" if seed_summary else ""
    airline_hint = ""
    if airline_pool_hint:
        airline_hint = (
            "\nFICTIONAL AIRLINE POOL (pick ONE for the operator field; do not "
            "invent new names outside this pool):\n"
            + "\n".join(f"  - {a}" for a in airline_pool_hint) + "\n"
        )
    name_hint = ""
    if name_pool_hint:
        name_hint = (
            "\nCHARACTER NAME POOL (pick from these for crew; keep diversity):\n"
            + "\n".join(f"  - {n}" for n in name_pool_hint) + "\n"
        )
    return (
        "__STEP__=concept\n"
        "You are a senior narrative designer for a long-form aviation "
        "channel. You are building a StoryBible for one incident.\n\n"
        f"WORKING TOPIC: {topic}\n\n"
        f"{_crew_hint(mode, facts)}\n"
        f"{seed_block}"
        f"{axes_block}"
        f"\nNARRATIVE STRUCTURE: {structure.value}\n"
        f"{STRUCTURE_GUIDANCE.get(structure, '')}\n\n"
        f"TARGET LENGTH: ~{target_words:,} words spanning ~{total_chapters} "
        f"chapters of ~{chapter_target} words each.\n\n"
        f"{AVIATION_STYLE_RULES}\n"
        f"{BLACKLIST_RULES}\n"
        f"{RETENTION_TIPS}\n"
        f"{forbidden_str}"
        f"{airline_hint}"
        f"{name_hint}"
        "\nReturn ONLY a JSON object matching this schema:\n"
        "{\n"
        '  "working_title": "…",\n'
        '  "logline": "…",\n'
        '  "premise": "…",\n'
        '  "aircraft": {"type":"…","operator":"…","registration":"…",\n'
        '               "flight_number":"…","engines":"…","seat_capacity": 0},\n'
        '  "route": {"origin":"…","destination":"…","alternate":"…",\n'
        '            "filed_altitude":"…","actual_diversion":"…"},\n'
        '  "crew": [{"name":"…","role":"captain","seat":"left",\n'
        '            "hours_total": 0, "hours_on_type": 0,\n'
        '            "traits":["…"], "arc":"…"}],\n'
        '  "other_characters": [{"name":"…","role":"…","traits":["…"],"arc":"…"}],\n'
        '  "timeline": [{"time_utc":"HH:MM:SSZ","time_local":"HH:MM local",\n'
        '                "altitude":"FL370","airspeed":".82M","phase":"cruise",\n'
        '                "description":"beat", "source_reference":""}],\n'
        '  "technical_facts": ["…"],\n'
        '  "causal_chain": [{"factor":"…","role":"primary","description":"…"}],\n'
        '  "cvr_excerpts": ["…"],\n'
        '  "glossary": {"ECAM":"…"},\n'
        '  "tone_description": "…",\n'
        '  "narrative_voice": "…",\n'
        '  "key_rules": ["…"],\n'
        '  "retention_plan": ["…"],\n'
        f'  "narrative_structure": "{structure.value}",\n'
        f'  "mode": "{mode.value}",\n'
        f'  "fictionalization_notice": "{fictionalization_notice}"\n'
        "}\n"
    )


def _format_forbidden(forbidden: dict[str, list[str]]) -> str:
    lines = ["\nFORBIDDEN — the following elements have appeared in "
             "earlier stories on this channel and MUST NOT be reused:"]
    for kind, values in forbidden.items():
        if not values:
            continue
        lines.append(f"  {kind}: {', '.join(sorted(set(values))[:40])}")
    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


# ── outline (chapter-list) ────────────────────────────────────────────

def outline_prompt(bible: AviationStoryBible, total_chapters: int, chapter_target: int) -> str:
    return (
        "__STEP__=outline\n"
        "Break the following StoryBible into a chapter-by-chapter plan.\n\n"
        f"STORY BIBLE:\n{bible.model_dump_json(indent=2)}\n\n"
        f"REQUIRED: {total_chapters} chapters of ~{chapter_target} words each. "
        "Chapters must obey the narrative structure declared in the "
        "bible (in_media_res / three_act / rashomon / "
        "reverse_chronological / frame_story). Every timeline event in "
        "the bible must land in one of the chapters. Each chapter ends "
        "with either an open loop or a scene shift; never end flatly.\n\n"
        "Return ONLY a JSON array — one object per chapter:\n"
        "[\n"
        '  {"index": 0, "title": "…", "outline_bullets": ["beat", "beat"],\n'
        '   "target_words": 1200, "opens_with": "…", "ends_with": "…"}\n'
        "]\n"
    )


# ── writer ────────────────────────────────────────────────────────────

def writer_prompt(
    *,
    bible: AviationStoryBible,
    chapter_index: int,
    chapter_title: str,
    outline_bullets: Iterable[str],
    target_words: int,
    story_so_far: str,
    ledger: Iterable[str],
    previous_tail: str,
    facts: ExtractedFacts | None,
) -> str:
    facts_block = (
        f"\nSOURCE-TRUTH FACTS (never contradict):\n{facts.model_dump_json(indent=2)}\n"
        if facts is not None else ""
    )
    return (
        "__STEP__=writer\n"
        f"You are writing CHAPTER {chapter_index + 1}: {chapter_title!r} of a "
        "long-form aviation-incident script for YouTube narration.\n\n"
        f"{AVIATION_STYLE_RULES}\n"
        f"{BLACKLIST_RULES}\n"
        f"{RETENTION_TIPS}\n\n"
        f"STORY BIBLE (do not contradict):\n{bible.model_dump_json(indent=2)}\n"
        f"{facts_block}\n"
        "STORY SO FAR (compressed memory of approved chapters):\n"
        f"{story_so_far or '(none — this is chapter 1)'}\n\n"
        "CONTINUITY LEDGER (do not contradict any of these):\n"
        + "\n".join(f"- {f}" for f in ledger) + "\n\n"
        + ("LAST 220 WORDS OF THE PREVIOUS CHAPTER (match the rhythm, "
           "vocabulary and register — start seamlessly):\n"
           "---\n"
           f"{previous_tail}\n"
           "---\n\n" if previous_tail else "")
        + "OUTLINE FOR THIS CHAPTER:\n"
        + "\n".join(f"- {b}" for b in outline_bullets) + "\n\n"
        + f"TARGET: {target_words} words (±10%). Write pure narration. No "
          "chapter heading. No section labels. No stage directions in "
          "brackets. No meta-commentary. Do not name yourself. Start "
          "immediately with the first sentence of narration."
    )


# ── fact-checker ──────────────────────────────────────────────────────

def fact_checker_prompt(
    *,
    facts: ExtractedFacts,
    chapter_text: str,
) -> str:
    return (
        "__STEP__=fact_check\n"
        "You are a fact-checker for an aviation-narrative studio. "
        "Compare the following chapter text against the source-truth "
        "facts. Flag every technical claim, timeline element, dialogue "
        "attribution or outcome that is not supported by (or "
        "contradicts) the facts. Ignore literary flourishes that add no "
        "new fact. Focus on: altitudes, airspeeds, times, phases, "
        "aircraft/system state, CVR wording, causal factors, casualty "
        "counts, regulatory findings.\n\n"
        f"SOURCE-TRUTH FACTS:\n{facts.model_dump_json(indent=2)}\n\n"
        "CHAPTER TEXT:\n"
        "---\n"
        f"{chapter_text}\n"
        "---\n\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "passed": true|false,\n'
        '  "confidence": 0.0..1.0,\n'
        '  "issues": [\n'
        '    {"severity":"high|medium|low","claim":"...","suggested_fix":"..."}\n'
        "  ],\n"
        '  "summary": "one-sentence overall verdict"\n'
        "}\n"
        "'passed' must be false if any issue is severity=high. Medium "
        "or low issues do not fail the chapter but should be noted."
    )


# ── critic ────────────────────────────────────────────────────────────

def critic_prompt(
    *,
    bible: AviationStoryBible,
    chapter_index: int,
    target_words: int,
    draft: str,
) -> str:
    return (
        "__STEP__=evaluate\n"
        f"Evaluate chapter {chapter_index + 1} of a long-form aviation "
        "script.\n\n"
        f"STYLE INTENT:\n{AVIATION_STYLE_RULES}\n"
        f"TARGET LENGTH: ~{target_words} words (currently "
        f"{len(draft.split()):,} words).\n\n"
        "Score each level 0.0-10.0 and return ONLY JSON:\n"
        "{\n"
        '  "l1_technical": {"score":0.0,"issues":[]},   \n'
        '  "l2_linguistic": {"score":0.0,"issues":[]},  \n'
        '  "l3_content": {"score":0.0,"issues":[]},     \n'
        '  "l4_voiceover": {"score":0.0,"issues":[]},   \n'
        '  "overall_score": 0.0,\n'
        '  "passed": true|false,\n'
        '  "summary": "…",\n'
        '  "critical_issues": [],\n'
        '  "continuity_facts_established": ["fact","fact"]  \n'
        "}\n"
        "L1 = technical (word count, markers, encoding).\n"
        "L2 = linguistic (grammar, natural prose).\n"
        "L3 = content (drama, pacing, structure adherence, aviation accuracy).\n"
        "L4 = voiceover (short-enough sentences, natural pauses).\n\n"
        "'continuity_facts_established' should list the discrete "
        "facts introduced by this chapter that later chapters must "
        "honour (e.g. 'captain deferred an APU write-up in "
        "Frankfurt', 'first officer's daughter is nine'). Terse "
        "one-sentence facts, no prose.\n\n"
        "CHAPTER DRAFT:\n"
        "---\n"
        f"{draft}\n"
        "---"
    )


# ── editor ────────────────────────────────────────────────────────────

def editor_prompt(
    *,
    bible: AviationStoryBible,
    chapter_index: int,
    chapter_title: str,
    draft: str,
    critic_notes: str,
    fact_check_issues: list[str],
    target_words: int,
) -> str:
    fc_block = ""
    if fact_check_issues:
        fc_block = (
            "FACT-CHECK CORRECTIONS (highest priority — fix these first):\n"
            + "\n".join(f"- {i}" for i in fact_check_issues) + "\n\n"
        )
    return (
        "__STEP__=writer\n"
        f"Rewrite chapter {chapter_index + 1} ({chapter_title!r}) "
        "to address the editor and fact-check notes below.\n\n"
        f"{AVIATION_STYLE_RULES}\n"
        f"{fc_block}"
        "EDITOR NOTES:\n"
        f"{critic_notes}\n\n"
        f"STORY BIBLE:\n{bible.model_dump_json(indent=2)}\n\n"
        f"TARGET LENGTH: ~{target_words} words.\n\n"
        "Return the FULL revised chapter, pure narration only.\n\n"
        "PREVIOUS DRAFT (rewrite this — do not add commentary):\n"
        "---\n"
        f"{draft}\n"
        "---"
    )


# ── chapter summariser ───────────────────────────────────────────────

def summary_prompt(chapter_index: int, chapter_title: str, text: str) -> str:
    return (
        "__STEP__=summary\n"
        f"Summarise chapter {chapter_index + 1} ({chapter_title!r}) in "
        "~120 words. Include: what happened, where the aircraft is now, "
        "what open threads remain. Terse, information-dense. No prose "
        "flourishes.\n\n"
        "CHAPTER:\n"
        "---\n"
        f"{text}\n"
        "---"
    )


# ── holistic evaluator ──────────────────────────────────────────────

def holistic_prompt(*, bible: AviationStoryBible, full_text: str) -> str:
    return (
        "__STEP__=evaluate\n"
        "You are a senior story editor reviewing the ENTIRE stitched "
        "manuscript of a long-form aviation script. Judge pacing, flow, "
        "continuity, repetition (word-level and beat-level), whether "
        "the technical detail lands, whether the emotional arc pays "
        "off.\n\n"
        f"STORY BIBLE:\n{bible.model_dump_json(indent=2)}\n\n"
        f"MANUSCRIPT ({len(full_text.split()):,} words):\n"
        "---\n"
        f"{full_text[:60000]}\n"  # cap to a reasonable size
        "---\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "l1_technical": {"score":0.0,"issues":[]},\n'
        '  "l2_linguistic": {"score":0.0,"issues":[]},\n'
        '  "l3_content": {"score":0.0,"issues":[]},\n'
        '  "l4_voiceover": {"score":0.0,"issues":[]},\n'
        '  "overall_score": 0.0,\n'
        '  "passed": true|false,\n'
        '  "summary": "…",\n'
        '  "flagged_chapters": [\n'
        '    {"index":0,"reason":"…","instructions":"specific rewrite ask"}\n'
        "  ]\n"
        "}\n"
    )


# ── YouTube metadata ────────────────────────────────────────────────

def metadata_prompt(*, bible: AviationStoryBible, total_words: int) -> str:
    return (
        "__STEP__=metadata\n"
        "Draft the YouTube metadata for the finished aviation story.\n\n"
        f"STORY BIBLE:\n{bible.model_dump_json(indent=2)}\n\n"
        f"MANUSCRIPT WORD COUNT: {total_words:,} words (≈ {total_words // 150} min).\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "titles": ["high-CTR title 1", "…", "…", "…", "…"],\n'
        '  "hook": "3-4 sentence hook for the description",\n'
        '  "personal_note": "100-150 word philosophical takeaway about '
        "systemic failure vs individual error — first person, warm but "
        'measured",\n'
        '  "tags": ["…", …],  // 20-30 SEO tags\n'
        '  "sources": ["…"]   // real citation lines or fictional-composite disclaimer\n'
        "}\n"
    )


# ── storyboard batch ────────────────────────────────────────────────

def storyboard_prompt(
    *,
    bible: AviationStoryBible,
    segments: list[str],
    batch_start_index: int,
) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(segments, start=batch_start_index))
    return (
        "__STEP__=storyboard\n"
        "For each ~5-second narration segment below, generate a single "
        "highly detailed IMAGE GENERATION PROMPT (Midjourney / Stable "
        "Diffusion style). One prompt per segment. Prompts describe a "
        "cinematic STILL image (I will animate them with pans / zooms "
        "in post). STRICTLY no B-roll suggestions, no video language, "
        "no motion verbs like 'panning' / 'zooming'. Every prompt must "
        "include: subject, camera framing (wide / medium / close / "
        "over-shoulder), lens (14mm / 35mm / 85mm / anamorphic), light "
        "source and quality, mood, colour palette, and end with "
        "'ultra-detailed, cinematic, 8k, no text, no watermark'.\n\n"
        f"CONTEXT (aircraft: {bible.aircraft.type}, operator: "
        f"{bible.aircraft.operator}, tail: {bible.aircraft.registration}):\n"
        f"tone: {bible.tone_description}\n\n"
        "SEGMENTS:\n"
        f"{numbered}\n\n"
        "Return ONLY JSON:\n"
        '{"prompts":[{"index":N,"text_segment":"…","image_prompt":"…"},…]}'
    )
