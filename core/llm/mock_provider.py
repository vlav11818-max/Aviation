"""Deterministic offline LLM used when no API key is configured.

The mock never calls the network. Instead it inspects the prompt for
one of a small number of well-known aviation-factory step markers
(``__STEP__=concept``, ``__STEP__=outline`` …) and returns a
JSON/text payload that satisfies the corresponding downstream Pydantic
schema. If no marker is present it falls back to a deterministic
"lorem-ipsum-style" narrative chunk sized to roughly match the
requested max_tokens.

The mock is what makes the whole pipeline runnable end-to-end from
this session without any provider keys. All real prompts inject the
``__STEP__=<name>`` marker at the top of the message so the mock knows
what shape to return.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

# One "word" ≈ 1.3 tokens in English (rough); the mock uses this to
# roughly honour the ``max_tokens`` cap so downstream word-count checks
# have something to work with.
_TOKENS_PER_WORD = 1.3

# Deterministic word pool for narrative chunks. Repeats but scrambles
# per-call via a hash so consecutive chunks look distinct.
_LOREM_WORDS = (
    "cockpit altitude airspeed rudder throttle horizon fuselage descent "
    "captain first-officer flaps trim vector runway approach clearance "
    "turbulence stall recovery pitch yaw roll cabin passenger stewardess "
    "checklist advisory transponder squawk mayday emergency divert "
    "engine failure hydraulic warning cascade sequence recorder timeline "
    "briefing weather cumulus icing wind-shear pattern holding vectors "
    "vertical horizontal separation controller tower center approach "
    "wisps of cloud rushed past the windscreen as she pushed the "
    "throttles forward and called out the callsign one last time".split()
)


def _extract_step(messages: list[dict[str, str]]) -> str:
    for msg in messages:
        content = msg.get("content", "")
        match = re.search(r"__STEP__=([a-z_]+)", content, flags=re.I)
        if match:
            return match.group(1).lower()
    return ""


def _seed(messages: list[dict[str, str]]) -> int:
    payload = "".join(m.get("content", "") for m in messages)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _lorem(target_words: int, seed: int) -> str:
    """Deterministic pseudo-random narrative of ``target_words`` words."""
    if target_words <= 0:
        return ""
    pool = list(_LOREM_WORDS)
    # Simple deterministic shuffle from the seed.
    rng = seed
    for i in range(len(pool) - 1, 0, -1):
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        j = rng % (i + 1)
        pool[i], pool[j] = pool[j], pool[i]
    out: list[str] = []
    while len(out) < target_words:
        out.extend(pool)
    words = out[:target_words]
    # Chunk into sentences and paragraphs so it looks like prose.
    sentences: list[str] = []
    for i in range(0, len(words), 14):
        chunk = words[i : i + 14]
        chunk[0] = chunk[0].capitalize()
        sentences.append(" ".join(chunk) + ".")
    paras: list[str] = []
    for i in range(0, len(sentences), 4):
        paras.append(" ".join(sentences[i : i + 4]))
    return "\n\n".join(paras)


def _mock_concept(seed: int) -> str:
    """Return an aviation-shaped payload that satisfies AviationStoryBible.

    Uses a deterministic-per-seed suffix so successive fictional mock
    runs do not collide on the global-history uniqueness constraint.
    """
    tag = f"{seed & 0xFFF:03X}"
    return json.dumps(
        {
            "working_title": f"Cascade at Flight Level 370 (mock/{tag})",
            "logline": "A cascading hydraulic failure at cruise, and the four minutes that separate a diversion from a headline.",
            "premise": (
                "A commercial jet begins a routine transoceanic flight when a "
                "hydraulic warning light turns steady. The crew has minutes to "
                "understand a cascading fault chain that no simulator ever "
                "taught them, and only one runway within reach."
            ),
            "aircraft": {
                "type": "Airbus A340-300",
                "operator": f"TransOcean Airways {tag}",
                "registration": f"N{tag}AB",
                "flight_number": f"TO{tag}",
                "engines": "4 × CFM56-5C4",
                "seat_capacity": 295,
            },
            "route": {
                "origin": "New York JFK",
                "destination": "London LHR",
                "alternate": "Reykjavík KEF",
                "filed_altitude": "FL370",
                "actual_diversion": "Reykjavík KEF",
            },
            "crew": [
                {
                    "name": f"Captain Ivo Rask {tag}",
                    "role": "captain",
                    "seat": "left",
                    "hours_total": 12400,
                    "hours_on_type": 4800,
                    "traits": ["deliberate", "instinctive", "self-doubting"],
                    "arc": "From by-the-book command to trusting his gut when the manuals go quiet.",
                },
                {
                    "name": f"First Officer Nadia Vella {tag}",
                    "role": "first_officer",
                    "seat": "right",
                    "hours_total": 3200,
                    "hours_on_type": 1100,
                    "traits": ["analytical", "young", "unshakably calm"],
                    "arc": "Learns the checklists have a limit — and that a captain who deviates is not always wrong.",
                },
            ],
            "other_characters": [],
            "timeline": [
                {
                    "time_utc": "02:14:00Z",
                    "time_local": "22:14 EST",
                    "altitude": "FL370",
                    "airspeed": ".82M",
                    "phase": "cruise",
                    "description": "First steady HYD Y caution illuminates.",
                    "source_reference": "",
                },
                {
                    "time_utc": "02:16:12Z",
                    "time_local": "22:16 EST",
                    "altitude": "FL370",
                    "airspeed": ".82M",
                    "phase": "cruise",
                    "description": "ECAM cascade: HYD Y RSVR LO, then GEN 2 FAULT.",
                    "source_reference": "",
                },
                {
                    "time_utc": "02:18:47Z",
                    "time_local": "22:18 EST",
                    "altitude": "FL370",
                    "airspeed": ".80M",
                    "phase": "cruise",
                    "description": "Captain elects diversion direct BIKF.",
                    "source_reference": "",
                },
                {
                    "time_utc": "02:47:03Z",
                    "time_local": "22:47 EST",
                    "altitude": "0 ft",
                    "airspeed": "135 KIAS",
                    "phase": "landing",
                    "description": "CAT I ILS RWY 19 to a full-stop landing at BIKF.",
                    "source_reference": "",
                },
            ],
            "technical_facts": [
                "Yellow hydraulic system fluid loss of 4 L/min.",
                "No cascade into the green system throughout the event.",
                "APU-driven electrical bus took the GEN 2 loss without a busbar drop.",
            ],
            "causal_chain": [
                {"factor": "cracked reservoir sight-gauge fitting", "role": "primary", "description": "Metal-fatigue crack in an aluminium fitting."},
                {"factor": "deferred maintenance advisory", "role": "contributing", "description": "MEL item left open past its calendar cap."},
                {"factor": "ambiguous ECAM sequencing", "role": "latent", "description": "System messages arrived in an order that suggested a different fault."},
            ],
            "cvr_excerpts": [
                '"Standby — I want to see the trend on Y before we call it."',
                '"Set course direct KEF, we\'ll brief on the way."',
            ],
            "glossary": {
                "ECAM": "Electronic Centralised Aircraft Monitoring — Airbus's system-status display.",
                "MEL": "Minimum Equipment List — deferred-defects register.",
                "BIKF": "ICAO code for Keflavík International Airport, Iceland.",
            },
            "tone_description": "Dramatic-cinematic, minute-by-minute, procedurally accurate.",
            "narrative_voice": "Third-person omniscient with cockpit intimacy.",
            "key_rules": [
                "No graphic passenger injury on-screen.",
                "Radio calls in real ATC phraseology.",
                "Time-stamps in UTC.",
            ],
            "retention_plan": [
                "Open loop: foreshadow the deferred MEL item in the pre-flight briefing.",
                "Sensory anchor: the yellow-limit tape on the HYD Y quantity gauge.",
                "Somatic detail: captain's hand hovering over the thrust levers before commanding an FL change.",
            ],
            "narrative_structure": "in_media_res",
            "mode": "fictional",
            "fictionalization_notice": "This story is entirely fictional; any resemblance to real airlines, aircraft or crew is unintentional.",
        },
        ensure_ascii=False,
    )


def _mock_outline(seed: int) -> str:
    """Return a JSON payload that satisfies Outline (~14400 words / 12 sections)."""
    sections = []
    for i in range(12):
        sections.append(
            {
                "index": i,
                "title": f"Chapter {i + 1} — Placeholder",
                "act_label": ["Act I — Setup", "Act II — Confrontation", "Act III — Resolution"][
                    min(2, i // 4)
                ],
                "target_words": 1200,
                "key_events": [
                    "Placeholder event A",
                    "Placeholder event B",
                    "Placeholder event C",
                ],
                "characters_present": ["Captain Ivo Rask", "First Officer Nadia Vella"],
                "transition_from": "" if i == 0 else "Previous scene fade.",
                "transition_to": "" if i == 11 else "Cue next scene.",
            }
        )
    return json.dumps(
        {"structure_type": "three_act", "total_target_words": 14400, "sections": sections},
        ensure_ascii=False,
    )


def _mock_evaluation(seed: int) -> str:
    """Return a JSON payload that satisfies EvaluationResult (passing)."""
    return json.dumps(
        {
            "l1_technical": {"score": 9.2, "issues": []},
            "l2_linguistic": {"score": 9.1, "issues": []},
            "l3_content": {"score": 9.0, "issues": []},
            "l4_voiceover": {"score": 9.3, "issues": []},
            "overall_score": 9.15,
            "passed": True,
            "summary": "Mock evaluation: passes all four levels.",
            "critical_issues": [],
        },
        ensure_ascii=False,
    )


def _mock_fact_check(seed: int) -> str:
    return json.dumps(
        {
            "passed": True,
            "confidence": 0.92,
            "issues": [],
            "summary": "Mock fact-check: chapter is consistent with the source facts.",
        },
        ensure_ascii=False,
    )


def _mock_summary(seed: int, tokens: int) -> str:
    words = max(20, min(150, int(tokens / _TOKENS_PER_WORD)))
    return _lorem(words, seed)


def _mock_metadata(seed: int) -> str:
    return json.dumps(
        {
            "titles": [
                "The 4-Minute Warning That Saved Everyone Onboard",
                "One Warning Light. Nine Hundred Miles from Land.",
                "The Captain Who Ignored The Checklist — And Was Right",
                "Cascade at 37,000 Feet",
                "Only One Runway Within Reach",
            ],
            "hook": "At 02:14 UTC, a hydraulic warning light on a routine "
            "transoceanic flight turned steady. Nine hundred miles from "
            "any land, the crew had minutes to understand a chain of "
            "faults no simulator had ever taught them.",
            "personal_note": "Every accident report is the same shape: a small "
            "thing that would have been nothing, on any other night, "
            "meeting another small thing that had been quietly waiting "
            "for months. What separates a diversion from a headline is "
            "usually not skill. It is the willingness, in the last minute, "
            "to say the manuals do not cover this — and to fly the airplane.",
            "tags": [
                "aviation",
                "aviation incident",
                "aviation accident",
                "aviation stories",
                "aircraft emergency",
                "cockpit voice recorder",
                "flight recorder",
                "aviation documentary",
                "air disaster",
                "close call",
                "captain decision",
                "airbus",
                "boeing",
                "transatlantic flight",
                "hydraulic failure",
                "north atlantic",
                "cascading failure",
                "checklists",
                "airline safety",
                "aviation analysis",
                "true aviation story",
                "aviation storyteller",
                "flight deck",
                "pilot decision making",
                "aviation history",
            ],
            "sources": [
                "Fictional composite (no single incident cited).",
                "Cross-referenced with NTSB narratives on similar hydraulic-cascade failures.",
            ],
        },
        ensure_ascii=False,
    )


def _mock_storyboard(seed: int) -> str:
    prompts = []
    for i in range(30):
        prompts.append(
            {
                "index": i,
                "text_segment": "Placeholder narration segment.",
                "image_prompt": (
                    "Cinematic wide shot of a modern twin-jet airliner cockpit at "
                    "night, single amber warning light lit on the overhead panel, "
                    "captain's left hand on the throttles, softly-lit instrument "
                    "cluster, faint reflection of instruments on the windscreen, "
                    "shallow depth of field, 35mm, ultra-detailed, moody cinematic "
                    "lighting, no text, no lens flare."
                ),
            }
        )
    return json.dumps({"prompts": prompts}, ensure_ascii=False)


def _mock_ingest(seed: int) -> str:
    return json.dumps(
        {
            "incident_name": "Mock Incident — TransOcean 447",
            "date": "2019-03-14",
            "location": "North Atlantic, FL370",
            "operator_and_flight": "TransOcean Airways TO447",
            "aircraft": {
                "type": "Airbus A340-300",
                "registration": "N/A (mock)",
                "flight_number": "TO447",
                "operator": "TransOcean Airways",
            },
            "crew": ["Captain Ivo Rask", "First Officer Nadia Vella"],
            "sequence_of_events": [
                "02:14Z steady hydraulic caution",
                "02:16Z ECAM: HYD Y RSVR LO",
                "02:18Z diversion decision to Keflavík",
                "02:47Z Category-I ILS RWY 19 to a full-stop landing",
            ],
            "probable_cause": "Cracked hydraulic reservoir sight-gauge fitting.",
            "contributing_factors": [
                "Deferred maintenance advisory left open.",
                "Ambiguous ECAM message sequence.",
            ],
            "technical_findings": [
                "Yellow system fluid loss 4L/min.",
                "No cascade to green system.",
            ],
            "cvr_highlights": [
                '"Standby — I want to see the trend on Y before we call it."',
                '"Set course direct KEF, we\'ll brief on the way."',
            ],
            "casualties": "None. 232 souls onboard.",
            "recommendations": [
                "Revise ECAM hydraulic-system sequencing.",
                "Mandatory eight-month inspection of reservoir fittings.",
            ],
            "notable_quotes": [
                '"We are not fighting an emergency. We are managing a leak."',
            ],
        },
        ensure_ascii=False,
    )


# Dispatch table for known step markers.
_STEP_HANDLERS = {
    "concept": lambda seed, tok: _mock_concept(seed),
    "outline": lambda seed, tok: _mock_outline(seed),
    "evaluate": lambda seed, tok: _mock_evaluation(seed),
    "evaluation": lambda seed, tok: _mock_evaluation(seed),
    "fact_check": lambda seed, tok: _mock_fact_check(seed),
    "summary": lambda seed, tok: _mock_summary(seed, tok),
    "metadata": lambda seed, tok: _mock_metadata(seed),
    "storyboard": lambda seed, tok: _mock_storyboard(seed),
    "ingest": lambda seed, tok: _mock_ingest(seed),
}


def mock_completion(
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Simulate a chat completion, returning a LiteLLM-shaped dict.

    Returns:
        A dict with the same shape as ``litellm.completion`` output:
        ``{"choices": [{"message": {"content": <str>}}], "usage": {...}}``.
    """
    step = _extract_step(messages)
    seed = _seed(messages)
    handler = _STEP_HANDLERS.get(step)
    if handler is not None:
        content = handler(seed, max_tokens)
    else:
        # Default: narrative chunk sized to max_tokens.
        target_words = max(60, min(2000, int(max_tokens / _TOKENS_PER_WORD)))
        content = _lorem(target_words, seed)

    # Rough token accounting.
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    return {
        "id": f"mock-{seed:x}",
        "model": "mock/demo",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": max(1, prompt_chars // 4),
            "completion_tokens": max(1, len(content) // 4),
            "total_tokens": max(2, (prompt_chars + len(content)) // 4),
        },
    }
