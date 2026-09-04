"""Three deliverables per finished story.

* ``01_tts_script.txt``       — ElevenLabs-flavoured SSML narration.
* ``02_youtube_metadata.md``  — titles + description + chapters + tags.
* ``03_storyboard.csv``       — ``Timestamp | Text Segment | Image Prompt``.

Plus two bonus files that make the run easy to audit:

* ``00_manuscript.md``        — full source-of-truth manuscript.
* ``04_story_bible.json``     — the aviation bible as JSON.

None of these functions call the LLM — they take already-generated
data and format it. Storyboard generation itself lives in
:mod:`aviation.agents`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from aviation.state import AviationJob, Deliverable
from aviation.text import (
    count_words,
    csv_escape,
    strip_for_tts,
    words_to_seconds,
    _format_timestamp,  # type: ignore[attr-defined]
)
from models.aviation_bible import AviationStoryBible, Mode


# ── ElevenLabs SSML ──────────────────────────────────────────────────

# Words / phrases that mark a beat where we want an extra-long pause.
# Regex fragments (case-insensitive).
_CLIFFHANGER_MARKERS = re.compile(
    r"\b(?:mayday|impact|silence|black(?:ed)?\s*out|crashed?|"
    r"stall(?:ed|ing)?|fire\s*warning|master\s*caution|then\s*(?:—|-)|"
    r"and\s*then\s*nothing|nothing\s*answered)\b",
    re.I,
)

# End-of-paragraph regex for inserting standard breaks.
_PARA_SPLIT = re.compile(r"\n\s*\n")


def to_elevenlabs_ssml(text: str, paragraph_ms: int = 800, cliff_ms: int = 1500) -> str:
    """Return ElevenLabs-flavoured SSML for the clean narration ``text``.

    ElevenLabs accepts a *subset* of SSML; the safe forms are
    ``<break time="…s"/>``. Nothing else is guaranteed to render. This
    exporter emits only ``<break/>`` tags on:

    * ``<break time="{cliff_ms}ms"/>`` — end of a paragraph that
      contains a cliffhanger marker (mayday / impact / crash / stall /
      fire warning …), before the paragraph break.
    * ``<break time="{paragraph_ms}ms"/>`` — end of every other
      paragraph.

    The output is a UTF-8 text file, one paragraph per line, ready to
    paste into ElevenLabs' long-form projects.
    """
    clean = strip_for_tts(text)
    paragraphs = [p.strip() for p in _PARA_SPLIT.split(clean) if p.strip()]
    lines: list[str] = []
    for para in paragraphs:
        # Emit the paragraph.
        lines.append(para)
        # Choose break duration.
        ms = cliff_ms if _CLIFFHANGER_MARKERS.search(para) else paragraph_ms
        lines.append(f'<break time="{ms/1000:.1f}s" />')
    return "\n\n".join(lines).strip() + "\n"


# ── YouTube description ──────────────────────────────────────────────

def render_youtube_metadata(
    *,
    metadata: dict[str, Any],
    bible: AviationStoryBible,
    chapter_titles_and_starts: list[tuple[str, float]],
    total_seconds: float,
    mode: Mode,
) -> str:
    """Render a Markdown file matching the spec's exact structure.

    ``metadata`` is the JSON returned by
    :func:`aviation.agents.generate_metadata`. ``chapter_titles_and_starts``
    is a list of ``(title, start_seconds)`` pairs computed from the
    approved chapters at 150 wpm.
    """
    titles = _coerce_str_list(metadata.get("titles"), ["Untitled aviation story"])
    hook = str(metadata.get("hook") or bible.logline or bible.premise).strip()
    note = str(metadata.get("personal_note") or "").strip()
    sources = _coerce_str_list(metadata.get("sources"), [])
    tags = _coerce_str_list(metadata.get("tags"), [])

    notice = bible.fictionalization_notice or _default_notice(mode)

    total_len = _format_timestamp(total_seconds)

    lines: list[str] = []
    lines.append(f"# {bible.working_title}\n")
    lines.append("## Title options\n")
    for t in titles[:5]:
        lines.append(f"- {t}")
    lines.append("")

    lines.append("## Description\n")
    lines.append(hook)
    lines.append("")
    lines.append(f"⚠️ **Fictionalization notice:** {notice}")
    lines.append("")
    lines.append("⏱ **CHAPTERS**  ")
    for title, start in chapter_titles_and_starts:
        ts = _format_timestamp(start)
        lines.append(f"{ts} – {title}")
    lines.append("")
    lines.append(f"_Total runtime: {total_len}._")
    lines.append("")
    lines.append("📚 **SOURCES**  ")
    if sources:
        for s in sources:
            lines.append(f"- {s}")
    else:
        lines.append("- (Fictional composite — no single incident cited.)")
    lines.append("")
    lines.append("🖋 **NOTE FROM ME**  ")
    if note:
        lines.append(note)
    else:
        lines.append(
            "Every accident report reads the same shape — a small thing that "
            "would have been nothing, on any other night, meeting another "
            "small thing that had been quietly waiting for months. "
            "What separates a diversion from a headline is rarely skill, "
            "but the willingness, in the last minute, to say 'the manuals "
            "do not cover this' — and to fly the airplane."
        )
    lines.append("")
    lines.append("**TAGS**  ")
    if len(tags) < 20:
        tags = _pad_tags(tags, bible)
    tag_line = ", ".join(tags[:30])
    lines.append(tag_line)
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _default_notice(mode: Mode) -> str:
    if mode == Mode.REAL:
        return (
            "Based on the official accident report. Some dialogue and interior "
            "moments have been dramatised where the record is silent."
        )
    return (
        "Fictional composite — any resemblance to real airlines, aircraft "
        "registrations, crew or specific incidents is unintentional."
    )


def _coerce_str_list(value: Any, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return list(default)


_FALLBACK_TAGS = [
    "aviation", "aviation incident", "aviation accident", "aviation stories",
    "aircraft emergency", "cockpit voice recorder", "flight recorder",
    "aviation documentary", "air disaster", "close call", "captain decision",
    "pilot decision making", "aviation analysis", "true aviation story",
    "aviation storyteller", "flight deck", "airline safety", "aviation history",
    "in the cockpit", "flight deck story",
]


def _pad_tags(existing: list[str], bible: AviationStoryBible) -> list[str]:
    tags = list(existing)
    for base in (bible.aircraft.type, bible.aircraft.operator, bible.route.origin, bible.route.destination):
        base = (base or "").strip()
        if base and base not in tags:
            tags.append(base)
    for t in _FALLBACK_TAGS:
        if t not in tags:
            tags.append(t)
        if len(tags) >= 25:
            break
    return tags


# ── Storyboard CSV ───────────────────────────────────────────────────

def render_storyboard_csv(rows: list[dict[str, Any]]) -> str:
    header = "timestamp,seconds,text_segment,image_prompt"
    body_lines: list[str] = []
    for r in rows:
        body_lines.append(
            ",".join(
                csv_escape(r.get(k, ""))
                for k in ("timestamp", "seconds", "text", "image_prompt")
            )
        )
    return header + "\n" + "\n".join(body_lines) + "\n"


def render_storyboard_pipe(rows: list[dict[str, Any]]) -> str:
    """Human-readable pipe-delimited variant (the spec's exact format)."""
    header = "Timestamp | Text Segment | Image Generation Prompt"
    lines = [header]
    for r in rows:
        text = str(r.get("text", "")).replace("|", "/")
        prompt = str(r.get("image_prompt", "")).replace("|", "/")
        lines.append(f'{r.get("timestamp", "")} | {text} | {prompt}')
    return "\n".join(lines) + "\n"


# ── Persistence helpers ──────────────────────────────────────────────

def write_deliverable(job: AviationJob, filename: str, content: str, kind: str) -> Deliverable:
    path = job.output_path(filename)
    path.write_text(content, encoding="utf-8")
    d = Deliverable(kind=kind, filename=filename, path=str(path), bytes=len(content.encode("utf-8")))
    # Replace an existing entry with the same filename.
    job.deliverables = [x for x in job.deliverables if x.filename != filename]
    job.deliverables.append(d)
    return d


def chapter_starts(
    approved_texts: list[str],
    wpm: int,
    titles: list[str],
) -> list[tuple[str, float]]:
    running = 0.0
    out: list[tuple[str, float]] = []
    for text, title in zip(approved_texts, titles):
        out.append((title, running))
        running += words_to_seconds(count_words(text), wpm)
    return out


def total_runtime_seconds(texts: list[str], wpm: int) -> float:
    return sum(words_to_seconds(count_words(t), wpm) for t in texts)


def build_manuscript(bible: AviationStoryBible, texts: list[str], titles: list[str]) -> str:
    parts = [f"# {bible.working_title}\n"]
    for i, (t, title) in enumerate(zip(texts, titles), start=1):
        parts.append(f"## Chapter {i} — {title}\n\n{t.strip()}\n")
    return "\n".join(parts)


def bible_json(bible: AviationStoryBible) -> str:
    return json.dumps(bible.model_dump(mode="json"), indent=2, ensure_ascii=False)
