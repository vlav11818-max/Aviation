"""The seven agents of the aviation content factory.

Each function is one async coroutine. They are stateless — the
orchestrator threads the :class:`~aviation.state.AviationJob` through
and each agent reads what it needs, calls the LLM once (or in a
batched loop for storyboard), and returns a typed result the
orchestrator writes back to the job.

Agents:
    * :func:`ingest_pdf`         — PDF → :class:`ExtractedFacts`
    * :func:`plan_story`         — topic → :class:`AviationStoryBible`
    * :func:`plan_chapters`      — bible → list of :class:`ChapterDraft`
    * :func:`write_chapter`      — bible + memory → chapter draft (text)
    * :func:`fact_check_chapter` — draft vs. facts → issues[]
    * :func:`critique_chapter`   — draft → score + notes + new facts
    * :func:`edit_chapter`       — draft + notes → revised draft
    * :func:`summarise_chapter`  — draft → ~120-word memo
    * :func:`holistic_review`    — full manuscript → flagged chapters
    * :func:`generate_metadata`  — bible + manuscript → YouTube metadata
    * :func:`generate_storyboard`— clean script → 5-sec segments + prompts

None of them look at LiteLLM directly — they go through
:mod:`aviation.llm_helpers`, which handles JSON parsing and cost tracking.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aviation import prompts as P
from aviation.llm_helpers import llm_json, llm_text
from aviation.state import AviationJob, ChapterDraft
from aviation.text import segment_with_timestamps
from core.pdf_ingest import PDFDocument, load_pdf
from models.aviation_bible import (
    Aircraft,
    AviationStoryBible,
    CausalLink,
    CrewMember,
    ExtractedFacts,
    Mode,
    NarrativeStructure,
    Route,
    TimelineEvent,
)
from models.story_bible import Character

logger = logging.getLogger(__name__)


# ── ingest ────────────────────────────────────────────────────────────


async def ingest_pdf(
    *,
    job: AviationJob,
    doc: PDFDocument,
    model: str,
) -> ExtractedFacts:
    """Run the ingest LLM over each ~18k-char chunk of ``doc`` and
    merge into one :class:`ExtractedFacts`."""
    merged = ExtractedFacts()
    for i, chunk in enumerate(doc.chunks()):
        job.append_log(
            "ingest",
            f"Extracting from {doc.filename} chunk {i + 1} ({len(chunk):,} chars)",
        )
        try:
            data = await llm_json(
                job=job,
                model=model,
                prompt=P.ingest_prompt(chunk, doc.filename),
                temperature=0.2,
                max_tokens=2400,
                agent="ingest",
            )
        except Exception as exc:
            job.append_log(
                "ingest",
                f"Chunk {i + 1} failed: {exc}. Continuing.",
                level="warn",
            )
            continue
        _merge_facts(merged, data)
    return merged


def _merge_facts(acc: ExtractedFacts, data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        return
    if not acc.incident_name and data.get("incident_name"):
        acc.incident_name = str(data["incident_name"])
    if not acc.date and data.get("date"):
        acc.date = str(data["date"])
    if not acc.location and data.get("location"):
        acc.location = str(data["location"])
    if not acc.operator_and_flight and data.get("operator_and_flight"):
        acc.operator_and_flight = str(data["operator_and_flight"])
    if not acc.probable_cause and data.get("probable_cause"):
        acc.probable_cause = str(data["probable_cause"])
    if not acc.casualties and data.get("casualties"):
        acc.casualties = str(data["casualties"])

    if isinstance(data.get("aircraft"), dict):
        ac = data["aircraft"]
        if not acc.aircraft.type and ac.get("type"):
            acc.aircraft.type = str(ac["type"])
        if not acc.aircraft.operator and ac.get("operator"):
            acc.aircraft.operator = str(ac["operator"])
        if not acc.aircraft.registration and ac.get("registration"):
            acc.aircraft.registration = str(ac["registration"])
        if not acc.aircraft.flight_number and ac.get("flight_number"):
            acc.aircraft.flight_number = str(ac["flight_number"])

    for key, target in (
        ("crew", acc.crew),
        ("sequence_of_events", acc.sequence_of_events),
        ("contributing_factors", acc.contributing_factors),
        ("technical_findings", acc.technical_findings),
        ("cvr_highlights", acc.cvr_highlights),
        ("recommendations", acc.recommendations),
        ("notable_quotes", acc.notable_quotes),
    ):
        for item in data.get(key, []) or []:
            s = str(item).strip()
            if s and s not in target:
                target.append(s)


# ── planner ───────────────────────────────────────────────────────────


async def plan_story(
    *,
    job: AviationJob,
    facts: ExtractedFacts | None,
    structure: NarrativeStructure,
    forbidden_elements: dict[str, list[str]],
    model: str,
) -> AviationStoryBible:
    """Ask the LLM to build the aviation StoryBible."""
    total_words = job.settings.target_words
    chapter_target = job.settings.chapter_target_words
    notice = _fictionalization_notice(job.settings.mode)
    data = await llm_json(
        job=job,
        model=model,
        prompt=P.planner_prompt(
            topic=job.topic,
            mode=job.settings.mode,
            structure=structure,
            target_words=total_words,
            chapter_target=chapter_target,
            facts=facts,
            forbidden_elements=forbidden_elements,
            fictionalization_notice=notice,
        ),
        temperature=0.6,
        max_tokens=6000,
        agent="planner",
    )
    return _hydrate_bible(data, structure, job.settings.mode, notice)


def _fictionalization_notice(mode: Mode) -> str:
    if mode == Mode.REAL:
        return (
            "This story is based on the official accident report. Some "
            "dialogue and interior moments have been dramatised where "
            "the record is silent."
        )
    return (
        "This story is entirely fictional. Any resemblance to real "
        "airlines, aircraft registrations, crew, or specific incidents "
        "is unintentional. Aircraft systems and phraseology are drawn "
        "from real practice."
    )


def _hydrate_bible(
    data: Any,
    structure: NarrativeStructure,
    mode: Mode,
    notice: str,
) -> AviationStoryBible:
    if not isinstance(data, dict):
        raise ValueError("Planner returned non-object payload.")

    aircraft = Aircraft.model_validate(data.get("aircraft") or {})
    route = Route.model_validate(data.get("route") or {})
    crew: list[CrewMember] = []
    for c in data.get("crew") or []:
        try:
            crew.append(CrewMember.model_validate(c))
        except Exception:
            continue
    others: list[Character] = []
    for c in data.get("other_characters") or []:
        try:
            others.append(Character.model_validate(c))
        except Exception:
            continue
    timeline: list[TimelineEvent] = []
    for t in data.get("timeline") or []:
        try:
            timeline.append(TimelineEvent.model_validate(t))
        except Exception:
            continue
    causal: list[CausalLink] = []
    for c in data.get("causal_chain") or []:
        try:
            causal.append(CausalLink.model_validate(c))
        except Exception:
            continue

    return AviationStoryBible(
        mode=mode,
        working_title=str(data.get("working_title") or "Untitled Incident"),
        logline=str(data.get("logline") or ""),
        premise=str(data.get("premise") or ""),
        aircraft=aircraft,
        route=route,
        crew=crew,
        other_characters=others,
        timeline=timeline,
        technical_facts=[str(x) for x in (data.get("technical_facts") or [])],
        causal_chain=causal,
        cvr_excerpts=[str(x) for x in (data.get("cvr_excerpts") or [])],
        glossary={str(k): str(v) for k, v in (data.get("glossary") or {}).items()},
        narrative_structure=structure,
        tone_description=str(data.get("tone_description") or ""),
        narrative_voice=str(data.get("narrative_voice") or ""),
        key_rules=[str(x) for x in (data.get("key_rules") or [])],
        retention_plan=[str(x) for x in (data.get("retention_plan") or [])],
        fictionalization_notice=notice,
    )


# ── chapter planner ───────────────────────────────────────────────────


async def plan_chapters(
    *,
    job: AviationJob,
    bible: AviationStoryBible,
    model: str,
) -> list[ChapterDraft]:
    total_words = job.settings.target_words
    chapter_target = job.settings.chapter_target_words
    total_chapters = max(6, min(18, total_words // max(600, chapter_target)))
    data = await llm_json(
        job=job,
        model=model,
        prompt=P.outline_prompt(bible, total_chapters, chapter_target),
        temperature=0.4,
        max_tokens=3000,
        agent="planner",
    )
    if isinstance(data, dict) and "chapters" in data:
        rows = data["chapters"]
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    chapters: list[ChapterDraft] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        chapters.append(
            ChapterDraft(
                index=int(row.get("index", i)),
                title=str(row.get("title", f"Chapter {i + 1}")),
                outline_bullets=[str(b) for b in (row.get("outline_bullets") or row.get("beats") or [])],
                target_words=int(row.get("target_words") or chapter_target),
            )
        )
    if not chapters:
        # Safety fallback: synthesize a flat outline so a mock/short
        # response still lets the pipeline run.
        for i in range(total_chapters):
            chapters.append(
                ChapterDraft(
                    index=i,
                    title=f"Chapter {i + 1}",
                    outline_bullets=[b.description for b in bible.timeline[i * 2 : i * 2 + 2]] or ["Placeholder beat"],
                    target_words=chapter_target,
                )
            )
    return chapters


# ── writer ────────────────────────────────────────────────────────────


async def write_chapter(
    *,
    job: AviationJob,
    bible: AviationStoryBible,
    chapter: ChapterDraft,
    facts: ExtractedFacts | None,
    model: str,
) -> str:
    return await llm_text(
        job=job,
        model=model,
        prompt=P.writer_prompt(
            bible=bible,
            chapter_index=chapter.index,
            chapter_title=chapter.title,
            outline_bullets=chapter.outline_bullets,
            target_words=chapter.target_words,
            story_so_far=job.story_so_far,
            ledger=job.ledger,
            previous_tail=job.previous_tail,
            facts=facts,
        ),
        temperature=0.8,
        max_tokens=min(4096, chapter.target_words * 3),
        agent="writer",
    )


async def edit_chapter(
    *,
    job: AviationJob,
    bible: AviationStoryBible,
    chapter: ChapterDraft,
    model: str,
) -> str:
    return await llm_text(
        job=job,
        model=model,
        prompt=P.editor_prompt(
            bible=bible,
            chapter_index=chapter.index,
            chapter_title=chapter.title,
            draft=chapter.draft_text,
            critic_notes=chapter.critic_notes,
            fact_check_issues=chapter.fact_check_issues,
            target_words=chapter.target_words,
        ),
        temperature=0.6,
        max_tokens=min(4096, chapter.target_words * 3),
        agent="editor",
    )


# ── fact-checker ──────────────────────────────────────────────────────


async def fact_check_chapter(
    *,
    job: AviationJob,
    facts: ExtractedFacts,
    chapter: ChapterDraft,
    model: str,
) -> dict[str, Any]:
    return await llm_json(
        job=job,
        model=model,
        prompt=P.fact_checker_prompt(facts=facts, chapter_text=chapter.draft_text),
        temperature=0.1,
        max_tokens=1200,
        agent="fact_checker",
    )  # type: ignore[return-value]


# ── critic ────────────────────────────────────────────────────────────


async def critique_chapter(
    *,
    job: AviationJob,
    bible: AviationStoryBible,
    chapter: ChapterDraft,
    model: str,
) -> dict[str, Any]:
    return await llm_json(
        job=job,
        model=model,
        prompt=P.critic_prompt(
            bible=bible,
            chapter_index=chapter.index,
            target_words=chapter.target_words,
            draft=chapter.draft_text,
        ),
        temperature=0.2,
        max_tokens=1600,
        agent="critic",
    )  # type: ignore[return-value]


# ── summariser ────────────────────────────────────────────────────────


async def summarise_chapter(
    *,
    job: AviationJob,
    chapter: ChapterDraft,
    model: str,
) -> str:
    return await llm_text(
        job=job,
        model=model,
        prompt=P.summary_prompt(chapter.index, chapter.title, chapter.clean_text or chapter.draft_text),
        temperature=0.2,
        max_tokens=400,
        agent="summariser",
    )


# ── holistic ─────────────────────────────────────────────────────────


async def holistic_review(
    *,
    job: AviationJob,
    bible: AviationStoryBible,
    manuscript: str,
    model: str,
) -> dict[str, Any]:
    return await llm_json(
        job=job,
        model=model,
        prompt=P.holistic_prompt(bible=bible, full_text=manuscript),
        temperature=0.2,
        max_tokens=2000,
        agent="holistic",
    )  # type: ignore[return-value]


# ── YouTube metadata ─────────────────────────────────────────────────


async def generate_metadata(
    *,
    job: AviationJob,
    bible: AviationStoryBible,
    total_words: int,
    model: str,
) -> dict[str, Any]:
    return await llm_json(
        job=job,
        model=model,
        prompt=P.metadata_prompt(bible=bible, total_words=total_words),
        temperature=0.6,
        max_tokens=1600,
        agent="metadata",
    )  # type: ignore[return-value]


# ── Storyboard ───────────────────────────────────────────────────────


STORYBOARD_BATCH = 30


async def generate_storyboard(
    *,
    job: AviationJob,
    bible: AviationStoryBible,
    clean_script: str,
    model: str,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> list[dict[str, Any]]:
    """Return the [{index,timestamp,duration,text_segment,image_prompt}, …] list."""
    rows = segment_with_timestamps(clean_script, wpm=job.settings.words_per_minute)
    total_batches = max(1, -(-len(rows) // STORYBOARD_BATCH))
    for b in range(total_batches):
        batch = rows[b * STORYBOARD_BATCH : (b + 1) * STORYBOARD_BATCH]
        data = await llm_json(
            job=job,
            model=model,
            prompt=P.storyboard_prompt(
                bible=bible,
                segments=[r["text"] for r in batch],
                batch_start_index=batch[0]["index"] if batch else 0,
            ),
            temperature=0.5,
            max_tokens=3500,
            agent="storyboard",
        )
        prompts = data.get("prompts") if isinstance(data, dict) else data
        if isinstance(prompts, list):
            # Match up by index for robustness.
            by_idx = {int(p.get("index", -1)): p for p in prompts if isinstance(p, dict)}
            for r in batch:
                p = by_idx.get(r["index"])
                if p and "image_prompt" in p:
                    r["image_prompt"] = str(p["image_prompt"])
                else:
                    r["image_prompt"] = _fallback_prompt(bible)
        else:
            for r in batch:
                r["image_prompt"] = _fallback_prompt(bible)
        if on_progress:
            await on_progress(b + 1, total_batches)
    return rows


def _fallback_prompt(bible: AviationStoryBible) -> str:
    return (
        f"Cinematic still, {bible.aircraft.type} cockpit interior at night, "
        "instrument-panel glow on the captain's face, wide 35mm, moody "
        "chiaroscuro lighting, ultra-detailed, 8k, no text, no watermark."
    )
