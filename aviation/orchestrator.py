"""Orchestrator — runs the aviation pipeline for one job.

Graph (checkpoint after every node into ``data/jobs/<job_id>.json``)::

    START
      → ingest        (real-mode only; parses PDFs → ExtractedFacts)
      → plan          (chooses narrative structure, builds StoryBible)
      → plan_chapters (splits into ~12 chapter drafts)
      → for each chapter:
          → write
          → fact_check                    (real-mode only)
          ├─ high issues       → editor → back to fact_check (bounded)
          → critique
          ├─ score < min       → editor → back to critique  (bounded)
          → summarise → approve
      → holistic
      ├─ flagged chapters      → editor per chapter → holistic  (bounded rounds)
      → post_process
          → tts_ssml
          → youtube_metadata
          → storyboard  (batched)
          → manuscript, bible.json
      → DONE

Every node persists the job. On resume the runner picks up at
``job.current_node`` and re-runs the failed node (idempotent).
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from aviation import agents as A
from aviation import deliverables as D
from aviation.axes import (
    AXES,
    HookPattern,
    PROTAGONIST_ARCHETYPES,
    SubGenre,
    AircraftClass,
    Setting as SettingAxis,
    IncidentType,
    TwistType,
    Resolution as ResolutionAxis,
    EmotionalBeat,
)
from aviation.persistence import save_job
from aviation.resources import IncidentSeed, filter_incidents, load_airlines, sample_airline, sample_names
from aviation.state import AviationJob, ChapterDraft, JobStatus, JobSettings
from aviation.text import (
    count_words,
    last_words,
    merge_ledger,
    strip_for_tts,
    words_to_seconds,
)
from core.history import HistoryStore, resolve_violations
from core.pdf_ingest import load_pdf
from core.settings import Settings
from models.aviation_bible import (
    AviationStoryBible,
    ExtractedFacts,
    Mode,
    NarrativeStructure,
)

logger = logging.getLogger(__name__)


class CancelledError(RuntimeError):
    """Raised when the job's cancel flag is set mid-run."""


# ── model resolution ─────────────────────────────────────────────────

_ROLES = ("primary", "evaluation", "fact_check", "summary", "storyboard")


def resolve_models(job: AviationJob, settings: Settings) -> dict[str, str]:
    """Return a role→model_id mapping honouring per-job overrides."""
    provider = "openrouter"
    # Try to read the settings.api.models block; fall back to safe defaults.
    provider_models = getattr(settings.api, "models", {}) or {}
    # Newer pydantic wraps this as a nested model; fall back to attribute access.
    if not isinstance(provider_models, dict):
        provider_models = provider_models.model_dump() if hasattr(provider_models, "model_dump") else {}

    # Prefer the mock block if forced.
    import os as _os
    force_mock = _os.environ.get("AVIATION_FORCE_MOCK", "").strip() == "1"
    if force_mock:
        provider = "mock"

    per = provider_models.get(provider) or provider_models.get("openrouter") or {}
    if not isinstance(per, dict):
        per = per.model_dump() if hasattr(per, "model_dump") else {}

    defaults = {
        "primary": per.get("primary") or "openrouter/anthropic/claude-3.5-sonnet",
        "evaluation": per.get("evaluation") or per.get("primary") or "openrouter/anthropic/claude-3-haiku",
        "fact_check": per.get("fact_check") or per.get("evaluation") or "openrouter/anthropic/claude-3-haiku",
        "summary": per.get("summary") or per.get("evaluation") or "openrouter/anthropic/claude-3-haiku",
        "storyboard": per.get("storyboard") or per.get("evaluation") or "openrouter/anthropic/claude-3-haiku",
    }
    if force_mock:
        for k in defaults:
            defaults[k] = "mock/demo"

    # Per-job override wins.
    return {
        "primary": job.settings.model_primary or defaults["primary"],
        "evaluation": job.settings.model_evaluation or defaults["evaluation"],
        "fact_check": job.settings.model_fact_check or defaults["fact_check"],
        "summary": job.settings.model_summary or defaults["summary"],
        "storyboard": job.settings.model_storyboard or defaults["storyboard"],
    }


# ── main entry ───────────────────────────────────────────────────────


async def run_job(
    job: AviationJob,
    *,
    settings: Settings | None = None,
    history: HistoryStore | None = None,
    on_progress: Any | None = None,
    cancel_check: Any | None = None,
) -> AviationJob:
    """Drive one job to completion (or failure). Persists after every node.

    ``cancel_check`` is a zero-arg callable returning True when the
    orchestrator should raise :class:`CancelledError`. When ``None``
    the job is never cancelled externally.
    """
    settings = settings or Settings()
    history = history or HistoryStore()
    models = resolve_models(job, settings)

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    def _ck() -> None:
        if _cancelled():
            raise CancelledError()

    job.status = JobStatus.RUNNING
    job.error = ""
    job.output_dir = job.output_dir or str(Path("data/outputs") / job.job_id)
    Path(job.output_dir).mkdir(parents=True, exist_ok=True)
    save_job(job)

    try:
        # ── ingest ────────────────────────────────────────────
        if job.settings.mode == Mode.REAL and job.facts is None:
            _ck()
            job.current_node = "ingest"
            job.progress = 0.02
            save_job(job)
            extracted = await _run_ingest(job, models["primary"])
            # If ingest returned an empty facts object (no PDFs), leave
            # job.facts as None so the planner falls back to the seed
            # catalog rather than injecting an empty facts blob.
            job.facts = extracted if extracted.incident_name else None
            save_job(job)

        # ── plan ──────────────────────────────────────────────
        if job.bible is None:
            _ck()
            job.current_node = "plan"
            job.progress = 0.06
            save_job(job)
            job.bible = await _run_plan(job, history, models["primary"])
            save_job(job)

        # ── plan_chapters ─────────────────────────────────────
        if not job.chapters:
            _ck()
            job.current_node = "plan_chapters"
            job.progress = 0.10
            save_job(job)
            job.chapters = await A.plan_chapters(
                job=job, bible=job.bible, model=models["primary"]
            )
            save_job(job)

        # ── chapter loop ──────────────────────────────────────
        total_chapters = len(job.chapters)
        while job.current_chapter < total_chapters:
            _ck()
            i = job.current_chapter
            chapter = job.chapters[i]
            job.current_node = f"chapter_{i + 1}"
            frac = 0.12 + 0.70 * (i / max(1, total_chapters))
            job.progress = frac
            save_job(job)

            await _run_one_chapter(
                job=job,
                chapter=chapter,
                models=models,
            )

            # Approved. Update running memory.
            chapter.clean_text = strip_for_tts(chapter.draft_text)
            job.previous_tail = last_words(chapter.clean_text, 220)
            job.story_so_far = _extend_memory(job.story_so_far, chapter)
            job.ledger = merge_ledger(job.ledger, chapter.established_facts)
            chapter.approved = True
            job.current_chapter = i + 1
            save_job(job)

        # ── holistic loop ─────────────────────────────────────
        while job.holistic_rounds < job.settings.max_holistic_rounds:
            _ck()
            job.current_node = f"holistic_{job.holistic_rounds + 1}"
            job.progress = 0.86
            save_job(job)
            manuscript = "\n\n".join(c.draft_text for c in job.chapters)
            verdict = await A.holistic_review(
                job=job, bible=job.bible, manuscript=manuscript, model=models["evaluation"]
            )
            score = float(verdict.get("overall_score", 0.0) or 0.0)
            flagged = verdict.get("flagged_chapters") or []
            job.append_log(
                "holistic",
                f"Round {job.holistic_rounds + 1}: score {score:.1f}/10, "
                f"{len(flagged)} chapter(s) flagged.",
                level="success" if score >= job.settings.min_score and not flagged else "warn",
            )
            if score >= job.settings.min_score and not flagged:
                break
            # Re-edit flagged chapters and loop.
            for f in flagged:
                idx = int(f.get("index", -1))
                if 0 <= idx < len(job.chapters):
                    ch = job.chapters[idx]
                    ch.critic_notes = str(f.get("instructions") or f.get("reason") or "")
                    ch.draft_text = await A.edit_chapter(
                        job=job, bible=job.bible, chapter=ch, model=models["primary"]
                    )
                    ch.clean_text = strip_for_tts(ch.draft_text)
                    save_job(job)
            job.holistic_rounds += 1

        # ── post-process ──────────────────────────────────────
        _ck()
        job.current_node = "post_process"
        job.progress = 0.90
        save_job(job)
        await _run_post_process(job, models)

        job.status = JobStatus.COMPLETED
        job.current_node = "done"
        job.progress = 1.0
        from datetime import datetime, timezone
        job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        job.touch()
        save_job(job)
        job.append_log("system", "Job complete.", level="success")
        save_job(job)

    except CancelledError:
        job.status = JobStatus.CANCELLED
        job.append_log("system", "Job cancelled.", level="warn")
        save_job(job)
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        job.append_log("system", f"FAILED at {job.current_node}: {exc}", level="error")
        save_job(job)
        raise

    return job


# ── node implementations ────────────────────────────────────────────


async def _run_ingest(job: AviationJob, model: str) -> ExtractedFacts:
    if not job.source_pdfs:
        # No PDFs → the planner falls back to the seed catalog. Return
        # an empty ExtractedFacts so downstream fact-check logic knows
        # there's nothing hard-authoritative to cross-check against.
        job.append_log(
            "ingest",
            "Real mode with no PDFs — planner will pick from the seed catalog.",
        )
        return ExtractedFacts()
    facts = ExtractedFacts()
    for pdf_path in job.source_pdfs:
        job.append_log("ingest", f"Loading {pdf_path}")
        doc = load_pdf(pdf_path)
        job.append_log(
            "ingest",
            f"Parsed via {doc.parser} ({doc.total_chars:,} chars, {len(doc.pages)} pages).",
        )
        partial = await A.ingest_pdf(job=job, doc=doc, model=model)
        # Merge each PDF's facts into the running accumulator.
        A._merge_facts(facts, partial.model_dump(mode="json"))
    return facts


def _pick_axes(
    job: AviationJob,
    history: HistoryStore,
    seed: IncidentSeed | None,
) -> tuple[dict[str, str], IncidentSeed | None, list[str], list[str]]:
    """Pick a value for every rotation axis, honouring cooldowns.

    * If a real-mode seed is provided, its sub_genre / aircraft_type /
      setting take precedence (a real story doesn't get to change what
      it is about).
    * Otherwise walks each axis's enum in a stable order and picks the
      first value that passes the cooldown check. Falls back to the
      first enum value if every candidate is on cooldown (extreme).
    * Hook pattern additionally honours the "max 3 uses in last 10"
      rule.
    * Returns (axis map, chosen seed, airline pool hints, name pool
      hints).
    """
    import random

    rng = random.Random(hash(job.job_id) & 0xFFFF)

    axis_choices: dict[str, str] = {}

    def _first_ok(axis: str, candidates: list[str]) -> str:
        for c in candidates:
            if history.cooldown_ok(axis, c):
                if axis == "hook_pattern" and not history.hook_pattern_allowed(c):
                    continue
                return c
        # All on cooldown — take the first anyway, prefer axis default.
        return candidates[0] if candidates else ""

    if seed is not None:
        axis_choices["sub_genre"] = seed.sub_genre_primary or SubGenre.MIRACLE_LANDING.value
        axis_choices["aircraft_type"] = seed.aircraft_type or AircraftClass.NARROW_BODY.value
        axis_choices["setting"] = seed.setting_layer2 or SettingAxis.NORTH_ATLANTIC.value
    else:
        genres = [g.value for g in SubGenre]
        rng.shuffle(genres)
        axis_choices["sub_genre"] = _first_ok("sub_genre", genres)
        acs = [a.value for a in AircraftClass]
        rng.shuffle(acs)
        axis_choices["aircraft_type"] = _first_ok("aircraft_type", acs)
        settings_v = [s.value for s in SettingAxis]
        rng.shuffle(settings_v)
        axis_choices["setting"] = _first_ok("setting", settings_v)

    # Protagonist archetype rotates freely.
    arches = list(PROTAGONIST_ARCHETYPES)
    rng.shuffle(arches)
    axis_choices["protagonist_archetype"] = _first_ok("protagonist_archetype", arches)

    # Inciting incident — bias by seed causation when available.
    incidents_pool = [i.value for i in IncidentType]
    if seed and seed.causation_type:
        # Sort so causation-related incidents come first.
        c = seed.causation_type.lower()
        incidents_pool.sort(key=lambda v: 0 if any(t in v.lower() for t in c.split("+")) else 1)
    else:
        rng.shuffle(incidents_pool)
    axis_choices["inciting_incident"] = _first_ok("inciting_incident", incidents_pool)

    twists = [t.value for t in TwistType]
    rng.shuffle(twists)
    axis_choices["twist_type"] = _first_ok("twist_type", twists)

    resolutions = [r.value for r in ResolutionAxis]
    rng.shuffle(resolutions)
    axis_choices["resolution"] = _first_ok("resolution", resolutions)

    beats = [b.value for b in EmotionalBeat]
    rng.shuffle(beats)
    axis_choices["emotional_beat"] = beats[0]

    hooks = [h.value for h in HookPattern]
    rng.shuffle(hooks)
    axis_choices["hook_pattern"] = _first_ok("hook_pattern", hooks)

    # Airline pool hint: 5 candidates the planner may pick from.
    used_airlines = set(history.forbidden_elements().get("airline", []))
    airline_hint = [a.name for a in load_airlines() if a.name not in used_airlines][:5]

    # Name-pool hint: 8 candidates.
    used_first_names = {c.split()[0] for c in history.forbidden_elements().get("character", []) if c}
    name_candidates = sample_names(8, excluded_first_names=used_first_names, rng=rng)
    name_hint = [f"{n.first_name} {n.last_name} ({n.role_hint or 'crew'})" for n in name_candidates]

    return axis_choices, seed, airline_hint, name_hint


async def _run_plan(
    job: AviationJob,
    history: HistoryStore,
    model: str,
) -> AviationStoryBible:
    # Structure: user override > next-in-rotation.
    if job.settings.force_structure:
        structure = job.settings.force_structure
    else:
        structure = history.next_structure()

    # Real-mode: pick a seed incident matching whatever the user picked
    # (falling back to sub_genre filtering only).
    seed: IncidentSeed | None = None
    if job.settings.mode == Mode.REAL and not job.facts and not job.source_pdfs:
        # No PDFs, no facts — try the seed catalog.
        used_seeds = set(history.used_seed_names())
        pool = filter_incidents(
            max_risk="MED",  # keep HIGH-risk incidents out unless the user opts in
            excluded_names=used_seeds,
        )
        if pool:
            import random
            seed = random.Random(hash(job.job_id) & 0xFFFF).choice(pool)
            job.append_log(
                "planner",
                f"Real-mode seed catalog picked: {seed.name} (risk {seed.monetization_risk}).",
            )

    axis_choices, seed, airline_hint, name_hint = _pick_axes(job, history, seed)
    axis_choices["narrative_structure"] = structure.value

    forbidden = history.forbidden_elements()

    job.append_log(
        "planner",
        f"Structure: {structure.value}. Axes picked: "
        + ", ".join(f"{k}={v[:40]}" for k, v in axis_choices.items() if v)
        + f". Forbidden elements loaded ({sum(len(v) for v in forbidden.values())} total).",
    )

    # Uniqueness-retry loop for fictional mode.
    from aviation.prompts import planner_prompt as _planner_prompt
    from aviation.llm_helpers import llm_json

    bible: AviationStoryBible | None = None
    seed_summary = seed.summary_for_prompt() if seed else ""
    for attempt in range(3):
        # Direct call — the standard A.plan_story doesn't yet accept axis/seed hints.
        prompt = _planner_prompt(
            topic=job.topic,
            mode=job.settings.mode,
            structure=structure,
            target_words=job.settings.target_words,
            chapter_target=job.settings.chapter_target_words,
            facts=job.facts,
            forbidden_elements=forbidden,
            fictionalization_notice=_fictionalization_notice_for_planner(job.settings.mode),
            axis_selection=axis_choices,
            seed_summary=seed_summary,
            airline_pool_hint=airline_hint,
            name_pool_hint=name_hint,
        )
        data = await llm_json(
            job=job,
            model=model,
            prompt=prompt,
            temperature=0.6,
            max_tokens=6000,
            agent="planner",
        )
        bible = A._hydrate_bible(  # type: ignore[attr-defined]
            data,
            structure=structure,
            mode=job.settings.mode,
            notice=_fictionalization_notice_for_planner(job.settings.mode),
        )
        violations = history.check_bible(bible)
        if not violations:
            break
        job.append_log(
            "history",
            f"Uniqueness violations on attempt {attempt + 1}: {violations}. Re-planning.",
            level="warn",
        )
        for v in violations:
            m = re.match(r"([a-z_]+):\s+\"(.+?)\"", v)
            if m:
                kind, value = m.group(1), m.group(2)
                forbidden.setdefault(kind, []).append(value)
    else:
        job.append_log("history", "Auto-renaming to break uniqueness tie.", level="warn")
        resolve_violations(bible, attempt=2)  # type: ignore[arg-type]

    assert bible is not None
    # Record with all axis values and the seed name (if any).
    history.record_bible(
        job.job_id,
        bible,
        axes=axis_choices,
        seed_incident_name=seed.name if seed else None,
    )
    job.title = bible.working_title
    return bible


def _fictionalization_notice_for_planner(mode: Mode) -> str:
    """Duplicated tiny helper (agents.py owns the canonical one)."""
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


async def _run_one_chapter(
    *,
    job: AviationJob,
    chapter: ChapterDraft,
    models: dict[str, str],
) -> None:
    # 1. Write
    chapter.draft_text = await A.write_chapter(
        job=job,
        bible=job.bible,
        chapter=chapter,
        facts=job.facts,
        model=models["primary"],
    )
    save_job(job)
    job.append_log("writer", f"Chapter {chapter.index + 1}: draft {count_words(chapter.draft_text):,} words.")

    # 2. Fact-check + critic loop, bounded by max_revisions.
    for rev in range(job.settings.max_revisions_per_chapter + 1):
        # Fact check (real mode only).
        if job.settings.mode == Mode.REAL and job.facts is not None:
            fc = await A.fact_check_chapter(
                job=job, facts=job.facts, chapter=chapter, model=models["fact_check"]
            )
            chapter.fact_check_passed = bool(fc.get("passed", True))
            chapter.fact_check_issues = _high_severity_issues(fc)
            job.append_log(
                "fact_check",
                f"Ch{chapter.index + 1} pass={chapter.fact_check_passed} "
                f"confidence={fc.get('confidence','?')} issues={len(fc.get('issues') or [])}",
                level="success" if chapter.fact_check_passed else "warn",
            )
        else:
            chapter.fact_check_passed = True
            chapter.fact_check_issues = []

        # Critic
        cr = await A.critique_chapter(
            job=job, bible=job.bible, chapter=chapter, model=models["evaluation"]
        )
        chapter.critic_score = float(cr.get("overall_score", 0.0) or 0.0)
        chapter.critic_notes = _summarise_notes(cr)
        chapter.established_facts = list(cr.get("continuity_facts_established") or [])
        save_job(job)
        job.append_log(
            "critic",
            f"Ch{chapter.index + 1}: {chapter.critic_score:.1f}/10 (rev {rev}).",
            level=(
                "success"
                if chapter.critic_score >= job.settings.min_score and chapter.fact_check_passed
                else "warn"
            ),
        )

        passes = (
            chapter.critic_score >= job.settings.min_score
            and chapter.fact_check_passed
        )
        if passes:
            break
        if rev >= job.settings.max_revisions_per_chapter:
            job.append_log(
                "system",
                f"Ch{chapter.index + 1}: max revisions reached; accepting best draft.",
                level="warn",
            )
            break
        # Editor pass
        chapter.draft_text = await A.edit_chapter(
            job=job, bible=job.bible, chapter=chapter, model=models["primary"]
        )
        chapter.revisions = rev + 1
        save_job(job)

    # 3. Summarise for the running memory
    chapter.summary = await A.summarise_chapter(
        job=job, chapter=chapter, model=models["summary"]
    )


def _high_severity_issues(fc: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for issue in fc.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        sev = str(issue.get("severity", "")).lower()
        claim = str(issue.get("claim") or "")
        fix = str(issue.get("suggested_fix") or "")
        if sev == "high":
            out.append(f"[HIGH] {claim} — {fix}".strip(" —"))
    return out


def _summarise_notes(cr: dict[str, Any]) -> str:
    parts: list[str] = []
    for level_key in ("l1_technical", "l2_linguistic", "l3_content", "l4_voiceover"):
        block = cr.get(level_key) or {}
        for issue in block.get("issues") or []:
            desc = issue.get("description") or issue.get("category") or issue.get("claim")
            if desc:
                parts.append(f"- {level_key}: {desc}")
    summary = cr.get("summary")
    if summary:
        parts.insert(0, str(summary))
    return "\n".join(parts).strip()


def _extend_memory(existing: str, chapter: ChapterDraft) -> str:
    memo = chapter.summary or f"(no summary; chapter {chapter.index + 1} landed.)"
    entry = f"Chapter {chapter.index + 1} — {chapter.title}: {memo}"
    return (existing + "\n\n" + entry).strip() if existing else entry


# ── post-process ────────────────────────────────────────────────────


async def _run_post_process(job: AviationJob, models: dict[str, str]) -> None:
    approved = [c for c in job.chapters if c.draft_text]
    texts = [c.draft_text for c in approved]
    titles = [c.title for c in approved]
    clean_chapters = [strip_for_tts(t) for t in texts]
    clean_script = "\n\n".join(clean_chapters).strip()
    total_words = count_words(clean_script)
    job.manuscript = D.build_manuscript(job.bible, texts, titles)

    # File 1: TTS SSML
    ssml = D.to_elevenlabs_ssml(clean_script)
    D.write_deliverable(job, "01_tts_script.txt", ssml, kind="tts")
    job.append_log("post_process", f"01_tts_script.txt written ({total_words:,} words).")

    # File 2: YouTube metadata
    meta = await A.generate_metadata(
        job=job, bible=job.bible, total_words=total_words, model=models["evaluation"]
    )
    starts = D.chapter_starts(clean_chapters, job.settings.words_per_minute, titles)
    total_seconds = D.total_runtime_seconds(clean_chapters, job.settings.words_per_minute)
    md = D.render_youtube_metadata(
        metadata=meta,
        bible=job.bible,
        chapter_titles_and_starts=starts,
        total_seconds=total_seconds,
        mode=job.settings.mode,
    )
    D.write_deliverable(job, "02_youtube_metadata.md", md, kind="metadata")
    job.append_log("post_process", "02_youtube_metadata.md written.")

    # File 3: Storyboard CSV (+ pipe-delimited + JSON variants).
    async def _stb_progress(done: int, total: int) -> None:
        job.progress = 0.90 + 0.09 * (done / max(1, total))
        save_job(job)

    rows = await A.generate_storyboard(
        job=job,
        bible=job.bible,
        clean_script=clean_script,
        model=models["storyboard"],
        on_progress=_stb_progress,
    )
    D.write_deliverable(job, "03_storyboard.csv", D.render_storyboard_csv(rows), kind="storyboard_csv")
    D.write_deliverable(job, "03_storyboard_pipe.txt", D.render_storyboard_pipe(rows), kind="storyboard_pipe")
    import json
    D.write_deliverable(job, "03_storyboard.json", json.dumps(rows, indent=2, ensure_ascii=False), kind="storyboard_json")
    job.append_log("post_process", f"03_storyboard.csv written ({len(rows)} segments).")

    # Bonus: manuscript + bible.
    D.write_deliverable(job, "00_manuscript.md", job.manuscript, kind="manuscript")
    D.write_deliverable(job, "04_story_bible.json", D.bible_json(job.bible), kind="bible")
    job.append_log("post_process", "00_manuscript.md and 04_story_bible.json written.")


# ── convenience: build a job from a queue row ───────────────────────


def new_job(
    *,
    topic: str,
    mode: Mode = Mode.FICTIONAL,
    source_pdfs: list[str] | None = None,
    settings_override: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> AviationJob:
    settings = JobSettings(mode=mode)
    if settings_override:
        settings = settings.model_copy(update=settings_override)
    return AviationJob(
        job_id=job_id or f"job_{uuid.uuid4().hex[:12]}",
        topic=topic,
        source_pdfs=source_pdfs or [],
        settings=settings,
    )


# ── sync wrapper ────────────────────────────────────────────────────


def run_job_sync(job: AviationJob, **kwargs: Any) -> AviationJob:
    return asyncio.run(run_job(job, **kwargs))
