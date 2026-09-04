"""In-memory + persisted state for one aviation-content-factory job.

An :class:`AviationJob` is created per incident and lives in the queue.
Its full state (settings, bible, per-chapter drafts, running ledger,
cost counters, deliverable paths) is serialisable to JSON so the run
can resume from disk after a crash.

Persistence is handled by :mod:`aviation.persistence`: after every
node the orchestrator calls ``save_job(job)`` which atomically writes
``data/jobs/<job_id>.json``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from models.aviation_bible import (
    AviationStoryBible,
    ExtractedFacts,
    Mode,
    NarrativeStructure,
)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobSettings(BaseModel):
    """User-facing knobs. One instance per job (usually cloned from
    the factory-wide defaults in ``settings.yaml``)."""

    mode: Mode = Field(default=Mode.FICTIONAL)
    target_words: int = Field(default=14400, ge=3000, le=25000)
    chapter_target_words: int = Field(default=1200, ge=400, le=4000)
    min_score: float = Field(default=8.0, ge=1.0, le=10.0)
    max_revisions_per_chapter: int = Field(default=2, ge=0, le=6)
    max_holistic_rounds: int = Field(default=2, ge=0, le=4)
    words_per_minute: int = Field(default=150, ge=100, le=220)
    force_structure: Optional[NarrativeStructure] = Field(
        default=None,
        description="If set, override the Global History Manager's rotation for this job.",
    )

    # LLM model overrides per role. Each is a LiteLLM-style model id.
    # Leave a role empty to use the value from settings.yaml.
    model_primary: str = Field(default="")
    model_evaluation: str = Field(default="")
    model_fact_check: str = Field(default="")
    model_summary: str = Field(default="")
    model_storyboard: str = Field(default="")


class ChapterDraft(BaseModel):
    """One chapter's evolving state."""

    index: int
    title: str = ""
    outline_bullets: list[str] = Field(default_factory=list)
    target_words: int = 1200
    draft_text: str = ""
    clean_text: str = ""
    revisions: int = 0
    critic_score: float = 0.0
    critic_notes: str = ""
    fact_check_passed: bool = True
    fact_check_issues: list[str] = Field(default_factory=list)
    summary: str = ""
    established_facts: list[str] = Field(default_factory=list)
    approved: bool = False


class Deliverable(BaseModel):
    kind: str
    filename: str
    path: str
    bytes: int = 0


class LogEntry(BaseModel):
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    agent: str
    level: str = "info"
    message: str


class AviationJob(BaseModel):
    """The full state of one aviation-story generation run."""

    job_id: str
    title: str = ""
    topic: str = ""
    source_pdfs: list[str] = Field(default_factory=list, description="Absolute paths.")
    settings: JobSettings = Field(default_factory=JobSettings)

    status: JobStatus = Field(default=JobStatus.PENDING)
    current_node: str = "created"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)

    facts: Optional[ExtractedFacts] = Field(default=None)
    bible: Optional[AviationStoryBible] = Field(default=None)

    chapters: list[ChapterDraft] = Field(default_factory=list)
    current_chapter: int = 0
    holistic_rounds: int = 0

    story_so_far: str = Field(default="")
    ledger: list[str] = Field(default_factory=list)
    previous_tail: str = Field(default="")

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    manuscript: str = ""
    deliverables: list[Deliverable] = Field(default_factory=list)
    output_dir: str = ""

    logs: list[LogEntry] = Field(default_factory=list)
    error: str = ""

    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    finished_at: str = ""

    # ── helpers ───────────────────────────────────────────────────

    def append_log(self, agent: str, message: str, level: str = "info") -> LogEntry:
        entry = LogEntry(agent=agent, level=level, message=message[:4000])
        self.logs.append(entry)
        # Bound the log to a reasonable size in memory / on disk.
        if len(self.logs) > 2000:
            self.logs = self.logs[-1500:]
        return entry

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def output_path(self, filename: str) -> Path:
        out = Path(self.output_dir or f"data/outputs/{self.job_id}")
        out.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(out)
        return out / filename
