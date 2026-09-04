"""Global History Manager.

Runs across the entire batch to guarantee two things:

1. **Forced narrative-structure rotation.** Consecutive stories don't
   repeat the same structure. Rotation walks
   :data:`~models.aviation_bible.ROTATION_ORDER`; the next structure
   is the one after whatever the previous incident used, wrapping.

2. **Fictional-element uniqueness.** In :attr:`Mode.FICTIONAL` mode
   the Planner must not reuse airlines, crew names, aircraft
   registrations, flight numbers, or origin/destination cities from
   any earlier fictional incident in the same history DB. Real-mode
   incidents also register their elements (so a later fictional story
   can't reuse them either) but are never rejected on uniqueness.

The store is a single SQLite file (default ``data/global_history.db``).
Callers do not need to manage connections — every public function
opens, does its work, and closes.

Public API:

.. code-block:: python

    hist = HistoryStore()
    structure = hist.next_structure(previous_incident_id=None)
    forbidden = hist.forbidden_elements()
    violations = hist.check_bible(bible)                # list[str]
    hist.record_bible(incident_id, bible)               # persists everything
    summary = hist.summary()

The manager is intentionally decoupled from the pipeline steps — a
step calls into it, but nothing in it depends on ``PipelineState``.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from models.aviation_bible import (
    AviationStoryBible,
    Mode,
    NarrativeStructure,
    ROTATION_ORDER,
)

logger = logging.getLogger(__name__)


DEFAULT_DB_PATH = Path("data/global_history.db")

_ELEMENT_KINDS = (
    "airline",
    "registration",
    "flight_number",
    "aircraft_type",  # recorded but never enforced
    "character",
    "city",
)

# Kinds where duplicates are rejected. Aircraft type is famously
# not unique — many stories can feature an A320. Registrations and
# flight numbers must be unique.
_ENFORCED_KINDS = ("airline", "registration", "flight_number", "character", "city")


@dataclass
class HistorySummary:
    """Read-only snapshot for the UI."""

    completed_incidents: int
    structures_used: list[str] = field(default_factory=list)
    elements_by_kind: dict[str, list[str]] = field(default_factory=dict)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


class HistoryStore:
    """SQLite-backed global history for the factory."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # sqlite3 connections aren't thread-safe by default; use a lock
        # and open per operation instead of caching a connection.
        self._lock = threading.RLock()
        self._ensure_schema()

    # ── schema ────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, isolation_level=None)  # autocommit
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id   TEXT UNIQUE,       -- caller's stable id (story_id)
                    title         TEXT,
                    mode          TEXT NOT NULL,     -- 'real' | 'fictional'
                    structure     TEXT NOT NULL,     -- NarrativeStructure value
                    completed_at  TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS elements (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id   INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
                    kind          TEXT NOT NULL,
                    value         TEXT NOT NULL,
                    norm          TEXT NOT NULL      -- normalised lower/alnum-only
                );
                CREATE INDEX IF NOT EXISTS elements_norm_kind ON elements(kind, norm);
                CREATE INDEX IF NOT EXISTS elements_incident ON elements(incident_id);
                """
            )

    # ── structure rotation ────────────────────────────────────────

    def last_structure(self) -> Optional[NarrativeStructure]:
        """Return the structure used by the most recent incident, or None."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT structure FROM incidents ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        try:
            return NarrativeStructure(row[0])
        except ValueError:
            return None

    def next_structure(self, previous: Optional[NarrativeStructure] = None) -> NarrativeStructure:
        """Pick the next structure by walking :data:`ROTATION_ORDER`.

        Uses ``previous`` if given, otherwise reads the last recorded
        structure from the DB. When no history exists yet, returns the
        first entry of the rotation order.
        """
        prev = previous or self.last_structure()
        if prev is None or prev not in ROTATION_ORDER:
            return ROTATION_ORDER[0]
        idx = ROTATION_ORDER.index(prev)
        return ROTATION_ORDER[(idx + 1) % len(ROTATION_ORDER)]

    # ── fictional-element uniqueness ──────────────────────────────

    def forbidden_elements(self) -> dict[str, list[str]]:
        """Return the current forbidden-elements map for the Planner.

        Contains every element ever recorded — real or fictional. The
        Planner uses this to steer away from reuse in a new fictional
        incident. Aircraft type is included in the return but the
        uniqueness check ignores it.
        """
        out: dict[str, list[str]] = {k: [] for k in _ELEMENT_KINDS}
        with self._lock, self._connect() as conn:
            for kind, value in conn.execute(
                "SELECT kind, value FROM elements ORDER BY id"
            ):
                if value not in out.setdefault(kind, []):
                    out[kind].append(value)
        return out

    def check_bible(self, bible: AviationStoryBible) -> list[str]:
        """Return violations of the uniqueness constraint (empty when clean).

        Real-mode incidents never produce violations — they are meant
        to name real airlines / flights / people.
        """
        if bible.mode == Mode.REAL:
            return []
        forbidden = self.forbidden_elements()
        violations: list[str] = []
        for kind, value in _extract_elements(bible):
            if kind not in _ENFORCED_KINDS:
                continue
            n = _norm(value)
            if not n:
                continue
            for f in forbidden.get(kind, []):
                if _norm(f) == n:
                    violations.append(f'{kind}: "{value}" already used')
                    break
        return violations

    # ── recording ─────────────────────────────────────────────────

    def record_bible(
        self,
        external_id: str,
        bible: AviationStoryBible,
    ) -> int:
        """Persist an incident's elements and structure choice.

        Returns the internal integer id (unique per external_id — a
        second call with the same external_id updates in place).
        """
        elements = list(_extract_elements(bible))
        title = bible.working_title or external_id
        structure = bible.narrative_structure.value

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT id FROM incidents WHERE external_id = ?", (external_id,)
            ).fetchone()
            if cur is None:
                cur = conn.execute(
                    "INSERT INTO incidents (external_id, title, mode, structure) "
                    "VALUES (?, ?, ?, ?)",
                    (external_id, title, bible.mode.value, structure),
                )
                incident_id = int(cur.lastrowid)
            else:
                incident_id = int(cur[0])
                conn.execute(
                    "UPDATE incidents SET title=?, mode=?, structure=?, completed_at=datetime('now') "
                    "WHERE id=?",
                    (title, bible.mode.value, structure, incident_id),
                )
                conn.execute("DELETE FROM elements WHERE incident_id=?", (incident_id,))
            conn.executemany(
                "INSERT INTO elements (incident_id, kind, value, norm) VALUES (?, ?, ?, ?)",
                [(incident_id, k, v, _norm(v)) for (k, v) in elements],
            )
        return incident_id

    # ── read-only helpers ─────────────────────────────────────────

    def summary(self) -> HistorySummary:
        with self._lock, self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()
            structures = [r[0] for r in conn.execute(
                "SELECT structure FROM incidents ORDER BY id"
            )]
            elements: dict[str, list[str]] = {k: [] for k in _ELEMENT_KINDS}
            for kind, value in conn.execute(
                "SELECT kind, value FROM elements ORDER BY id"
            ):
                if value not in elements.setdefault(kind, []):
                    elements[kind].append(value)
        return HistorySummary(
            completed_incidents=int(n),
            structures_used=structures,
            elements_by_kind=elements,
        )

    def reset(self) -> None:
        """Wipe the history (for tests / fresh starts)."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM elements")
            conn.execute("DELETE FROM incidents")


# ── helpers ───────────────────────────────────────────────────────────


def _extract_elements(bible: AviationStoryBible) -> Iterable[tuple[str, str]]:
    """Yield (kind, value) pairs for everything trackable in a bible."""
    if v := bible.aircraft.operator.strip():
        yield "airline", v
    if v := bible.aircraft.registration.strip():
        yield "registration", v
    if v := bible.aircraft.flight_number.strip():
        yield "flight_number", v
    if v := bible.aircraft.type.strip():
        yield "aircraft_type", v
    for member in bible.crew:
        if v := (member.name or "").strip():
            yield "character", v
    for character in bible.other_characters:
        if v := (character.name or "").strip():
            yield "character", v
    for city in (bible.route.origin, bible.route.destination, bible.route.alternate, bible.route.actual_diversion):
        v = (city or "").strip()
        if v:
            yield "city", v


def resolve_violations(bible: AviationStoryBible, attempt: int) -> None:
    """Best-effort in-place rename to break uniqueness ties.

    Called by the Planner after :meth:`HistoryStore.check_bible` still
    returns violations on the final retry. Nothing sophisticated —
    appends a distinguishing suffix so the DB constraint is satisfied
    and the story can proceed.
    """
    suffix_pool = ["Nova", "Prime", "Atlas", "Beacon", "II", "III"]
    suffix = suffix_pool[attempt % len(suffix_pool)]
    if bible.aircraft.operator:
        bible.aircraft.operator = f"{bible.aircraft.operator} {suffix}".strip()
    if bible.aircraft.registration:
        # Bump the last character.
        reg = bible.aircraft.registration
        bible.aircraft.registration = reg[:-1] + ("A" if reg[-1].upper() == "Z" else chr(ord(reg[-1]) + 1))
    if bible.aircraft.flight_number:
        bible.aircraft.flight_number = f"{bible.aircraft.flight_number}{attempt + 1}"
    for member in bible.crew:
        if member.name:
            member.name = f"{member.name} {suffix}"
