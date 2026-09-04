"""Global History Manager.

Runs across the entire batch to guarantee three things:

1. **Per-axis cooldowns** on 8+ rotation axes (sub-genre, aircraft,
   setting, protagonist archetype, incident type, twist, resolution,
   hook pattern, narrator voice, character first name, fictional
   airline, narrative structure) — a value cannot repeat within its
   cooldown window in the batch history.

2. **Parameter-tuple uniqueness.** The triple
   {sub_genre, aircraft_type, setting} must never repeat across the
   entire history of the channel.

3. **Fictional-element uniqueness.** Fictional airlines, crew names,
   aircraft registrations, flight numbers, and cities can't be reused
   in later fictional-mode stories. Real-mode never violates.

The store is a single SQLite file (default
``data/global_history.db``). Callers don't manage connections —
every public method opens, does its work, and closes.

Public API — the four things a Planner asks:

.. code-block:: python

    hist = HistoryStore()
    # 1. Structure rotation walker (backwards-compatible).
    structure = hist.next_structure(previous=None)

    # 2. Axis picking + cooldown check.
    forbidden_values = hist.recent_axis_values("sub_genre")
    is_ok = hist.cooldown_ok("sub_genre", proposed_value)

    # 3. Parameter-tuple uniqueness (for {sub_genre, aircraft, setting}).
    tuple_ok = hist.parameter_tuple_unique(sub_genre, aircraft, setting)

    # 4. Fictional-element uniqueness (unchanged from v1).
    violations = hist.check_bible(bible)

    # After the story is finalised:
    hist.record_bible(external_id, bible, axes={"sub_genre": ..., "hook_pattern": ...})
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from aviation.axes import AXES, HOOK_MAX_USES, HOOK_MAX_USES_WINDOW, HookPattern
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

_ENFORCED_KINDS = ("airline", "registration", "flight_number", "character", "city")


@dataclass
class HistorySummary:
    """Read-only snapshot for the UI."""

    completed_incidents: int
    structures_used: list[str] = field(default_factory=list)
    elements_by_kind: dict[str, list[str]] = field(default_factory=dict)
    axis_recent_values: dict[str, list[str]] = field(default_factory=dict)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


class HistoryStore:
    """SQLite-backed global history for the aviation factory."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
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
                    external_id   TEXT UNIQUE,
                    title         TEXT,
                    mode          TEXT NOT NULL,
                    structure     TEXT NOT NULL,
                    completed_at  TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS elements (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id   INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
                    kind          TEXT NOT NULL,
                    value         TEXT NOT NULL,
                    norm          TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS elements_norm_kind ON elements(kind, norm);
                CREATE INDEX IF NOT EXISTS elements_incident ON elements(incident_id);

                CREATE TABLE IF NOT EXISTS axis_values (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id   INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
                    axis          TEXT NOT NULL,
                    value         TEXT NOT NULL,
                    norm          TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS axis_values_axis ON axis_values(axis);
                CREATE INDEX IF NOT EXISTS axis_values_incident ON axis_values(incident_id);

                CREATE TABLE IF NOT EXISTS seed_incidents (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id   INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
                    seed_name     TEXT NOT NULL,
                    seed_norm     TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS seed_norm ON seed_incidents(seed_norm);
                """
            )

    # ── structure rotation (legacy walker) ────────────────────────

    def last_structure(self) -> Optional[NarrativeStructure]:
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
        prev = previous or self.last_structure()
        if prev is None or prev not in ROTATION_ORDER:
            return ROTATION_ORDER[0]
        idx = ROTATION_ORDER.index(prev)
        return ROTATION_ORDER[(idx + 1) % len(ROTATION_ORDER)]

    # ── axis cooldowns ────────────────────────────────────────────

    def recent_axis_values(self, axis: str, limit: int | None = None) -> list[str]:
        """Return the most-recent values for ``axis``, newest first.

        ``limit`` defaults to the axis's cooldown window from ``AXES``.
        """
        spec = AXES.get(axis)
        window = limit if limit is not None else (spec.cooldown if spec else 3)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT av.value FROM axis_values av "
                "JOIN incidents i ON i.id = av.incident_id "
                "WHERE av.axis = ? ORDER BY i.id DESC LIMIT ?",
                (axis, window),
            ).fetchall()
        return [r[0] for r in rows]

    def cooldown_ok(self, axis: str, value: str) -> bool:
        """True if ``value`` is not in the axis's cooldown window."""
        if not value:
            return True
        spec = AXES.get(axis)
        if spec is None:
            return True
        recent = self.recent_axis_values(axis, spec.cooldown)
        n = _norm(value)
        return not any(_norm(r) == n for r in recent)

    def hook_pattern_allowed(self, value: str) -> bool:
        """Hook pattern has an extra rule: no more than ``HOOK_MAX_USES``
        uses in the last ``HOOK_MAX_USES_WINDOW`` stories.
        """
        recent = self.recent_axis_values("hook_pattern", HOOK_MAX_USES_WINDOW)
        n = _norm(value)
        matches = sum(1 for r in recent if _norm(r) == n)
        return matches < HOOK_MAX_USES

    def parameter_tuple_unique(
        self, sub_genre: str, aircraft_type: str, setting: str
    ) -> bool:
        """True when the {sub_genre, aircraft, setting} triple has never
        appeared in this history."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT incident_id FROM axis_values "
                "WHERE axis IN ('sub_genre', 'aircraft_type', 'setting')"
            ).fetchall()
            for (iid,) in rows:
                trip = conn.execute(
                    "SELECT axis, value FROM axis_values WHERE incident_id = ? "
                    "AND axis IN ('sub_genre', 'aircraft_type', 'setting')",
                    (iid,),
                ).fetchall()
                mapping = {a: v for a, v in trip}
                if (
                    _norm(mapping.get("sub_genre", "")) == _norm(sub_genre)
                    and _norm(mapping.get("aircraft_type", "")) == _norm(aircraft_type)
                    and _norm(mapping.get("setting", "")) == _norm(setting)
                ):
                    return False
        return True

    # ── seed incident reuse ───────────────────────────────────────

    def seed_used(self, seed_name: str) -> bool:
        """True when a seed incident with this name has already been consumed."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM seed_incidents WHERE seed_norm = ? LIMIT 1",
                (_norm(seed_name),),
            ).fetchone()
        return row is not None

    def used_seed_names(self) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT seed_name FROM seed_incidents ORDER BY id"
            ).fetchall()
        return [r[0] for r in rows]

    # ── fictional-element uniqueness (v1) ─────────────────────────

    def forbidden_elements(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {k: [] for k in _ELEMENT_KINDS}
        with self._lock, self._connect() as conn:
            for kind, value in conn.execute(
                "SELECT kind, value FROM elements ORDER BY id"
            ):
                if value not in out.setdefault(kind, []):
                    out[kind].append(value)
        return out

    def check_bible(self, bible: AviationStoryBible) -> list[str]:
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
        *,
        axes: dict[str, str] | None = None,
        seed_incident_name: str | None = None,
    ) -> int:
        """Persist an incident's elements, axes, and (optional) seed."""
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
                conn.execute("DELETE FROM axis_values WHERE incident_id=?", (incident_id,))
                conn.execute("DELETE FROM seed_incidents WHERE incident_id=?", (incident_id,))
            conn.executemany(
                "INSERT INTO elements (incident_id, kind, value, norm) VALUES (?, ?, ?, ?)",
                [(incident_id, k, v, _norm(v)) for (k, v) in elements],
            )
            if axes:
                conn.executemany(
                    "INSERT INTO axis_values (incident_id, axis, value, norm) VALUES (?, ?, ?, ?)",
                    [(incident_id, ax, val, _norm(val)) for ax, val in axes.items() if val],
                )
            if seed_incident_name:
                conn.execute(
                    "INSERT INTO seed_incidents (incident_id, seed_name, seed_norm) VALUES (?, ?, ?)",
                    (incident_id, seed_incident_name, _norm(seed_incident_name)),
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
            axis_recent: dict[str, list[str]] = {}
            for axis in AXES:
                rows = conn.execute(
                    "SELECT av.value FROM axis_values av "
                    "JOIN incidents i ON i.id = av.incident_id "
                    "WHERE av.axis = ? ORDER BY i.id DESC LIMIT ?",
                    (axis, max(3, AXES[axis].cooldown)),
                ).fetchall()
                axis_recent[axis] = [r[0] for r in rows]
        return HistorySummary(
            completed_incidents=int(n),
            structures_used=structures,
            elements_by_kind=elements,
            axis_recent_values=axis_recent,
        )

    def reset(self) -> None:
        """Wipe the history (for tests / fresh starts)."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM axis_values")
            conn.execute("DELETE FROM elements")
            conn.execute("DELETE FROM seed_incidents")
            conn.execute("DELETE FROM incidents")


# ── helpers ─────────────────────────────────────────────────────────


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
    """Best-effort in-place rename to break uniqueness ties."""
    suffix_pool = ["Nova", "Prime", "Atlas", "Beacon", "II", "III"]
    suffix = suffix_pool[attempt % len(suffix_pool)]
    if bible.aircraft.operator:
        bible.aircraft.operator = f"{bible.aircraft.operator} {suffix}".strip()
    if bible.aircraft.registration:
        reg = bible.aircraft.registration
        bible.aircraft.registration = reg[:-1] + ("A" if reg[-1].upper() == "Z" else chr(ord(reg[-1]) + 1))
    if bible.aircraft.flight_number:
        bible.aircraft.flight_number = f"{bible.aircraft.flight_number}{attempt + 1}"
    for member in bible.crew:
        if member.name:
            member.name = f"{member.name} {suffix}"
