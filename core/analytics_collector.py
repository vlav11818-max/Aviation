"""Analytics collector for AI Story Generator Pro.

``AnalyticsCollector`` persists per-story generation records to a SQLite
database (``data/analytics/analytics.db``) and provides aggregated views
by language, provider, score distribution, common issues, and quality
trend.

Why SQLite instead of a flat JSON file
---------------------------------------
The original implementation stored all records in a single
``stats.json`` file and rewrote the **entire file** on every
``record_story()`` call.  At scale this becomes a serious bottleneck:

* 10,000 stories ≈ 3 MB read + 3 MB write per completion event
* 100,000 stories ≈ 30 MB per event, blocking the write lock for
  hundreds of milliseconds while other workers wait

SQLite solves this with:

* **O(1) INSERT** — each story appends one row; no full-file rewrite
* **Indexed GROUP-BY queries** — aggregations (by language, by provider)
  are executed in the DB engine, not by scanning a Python list
* **Crash safety** — WAL mode provides atomic, durable writes
* **No new dependency** — ``sqlite3`` is Python stdlib

Backward compatibility
-----------------------
If a ``stats.json`` file exists in the analytics directory on first
startup, its records are automatically migrated into SQLite and the
JSON file is renamed to ``stats.json.migrated`` so the migration runs
only once.  Corrupt JSON files are renamed to ``stats.json.corrupt``.

The ``get_stats()`` method still returns an ``AnalyticsData`` Pydantic
model (with a ``stories`` list) so all callers — GUI panels, export
functions, tests — require no changes.

All DB operations are protected by a ``threading.Lock`` in addition to
SQLite's internal locking, because the GUI thread may call read methods
concurrently with worker threads calling ``record_story()``.

Typical usage::

    collector = AnalyticsCollector("data/analytics")
    collector.record_story(metadata)
    stats = collector.get_stats()
    by_lang = collector.get_by_language()
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from models.metadata import StoryMetadata

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS stories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id         TEXT    NOT NULL DEFAULT '',
    topic            TEXT    NOT NULL DEFAULT '',
    language         TEXT    NOT NULL DEFAULT '',
    provider         TEXT    NOT NULL DEFAULT '',
    model            TEXT    NOT NULL DEFAULT '',
    score            REAL    NOT NULL DEFAULT 0.0,
    attempts         INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL    NOT NULL DEFAULT 0.0,
    cost             REAL    NOT NULL DEFAULT 0.0,
    word_count       INTEGER NOT NULL DEFAULT 0,
    strategy         TEXT    NOT NULL DEFAULT '',
    completed_at     TEXT    NOT NULL DEFAULT '',
    status           TEXT    NOT NULL DEFAULT 'completed',
    errors           TEXT    NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_language     ON stories (language);
CREATE INDEX IF NOT EXISTS idx_provider     ON stories (provider);
CREATE INDEX IF NOT EXISTS idx_completed_at ON stories (completed_at);
CREATE INDEX IF NOT EXISTS idx_status       ON stories (status);
"""

# Canonical column order for INSERT / SELECT / CSV — excludes auto-increment id.
_COLS: list[str] = [
    "story_id",
    "topic",
    "language",
    "provider",
    "model",
    "score",
    "attempts",
    "duration_seconds",
    "cost",
    "word_count",
    "strategy",
    "completed_at",
    "status",
    "errors",
]

_SELECT_COLS = ", ".join(_COLS)


# ── Analytics data model ──────────────────────────────────────────────────────


class AnalyticsData(BaseModel):
    """Aggregated analytics data returned by ``get_stats()``.

    Attributes:
        stories: Raw per-story records as plain dicts.
        summary: All-time summary counters.
        updated_at: Timestamp of the most-recently recorded story, or
            ``None`` if no stories exist.
    """

    stories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-story records.",
    )
    summary: dict[str, Any] = Field(
        default_factory=lambda: {
            "total_stories": 0,
            "total_words": 0,
            "total_cost": 0.0,
        },
        description="All-time summary counters.",
    )
    updated_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp of the most-recent story, or None.",
    )


# ── Collector ─────────────────────────────────────────────────────────────────


class AnalyticsCollector:
    """Thread-safe analytics persistence and aggregation backed by SQLite.

    Args:
        analytics_dir: Path to the analytics directory.
            Defaults to ``data/analytics``.
    """

    def __init__(self, analytics_dir: str = "data/analytics") -> None:
        self._dir = Path(analytics_dir)
        self._db_path = self._dir / "analytics.db"
        self._legacy_json = self._dir / "stats.json"
        self._lock = threading.Lock()

        self._dir.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_connection()
        self._apply_schema()
        self._migrate_legacy_json()

        logger.debug("AnalyticsCollector initialised: %s", self._db_path)

    # ── Connection & schema ───────────────────────────────────────────────────

    def _open_connection(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode for crash safety.

        Returns:
            An open ``sqlite3.Connection`` with ``row_factory`` set.
        """
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,  # protected externally by self._lock
        )
        conn.row_factory = sqlite3.Row
        # WAL: readers never block writers; writes are atomic and durable.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.commit()
        return conn

    def _apply_schema(self) -> None:
        """Create tables and indexes if they do not yet exist."""
        self._conn.executescript(_DDL)
        self._conn.commit()
        logger.debug("AnalyticsCollector: schema ready")

    # ── Legacy JSON migration ─────────────────────────────────────────────────

    def _migrate_legacy_json(self) -> None:
        """One-time migration of ``stats.json`` records into SQLite.

        After migration the JSON file is renamed to
        ``stats.json.migrated``.  Subsequent startups skip this entirely
        because the source file no longer exists at its original path.
        Corrupt JSON is renamed to ``stats.json.corrupt`` and skipped.
        """
        if not self._legacy_json.exists():
            return

        migrated_path = self._dir / "stats.json.migrated"
        corrupt_path = self._dir / "stats.json.corrupt"

        try:
            raw = json.loads(self._legacy_json.read_text(encoding="utf-8"))
            stories: list[dict[str, Any]] = raw.get("stories", [])
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "AnalyticsCollector: failed to read legacy stats.json "
                "(renaming to .corrupt): %s",
                exc,
            )
            self._legacy_json.replace(corrupt_path)
            return

        if not stories:
            self._legacy_json.replace(migrated_path)
            logger.debug("AnalyticsCollector: stats.json was empty — migrated cleanly")
            return

        migrated = 0
        skipped = 0
        with self._lock:
            for record in stories:
                try:
                    self._insert_record(record)
                    migrated += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "AnalyticsCollector: skipped record during migration: %s",
                        exc,
                    )
                    skipped += 1

        self._legacy_json.replace(migrated_path)
        logger.info(
            "AnalyticsCollector: migrated %d records from stats.json "
            "(%d skipped) → analytics.db",
            migrated,
            skipped,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _insert_record(self, record: dict[str, Any]) -> None:
        """Insert one story record into the ``stories`` table.

        Must be called with ``self._lock`` held.

        Args:
            record: Dict with keys matching ``_COLS``.
        """
        errors = record.get("errors", [])
        errors_str = (
            json.dumps(errors, ensure_ascii=False)
            if isinstance(errors, list)
            else str(errors)
        )

        self._conn.execute(
            f"INSERT INTO stories ({_SELECT_COLS}) "
            f"VALUES (:story_id, :topic, :language, :provider, :model, "
            f":score, :attempts, :duration_seconds, :cost, :word_count, "
            f":strategy, :completed_at, :status, :errors)",
            {
                "story_id":         str(record.get("story_id", "")),
                "topic":            str(record.get("topic", "")),
                "language":         str(record.get("language", "")),
                "provider":         str(record.get("provider", "")),
                "model":            str(record.get("model", "")),
                "score":            float(record.get("score", 0.0)),
                "attempts":         int(record.get("attempts", 0)),
                "duration_seconds": float(record.get("duration_seconds", 0.0)),
                "cost":             float(record.get("cost", 0.0)),
                "word_count":       int(record.get("word_count", 0)),
                "strategy":         str(record.get("strategy", "")),
                "completed_at":     str(record.get("completed_at", "")),
                "status":           str(record.get("status", "completed")),
                "errors":           errors_str,
            },
        )
        self._conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a ``sqlite3.Row`` to a plain dict.

        Deserialises the ``errors`` JSON string back to a Python list.

        Args:
            row: A row from a ``stories`` SELECT.

        Returns:
            Plain dictionary suitable for ``AnalyticsData.stories``.
        """
        d = dict(row)
        try:
            d["errors"] = json.loads(d.get("errors", "[]"))
        except (json.JSONDecodeError, TypeError):
            d["errors"] = []
        return d

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_story(self, metadata: StoryMetadata) -> None:
        """Append a completed story record to the analytics store.

        Thread-safe.  Performs a single SQL INSERT — O(1) regardless of
        how many stories are already stored.

        Args:
            metadata: Story metadata to record.
        """
        record: dict[str, Any] = {
            "story_id":         metadata.story_id,
            "topic":            metadata.topic,
            "language":         metadata.language,
            "provider":         metadata.provider,
            "model":            metadata.model,
            "score":            metadata.final_score,
            "attempts":         metadata.attempts,
            "duration_seconds": metadata.duration_seconds,
            "cost":             metadata.estimated_cost_usd,
            "word_count":       metadata.word_count,
            "strategy":         metadata.strategy_used,
            "completed_at":     (
                metadata.completed_at
                or datetime.now(timezone.utc).isoformat()
            ),
            "status":  "completed",
            "errors":  [],
        }

        with self._lock:
            self._insert_record(record)

        logger.info(
            "Analytics: recorded story '%s' (lang=%s, score=%.2f, cost=$%.4f)",
            metadata.topic,
            metadata.language,
            metadata.final_score,
            metadata.estimated_cost_usd,
        )

    def record_failure(
        self,
        topic: str,
        language: str,
        error: str,
        provider: str = "",
        model: str = "",
        duration_seconds: float = 0.0,
    ) -> None:
        """Record a failed story generation.

        Thread-safe.  Performs a single SQL INSERT.

        Args:
            topic: The topic that failed.
            language: Language code.
            error: Error description.
            provider: Provider used.
            model: Model used.
            duration_seconds: Time elapsed before failure.
        """
        record: dict[str, Any] = {
            "story_id":         "",
            "topic":            topic,
            "language":         language,
            "provider":         provider,
            "model":            model,
            "score":            0.0,
            "attempts":         0,
            "duration_seconds": duration_seconds,
            "cost":             0.0,
            "word_count":       0,
            "strategy":         "",
            "completed_at":     datetime.now(timezone.utc).isoformat(),
            "status":           "failed",
            "errors":           [error],
        }

        with self._lock:
            self._insert_record(record)

        logger.info("Analytics: recorded failure for '%s': %s", topic, error)

    # ── Read-only queries ─────────────────────────────────────────────────────

    def get_stats(self) -> AnalyticsData:
        """Return the full analytics data.

        Queries SQLite for all story records and assembles an
        ``AnalyticsData`` instance with a recomputed summary.

        Returns:
            ``AnalyticsData`` with ``stories``, ``summary``, and
            ``updated_at`` populated.
        """
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM stories ORDER BY id ASC"
            )
            rows = cursor.fetchall()

        if not rows:
            return AnalyticsData()

        stories = [self._row_to_dict(r) for r in rows]
        total_words = sum(s.get("word_count", 0) for s in stories)
        total_cost = sum(s.get("cost", 0.0) for s in stories)
        timestamps = [s["completed_at"] for s in stories if s.get("completed_at")]
        updated_at = max(timestamps) if timestamps else None

        return AnalyticsData(
            stories=stories,
            summary={
                "total_stories": len(stories),
                "total_words":   total_words,
                "total_cost":    round(total_cost, 6),
            },
            updated_at=updated_at,
        )

    def get_by_language(self) -> dict[str, dict[str, Any]]:
        """Aggregate analytics by language using a SQL GROUP BY.

        Returns:
            Mapping of language code → dict with keys ``count``,
            ``avg_score``, ``total_cost``, ``total_words``.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT
                    language,
                    COUNT(*)        AS count,
                    AVG(score)      AS avg_score,
                    SUM(cost)       AS total_cost,
                    SUM(word_count) AS total_words
                FROM stories
                GROUP BY language
                ORDER BY count DESC
                """
            )
            rows = cursor.fetchall()

        return {
            row["language"]: {
                "count":       row["count"],
                "avg_score":   round(row["avg_score"] or 0.0, 2),
                "total_cost":  round(row["total_cost"] or 0.0, 6),
                "total_words": row["total_words"] or 0,
            }
            for row in rows
        }

    def get_by_provider(self) -> dict[str, dict[str, Any]]:
        """Aggregate analytics by provider using a SQL GROUP BY.

        Returns:
            Mapping of provider → dict with keys ``count``,
            ``avg_score``, ``total_cost``.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT
                    provider,
                    COUNT(*)   AS count,
                    AVG(score) AS avg_score,
                    SUM(cost)  AS total_cost
                FROM stories
                GROUP BY provider
                ORDER BY count DESC
                """
            )
            rows = cursor.fetchall()

        return {
            row["provider"]: {
                "count":      row["count"],
                "avg_score":  round(row["avg_score"] or 0.0, 2),
                "total_cost": round(row["total_cost"] or 0.0, 6),
            }
            for row in rows
        }

    def get_score_distribution(self) -> dict[str, int]:
        """Compute score distribution across four buckets using SQL CASE.

        Returns:
            Mapping of bucket label → count.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT
                    SUM(CASE WHEN score >= 9.0                    THEN 1 ELSE 0 END) AS high,
                    SUM(CASE WHEN score >= 8.0 AND score < 9.0   THEN 1 ELSE 0 END) AS mid_high,
                    SUM(CASE WHEN score >= 7.0 AND score < 8.0   THEN 1 ELSE 0 END) AS mid_low,
                    SUM(CASE WHEN score  > 0.0 AND score < 7.0   THEN 1 ELSE 0 END) AS low
                FROM stories
                WHERE score > 0
                """
            )
            row = cursor.fetchone()

        if row is None:
            return {"9.0-10.0": 0, "8.0-9.0": 0, "7.0-8.0": 0, "< 7.0": 0}

        return {
            "9.0-10.0": row["high"]     or 0,
            "8.0-9.0":  row["mid_high"] or 0,
            "7.0-8.0":  row["mid_low"]  or 0,
            "< 7.0":    row["low"]      or 0,
        }

    def get_common_issues(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return the most common issues across all stories.

        JSON error arrays are stored as text and cannot be GROUP-BY'd
        in SQLite without extensions, so this method fetches only the
        non-empty error rows and counts in Python.

        Args:
            top_n: Maximum number of issues to return.

        Returns:
            List of ``(issue_description, count)`` tuples sorted by
            frequency descending.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT errors FROM stories "
                "WHERE errors != '[]' AND errors != '' AND errors IS NOT NULL"
            )
            rows = cursor.fetchall()

        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            try:
                errors = json.loads(row["errors"])
            except (json.JSONDecodeError, TypeError):
                errors = [str(row["errors"])]
            for issue in errors:
                counts[str(issue)[:120]] += 1

        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_trend(self, days: int = 30) -> list[dict[str, Any]]:
        """Return daily aggregates for the last N days using SQL.

        Uses ``substr(completed_at, 1, 10)`` to extract the date portion
        from ISO-8601 timestamps (``YYYY-MM-DD``).

        Args:
            days: Maximum number of days to return, newest first.

        Returns:
            List of dicts with ``date``, ``count``, ``avg_score``,
            ``cost``, ordered newest-first.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT
                    substr(completed_at, 1, 10) AS date,
                    COUNT(*)                    AS count,
                    AVG(score)                  AS avg_score,
                    SUM(cost)                   AS cost
                FROM stories
                WHERE completed_at != ''
                GROUP BY date
                ORDER BY date DESC
                LIMIT ?
                """,
                (days,),
            )
            rows = cursor.fetchall()

        return [
            {
                "date":      row["date"],
                "count":     row["count"],
                "avg_score": round(row["avg_score"] or 0.0, 2),
                "cost":      round(row["cost"] or 0.0, 4),
            }
            for row in rows
        ]

    # ── Export ────────────────────────────────────────────────────────────────

    def export_json(self, path: str | Path) -> None:
        """Export full analytics as JSON.

        Args:
            path: Destination file path.
        """
        data = self.get_stats()
        Path(path).write_text(
            json.dumps(data.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Analytics exported to JSON: %s", path)

    def export_csv(self, path: str | Path) -> None:
        """Export per-story records as CSV.

        Streams directly from SQLite using ``csv.DictWriter`` so all
        quoting, escaping, and line endings are handled correctly by
        Python's stdlib.  The ``errors`` JSON array is serialised as a
        semicolon-separated string so each story occupies one row.

        Args:
            path: Destination file path.
        """
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM stories ORDER BY id ASC"
            )
            rows = cursor.fetchall()

        if not rows:
            logger.warning("No stories to export as CSV")
            return

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=_COLS,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\r\n",  # RFC 4180
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            record = dict(row)
            try:
                errors = json.loads(record.get("errors", "[]"))
            except (json.JSONDecodeError, TypeError):
                errors = [str(record.get("errors", ""))]
            record["errors"] = "; ".join(str(e) for e in errors)
            writer.writerow(record)

        Path(path).write_text(output.getvalue(), encoding="utf-8")
        logger.info("Analytics exported to CSV (%d rows): %s", len(rows), path)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection.

        Call this on application shutdown to flush WAL and release the
        file handle.  Subsequent calls to any other method will raise
        ``sqlite3.ProgrammingError``.
        """
        with self._lock:
            self._conn.close()
        logger.debug("AnalyticsCollector: connection closed")
