"""Unit tests for ``core.analytics_collector``.

Tests cover: SQLite schema creation, record_story, record_failure,
get_stats, get_by_language, get_by_provider, get_score_distribution,
get_common_issues, get_trend, export_json, export_csv, legacy
stats.json migration, thread safety, and the close() lifecycle.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pytest

from core.analytics_collector import AnalyticsCollector, AnalyticsData
from models.metadata import StoryMetadata


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_metadata(
    topic: str = "Test Topic",
    language: str = "en",
    provider: str = "openrouter",
    model: str = "gpt-4o",
    score: float = 9.5,
    attempts: int = 1,
    duration: float = 10.0,
    cost: float = 0.05,
    words: int = 1000,
    strategy: str = "full_pipeline",
) -> StoryMetadata:
    """Build a minimal ``StoryMetadata`` for testing."""
    return StoryMetadata(
        story_id=f"{topic.lower().replace(' ', '_')}_{language}",
        topic=topic,
        language=language,
        provider=provider,
        model=model,
        final_score=score,
        attempts=attempts,
        duration_seconds=duration,
        estimated_cost_usd=cost,
        word_count=words,
        strategy_used=strategy,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def analytics_dir(tmp_path: Path) -> Path:
    """Return a temp directory for analytics files."""
    d = tmp_path / "analytics"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def collector(analytics_dir: Path) -> Iterator[AnalyticsCollector]:
    """Return a fresh AnalyticsCollector; closes after each test."""
    ac = AnalyticsCollector(analytics_dir=str(analytics_dir))
    yield ac
    ac.close()


# ── Tests: schema ─────────────────────────────────────────────────────────────


class TestSchema:
    """Tests that the SQLite schema is created correctly."""

    def test_db_file_created(self, analytics_dir: Path) -> None:
        """analytics.db should be created on construction."""
        ac = AnalyticsCollector(analytics_dir=str(analytics_dir))
        ac.close()
        assert (analytics_dir / "analytics.db").exists()

    def test_stories_table_exists(self, analytics_dir: Path) -> None:
        """The stories table must exist with all required columns."""
        ac = AnalyticsCollector(analytics_dir=str(analytics_dir))
        ac.close()

        conn = sqlite3.connect(str(analytics_dir / "analytics.db"))
        cursor = conn.execute("PRAGMA table_info(stories)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()

        required = {
            "id", "story_id", "topic", "language", "provider", "model",
            "score", "attempts", "duration_seconds", "cost", "word_count",
            "strategy", "completed_at", "status", "errors",
        }
        assert required.issubset(cols)

    def test_idempotent_schema(self, analytics_dir: Path) -> None:
        """Creating two collectors in the same dir must not raise."""
        ac1 = AnalyticsCollector(analytics_dir=str(analytics_dir))
        ac1.close()
        ac2 = AnalyticsCollector(analytics_dir=str(analytics_dir))
        ac2.close()


# ── Tests: record_story ───────────────────────────────────────────────────────


class TestRecordStory:
    """Tests for record_story()."""

    def test_record_increases_count(self, collector: AnalyticsCollector) -> None:
        """Recording a story should increase the story count to 1."""
        collector.record_story(_make_metadata())
        stats = collector.get_stats()
        assert stats.summary["total_stories"] == 1

    def test_record_fields_round_trip(self, collector: AnalyticsCollector) -> None:
        """All metadata fields must survive the insert→select round-trip."""
        meta = _make_metadata(
            topic="Ancient Temple",
            language="de",
            provider="anthropic",
            model="claude-3",
            score=9.2,
            attempts=2,
            duration=42.5,
            cost=0.12,
            words=3500,
            strategy="two_pass",
        )
        collector.record_story(meta)
        stats = collector.get_stats()
        assert len(stats.stories) == 1
        s = stats.stories[0]
        assert s["topic"] == "Ancient Temple"
        assert s["language"] == "de"
        assert s["provider"] == "anthropic"
        assert s["model"] == "claude-3"
        assert s["score"] == pytest.approx(9.2)
        assert s["attempts"] == 2
        assert s["duration_seconds"] == pytest.approx(42.5)
        assert s["cost"] == pytest.approx(0.12)
        assert s["word_count"] == 3500
        assert s["strategy"] == "two_pass"
        assert s["status"] == "completed"
        assert s["errors"] == []

    def test_multiple_records(self, collector: AnalyticsCollector) -> None:
        """Recording 5 stories should give count=5."""
        for i in range(5):
            collector.record_story(_make_metadata(topic=f"Topic {i}"))
        assert collector.get_stats().summary["total_stories"] == 5

    def test_summary_totals(self, collector: AnalyticsCollector) -> None:
        """Summary total_words and total_cost must aggregate correctly."""
        collector.record_story(_make_metadata(words=1000, cost=0.10))
        collector.record_story(_make_metadata(words=2000, cost=0.20))
        stats = collector.get_stats()
        assert stats.summary["total_words"] == 3000
        assert stats.summary["total_cost"] == pytest.approx(0.30)


# ── Tests: record_failure ─────────────────────────────────────────────────────


class TestRecordFailure:
    """Tests for record_failure()."""

    def test_failure_recorded_as_failed_status(
        self, collector: AnalyticsCollector
    ) -> None:
        """Failed stories must have status='failed'."""
        collector.record_failure("Bad Topic", "en", "API timeout")
        stats = collector.get_stats()
        assert len(stats.stories) == 1
        assert stats.stories[0]["status"] == "failed"
        assert stats.stories[0]["score"] == 0.0

    def test_failure_error_message_stored(
        self, collector: AnalyticsCollector
    ) -> None:
        """The error message must appear in the errors list."""
        collector.record_failure("Topic", "fr", "Rate limit exceeded")
        stats = collector.get_stats()
        assert "Rate limit exceeded" in stats.stories[0]["errors"]


# ── Tests: get_stats ──────────────────────────────────────────────────────────


class TestGetStats:
    """Tests for get_stats()."""

    def test_empty_returns_default(self, collector: AnalyticsCollector) -> None:
        """Empty DB must return an AnalyticsData with zero counts."""
        stats = collector.get_stats()
        assert isinstance(stats, AnalyticsData)
        assert stats.stories == []
        assert stats.summary["total_stories"] == 0
        assert stats.updated_at is None

    def test_updated_at_set(self, collector: AnalyticsCollector) -> None:
        """updated_at must be non-None after at least one story."""
        collector.record_story(_make_metadata())
        stats = collector.get_stats()
        assert stats.updated_at is not None


# ── Tests: get_by_language ────────────────────────────────────────────────────


class TestGetByLanguage:
    """Tests for get_by_language() SQL GROUP BY."""

    def test_aggregates_correctly(self, collector: AnalyticsCollector) -> None:
        """Stories in 'en' and 'de' should produce two separate buckets."""
        collector.record_story(_make_metadata(language="en", score=9.0, cost=0.10, words=1000))
        collector.record_story(_make_metadata(language="en", score=8.0, cost=0.20, words=2000))
        collector.record_story(_make_metadata(language="de", score=7.0, cost=0.05, words=500))

        by_lang = collector.get_by_language()
        assert "en" in by_lang
        assert "de" in by_lang
        assert by_lang["en"]["count"] == 2
        assert by_lang["en"]["avg_score"] == pytest.approx(8.5)
        assert by_lang["en"]["total_cost"] == pytest.approx(0.30)
        assert by_lang["en"]["total_words"] == 3000
        assert by_lang["de"]["count"] == 1

    def test_empty_returns_empty_dict(self, collector: AnalyticsCollector) -> None:
        """Empty DB must return empty dict."""
        assert collector.get_by_language() == {}


# ── Tests: get_by_provider ────────────────────────────────────────────────────


class TestGetByProvider:
    """Tests for get_by_provider() SQL GROUP BY."""

    def test_aggregates_correctly(self, collector: AnalyticsCollector) -> None:
        """Stories by two providers should aggregate independently."""
        collector.record_story(_make_metadata(provider="openai", score=9.0, cost=0.10))
        collector.record_story(_make_metadata(provider="openai", score=8.0, cost=0.20))
        collector.record_story(_make_metadata(provider="anthropic", score=7.5, cost=0.15))

        by_prov = collector.get_by_provider()
        assert by_prov["openai"]["count"] == 2
        assert by_prov["openai"]["avg_score"] == pytest.approx(8.5)
        assert by_prov["anthropic"]["count"] == 1

    def test_empty_returns_empty_dict(self, collector: AnalyticsCollector) -> None:
        """Empty DB must return empty dict."""
        assert collector.get_by_provider() == {}


# ── Tests: get_score_distribution ────────────────────────────────────────────


class TestGetScoreDistribution:
    """Tests for get_score_distribution() SQL CASE buckets."""

    def test_buckets_correct(self, collector: AnalyticsCollector) -> None:
        """Scores must be placed in the correct bucket."""
        collector.record_story(_make_metadata(score=9.5))   # 9.0-10.0
        collector.record_story(_make_metadata(score=8.5))   # 8.0-9.0
        collector.record_story(_make_metadata(score=7.5))   # 7.0-8.0
        collector.record_story(_make_metadata(score=6.0))   # < 7.0
        collector.record_failure("x", "en", "err")          # score=0, excluded

        dist = collector.get_score_distribution()
        assert dist["9.0-10.0"] == 1
        assert dist["8.0-9.0"] == 1
        assert dist["7.0-8.0"] == 1
        assert dist["< 7.0"] == 1

    def test_empty_returns_zero_buckets(self, collector: AnalyticsCollector) -> None:
        """Empty DB must return zero counts in all four buckets."""
        dist = collector.get_score_distribution()
        assert all(v == 0 for v in dist.values())
        assert set(dist.keys()) == {"9.0-10.0", "8.0-9.0", "7.0-8.0", "< 7.0"}


# ── Tests: get_common_issues ──────────────────────────────────────────────────


class TestGetCommonIssues:
    """Tests for get_common_issues()."""

    def test_returns_top_issues(self, collector: AnalyticsCollector) -> None:
        """Most frequent error strings must rank first."""
        for _ in range(3):
            collector.record_failure("T", "en", "Rate limit")
        for _ in range(1):
            collector.record_failure("T", "en", "Timeout")

        issues = collector.get_common_issues(top_n=10)
        assert issues[0] == ("Rate limit", 3)
        assert issues[1] == ("Timeout", 1)

    def test_empty_returns_empty_list(self, collector: AnalyticsCollector) -> None:
        """Empty DB must return empty list."""
        assert collector.get_common_issues() == []

    def test_respects_top_n(self, collector: AnalyticsCollector) -> None:
        """top_n=1 must return at most one result."""
        collector.record_failure("T", "en", "A")
        collector.record_failure("T", "en", "B")
        assert len(collector.get_common_issues(top_n=1)) == 1


# ── Tests: get_trend ─────────────────────────────────────────────────────────


class TestGetTrend:
    """Tests for get_trend()."""

    def test_returns_daily_aggregates(self, collector: AnalyticsCollector) -> None:
        """Two stories on the same day should aggregate into one row."""
        ts = "2025-06-15T10:00:00+00:00"
        meta1 = _make_metadata(score=9.0, cost=0.10)
        meta2 = _make_metadata(score=8.0, cost=0.20)
        # Override completed_at directly after construction
        object.__setattr__(meta1, "completed_at", ts)
        object.__setattr__(meta2, "completed_at", ts)
        collector.record_story(meta1)
        collector.record_story(meta2)

        trend = collector.get_trend(days=30)
        assert len(trend) == 1
        assert trend[0]["date"] == "2025-06-15"
        assert trend[0]["count"] == 2
        assert trend[0]["avg_score"] == pytest.approx(8.5)

    def test_empty_returns_empty_list(self, collector: AnalyticsCollector) -> None:
        """Empty DB must return empty list."""
        assert collector.get_trend() == []


# ── Tests: export_json ────────────────────────────────────────────────────────


class TestExportJson:
    """Tests for export_json()."""

    def test_creates_valid_json(
        self, collector: AnalyticsCollector, tmp_path: Path
    ) -> None:
        """Exported file must be valid JSON with stories and summary."""
        collector.record_story(_make_metadata(topic="Forest"))
        out = tmp_path / "report.json"
        collector.export_json(out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert "stories" in data
        assert "summary" in data
        assert data["summary"]["total_stories"] == 1
        assert data["stories"][0]["topic"] == "Forest"

    def test_empty_export(
        self, collector: AnalyticsCollector, tmp_path: Path
    ) -> None:
        """Exporting an empty collector must produce valid JSON with 0 stories."""
        out = tmp_path / "empty.json"
        collector.export_json(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["total_stories"] == 0


# ── Tests: export_csv ─────────────────────────────────────────────────────────


class TestExportCsv:
    """Tests for export_csv() — verifies RFC 4180 CSV output."""

    def test_creates_valid_csv(
        self, collector: AnalyticsCollector, tmp_path: Path
    ) -> None:
        """Exported CSV must be parseable with the csv module."""
        collector.record_story(_make_metadata(topic="Ocean"))
        out = tmp_path / "report.csv"
        collector.export_csv(out)

        text = out.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["topic"] == "Ocean"

    def test_fields_with_commas_quoted(
        self, collector: AnalyticsCollector, tmp_path: Path
    ) -> None:
        """Topics containing commas must be quoted correctly."""
        collector.record_story(_make_metadata(topic='Topic, with "commas" and quotes'))
        out = tmp_path / "quoted.csv"
        collector.export_csv(out)

        reader = csv.DictReader(io.StringIO(out.read_text(encoding="utf-8")))
        rows = list(reader)
        assert rows[0]["topic"] == 'Topic, with "commas" and quotes'

    def test_errors_serialised_as_string(
        self, collector: AnalyticsCollector, tmp_path: Path
    ) -> None:
        """Error lists must appear as semicolon-separated strings in CSV."""
        collector.record_failure("Topic", "en", "Rate limit")
        out = tmp_path / "errors.csv"
        collector.export_csv(out)

        reader = csv.DictReader(io.StringIO(out.read_text(encoding="utf-8")))
        rows = list(reader)
        assert "Rate limit" in rows[0]["errors"]

    def test_empty_export_skips_file(
        self, collector: AnalyticsCollector, tmp_path: Path
    ) -> None:
        """Exporting an empty collector must NOT create a file (warning logged)."""
        out = tmp_path / "empty.csv"
        collector.export_csv(out)
        # File should not be written because there are no rows
        assert not out.exists()


# ── Tests: legacy JSON migration ──────────────────────────────────────────────


class TestLegacyMigration:
    """Tests for automatic stats.json → SQLite migration."""

    def test_migrates_records(self, analytics_dir: Path) -> None:
        """Records in stats.json must appear in SQLite after construction."""
        stats_json = analytics_dir / "stats.json"
        stats_json.write_text(
            json.dumps({
                "stories": [
                    {
                        "story_id": "migrated_01",
                        "topic": "Migrated Story",
                        "language": "en",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "score": 9.1,
                        "attempts": 1,
                        "duration_seconds": 20.0,
                        "cost": 0.08,
                        "word_count": 1200,
                        "strategy": "single_shot",
                        "completed_at": "2025-01-01T10:00:00+00:00",
                        "status": "completed",
                        "errors": [],
                    }
                ],
                "summary": {"total_stories": 1},
                "updated_at": "2025-01-01T10:00:00+00:00",
            }),
            encoding="utf-8",
        )

        ac = AnalyticsCollector(analytics_dir=str(analytics_dir))
        try:
            stats = ac.get_stats()
            assert stats.summary["total_stories"] == 1
            assert stats.stories[0]["topic"] == "Migrated Story"
        finally:
            ac.close()

    def test_migration_renames_json(self, analytics_dir: Path) -> None:
        """After migration stats.json must be renamed to stats.json.migrated."""
        stats_json = analytics_dir / "stats.json"
        stats_json.write_text(
            json.dumps({"stories": [], "summary": {}, "updated_at": None}),
            encoding="utf-8",
        )

        ac = AnalyticsCollector(analytics_dir=str(analytics_dir))
        ac.close()

        assert not stats_json.exists()
        assert (analytics_dir / "stats.json.migrated").exists()

    def test_corrupt_json_renamed_to_corrupt(self, analytics_dir: Path) -> None:
        """Corrupt stats.json must be renamed to stats.json.corrupt."""
        stats_json = analytics_dir / "stats.json"
        stats_json.write_text("{ this is not valid json !!!", encoding="utf-8")

        ac = AnalyticsCollector(analytics_dir=str(analytics_dir))
        ac.close()

        assert not stats_json.exists()
        assert (analytics_dir / "stats.json.corrupt").exists()

    def test_no_stats_json_no_migration(self, analytics_dir: Path) -> None:
        """No stats.json must mean zero stories and no .migrated file."""
        ac = AnalyticsCollector(analytics_dir=str(analytics_dir))
        try:
            assert ac.get_stats().summary["total_stories"] == 0
            assert not (analytics_dir / "stats.json.migrated").exists()
        finally:
            ac.close()

    def test_migration_runs_only_once(self, analytics_dir: Path) -> None:
        """Second construction must not re-migrate (source file is gone)."""
        stats_json = analytics_dir / "stats.json"
        stats_json.write_text(
            json.dumps({
                "stories": [
                    {
                        "story_id": "once", "topic": "Once", "language": "en",
                        "provider": "x", "model": "x", "score": 9.0,
                        "attempts": 1, "duration_seconds": 5.0, "cost": 0.01,
                        "word_count": 100, "strategy": "single_shot",
                        "completed_at": "2025-01-01T00:00:00+00:00",
                        "status": "completed", "errors": [],
                    }
                ],
                "summary": {}, "updated_at": None,
            }),
            encoding="utf-8",
        )

        ac1 = AnalyticsCollector(analytics_dir=str(analytics_dir))
        ac1.close()

        # Second construction — source is gone; count must still be 1.
        ac2 = AnalyticsCollector(analytics_dir=str(analytics_dir))
        try:
            assert ac2.get_stats().summary["total_stories"] == 1
        finally:
            ac2.close()


# ── Tests: thread safety ──────────────────────────────────────────────────────


class TestThreadSafety:
    """Tests for concurrent record_story() calls."""

    def test_concurrent_inserts(self, collector: AnalyticsCollector) -> None:
        """20 concurrent threads must each insert without corruption."""
        errors: list[Exception] = []

        def worker(n: int) -> None:
            try:
                collector.record_story(_make_metadata(topic=f"Concurrent {n}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert collector.get_stats().summary["total_stories"] == 20


# ── Tests: lifecycle ──────────────────────────────────────────────────────────


class TestLifecycle:
    """Tests for close()."""

    def test_close_does_not_raise(self, analytics_dir: Path) -> None:
        """close() on a live collector must not raise."""
        ac = AnalyticsCollector(analytics_dir=str(analytics_dir))
        ac.close()  # must not raise

    def test_operations_after_close_raise(self, analytics_dir: Path) -> None:
        """Any query after close() must raise ProgrammingError."""
        ac = AnalyticsCollector(analytics_dir=str(analytics_dir))
        ac.close()
        with pytest.raises(Exception):
            # sqlite3.ProgrammingError: Cannot operate on a closed database.
            ac.get_stats()
