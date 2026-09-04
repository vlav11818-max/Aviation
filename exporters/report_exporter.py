"""Batch report exporter for AI Story Generator Pro.

``ReportExporter`` generates JSON and CSV reports after a batch run.
Each report includes per-story records (topic, language, score,
attempts, time, cost, model, status, errors), a batch summary, and
a configuration snapshot.

Typical usage::

    exporter = ReportExporter()
    exporter.export_json(batch_result, Path("output/report.json"))
    exporter.export_csv(batch_result, Path("output/report.csv"))
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.exceptions import ExportError
from utils.file_handler import write_file

logger = logging.getLogger(__name__)


# ── Data models ────────────────────────────────────────────────────────


class StoryRecord(BaseModel):
    """Per-story record in a batch report.

    Attributes:
        topic: The story topic.
        language: Two-letter language code.
        score: Final evaluation score.
        passed: Whether the quality threshold was met.
        attempts: Number of evaluation/revision attempts.
        duration_seconds: Wall-clock time for this story.
        cost_usd: Estimated cost for this story.
        model: Model identifier used.
        provider: API provider used.
        status: Final status (completed, failed, etc.).
        word_count: Final word count.
        errors: List of error messages (empty if no errors).
    """

    topic: str = Field(default="", description="Story topic.")
    language: str = Field(default="en", description="Language code.")
    score: float = Field(default=0.0, description="Final evaluation score.")
    passed: bool = Field(default=False, description="Quality threshold met.")
    attempts: int = Field(default=1, ge=0, description="Eval/revision attempts.")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Wall-clock time.")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated cost.")
    model: str = Field(default="", description="Model used.")
    provider: str = Field(default="", description="API provider used.")
    status: str = Field(default="", description="Final status.")
    word_count: int = Field(default=0, ge=0, description="Final word count.")
    errors: list[str] = Field(default_factory=list, description="Error messages.")


class BatchSummary(BaseModel):
    """Aggregate summary for an entire batch run.

    Attributes:
        total: Total stories in the batch.
        completed: Number of successfully completed stories.
        failed: Number of failed stories.
        avg_score: Average evaluation score across completed stories.
        total_cost: Total estimated cost in USD.
        total_time: Total wall-clock time in seconds.
        total_words: Total words across all completed stories.
        stories_per_hour: Throughput (stories per hour).
    """

    total: int = Field(default=0, ge=0, description="Total stories.")
    completed: int = Field(default=0, ge=0, description="Completed stories.")
    failed: int = Field(default=0, ge=0, description="Failed stories.")
    avg_score: float = Field(default=0.0, ge=0.0, description="Average score.")
    total_cost: float = Field(default=0.0, ge=0.0, description="Total cost USD.")
    total_time: float = Field(default=0.0, ge=0.0, description="Total time seconds.")
    total_words: int = Field(default=0, ge=0, description="Total words.")
    stories_per_hour: float = Field(default=0.0, ge=0.0, description="Throughput.")


class ConfigSnapshot(BaseModel):
    """Configuration snapshot included in the report.

    Attributes:
        language: Target language.
        target_words: Target word count.
        tone: Story tone.
        min_score: Quality threshold.
        max_attempts: Maximum revision attempts.
        model: Primary model.
        provider: Primary provider.
        strategy: Strategy used.
    """

    language: str = Field(default="", description="Target language.")
    target_words: int = Field(default=0, description="Target word count.")
    tone: str = Field(default="", description="Story tone.")
    min_score: float = Field(default=0.0, description="Quality threshold.")
    max_attempts: int = Field(default=0, description="Max attempts.")
    model: str = Field(default="", description="Primary model.")
    provider: str = Field(default="", description="Primary provider.")
    strategy: str = Field(default="", description="Strategy used.")


class BatchResult(BaseModel):
    """Complete batch result data for report generation.

    Attributes:
        stories: List of per-story records.
        summary: Aggregate batch summary.
        config: Configuration snapshot.
        started_at: Batch start timestamp (ISO-8601).
        completed_at: Batch completion timestamp (ISO-8601).
    """

    stories: list[StoryRecord] = Field(
        default_factory=list,
        description="Per-story records.",
    )
    summary: BatchSummary = Field(
        default_factory=BatchSummary,
        description="Batch summary.",
    )
    config: ConfigSnapshot = Field(
        default_factory=ConfigSnapshot,
        description="Config snapshot.",
    )
    started_at: str = Field(
        default="",
        description="Batch start timestamp (ISO-8601).",
    )
    completed_at: str = Field(
        default="",
        description="Batch completion timestamp (ISO-8601).",
    )

    def compute_summary(self) -> None:
        """Recompute the summary from the current story records.

        Updates ``self.summary`` in place based on ``self.stories``.
        """
        total = len(self.stories)
        completed_stories = [s for s in self.stories if s.status == "completed"]
        failed_stories = [s for s in self.stories if s.status == "failed"]

        completed = len(completed_stories)
        failed = len(failed_stories)

        scores = [s.score for s in completed_stories if s.score > 0.0]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        total_cost = sum(s.cost_usd for s in self.stories)
        total_time = sum(s.duration_seconds for s in self.stories)
        total_words = sum(s.word_count for s in completed_stories)

        stories_per_hour = (completed / (total_time / 3600.0)) if total_time > 0 else 0.0

        self.summary = BatchSummary(
            total=total,
            completed=completed,
            failed=failed,
            avg_score=round(avg_score, 2),
            total_cost=round(total_cost, 6),
            total_time=round(total_time, 2),
            total_words=total_words,
            stories_per_hour=round(stories_per_hour, 2),
        )


# ── ReportExporter ─────────────────────────────────────────────────────


class ReportExporter:
    """Generates JSON and CSV batch reports."""

    def export_json(
        self,
        batch_result: BatchResult,
        output_path: str | Path,
    ) -> Path:
        """Export batch results as a JSON report.

        Args:
            batch_result: The complete batch result data.
            output_path: Target ``.json`` file path.

        Returns:
            The ``Path`` of the written file.

        Raises:
            ExportError: If serialisation or writing fails.
        """
        logger.info(
            "ReportExporter: exporting JSON report to %s (%d stories)",
            output_path,
            len(batch_result.stories),
        )

        try:
            report_data = self._build_report_dict(batch_result)
            json_str = json.dumps(report_data, indent=2, ensure_ascii=False)
        except Exception as exc:
            raise ExportError(
                f"Failed to serialise JSON report: {exc}",
                export_format="json",
            ) from exc

        try:
            result_path = write_file(output_path, json_str)
        except OSError as exc:
            raise ExportError(
                f"Failed to write JSON report {output_path}: {exc}",
                export_format="json",
            ) from exc

        logger.info(
            "ReportExporter: JSON report written: %s (%d chars)",
            result_path,
            len(json_str),
        )
        return result_path

    def export_csv(
        self,
        batch_result: BatchResult,
        output_path: str | Path,
    ) -> Path:
        """Export per-story records as a CSV report.

        The CSV contains one row per story with all fields from
        ``StoryRecord``.  The summary and config are not included in
        the CSV (use ``export_json`` for those).

        Args:
            batch_result: The complete batch result data.
            output_path: Target ``.csv`` file path.

        Returns:
            The ``Path`` of the written file.

        Raises:
            ExportError: If building or writing fails.
        """
        logger.info(
            "ReportExporter: exporting CSV report to %s (%d stories)",
            output_path,
            len(batch_result.stories),
        )

        try:
            csv_str = self._build_csv(batch_result)
        except Exception as exc:
            raise ExportError(
                f"Failed to build CSV report: {exc}",
                export_format="csv",
            ) from exc

        try:
            result_path = write_file(output_path, csv_str)
        except OSError as exc:
            raise ExportError(
                f"Failed to write CSV report {output_path}: {exc}",
                export_format="csv",
            ) from exc

        logger.info(
            "ReportExporter: CSV report written: %s",
            result_path,
        )
        return result_path

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_report_dict(batch_result: BatchResult) -> dict[str, Any]:
        """Build a complete report dictionary from the batch result.

        Args:
            batch_result: Batch result data.

        Returns:
            Dict suitable for JSON serialisation.
        """
        now = datetime.now(timezone.utc).isoformat()

        return {
            "report_generated_at": now,
            "batch": {
                "started_at": batch_result.started_at,
                "completed_at": batch_result.completed_at,
            },
            "summary": batch_result.summary.model_dump(),
            "config": batch_result.config.model_dump(),
            "stories": [
                story.model_dump() for story in batch_result.stories
            ],
        }

    @staticmethod
    def _build_csv(batch_result: BatchResult) -> str:
        """Build CSV string from per-story records.

        Args:
            batch_result: Batch result data.

        Returns:
            CSV string with header row and one data row per story.
        """
        fieldnames = [
            "topic",
            "language",
            "score",
            "passed",
            "attempts",
            "duration_seconds",
            "cost_usd",
            "model",
            "provider",
            "status",
            "word_count",
            "errors",
        ]

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )

        writer.writeheader()

        for story in batch_result.stories:
            row = story.model_dump()
            # Join error list into a semicolon-separated string for CSV.
            row["errors"] = "; ".join(row["errors"]) if row["errors"] else ""
            writer.writerow(row)

        return output.getvalue()
