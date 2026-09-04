"""Crash recovery management for AI Story Generator Pro.

``RecoveryManager`` persists batch processing state to
``data/recovery/recovery_state.json`` so that the application can
resume after an unexpected shutdown.  After every topic completes
(or fails) the manager updates the on-disk state.

On startup the caller checks ``has_unfinished()`` and, if ``True``,
presents the user with ``get_recovery_options()`` to continue, restart,
or retry only the errored topics.

Typical usage::

    rm = RecoveryManager(settings)
    if rm.has_unfinished():
        opts = rm.get_recovery_options()
        # present opts to user …
    rm.save_batch_state(batch)
    # … later …
    batch = rm.load_batch_state()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.exceptions import StateError
from core.settings import Settings
from utils.file_handler import ensure_dir, read_file, write_file

logger = logging.getLogger(__name__)

# Filename for the recovery state JSON.
_RECOVERY_FILENAME = "recovery_state.json"


# ── Recovery option enum ─────────────────────────────────────────────


class RecoveryAction(str, Enum):
    """Actions available when unfinished work is detected."""

    CONTINUE = "continue"
    RESTART = "restart"
    RETRY_ERRORS = "retry_errors"


# ── Data models (pydantic) ───────────────────────────────────────────


class TopicProgress(BaseModel):
    """In-progress snapshot of a single topic.

    Attributes:
        stage: Name of the pipeline step currently executing.
        section: Section index if in SectionStep, else 0.
        attempt: Current evaluation/revision attempt (1-based).
    """

    stage: str = Field(default="", description="Current pipeline step name.")
    section: int = Field(default=0, ge=0, description="Current section index.")
    attempt: int = Field(default=1, ge=1, description="Current attempt number.")


class FailedTopic(BaseModel):
    """Record of a topic that failed during processing.

    Attributes:
        topic: The topic string.
        error: Error description.
        stage: The pipeline step where failure occurred.
        attempt: The attempt number at time of failure.
    """

    topic: str = Field(description="The topic that failed.")
    error: str = Field(default="", description="Error description.")
    stage: str = Field(default="", description="Step where failure occurred.")
    attempt: int = Field(default=1, ge=1, description="Attempt at failure time.")


class ConfigSnapshot(BaseModel):
    """Minimal snapshot of generation config for recovery comparison.

    Attributes:
        language: Target language code.
        target_words: Target word count.
        tone: Tone preset.
        model: Model identifier.
        provider: API provider.
    """

    language: str = Field(default="en")
    target_words: int = Field(default=3000)
    tone: str = Field(default="dramatic_cinematic")
    model: str = Field(default="")
    provider: str = Field(default="")


class BatchState(BaseModel):
    """Full state of a batch processing run.

    Serialised to ``recovery_state.json`` and reloaded on startup if
    the previous run did not complete cleanly.

    Attributes:
        batch_id: Unique identifier for this batch run.
        total_topics: Total number of topics in the batch.
        completed: List of topics that completed successfully.
        in_progress: Mapping of topic → progress snapshot for topics
            that were being processed when the crash occurred.
        failed: List of topics that failed with error info.
        queued: List of topics not yet started.
        started_at: ISO-8601 timestamp of batch start.
        updated_at: ISO-8601 timestamp of last state update.
        config_snapshot: Snapshot of key configuration values.
    """

    batch_id: str = Field(default="", description="Unique batch identifier.")
    total_topics: int = Field(default=0, ge=0, description="Total topics in batch.")
    completed: list[str] = Field(
        default_factory=list, description="Successfully completed topics."
    )
    in_progress: dict[str, TopicProgress] = Field(
        default_factory=dict,
        description="Topics currently being processed.",
    )
    failed: list[FailedTopic] = Field(
        default_factory=list, description="Topics that failed."
    )
    queued: list[str] = Field(
        default_factory=list, description="Topics not yet started."
    )
    started_at: str = Field(default="", description="Batch start timestamp (ISO-8601).")
    updated_at: str = Field(default="", description="Last update timestamp (ISO-8601).")
    config_snapshot: ConfigSnapshot = Field(
        default_factory=ConfigSnapshot,
        description="Key config values at batch start.",
    )


@dataclass(frozen=True)
class RecoveryOptions:
    """Options presented to the user when unfinished work is found.

    Attributes:
        batch_id: The batch that was interrupted.
        total_topics: Total number of topics in the batch.
        completed_count: How many topics completed successfully.
        in_progress_count: How many topics were in-progress at crash.
        failed_count: How many topics failed.
        queued_count: How many topics were not yet started.
        available_actions: List of actions the user can choose from.
        started_at: When the batch was originally started.
    """

    batch_id: str
    total_topics: int
    completed_count: int
    in_progress_count: int
    failed_count: int
    queued_count: int
    available_actions: list[RecoveryAction] = field(
        default_factory=lambda: [
            RecoveryAction.CONTINUE,
            RecoveryAction.RESTART,
            RecoveryAction.RETRY_ERRORS,
        ]
    )
    started_at: str = ""


# ── RecoveryManager ──────────────────────────────────────────────────


class RecoveryManager:
    """Manages batch state persistence for crash recovery.

    Args:
        settings: Application settings (provides ``paths.recovery_dir``).
    """

    def __init__(self, settings: Settings) -> None:
        self._recovery_dir = Path(settings.paths.recovery_dir)
        self._recovery_file = self._recovery_dir / _RECOVERY_FILENAME
        ensure_dir(self._recovery_dir)
        logger.debug(
            "RecoveryManager initialised: recovery_file=%s",
            self._recovery_file,
        )

    # ── Query ─────────────────────────────────────────────────────────

    def has_unfinished(self) -> bool:
        """Check whether an unfinished batch exists.

        Returns:
            ``True`` if a recovery state file exists and contains
            topics that are queued, in-progress, or (if action is
            retry-errors) failed.
        """
        if not self._recovery_file.exists():
            return False

        try:
            batch = self._read_state()
        except StateError:
            return False

        has_work = bool(batch.queued or batch.in_progress or batch.failed)
        if has_work:
            logger.info(
                "RecoveryManager: unfinished batch '%s' found — "
                "completed=%d, in_progress=%d, failed=%d, queued=%d",
                batch.batch_id,
                len(batch.completed),
                len(batch.in_progress),
                len(batch.failed),
                len(batch.queued),
            )
        return has_work

    def get_recovery_options(self) -> RecoveryOptions:
        """Build recovery options from the persisted batch state.

        Returns:
            A ``RecoveryOptions`` instance describing the interrupted
            batch and available actions.

        Raises:
            StateError: If no recovery state exists or it cannot be
                loaded.
        """
        batch = self._read_state()

        actions: list[RecoveryAction] = [RecoveryAction.RESTART]

        # Continue is available if there are queued or in-progress topics.
        if batch.queued or batch.in_progress:
            actions.insert(0, RecoveryAction.CONTINUE)

        # Retry-errors is available only if there are failed topics.
        if batch.failed:
            actions.append(RecoveryAction.RETRY_ERRORS)

        options = RecoveryOptions(
            batch_id=batch.batch_id,
            total_topics=batch.total_topics,
            completed_count=len(batch.completed),
            in_progress_count=len(batch.in_progress),
            failed_count=len(batch.failed),
            queued_count=len(batch.queued),
            available_actions=actions,
            started_at=batch.started_at,
        )

        logger.info(
            "RecoveryManager: options built for batch '%s': actions=%s",
            batch.batch_id,
            [a.value for a in actions],
        )
        return options

    # ── Mutations ─────────────────────────────────────────────────────

    def save_batch_state(self, batch: BatchState) -> Path:
        """Persist the current batch state to disk.

        Args:
            batch: The batch state to persist.

        Returns:
            Path to the written recovery file.

        Raises:
            StateError: If serialisation or writing fails.
        """
        batch.updated_at = datetime.now(timezone.utc).isoformat()

        try:
            json_str = batch.model_dump_json(indent=2)
            path = write_file(self._recovery_file, json_str)
        except Exception as exc:
            raise StateError(
                f"Failed to save recovery state: {exc}"
            ) from exc

        logger.debug(
            "RecoveryManager: batch state saved — completed=%d, "
            "in_progress=%d, failed=%d, queued=%d",
            len(batch.completed),
            len(batch.in_progress),
            len(batch.failed),
            len(batch.queued),
        )
        return path

    def load_batch_state(self) -> BatchState:
        """Load the persisted batch state from disk.

        Returns:
            The deserialised ``BatchState``.

        Raises:
            StateError: If the file does not exist, cannot be read,
                or fails validation.
        """
        return self._read_state()

    def mark_topic_started(
        self, batch: BatchState, topic: str, stage: str = "concept"
    ) -> BatchState:
        """Move a topic from queued to in-progress.

        Args:
            batch: The current batch state.
            topic: The topic string to move.
            stage: The starting pipeline step name.

        Returns:
            The updated batch state.
        """
        if topic in batch.queued:
            batch.queued.remove(topic)

        batch.in_progress[topic] = TopicProgress(stage=stage, section=0, attempt=1)

        logger.debug(
            "RecoveryManager: topic '%s' started (stage=%s)",
            topic,
            stage,
        )
        return batch

    def update_topic_progress(
        self,
        batch: BatchState,
        topic: str,
        stage: str,
        section: int = 0,
        attempt: int = 1,
    ) -> BatchState:
        """Update the progress of an in-progress topic.

        Args:
            batch: The current batch state.
            topic: The topic string.
            stage: Current pipeline step name.
            section: Current section index (if applicable).
            attempt: Current attempt number.

        Returns:
            The updated batch state.
        """
        batch.in_progress[topic] = TopicProgress(
            stage=stage, section=section, attempt=attempt
        )
        return batch

    def mark_topic_completed(self, batch: BatchState, topic: str) -> BatchState:
        """Move a topic from in-progress to completed.

        Args:
            batch: The current batch state.
            topic: The topic string to mark as completed.

        Returns:
            The updated batch state.
        """
        batch.in_progress.pop(topic, None)

        if topic not in batch.completed:
            batch.completed.append(topic)

        logger.debug("RecoveryManager: topic '%s' completed", topic)
        return batch

    def mark_topic_failed(
        self,
        batch: BatchState,
        topic: str,
        error: str,
        stage: str = "",
        attempt: int = 1,
    ) -> BatchState:
        """Move a topic from in-progress to failed.

        Args:
            batch: The current batch state.
            topic: The topic string that failed.
            error: Error description.
            stage: Pipeline step where failure occurred.
            attempt: Attempt number at time of failure.

        Returns:
            The updated batch state.
        """
        progress = batch.in_progress.pop(topic, None)
        if progress is not None and not stage:
            stage = progress.stage

        batch.failed.append(
            FailedTopic(topic=topic, error=error, stage=stage, attempt=attempt)
        )

        logger.debug(
            "RecoveryManager: topic '%s' failed at stage '%s': %s",
            topic,
            stage,
            error,
        )
        return batch

    def clear(self) -> None:
        """Delete the recovery state file.

        Safe to call even if the file does not exist.
        """
        if self._recovery_file.exists():
            try:
                self._recovery_file.unlink()
                logger.info(
                    "RecoveryManager: recovery state cleared (%s)",
                    self._recovery_file,
                )
            except OSError as exc:
                logger.error(
                    "RecoveryManager: failed to delete recovery file: %s", exc
                )
        else:
            logger.debug("RecoveryManager: nothing to clear (no recovery file)")

    # ── Batch creation helpers ────────────────────────────────────────

    @staticmethod
    def create_batch_state(
        topics: list[str],
        language: str = "en",
        target_words: int = 3000,
        tone: str = "dramatic_cinematic",
        model: str = "",
        provider: str = "",
    ) -> BatchState:
        """Create a new ``BatchState`` for a list of topics.

        Args:
            topics: List of topic strings.
            language: Target language code.
            target_words: Target word count.
            tone: Tone preset name.
            model: Model identifier.
            provider: Provider name.

        Returns:
            A fresh ``BatchState`` with all topics in ``queued``.
        """
        now = datetime.now(timezone.utc).isoformat()
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        batch = BatchState(
            batch_id=batch_id,
            total_topics=len(topics),
            completed=[],
            in_progress={},
            failed=[],
            queued=list(topics),
            started_at=now,
            updated_at=now,
            config_snapshot=ConfigSnapshot(
                language=language,
                target_words=target_words,
                tone=tone,
                model=model,
                provider=provider,
            ),
        )

        logger.info(
            "RecoveryManager: created BatchState '%s' with %d topics",
            batch_id,
            len(topics),
        )
        return batch

    # ── Private helpers ───────────────────────────────────────────────

    def _read_state(self) -> BatchState:
        """Read and parse the recovery state file.

        Returns:
            The deserialised ``BatchState``.

        Raises:
            StateError: If the file does not exist, cannot be read, or
                fails validation.
        """
        if not self._recovery_file.exists():
            raise StateError(
                f"Recovery state file not found: {self._recovery_file}"
            )

        try:
            raw = read_file(self._recovery_file)
        except OSError as exc:
            raise StateError(
                f"Failed to read recovery state file: {exc}"
            ) from exc

        try:
            batch = BatchState.model_validate_json(raw)
        except Exception as exc:
            raise StateError(
                f"Failed to parse recovery state file: {exc}"
            ) from exc

        logger.debug(
            "RecoveryManager: loaded batch '%s' from %s",
            batch.batch_id,
            self._recovery_file,
        )
        return batch
