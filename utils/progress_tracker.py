"""Thread-safe progress tracking for AI Story Generator Pro.

``ProgressTracker`` maintains a real-time view of batch processing
progress including per-worker status, overall completion percentage,
elapsed time, cost, average score, error count, and throughput.

The GUI polls ``get_snapshot()`` to obtain an immutable snapshot of
the current state.  Workers call the ``update_*`` methods from their
own threads; all mutations are protected by a ``threading.Lock``.

Typical usage::

    pt = ProgressTracker()
    pt.update_overall(completed=5, total=50)
    pt.update_worker(worker_id=1, topic="Ancient Temple", step_name="concept")
    snapshot = pt.get_snapshot()
    # snapshot.overall_percent  -> 10.0
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerSnapshot:
    """Immutable snapshot of a single worker's current state.

    Attributes:
        worker_id: Numeric identifier for the worker.
        topic: Topic currently being processed (empty if idle).
        step: Pipeline step name (empty if idle).
        detail: Extra detail string (e.g. section index).
    """

    worker_id: int = 0
    topic: str = ""
    step: str = ""
    detail: str = ""


@dataclass(frozen=True)
class ProgressSnapshot:
    """Immutable snapshot of overall batch progress.

    Attributes:
        overall_completed: Number of topics completed so far.
        overall_total: Total number of topics in the batch.
        overall_percent: Completion percentage (0.0–100.0).
        workers: Per-worker status snapshots.
        elapsed_time: Seconds elapsed since tracking started.
        total_cost: Cumulative cost in USD.
        avg_score: Average evaluation score across completed topics.
        error_count: Number of topics that failed.
        speed_stories_per_hour: Throughput (stories per hour based on
            completed count and elapsed time).
    """

    overall_completed: int = 0
    overall_total: int = 0
    overall_percent: float = 0.0
    workers: tuple[WorkerSnapshot, ...] = ()
    elapsed_time: float = 0.0
    total_cost: float = 0.0
    avg_score: float = 0.0
    error_count: int = 0
    speed_stories_per_hour: float = 0.0


class ProgressTracker:
    """Thread-safe progress reporter.

    All mutation methods acquire the internal lock before modifying
    state.  ``get_snapshot()`` returns an immutable copy for the GUI.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Overall counters.
        self._completed: int = 0
        self._total: int = 0

        # Workers: worker_id → mutable state dict.
        self._workers: dict[int, dict[str, str]] = {}

        # Cost tracking.
        self._total_cost: float = 0.0

        # Score tracking.
        self._score_sum: float = 0.0
        self._score_count: int = 0

        # Error count.
        self._error_count: int = 0

        # Timing.
        self._start_time: float = time.monotonic()

        logger.debug("ProgressTracker initialised")

    # ── Mutations (thread-safe) ───────────────────────────────────────

    def update_overall(self, completed: int, total: int) -> None:
        """Update the overall completion counters.

        Args:
            completed: Number of topics completed.
            total: Total number of topics in the batch.
        """
        with self._lock:
            self._completed = completed
            self._total = total
        logger.debug(
            "ProgressTracker: overall updated — %d/%d",
            completed,
            total,
        )

    def update_worker(
        self,
        worker_id: int,
        topic: str,
        step_name: str,
        detail: str = "",
    ) -> None:
        """Update a worker's current status.

        Args:
            worker_id: Numeric identifier for the worker.
            topic: Topic currently being processed.
            step_name: Pipeline step name.
            detail: Optional extra detail string.
        """
        with self._lock:
            self._workers[worker_id] = {
                "topic": topic,
                "step": step_name,
                "detail": detail,
            }
        logger.debug(
            "ProgressTracker: worker %d → topic='%s', step='%s'",
            worker_id,
            topic,
            step_name,
        )

    def clear_worker(self, worker_id: int) -> None:
        """Mark a worker as idle.

        Args:
            worker_id: Numeric identifier for the worker.
        """
        with self._lock:
            self._workers[worker_id] = {
                "topic": "",
                "step": "",
                "detail": "",
            }

    def update_cost(self, cost_usd: float) -> None:
        """Add to the cumulative cost.

        Args:
            cost_usd: Cost in USD to add.
        """
        with self._lock:
            self._total_cost += cost_usd

    def record_score(self, score: float) -> None:
        """Record an evaluation score for average computation.

        Args:
            score: The evaluation score (0.0–10.0).
        """
        with self._lock:
            self._score_sum += score
            self._score_count += 1

    def record_error(self) -> None:
        """Increment the error count by one."""
        with self._lock:
            self._error_count += 1

    def increment_completed(self) -> None:
        """Increment the completed counter by one."""
        with self._lock:
            self._completed += 1

    def set_total(self, total: int) -> None:
        """Set the total number of topics.

        Args:
            total: Total topic count.
        """
        with self._lock:
            self._total = total

    def reset(self) -> None:
        """Reset all tracking state for a new batch."""
        with self._lock:
            self._completed = 0
            self._total = 0
            self._workers.clear()
            self._total_cost = 0.0
            self._score_sum = 0.0
            self._score_count = 0
            self._error_count = 0
            self._start_time = time.monotonic()
        logger.debug("ProgressTracker: reset")

    # ── Snapshot (thread-safe) ────────────────────────────────────────

    def get_snapshot(self) -> ProgressSnapshot:
        """Return an immutable snapshot of the current progress.

        Returns:
            A ``ProgressSnapshot`` with all current state.
        """
        with self._lock:
            elapsed = time.monotonic() - self._start_time

            total = self._total
            completed = self._completed
            percent = (completed / total * 100.0) if total > 0 else 0.0

            avg_score = (
                self._score_sum / self._score_count
                if self._score_count > 0
                else 0.0
            )

            hours_elapsed = elapsed / 3600.0
            speed = (
                completed / hours_elapsed
                if hours_elapsed > 0 and completed > 0
                else 0.0
            )

            worker_snapshots = tuple(
                WorkerSnapshot(
                    worker_id=wid,
                    topic=state.get("topic", ""),
                    step=state.get("step", ""),
                    detail=state.get("detail", ""),
                )
                for wid, state in sorted(self._workers.items())
            )

            return ProgressSnapshot(
                overall_completed=completed,
                overall_total=total,
                overall_percent=round(percent, 1),
                workers=worker_snapshots,
                elapsed_time=round(elapsed, 1),
                total_cost=round(self._total_cost, 4),
                avg_score=round(avg_score, 2),
                error_count=self._error_count,
                speed_stories_per_hour=round(speed, 1),
            )
