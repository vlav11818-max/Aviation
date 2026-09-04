"""Parallel batch processor for AI Story Generator Pro.

``ParallelProcessor`` takes a list of topics and processes them
concurrently using N async workers.  Each worker:

1. Checks the cache (skip if already done).
2. Creates a pipeline state.
3. Runs ``StepRunner.execute()`` for the selected strategy.
4. Saves the result and updates recovery state.
5. Marks the topic as done in the source topics file (``OK `` prefix).

Shared resources (rate limiter, file locks, event bus) ensure safe
concurrent access.  Pause/stop signals from the GUI are honoured via
``asyncio.Event`` flags.

Typical usage::

    pp = ParallelProcessor(
        step_runner=runner,
        api_client=client,
        state_manager=state_mgr,
        prompt_manager=prompt_mgr,
        cache_manager=cache_mgr,
        recovery_manager=recovery_mgr,
        event_bus=event_bus,
        settings=settings,
    )
    result = await pp.process_batch(
        topics, gen_config, api_config, language,
        topics_file_path=Path("themes.txt"),
    )
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from core.events import EventBus, EventType
from core.exceptions import PipelineError, StepError, StoryGeneratorError
from core.recovery_manager import BatchState, RecoveryManager
from core.strategies import select_strategy
from models.state import PipelineState, PipelineStatus

if TYPE_CHECKING:
    from core.api_client import APIClient
    from core.cache_manager import CacheManager
    from core.prompt_manager import PromptManager
    from core.settings import Settings
    from core.state_manager import StateManager
    from core.step_runner import StepRunner
    from models.config import APIConfig, GenerationConfig

logger = logging.getLogger(__name__)

# Prefix written before a topic line in the source file to mark it
# as successfully completed.  Must match ``DONE_PREFIX`` in
# ``core.input_validator``.
_DONE_PREFIX: str = "OK "


@dataclass
class TopicResult:
    """Result of processing a single topic.

    Attributes:
        topic: The topic string.
        success: Whether the pipeline completed successfully.
        output_dir: Path to the output directory (if successful).
        score: Final evaluation score (0.0 if not evaluated).
        attempts: Number of evaluation attempts.
        error: Error message (empty if successful).
        cached: Whether the result was served from cache.
        elapsed_seconds: Wall-clock time for this topic.
    """

    topic: str
    success: bool = False
    output_dir: str = ""
    score: float = 0.0
    attempts: int = 0
    error: str = ""
    cached: bool = False
    elapsed_seconds: float = 0.0


@dataclass
class BatchResult:
    """Aggregated result of a batch processing run.

    Attributes:
        batch_id: The batch identifier.
        total: Total topics submitted.
        completed: Number of successfully completed topics.
        failed: Number of failed topics.
        cached: Number of topics served from cache.
        results: Per-topic results.
        elapsed_seconds: Total wall-clock time for the batch.
    """

    batch_id: str = ""
    total: int = 0
    completed: int = 0
    failed: int = 0
    cached: int = 0
    results: list[TopicResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class ParallelProcessor:
    """Processes a batch of topics concurrently with N async workers.

    Integrates caching, recovery, rate limiting, and event emission.
    Supports pause/stop signals via ``asyncio.Event`` flags.

    After each successfully completed topic the processor updates the
    source topics file, prepending ``OK `` to the matching line so
    that subsequent runs automatically skip already-done topics.

    Args:
        step_runner: The ``StepRunner`` that executes strategies.
        api_client: Unified LLM API client.
        state_manager: Manager for pipeline state CRUD.
        prompt_manager: Template loader and renderer.
        cache_manager: Result cache manager.
        recovery_manager: Crash recovery manager.
        event_bus: Thread-safe event bus for GUI communication.
        settings: Application settings (parallelism config).
    """

    def __init__(
        self,
        step_runner: "StepRunner",
        api_client: "APIClient",
        state_manager: "StateManager",
        prompt_manager: "PromptManager",
        cache_manager: "CacheManager",
        recovery_manager: RecoveryManager,
        event_bus: EventBus,
        settings: "Settings",
    ) -> None:
        self._runner = step_runner
        self._api_client = api_client
        self._state_mgr = state_manager
        self._prompt_mgr = prompt_manager
        self._cache_mgr = cache_manager
        self._recovery_mgr = recovery_manager
        self._event_bus = event_bus
        self._settings = settings

        # Control signals
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially

        # Active batch state (set during process_batch)
        self._batch_state: BatchState | None = None

        # Path to the source topics file (set per batch, may be None).
        self._topics_file_path: Path | None = None

        # Lock that serialises writes to the topics file so that
        # concurrent workers do not corrupt it.
        self._topics_file_lock = threading.Lock()

    # ── Control signals ───────────────────────────────────────────────

    def request_stop(self) -> None:
        """Signal all workers to stop after their current topic.

        Workers will finish the topic they are working on, then exit.
        Also unblocks any workers waiting on the pause event to prevent
        deadlock when stop is requested while paused.
        """
        self._stop_event.set()
        # Unblock workers that may be waiting on the pause event so they
        # can check the stop flag and exit cleanly.
        self._pause_event.set()
        logger.info("ParallelProcessor: stop requested")

    def request_pause(self) -> None:
        """Pause all workers after their current step.

        Workers will block before starting their next topic until
        ``resume()`` is called.
        """
        self._pause_event.clear()
        logger.info("ParallelProcessor: pause requested")

    def resume(self) -> None:
        """Resume paused workers."""
        self._pause_event.set()
        logger.info("ParallelProcessor: resumed")

    def reset_signals(self) -> None:
        """Reset stop/pause signals for a new batch."""
        self._stop_event.clear()
        self._pause_event.set()

    @property
    def is_stopped(self) -> bool:
        """Whether a stop has been requested."""
        return self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        """Whether processing is currently paused."""
        return not self._pause_event.is_set()

    # ── Main batch entry point ────────────────────────────────────────

    async def process_batch(
        self,
        topics: list[str],
        gen_config: "GenerationConfig",
        api_config: "APIConfig",
        language: str = "en",
        topics_file_path: Path | str | None = None,
    ) -> BatchResult:
        """Process a list of topics concurrently.

        Args:
            topics: List of topic strings to process.
            gen_config: Creative parameter config for all topics.
            api_config: API settings for this batch.
            language: Target language code.
            topics_file_path: Optional path to the source topics file.
                When provided, successfully completed topics are marked
                with an ``OK `` prefix in-place so that subsequent runs
                skip them automatically.

        Returns:
            A ``BatchResult`` with per-topic results and aggregates.
        """
        self.reset_signals()
        await self._api_client.reconfigure(api_config)
        batch_start = time.monotonic()

        # Store the topics file path for this batch.
        if topics_file_path is not None:
            self._topics_file_path = Path(topics_file_path)
        else:
            self._topics_file_path = None

        max_workers = self._settings.parallelism.max_workers

        # Create or load batch state for recovery
        batch = RecoveryManager.create_batch_state(
            topics=topics,
            language=language,
            target_words=gen_config.target_words,
            tone=gen_config.tone.value if hasattr(gen_config.tone, "value") else str(gen_config.tone),
            model=api_config.primary_model,
            provider=api_config.primary_provider.value if hasattr(api_config.primary_provider, "value") else str(api_config.primary_provider),
        )
        self._batch_state = batch
        self._recovery_mgr.save_batch_state(batch)

        logger.info(
            "ParallelProcessor: starting batch '%s' — %d topics, %d workers",
            batch.batch_id,
            len(topics),
            max_workers,
        )

        # Build async task queue
        topic_queue: asyncio.Queue[str] = asyncio.Queue()
        for topic in topics:
            await topic_queue.put(topic)

        # Shared results list (safe because only workers append, and
        # we gather them after all workers finish)
        results: list[TopicResult] = []
        results_lock = asyncio.Lock()

        # Launch workers
        workers = [
            asyncio.create_task(
                self._worker(
                    worker_id=i + 1,
                    topic_queue=topic_queue,
                    gen_config=gen_config,
                    api_config=api_config,
                    language=language,
                    results=results,
                    results_lock=results_lock,
                )
            )
            for i in range(min(max_workers, len(topics)))
        ]

        # Wait for all workers to finish
        await asyncio.gather(*workers, return_exceptions=True)

        elapsed = time.monotonic() - batch_start

        # Build aggregated result
        completed_count = sum(1 for r in results if r.success)
        failed_count = sum(1 for r in results if not r.success and not r.cached)
        cached_count = sum(1 for r in results if r.cached)

        batch_result = BatchResult(
            batch_id=batch.batch_id,
            total=len(topics),
            completed=completed_count,
            failed=failed_count,
            cached=cached_count,
            results=results,
            elapsed_seconds=elapsed,
        )

        # Emit batch completed event
        self._event_bus.emit(
            EventType.BATCH_COMPLETED,
            batch_id=batch.batch_id,
            total=batch_result.total,
            completed=batch_result.completed,
            failed=batch_result.failed,
            cached=batch_result.cached,
            elapsed_seconds=elapsed,
        )

        # Clear recovery state if everything completed
        if not batch.queued and not batch.in_progress:
            self._recovery_mgr.clear()

        logger.info(
            "ParallelProcessor: batch '%s' finished — "
            "completed=%d, failed=%d, cached=%d, elapsed=%.1fs",
            batch.batch_id,
            batch_result.completed,
            batch_result.failed,
            batch_result.cached,
            elapsed,
        )

        return batch_result

    # ── Worker ────────────────────────────────────────────────────────

    async def _worker(
        self,
        worker_id: int,
        topic_queue: asyncio.Queue[str],
        gen_config: "GenerationConfig",
        api_config: "APIConfig",
        language: str,
        results: list[TopicResult],
        results_lock: asyncio.Lock,
    ) -> None:
        """Async worker that pulls topics from the queue and processes them.

        Args:
            worker_id: Numeric identifier for this worker (for logging).
            topic_queue: Shared queue of topics to process.
            gen_config: Creative parameter config.
            api_config: API settings.
            language: Target language code.
            results: Shared results list.
            results_lock: Lock for appending to the results list.
        """
        logger.debug("Worker %d started", worker_id)

        while not self._stop_event.is_set():
            # Wait if paused
            await self._pause_event.wait()

            # Get next topic (non-blocking)
            try:
                topic = topic_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            result = await self._process_single_topic(
                worker_id=worker_id,
                topic=topic,
                gen_config=gen_config,
                api_config=api_config,
                language=language,
            )

            async with results_lock:
                results.append(result)

            topic_queue.task_done()

        logger.debug("Worker %d finished", worker_id)

    # ── Single-topic processing ───────────────────────────────────────

    async def _process_single_topic(
        self,
        worker_id: int,
        topic: str,
        gen_config: "GenerationConfig",
        api_config: "APIConfig",
        language: str,
    ) -> TopicResult:
        """Process a single topic through the full pipeline.

        Args:
            worker_id: Worker identifier for logging.
            topic: The topic string.
            gen_config: Creative parameter config.
            api_config: API settings.
            language: Target language code.

        Returns:
            A ``TopicResult`` describing the outcome.
        """
        topic_start = time.monotonic()

        logger.info(
            "Worker %d: processing topic '%s' (lang=%s)",
            worker_id,
            topic,
            language,
        )

        # 1. Check cache
        if self._settings.cache.skip_processed:
            cache_key = self._cache_mgr.make_key(
                topic, language, gen_config, api_config.primary_model
            )
            cached_dir = self._cache_mgr.get(cache_key)
            if cached_dir is not None:
                elapsed = time.monotonic() - topic_start
                logger.info(
                    "Worker %d: topic '%s' found in cache → %s",
                    worker_id,
                    topic,
                    cached_dir,
                )

                # Update batch recovery state
                if self._batch_state is not None:
                    self._recovery_mgr.mark_topic_completed(
                        self._batch_state, topic
                    )
                    self._recovery_mgr.save_batch_state(self._batch_state)

                # Mark in topics file (cached = already successful)
                self._mark_topic_done_in_file(topic)

                return TopicResult(
                    topic=topic,
                    success=True,
                    output_dir=str(cached_dir),
                    cached=True,
                    elapsed_seconds=elapsed,
                )

        # 2. Select strategy
        strategy_name, strategy_steps = select_strategy(
            target_words=gen_config.target_words,
            settings=self._settings,
        )

        # 3. Build output directory
        slug = topic.lower().replace(" ", "_")[:40]
        output_dir = Path(self._settings.paths.output_dir) / language / slug

        # 4. Create pipeline state
        state = self._state_mgr.create_new(
            topic=topic,
            language=language,
            gen_config=gen_config,
            api_config=api_config,
            strategy_name=strategy_name,
            output_dir=output_dir,
        )

        # 5. Update recovery state
        if self._batch_state is not None:
            self._recovery_mgr.mark_topic_started(
                self._batch_state, topic, stage="starting"
            )
            self._recovery_mgr.save_batch_state(self._batch_state)

        # 6. Execute pipeline
        try:
            state = await self._runner.execute(state, strategy_steps)
        except StoryGeneratorError as exc:
            elapsed = time.monotonic() - topic_start
            logger.error(
                "Worker %d: topic '%s' failed: %s",
                worker_id,
                topic,
                exc,
            )

            if self._batch_state is not None:
                self._recovery_mgr.mark_topic_failed(
                    self._batch_state,
                    topic=topic,
                    error=str(exc),
                    stage=getattr(exc, "step_name", "unknown"),
                    attempt=state.current_attempt,
                )
                self._recovery_mgr.save_batch_state(self._batch_state)

            return TopicResult(
                topic=topic,
                success=False,
                error=str(exc),
                attempts=state.current_attempt,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - topic_start
            logger.error(
                "Worker %d: topic '%s' unexpected error: %s",
                worker_id,
                topic,
                exc,
                exc_info=True,
            )

            if self._batch_state is not None:
                self._recovery_mgr.mark_topic_failed(
                    self._batch_state,
                    topic=topic,
                    error=f"Unexpected: {exc}",
                    stage="unknown",
                    attempt=state.current_attempt,
                )
                self._recovery_mgr.save_batch_state(self._batch_state)

            return TopicResult(
                topic=topic,
                success=False,
                error=f"Unexpected error: {exc}",
                attempts=state.current_attempt,
                elapsed_seconds=elapsed,
            )

        elapsed = time.monotonic() - topic_start

        # 7. Evaluate result
        success = state.status == PipelineStatus.COMPLETED
        score = 0.0
        if state.evaluations:
            score = state.evaluations[-1].overall_score

        # 8. Update cache
        if success and self._settings.cache.enabled:
            cache_key = self._cache_mgr.make_key(
                topic, language, gen_config, api_config.primary_model
            )
            self._cache_mgr.put(cache_key, output_dir)

        # 9. Update recovery state
        if self._batch_state is not None:
            if success:
                self._recovery_mgr.mark_topic_completed(
                    self._batch_state, topic
                )
            else:
                self._recovery_mgr.mark_topic_failed(
                    self._batch_state,
                    topic=topic,
                    error=state.error_message,
                    stage="pipeline",
                    attempt=state.current_attempt,
                )
            self._recovery_mgr.save_batch_state(self._batch_state)

        # 10. Mark completed topic in the source file
        if success:
            self._mark_topic_done_in_file(topic)

        # NOTE: STORY_COMPLETED event is already emitted by StepRunner.
        # Do NOT emit here to avoid duplicate log lines in the GUI.

        logger.info(
            "Worker %d: topic '%s' finished — success=%s, score=%.2f, "
            "attempts=%d, elapsed=%.1fs",
            worker_id,
            topic,
            success,
            score,
            state.current_attempt,
            elapsed,
        )

        return TopicResult(
            topic=topic,
            success=success,
            output_dir=str(output_dir),
            score=score,
            attempts=state.current_attempt,
            error=state.error_message if not success else "",
            cached=False,
            elapsed_seconds=elapsed,
        )

    # ── Topics file marking ───────────────────────────────────────────

    def _mark_topic_done_in_file(self, topic: str) -> None:
        """Prepend ``OK `` to the matching line in the source topics file.

        The operation is atomic: the file is read, the first line whose
        stripped content matches ``topic`` (case-sensitive) is prefixed,
        and the whole file is written to a temporary path then renamed
        over the original.  A ``threading.Lock`` ensures that concurrent
        workers do not interleave reads and writes.

        If no topics file was provided for this batch, or if the topic
        is not found in the file (e.g., already marked), the method
        returns silently.

        Args:
            topic: The topic string to mark as done.
        """
        if self._topics_file_path is None:
            return

        with self._topics_file_lock:
            try:
                content = self._topics_file_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "Cannot read topics file for marking: %s", exc,
                )
                return

            lines = content.splitlines(keepends=True)
            updated = False

            for idx, line in enumerate(lines):
                stripped = line.strip()
                # Skip lines already marked or empty.
                if stripped.startswith(_DONE_PREFIX) or not stripped:
                    continue
                if stripped == topic:
                    # Preserve the original line ending (if any).
                    # Replace only the first occurrence.
                    leading_ws = line[: len(line) - len(line.lstrip())]
                    trailing = line[len(leading_ws) + len(stripped):]
                    lines[idx] = f"{leading_ws}{_DONE_PREFIX}{stripped}{trailing}"
                    updated = True
                    break

            if not updated:
                logger.debug(
                    "Topic '%s' not found (or already marked) in %s",
                    topic,
                    self._topics_file_path,
                )
                return

            # Atomic write: temp file → rename.
            import os
            import tempfile

            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self._topics_file_path.parent),
                prefix=".topics_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
                    tmp_f.writelines(lines)
                os.replace(tmp_path, str(self._topics_file_path))
            except OSError as exc:
                logger.error(
                    "Failed to update topics file for topic '%s': %s",
                    topic,
                    exc,
                )
                # Clean up temp file on failure.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                return

            logger.info(
                "Marked topic as done in %s: %s",
                self._topics_file_path.name,
                topic,
            )
