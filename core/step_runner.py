"""Step runner for AI Story Generator Pro.

Executes a strategy (ordered list of ``BaseStep`` subclasses) against
a ``PipelineState``.  Handles the evaluate→revise loop automatically,
injects revision steps when the score is below the configured minimum,
detects score plateaus to avoid wasting API credits, auto-saves state
after every step, records analytics on completion, and emits events for
the GUI throughout.

Typical usage::

    runner = StepRunner(state_mgr, api_client, prompt_mgr, event_bus, settings)
    state = await runner.execute(state, strategy)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.events import EventBus, EventType
from core.exceptions import PipelineError, StepError
from core.state_manager import StateManager
from core.steps.base_step import BaseStep
from core.steps.clean_step import CleanStep
from core.steps.evaluate_step import EvaluateStep
from core.steps.revise_step import ReviseStep
from models.metadata import StoryMetadata
from models.state import PipelineState, PipelineStatus

if TYPE_CHECKING:
    from core.analytics_collector import AnalyticsCollector
    from core.api_client import APIClient
    from core.prompt_manager import PromptManager
    from core.settings import Settings

logger = logging.getLogger(__name__)

# Default minimum score improvement between consecutive evaluations
# to justify another revision attempt.  If the improvement is below
# this threshold, the revision loop stops early.
DEFAULT_MIN_SCORE_IMPROVEMENT: float = 0.3


class StepRunner:
    """Executes a strategy (list of step classes) against pipeline state.

    Handles the evaluate→revise loop automatically: after each
    ``EvaluateStep`` execution, if the score is below ``min_score``
    and the attempt count is below ``max_attempts``, it injects
    ``ReviseStep``, ``CleanStep``, and ``EvaluateStep`` into the
    remaining step queue.

    **Score plateau detection** prevents infinite cost loops: if the
    score improvement between the last two evaluations is below
    ``min_score_improvement``, revision stops even if attempts remain.

    Args:
        state_manager: Manager for state mutations and persistence.
        api_client: Unified LLM API client.
        prompt_manager: Template loader and renderer.
        event_bus: Thread-safe event bus for GUI communication.
        settings: Application settings (used for min_score/max_attempts
            defaults, though the generation_config on the state takes
            precedence).
        analytics_collector: Optional analytics collector for recording
            completed stories.
    """

    def __init__(
        self,
        state_manager: StateManager,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
        settings: "Settings",
        analytics_collector: "AnalyticsCollector | None" = None,
    ) -> None:
        self._state_mgr = state_manager
        self._api_client = api_client
        self._prompt_mgr = prompt_manager
        self._event_bus = event_bus
        self._settings = settings
        self._analytics = analytics_collector

        # Score plateau detection threshold.
        self._min_score_improvement = getattr(
            getattr(settings, "generation", None),
            "min_score_improvement",
            DEFAULT_MIN_SCORE_IMPROVEMENT,
        )

    def set_analytics_collector(
        self, collector: "AnalyticsCollector",
    ) -> None:
        """Wire the analytics collector after construction.

        Args:
            collector: The ``AnalyticsCollector`` instance.
        """
        self._analytics = collector
        logger.debug("AnalyticsCollector wired to StepRunner")

    async def execute(
        self,
        state: PipelineState,
        strategy: list[type[BaseStep]],
    ) -> PipelineState:
        """Execute a strategy against the given state.

        Args:
            state: The pipeline state (typically freshly created or
                recovered).
            strategy: Ordered list of ``BaseStep`` subclasses to
                instantiate and execute.

        Returns:
            The final pipeline state after all steps (or after failure).
        """
        logger.info(
            "StepRunner: executing strategy for '%s' (%d steps)",
            state.topic,
            len(strategy),
        )

        state = self._state_mgr.mark_in_progress(state)

        # Build a mutable queue from the strategy.  We work with
        # instances so that dynamic injection (revise loop) is easy.
        step_queue: list[BaseStep] = [cls() for cls in strategy]

        step_index = 0
        while step_index < len(step_queue):
            step = step_queue[step_index]

            logger.info(
                "StepRunner: step %d/%d — %s (%s) for '%s'",
                step_index + 1,
                len(step_queue),
                step.name,
                step.description,
                state.topic,
            )

            state = self._state_mgr.update_step_index(state, step_index)

            self._event_bus.emit(
                EventType.STEP_STARTED,
                step=step.name,
                topic=state.topic,
                story_id=state.story_id,
                step_index=step_index,
                total_steps=len(step_queue),
            )

            try:
                state = await step.execute(
                    state,
                    self._api_client,
                    self._prompt_mgr,
                    self._event_bus,
                )
            except StepError as exc:
                logger.error(
                    "StepRunner: step '%s' failed for '%s': %s (recoverable=%s)",
                    step.name,
                    state.topic,
                    exc,
                    exc.recoverable,
                )
                self._event_bus.emit(
                    EventType.STEP_FAILED,
                    step=step.name,
                    topic=state.topic,
                    story_id=state.story_id,
                    error=str(exc),
                    recoverable=exc.recoverable,
                )

                if exc.recoverable:
                    logger.warning(
                        "StepRunner: skipping recoverable step '%s'",
                        step.name,
                    )
                    step_index += 1
                    continue
                else:
                    state = self._state_mgr.mark_failed(state, str(exc))
                    self._auto_save(state)
                    return state

            except PipelineError as exc:
                logger.error(
                    "StepRunner: pipeline error in step '%s' for '%s': %s",
                    step.name,
                    state.topic,
                    exc,
                )
                state = self._state_mgr.mark_failed(state, str(exc))
                self._auto_save(state)
                return state

            except Exception as exc:
                logger.error(
                    "StepRunner: unexpected error in step '%s' for '%s': %s",
                    step.name,
                    state.topic,
                    exc,
                    exc_info=True,
                )
                state = self._state_mgr.mark_failed(
                    state, f"Unexpected error in {step.name}: {exc}"
                )
                self._auto_save(state)
                return state

            # Emit completion.
            self._event_bus.emit(
                EventType.STEP_COMPLETED,
                step=step.name,
                topic=state.topic,
                story_id=state.story_id,
                step_index=step_index,
                total_steps=len(step_queue),
            )

            # Auto-save after every step.
            self._auto_save(state)

            # After EvaluateStep: check score and possibly inject revise loop.
            if isinstance(step, EvaluateStep):
                step_queue, step_index = self._handle_evaluation(
                    state, step_queue, step_index
                )
            else:
                step_index += 1

        # All steps completed successfully.
        state = self._state_mgr.mark_completed(state)
        self._auto_save(state)

        # Record analytics for completed story.
        self._record_analytics(state)

        # Emit story completion event.
        final_score = state.latest_evaluation.overall_score if state.latest_evaluation else 0.0
        self._event_bus.emit(
            EventType.STORY_COMPLETED,
            topic=state.topic,
            story_id=state.story_id,
            score=final_score,
            attempts=state.current_attempt,
            word_count=state.word_count,
        )

        logger.info(
            "StepRunner: strategy complete for '%s': status=%s, "
            "score=%.2f, attempts=%d",
            state.topic,
            state.status.value,
            final_score,
            state.current_attempt,
        )
        return state

    def _handle_evaluation(
        self,
        state: PipelineState,
        step_queue: list[BaseStep],
        current_index: int,
    ) -> tuple[list[BaseStep], int]:
        """Check evaluation result and inject revision steps if needed.

        Implements score plateau detection: if the improvement between
        the last two evaluations is below ``min_score_improvement``,
        revision stops early even if attempts remain.

        Args:
            state: Current pipeline state (with latest evaluation).
            step_queue: The mutable step queue.
            current_index: Index of the EvaluateStep just completed.

        Returns:
            Tuple of (updated step_queue, next step_index).
        """
        latest_eval = state.latest_evaluation
        gen = state.generation_config
        min_score = gen.min_score
        max_attempts = gen.max_attempts

        if latest_eval is None:
            logger.warning(
                "StepRunner: EvaluateStep completed but no evaluation on state"
            )
            return step_queue, current_index + 1

        score = latest_eval.overall_score
        passed = latest_eval.passed

        logger.info(
            "StepRunner: evaluation for '%s': score=%.2f, min=%.1f, "
            "passed=%s, attempt=%d/%d",
            state.topic,
            score,
            min_score,
            passed,
            state.current_attempt,
            max_attempts,
        )

        self._event_bus.emit(
            EventType.EVALUATION_RESULT,
            topic=state.topic,
            story_id=state.story_id,
            score=score,
            passed=passed,
            attempt=state.current_attempt,
        )

        # ── Check 1: passed or max attempts reached ───────────────
        if passed or state.current_attempt >= max_attempts:
            if not passed:
                logger.warning(
                    "StepRunner: max attempts (%d) reached for '%s' "
                    "with score %.2f (min %.1f) — accepting as-is",
                    max_attempts,
                    state.topic,
                    score,
                    min_score,
                )
            return step_queue, current_index + 1

        # ── Check 2: score plateau detection ──────────────────────
        if self._is_score_plateaued(state, score):
            logger.warning(
                "StepRunner: score plateau detected for '%s' — "
                "improvement below %.2f over last evaluations. "
                "Stopping revision early (attempt %d/%d, score %.2f)",
                state.topic,
                self._min_score_improvement,
                state.current_attempt,
                max_attempts,
                score,
            )
            self._event_bus.emit(
                EventType.EVALUATION_RESULT,
                topic=state.topic,
                story_id=state.story_id,
                score=score,
                passed=False,
                attempt=state.current_attempt,
                plateau_detected=True,
            )
            return step_queue, current_index + 1

        # Score too low and attempts remain: inject revise → clean → evaluate.
        self._state_mgr.increment_attempt(state)

        revision_steps: list[BaseStep] = [
            ReviseStep(),
            CleanStep(),
            EvaluateStep(),
        ]

        # Insert after the current evaluate step.
        insert_at = current_index + 1
        for i, rs in enumerate(revision_steps):
            step_queue.insert(insert_at + i, rs)

        logger.info(
            "StepRunner: injected revision loop for '%s' "
            "(attempt %d, %d steps added)",
            state.topic,
            state.current_attempt,
            len(revision_steps),
        )

        self._event_bus.emit(
            EventType.REVISION_STARTED,
            topic=state.topic,
            story_id=state.story_id,
            attempt=state.current_attempt,
            score=score,
        )

        return step_queue, current_index + 1

    def _is_score_plateaued(
        self,
        state: PipelineState,
        current_score: float,
    ) -> bool:
        """Check if the score has plateaued across recent evaluations.

        A plateau is detected when we have at least 3 evaluations
        (including the current one already appended to
        ``state.evaluations`` by ``EvaluateStep``) and the maximum
        improvement between consecutive scores over the last 3
        evaluations is below ``min_score_improvement``.

        Note: ``current_score`` is accepted for API compatibility but
        is not appended separately — it is already the last entry in
        ``state.evaluations``.

        Args:
            state: Pipeline state with evaluation history.
            current_score: Score from the evaluation just completed.
                Already present as the last element of
                ``state.evaluations``.

        Returns:
            True if the score is plateaued and further revision is
            unlikely to yield meaningful improvement.
        """
        evaluations = state.evaluations if state.evaluations else []

        # Need at least 3 evaluations (the current one is already in
        # the list, appended by EvaluateStep) to detect a plateau.
        if len(evaluations) < 3:
            return False

        # Build score history from the last 3 evaluations (already
        # includes the current score as the final element).
        recent_scores: list[float] = [
            ev.overall_score for ev in evaluations[-3:]
        ]

        # Check maximum improvement between consecutive pairs.
        max_improvement = 0.0
        for i in range(1, len(recent_scores)):
            improvement = recent_scores[i] - recent_scores[i - 1]
            max_improvement = max(max_improvement, improvement)

        return max_improvement < self._min_score_improvement

    def _auto_save(self, state: PipelineState) -> None:
        """Attempt to auto-save state; log warning on failure.

        Args:
            state: The pipeline state to persist.
        """
        if not state.output_dir:
            return

        try:
            self._state_mgr.save(state)
        except Exception as exc:
            logger.warning(
                "StepRunner: auto-save failed for '%s': %s",
                state.topic,
                exc,
            )

    def _record_analytics(self, state: PipelineState) -> None:
        """Build StoryMetadata from final state and record it.

        Silently skips if analytics collector is not wired or if
        recording fails.

        Args:
            state: The completed pipeline state.
        """
        if self._analytics is None:
            logger.debug("StepRunner: no analytics collector — skipping record")
            return

        try:
            latest_eval = state.latest_evaluation
            final_score = latest_eval.overall_score if latest_eval else 0.0
            now_str = datetime.now(timezone.utc).isoformat()

            # Compute wall-clock duration from started_at to now.
            duration = 0.0
            if state.started_at:
                try:
                    start_dt = datetime.fromisoformat(state.started_at)
                    end_dt = datetime.now(timezone.utc)
                    duration = (end_dt - start_dt).total_seconds()
                except (ValueError, TypeError):
                    logger.debug(
                        "StepRunner: could not parse started_at '%s' for duration",
                        state.started_at,
                    )

            metadata = StoryMetadata(
                story_id=state.story_id,
                topic=state.topic,
                language=state.language,
                provider=state.api_config.primary_provider.value if state.api_config else "",
                model=state.api_config.primary_model if state.api_config else "",
                generation_config=state.generation_config,
                strategy_used=state.strategy_name,
                final_score=final_score,
                evaluation_history=list(state.evaluations),
                attempts=state.current_attempt,
                started_at=state.started_at,
                completed_at=now_str,
                duration_seconds=duration,
                total_tokens_in=state.tokens_used_in,
                total_tokens_out=state.tokens_used_out,
                estimated_cost_usd=state.cost_accumulated,
                word_count=state.word_count,
                section_count=len(state.sections_completed),
                output_files=[],
            )

            self._analytics.record_story(metadata)
            logger.info(
                "StepRunner: analytics recorded for '%s' "
                "(score=%.2f, cost=$%.4f)",
                state.topic,
                final_score,
                state.cost_accumulated,
            )

        except Exception as exc:
            logger.warning(
                "StepRunner: failed to record analytics for '%s': %s",
                state.topic,
                exc,
            )
