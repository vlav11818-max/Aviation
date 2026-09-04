"""Text adaptation engine for AI Story Generator Pro.

``TextAdapter`` adapts existing stories from one language to another in
three modes: *literal* (faithful translation), *cultural* (localised
adaptation), and *free* (creative reimagining).

For long texts (>4 000 words) the source is split at paragraph
boundaries and each chunk is adapted individually with overlapping
context from the previous chunk (last 200 words).

After adaptation, an optional evaluate→revise loop can be run to
ensure the adapted text meets the project's quality bar.

Typical usage::

    adapter = TextAdapter(settings=settings)
    result = await adapter.adapt(
        source_text="Once upon a time ...",
        source_lang="en",
        target_lang="de",
        mode=AdaptationMode.CULTURAL,
        params=AdaptationParams(),
        api_client=api_client,
        prompt_manager=prompt_manager,
        event_bus=event_bus,
    )
    # result.adapted_text  -> the adapted story text
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from core.events import EventBus, EventType
from core.exceptions import PipelineError, StepError
from core.steps.clean_step import CleanStep
from core.steps.evaluate_step import EvaluateStep
from core.steps.revise_step import ReviseStep
from core.state_manager import StateManager
from models.config import APIConfig, GenerationConfig
from models.state import PipelineState, PipelineStatus

if TYPE_CHECKING:
    from core.api_client import APIClient
    from core.prompt_manager import PromptManager
    from core.settings import Settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

_LONG_TEXT_THRESHOLD_WORDS: int = 4000
_OVERLAP_WORDS: int = 200
_DEFAULT_TEMPERATURE: float = 0.7
_DEFAULT_MAX_TOKENS: int = 8192


# ── Enum ───────────────────────────────────────────────────────────────


class AdaptationMode(str, Enum):
    """Available adaptation modes."""

    LITERAL = "literal"
    CULTURAL = "cultural"
    FREE = "free"


# ── Data models ────────────────────────────────────────────────────────


class AdaptationParams(BaseModel):
    """Parameters controlling adaptation behaviour.

    Attributes:
        adapt_names: Adapt character names to target culture.
        adapt_references: Adapt cultural references.
        adapt_units: Convert units of measurement.
        adapt_setting: Relocate setting to target culture.
        preserve_length: Keep adapted text within 10% of original length.
        voiceover_optimize: Optimise for voiceover/TTS.
        run_evaluation: Run evaluation loop after adaptation.
    """

    adapt_names: bool = Field(
        default=True,
        description="Adapt character names to target culture.",
    )
    adapt_references: bool = Field(
        default=True,
        description="Adapt cultural references.",
    )
    adapt_units: bool = Field(
        default=True,
        description="Convert units of measurement.",
    )
    adapt_setting: bool = Field(
        default=False,
        description="Relocate setting to target culture.",
    )
    preserve_length: bool = Field(
        default=True,
        description="Keep adapted text within 10% of original length.",
    )
    voiceover_optimize: bool = Field(
        default=True,
        description="Optimise for voiceover/TTS.",
    )
    run_evaluation: bool = Field(
        default=True,
        description="Run evaluation loop after adaptation.",
    )


class AdaptationResult(BaseModel):
    """Result of a text adaptation operation.

    Attributes:
        adapted_text: The adapted/translated text.
        source_lang: Source language code.
        target_lang: Target language code.
        mode: Adaptation mode used.
        score: Final evaluation score (0.0 if evaluation was skipped).
        passed: Whether the evaluation threshold was met.
        attempts: Number of evaluation/revision attempts.
        tokens_in: Total input tokens consumed.
        tokens_out: Total output tokens consumed.
        cost_usd: Estimated cost in USD.
        duration_seconds: Wall-clock duration.
        source_word_count: Word count of the source text.
        adapted_word_count: Word count of the adapted text.
        chunks_used: Number of chunks (1 for short texts).
        error: Error message if adaptation failed.
    """

    adapted_text: str = Field(
        default="",
        description="The adapted text.",
    )
    source_lang: str = Field(
        default="",
        description="Source language code.",
    )
    target_lang: str = Field(
        default="",
        description="Target language code.",
    )
    mode: str = Field(
        default="",
        description="Adaptation mode used.",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Final evaluation score.",
    )
    passed: bool = Field(
        default=False,
        description="Whether evaluation threshold was met.",
    )
    attempts: int = Field(
        default=0,
        ge=0,
        description="Number of evaluation/revision attempts.",
    )
    tokens_in: int = Field(
        default=0,
        ge=0,
        description="Total input tokens consumed.",
    )
    tokens_out: int = Field(
        default=0,
        ge=0,
        description="Total output tokens consumed.",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated cost in USD.",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration in seconds.",
    )
    source_word_count: int = Field(
        default=0,
        ge=0,
        description="Word count of source text.",
    )
    adapted_word_count: int = Field(
        default=0,
        ge=0,
        description="Word count of adapted text.",
    )
    chunks_used: int = Field(
        default=1,
        ge=1,
        description="Number of chunks processed.",
    )
    error: str = Field(
        default="",
        description="Error message if adaptation failed.",
    )


# ── TextAdapter ────────────────────────────────────────────────────────


class TextAdapter:
    """Adapts existing texts between languages in three modes.

    Handles both short texts (single API call) and long texts (chunked
    at paragraph boundaries with overlapping context).

    Args:
        settings: Application-wide settings for quality thresholds and
            generation defaults.
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._state_mgr = StateManager()
        logger.info("TextAdapter initialised")

    # ── Public API ─────────────────────────────────────────────────────

    async def adapt(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        mode: AdaptationMode,
        params: AdaptationParams,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
        output_dir: str = "",
    ) -> AdaptationResult:
        """Adapt a source text to the target language.

        Args:
            source_text: The original text to adapt.
            source_lang: Two-letter source language code.
            target_lang: Two-letter target language code.
            mode: Adaptation mode (literal, cultural, free).
            params: Adaptation parameters.
            api_client: Unified API client for LLM calls.
            prompt_manager: Template loader and renderer.
            event_bus: Event bus for GUI communication.

        Returns:
            An ``AdaptationResult`` with the adapted text and metadata.
        """
        start_time = time.monotonic()
        source_word_count = len(source_text.split())

        logger.info(
            "TextAdapter: adapting %d words from %s to %s (mode=%s)",
            source_word_count,
            source_lang,
            target_lang,
            mode.value,
        )

        event_bus.emit(
            EventType.STEP_STARTED,
            step="adaptation",
            topic=f"adapt_{source_lang}_to_{target_lang}",
            story_id="",
        )

        try:
            # Decide chunking strategy.
            if source_word_count > _LONG_TEXT_THRESHOLD_WORDS:
                adapted_text, chunks_used = await self._adapt_chunked(
                    source_text=source_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    mode=mode,
                    params=params,
                    api_client=api_client,
                    prompt_manager=prompt_manager,
                    event_bus=event_bus,
                )
            else:
                adapted_text = await self._adapt_single(
                    source_text=source_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    mode=mode,
                    params=params,
                    api_client=api_client,
                    prompt_manager=prompt_manager,
                    event_bus=event_bus,
                )
                chunks_used = 1

        except (PipelineError, StepError) as exc:
            logger.error(
                "TextAdapter: adaptation failed for %s→%s: %s",
                source_lang,
                target_lang,
                exc,
            )
            elapsed = time.monotonic() - start_time
            return AdaptationResult(
                source_lang=source_lang,
                target_lang=target_lang,
                mode=mode.value,
                source_word_count=source_word_count,
                duration_seconds=elapsed,
                error=str(exc),
            )
        except Exception as exc:
            logger.error(
                "TextAdapter: unexpected error for %s→%s: %s",
                source_lang,
                target_lang,
                exc,
                exc_info=True,
            )
            elapsed = time.monotonic() - start_time
            return AdaptationResult(
                source_lang=source_lang,
                target_lang=target_lang,
                mode=mode.value,
                source_word_count=source_word_count,
                duration_seconds=elapsed,
                error=f"Unexpected error: {exc}",
            )

        # Run optional evaluation → revision loop.
        score = 0.0
        passed = False
        attempts = 0
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0

        if params.run_evaluation:
            eval_result = await self._run_evaluation_loop(
                adapted_text=adapted_text,
                target_lang=target_lang,
                api_client=api_client,
                prompt_manager=prompt_manager,
                event_bus=event_bus,
                output_dir=output_dir,
            )
            adapted_text = eval_result["text"]
            score = eval_result["score"]
            passed = eval_result["passed"]
            attempts = eval_result["attempts"]
            total_tokens_in = eval_result["tokens_in"]
            total_tokens_out = eval_result["tokens_out"]
            total_cost = eval_result["cost_usd"]

        elapsed = time.monotonic() - start_time
        adapted_word_count = len(adapted_text.split())

        event_bus.emit(
            EventType.STEP_COMPLETED,
            step="adaptation",
            topic=f"adapt_{source_lang}_to_{target_lang}",
            story_id="",
        )

        logger.info(
            "TextAdapter: adaptation complete %s→%s: %d→%d words, "
            "score=%.2f, passed=%s, attempts=%d, %.1fs",
            source_lang,
            target_lang,
            source_word_count,
            adapted_word_count,
            score,
            passed,
            attempts,
            elapsed,
        )

        return AdaptationResult(
            adapted_text=adapted_text,
            source_lang=source_lang,
            target_lang=target_lang,
            mode=mode.value,
            score=score,
            passed=passed,
            attempts=attempts,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            cost_usd=total_cost,
            duration_seconds=elapsed,
            source_word_count=source_word_count,
            adapted_word_count=adapted_word_count,
            chunks_used=chunks_used,
        )

    # ── Single-call adaptation ─────────────────────────────────────────

    async def _adapt_single(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        mode: AdaptationMode,
        params: AdaptationParams,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
    ) -> str:
        """Adapt a text that fits in a single API call.

        Args:
            source_text: The text to adapt.
            source_lang: Source language code.
            target_lang: Target language code.
            mode: Adaptation mode.
            params: Adaptation parameters.
            api_client: API client.
            prompt_manager: Prompt manager.
            event_bus: Event bus.

        Returns:
            The adapted text.

        Raises:
            PipelineError: If the API call or prompt rendering fails.
        """
        prompt = self._render_adaptation_prompt(
            source_text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            mode=mode,
            params=params,
            prompt_manager=prompt_manager,
        )

        try:
            response = await api_client.send(
                messages=[{"role": "user", "content": prompt}],
                temperature=_DEFAULT_TEMPERATURE,
                max_tokens=_DEFAULT_MAX_TOKENS,
            )
        except Exception as exc:
            raise PipelineError(
                f"Adaptation API call failed for {source_lang}→{target_lang}: {exc}"
            ) from exc

        adapted = response.strip()
        if not adapted:
            raise PipelineError(
                f"Adaptation returned empty text for {source_lang}→{target_lang}"
            )

        logger.debug(
            "Single adaptation complete: %d→%d words",
            len(source_text.split()),
            len(adapted.split()),
        )
        return adapted

    # ── Chunked adaptation ─────────────────────────────────────────────

    async def _adapt_chunked(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        mode: AdaptationMode,
        params: AdaptationParams,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
    ) -> tuple[str, int]:
        """Adapt a long text by splitting at paragraph boundaries.

        Each chunk gets the last 200 words of the previous chunk's
        *adapted* output as overlapping context.

        Args:
            source_text: The full source text.
            source_lang: Source language code.
            target_lang: Target language code.
            mode: Adaptation mode.
            params: Adaptation parameters.
            api_client: API client.
            prompt_manager: Prompt manager.
            event_bus: Event bus.

        Returns:
            Tuple of (adapted text, number of chunks).

        Raises:
            PipelineError: If any chunk fails.
        """
        chunks = self._split_into_chunks(source_text)
        total_chunks = len(chunks)

        logger.info(
            "TextAdapter: adapting in %d chunks (source %d words)",
            total_chunks,
            len(source_text.split()),
        )

        adapted_chunks: list[str] = []
        previous_context = ""

        for idx, chunk in enumerate(chunks):
            logger.info(
                "TextAdapter: adapting chunk %d/%d (%d words)",
                idx + 1,
                total_chunks,
                len(chunk.split()),
            )

            event_bus.emit(
                EventType.SECTION_COMPLETED,
                step="adaptation",
                topic=f"adapt_{source_lang}_to_{target_lang}",
                section_index=idx,
                total_sections=total_chunks,
            )

            # Build prompt with context from previous adapted chunk.
            chunk_with_context = chunk
            if previous_context:
                chunk_with_context = (
                    f"[CONTEXT FROM PREVIOUS SECTION — do NOT include "
                    f"this in your output, use it only for continuity "
                    f"and consistency:]\n{previous_context}\n\n"
                    f"[TEXT TO ADAPT:]\n{chunk}"
                )

            adapted_chunk = await self._adapt_single(
                source_text=chunk_with_context,
                source_lang=source_lang,
                target_lang=target_lang,
                mode=mode,
                params=params,
                api_client=api_client,
                prompt_manager=prompt_manager,
                event_bus=event_bus,
            )

            adapted_chunks.append(adapted_chunk)

            # Build context for the next chunk: last 200 words of adapted output.
            words = adapted_chunk.split()
            if len(words) > _OVERLAP_WORDS:
                previous_context = " ".join(words[-_OVERLAP_WORDS:])
            else:
                previous_context = adapted_chunk

        full_adapted = "\n\n".join(adapted_chunks)
        return full_adapted, total_chunks

    # ── Evaluation loop ────────────────────────────────────────────────

    async def _run_evaluation_loop(
        self,
        adapted_text: str,
        target_lang: str,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
        output_dir: str = "",
    ) -> dict:
        """Run evaluate → revise loop on adapted text.

        Creates a temporary ``PipelineState``, injects the adapted text
        as its draft, then runs ``EvaluateStep`` (and ``ReviseStep`` +
        ``CleanStep`` if the score is below threshold) up to
        ``max_attempts`` times.

        Args:
            adapted_text: The text to evaluate and potentially revise.
            target_lang: Target language code.
            api_client: API client.
            prompt_manager: Prompt manager.
            event_bus: Event bus.

        Returns:
            Dict with keys: text, score, passed, attempts, tokens_in,
            tokens_out, cost_usd.
        """
        gen_config = GenerationConfig(
            target_words=len(adapted_text.split()),
            min_score=self._settings.generation.min_score,
            max_attempts=self._settings.generation.max_attempts,
            voiceover_optimized=True,
            no_headers=True,
            no_meta_comments=True,
            pause_markers=True,
            avoid_complex_sentences=True,
        )

        api_config = APIConfig(
            provider=self._settings.api.primary_provider,
            model=self._settings.api.primary_model,
        )

        state = self._state_mgr.create_new(
            topic=f"adaptation_{target_lang}",
            language=target_lang,
            gen_config=gen_config,
            api_config=api_config,
            strategy_name="adaptation_eval",
            output_dir=output_dir,
        )

        # Inject adapted text as draft v1.
        state = self._state_mgr.add_draft(state, adapted_text, version=1)
        state.status = PipelineStatus.IN_PROGRESS
        state.touch()

        evaluate_step = EvaluateStep()
        revise_step = ReviseStep()
        clean_step = CleanStep()

        min_score = gen_config.min_score
        max_attempts = gen_config.max_attempts
        current_attempt = 1

        logger.info(
            "TextAdapter: starting evaluation loop (min_score=%.1f, max_attempts=%d)",
            min_score,
            max_attempts,
        )

        while current_attempt <= max_attempts:
            # Evaluate.
            try:
                state = await evaluate_step.execute(
                    state, api_client, prompt_manager, event_bus
                )
            except StepError as exc:
                logger.error(
                    "TextAdapter: evaluation failed at attempt %d: %s",
                    current_attempt,
                    exc,
                )
                break

            latest_eval = state.latest_evaluation
            if latest_eval is None:
                logger.warning(
                    "TextAdapter: evaluation step produced no result at attempt %d",
                    current_attempt,
                )
                break

            score = latest_eval.overall_score
            passed = latest_eval.passed

            logger.info(
                "TextAdapter: evaluation attempt %d: score=%.2f, passed=%s",
                current_attempt,
                score,
                passed,
            )

            event_bus.emit(
                EventType.EVALUATION_RESULT,
                step="adaptation_eval",
                score=score,
                passed=passed,
                attempt=current_attempt,
            )

            if passed or current_attempt >= max_attempts:
                if not passed:
                    logger.warning(
                        "TextAdapter: max attempts (%d) reached with score %.2f",
                        max_attempts,
                        score,
                    )
                break

            # Revise.
            self._state_mgr.increment_attempt(state)
            current_attempt += 1

            event_bus.emit(
                EventType.REVISION_STARTED,
                step="adaptation_revise",
                attempt=current_attempt,
                score=score,
            )

            try:
                state = await revise_step.execute(
                    state, api_client, prompt_manager, event_bus
                )
            except StepError as exc:
                logger.error(
                    "TextAdapter: revision failed at attempt %d: %s",
                    current_attempt,
                    exc,
                )
                break

            # Clean.
            try:
                state = await clean_step.execute(
                    state, api_client, prompt_manager, event_bus
                )
            except StepError as exc:
                logger.warning(
                    "TextAdapter: clean step failed at attempt %d: %s",
                    current_attempt,
                    exc,
                )
                # Non-critical — continue with unclean text.

        final_eval = state.latest_evaluation
        return {
            "text": state.latest_draft,
            "score": final_eval.overall_score if final_eval else 0.0,
            "passed": final_eval.passed if final_eval else False,
            "attempts": state.current_attempt,
            "tokens_in": state.tokens_used_in,
            "tokens_out": state.tokens_used_out,
            "cost_usd": state.cost_accumulated,
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _render_adaptation_prompt(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        mode: AdaptationMode,
        params: AdaptationParams,
        prompt_manager: "PromptManager",
    ) -> str:
        """Render the adaptation prompt template.

        Args:
            source_text: The text to adapt.
            source_lang: Source language code.
            target_lang: Target language code.
            mode: Adaptation mode.
            params: Adaptation parameters.
            prompt_manager: Template loader and renderer.

        Returns:
            The fully rendered prompt string.

        Raises:
            PipelineError: If rendering fails.
        """
        voiceover_rules = self._build_voiceover_rules(params)

        try:
            prompt = prompt_manager.render(
                "adaptation",
                language=target_lang,
                source_text=source_text,
                source_language=source_lang,
                target_language=target_lang,
                adaptation_mode=mode.value,
                adapt_names=str(params.adapt_names),
                adapt_references=str(params.adapt_references),
                adapt_units=str(params.adapt_units),
                adapt_setting=str(params.adapt_setting),
                preserve_length=str(params.preserve_length),
                voiceover_optimize=str(params.voiceover_optimize),
                voiceover_rules=voiceover_rules,
            )
        except Exception as exc:
            raise PipelineError(
                f"Failed to render adaptation prompt: {exc}"
            ) from exc

        return prompt

    @staticmethod
    def _build_voiceover_rules(params: AdaptationParams) -> str:
        """Build voiceover rules string from adaptation params.

        Args:
            params: Adaptation parameters.

        Returns:
            Voiceover instructions string.
        """
        if not params.voiceover_optimize:
            return "No specific voiceover requirements."

        rules: list[str] = [
            "Text must be optimized for voiceover/TTS narration.",
            "Use simple, clear sentences. Avoid nested clauses.",
            "Do NOT include chapter headers, section markers, or titles.",
            "Do NOT include meta-comments, author's notes, or AI commentary.",
            "Use natural punctuation for pauses: commas, periods, "
            "ellipses, em dashes.",
        ]
        return "\n".join(rules)

    @staticmethod
    def _split_into_chunks(text: str) -> list[str]:
        """Split text into chunks at paragraph boundaries.

        Aims for chunks of approximately 2 000 words each, but never
        breaks mid-paragraph.  If a single paragraph exceeds the target
        chunk size, it becomes its own chunk.

        Args:
            text: The full text to split.

        Returns:
            List of text chunks.
        """
        target_chunk_words = 2000
        paragraphs = text.split("\n\n")

        # Remove empty paragraphs that arise from triple newlines.
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text] if text.strip() else []

        chunks: list[str] = []
        current_chunk_paragraphs: list[str] = []
        current_word_count = 0

        for para in paragraphs:
            para_words = len(para.split())

            if current_word_count + para_words > target_chunk_words and current_chunk_paragraphs:
                # Flush current chunk.
                chunks.append("\n\n".join(current_chunk_paragraphs))
                current_chunk_paragraphs = [para]
                current_word_count = para_words
            else:
                current_chunk_paragraphs.append(para)
                current_word_count += para_words

        # Flush remaining paragraphs.
        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))

        logger.debug(
            "Split text into %d chunks (source: %d paragraphs)",
            len(chunks),
            len(paragraphs),
        )
        return chunks
