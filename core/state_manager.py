"""Unified state management for AI Story Generator Pro.

``StateManager`` is the exclusive API for creating, loading, saving,
and mutating ``PipelineState`` objects.  All pipeline steps go through
this manager rather than modifying state directly, ensuring a single
source of truth and consistent serialisation for crash recovery.

Typical usage::

    sm = StateManager()
    state = sm.create_new("Ancient Temple", "en", gen_config, api_config, "full")
    sm.save(state, Path("output/ancient_temple_en"))
    state = sm.load(Path("output/ancient_temple_en"))
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.exceptions import StateError
from models.evaluation import EvaluationResult
from models.config import APIConfig, GenerationConfig
from models.outline import Outline
from models.section import Section
from models.state import PipelineState, PipelineStatus
from models.story_bible import StoryBible
from utils.file_handler import read_file, write_file

logger = logging.getLogger(__name__)

# State is persisted as this filename inside the output directory.
_STATE_FILENAME = "state.json"

# Number of trailing words from the last section to include in context.
_CONTEXT_LAST_WORDS = 500


class StateManager:
    """Creates, loads, saves, and mutates ``PipelineState`` objects.

    Every mutation method updates the ``updated_at`` timestamp and
    returns the modified state.  ``save()`` serialises the state to
    ``state.json`` in the story's output directory.
    """

    # ── Factory ────────────────────────────────────────────────────────

    def create_new(
        self,
        topic: str,
        language: str,
        gen_config: GenerationConfig,
        api_config: APIConfig,
        strategy_name: str,
        output_dir: str | Path = "",
    ) -> PipelineState:
        """Create a fresh ``PipelineState`` for a new topic.

        Args:
            topic: The story topic or theme string.
            language: Two-letter language code.
            gen_config: Creative parameter snapshot.
            api_config: API settings snapshot.
            strategy_name: Name of the selected strategy.
            output_dir: Path to the output directory (optional; can
                be set later before the first ``save()``).

        Returns:
            A fully initialised ``PipelineState`` in PENDING status.
        """
        story_id = self._generate_story_id(topic, language)
        now = datetime.now(timezone.utc).isoformat()

        state = PipelineState(
            story_id=story_id,
            topic=topic,
            language=language,
            generation_config=gen_config,
            api_config=api_config,
            strategy_name=strategy_name,
            status=PipelineStatus.PENDING,
            current_step_index=0,
            current_attempt=1,
            started_at=now,
            updated_at=now,
            output_dir=str(output_dir) if output_dir else "",
        )

        logger.info(
            "Created new PipelineState: story_id=%s, topic='%s', lang=%s, strategy=%s",
            story_id,
            topic,
            language,
            strategy_name,
        )
        return state

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, state: PipelineState, output_dir: str | Path | None = None) -> Path:
        """Serialise state to ``state.json`` in the output directory.

        If *output_dir* is given it overrides ``state.output_dir`` and
        the state object is updated in-place.

        Args:
            state: The pipeline state to persist.
            output_dir: Optional override for the output directory.

        Returns:
            Path to the written ``state.json`` file.

        Raises:
            StateError: If no output directory is configured and none
                is provided, or if serialisation/write fails.
        """
        if output_dir is not None:
            state.output_dir = str(output_dir)

        if not state.output_dir:
            raise StateError(
                "Cannot save state: no output_dir configured and none provided"
            )

        out_path = Path(state.output_dir) / _STATE_FILENAME
        state.touch()

        try:
            json_str = state.model_dump_json(indent=2)
            write_file(out_path, json_str)
        except Exception as exc:
            raise StateError(
                f"Failed to save state to {out_path}: {exc}"
            ) from exc

        logger.debug("State saved: %s", out_path)
        return out_path

    def load(self, output_dir: str | Path) -> PipelineState:
        """Load a ``PipelineState`` from a directory's ``state.json``.

        Args:
            output_dir: Directory containing ``state.json``.

        Returns:
            The deserialised ``PipelineState``.

        Raises:
            StateError: If the file does not exist, cannot be read,
                or fails validation.
        """
        state_path = Path(output_dir) / _STATE_FILENAME

        if not state_path.exists():
            raise StateError(
                f"State file not found: {state_path}"
            )

        try:
            raw = read_file(state_path)
        except OSError as exc:
            raise StateError(
                f"Failed to read state file {state_path}: {exc}"
            ) from exc

        try:
            state = PipelineState.model_validate_json(raw)
        except Exception as exc:
            raise StateError(
                f"Failed to parse state file {state_path}: {exc}"
            ) from exc

        logger.info(
            "Loaded state: story_id=%s, status=%s, step=%d",
            state.story_id,
            state.status.value,
            state.current_step_index,
        )
        return state

    # ── Mutations ──────────────────────────────────────────────────────

    def update_story_bible(
        self, state: PipelineState, story_bible: StoryBible
    ) -> PipelineState:
        """Set the Story Bible on the state.

        Args:
            state: Current pipeline state.
            story_bible: The Story Bible produced by the concept step.

        Returns:
            The updated state.
        """
        state.story_bible = story_bible
        state.touch()
        logger.debug(
            "Story Bible set for '%s': premise='%s...'",
            state.topic,
            story_bible.premise[:60] if story_bible.premise else "",
        )
        return state

    def update_outline(
        self, state: PipelineState, outline: Outline
    ) -> PipelineState:
        """Set the outline on the state.

        Args:
            state: Current pipeline state.
            outline: The structural outline produced by the outline step.

        Returns:
            The updated state.
        """
        state.outline = outline
        state.touch()
        logger.debug(
            "Outline set for '%s': %d sections, structure=%s",
            state.topic,
            len(outline.sections),
            outline.structure_type,
        )
        return state

    def add_section(
        self, state: PipelineState, section: Section
    ) -> PipelineState:
        """Append a completed section to the state.

        Args:
            state: Current pipeline state.
            section: The completed section with text and summary.

        Returns:
            The updated state.
        """
        state.sections_completed.append(section)
        if section.summary:
            state.section_summaries.append(section.summary)
        state.touch()
        logger.debug(
            "Section %d added for '%s': %d words",
            section.index,
            state.topic,
            section.word_count,
        )
        return state

    def add_section_summary(
        self, state: PipelineState, summary: str
    ) -> PipelineState:
        """Append a section summary without a full section object.

        Args:
            state: Current pipeline state.
            summary: Summary text for context propagation.

        Returns:
            The updated state.
        """
        state.section_summaries.append(summary)
        state.touch()
        return state

    def add_draft(
        self, state: PipelineState, draft_text: str, version: int
    ) -> PipelineState:
        """Add or replace a draft version.

        Drafts are stored in a list where index 0 = v1, index 1 = v2, etc.
        If *version* exceeds the current length, the draft is appended.
        If it matches an existing slot, it replaces that slot.

        Args:
            state: Current pipeline state.
            draft_text: The full draft text.
            version: 1-based draft version number.

        Returns:
            The updated state.
        """
        idx = version - 1
        if idx < 0:
            raise StateError(f"Draft version must be >= 1, got {version}")

        if idx < len(state.drafts):
            state.drafts[idx] = draft_text
        else:
            # Fill any gaps with empty strings (shouldn't happen normally).
            while len(state.drafts) < idx:
                state.drafts.append("")
            state.drafts.append(draft_text)

        state.touch()
        logger.debug(
            "Draft v%d set for '%s': %d words",
            version,
            state.topic,
            len(draft_text.split()),
        )
        return state

    def add_evaluation(
        self, state: PipelineState, eval_result: EvaluationResult
    ) -> PipelineState:
        """Append an evaluation result to the state.

        Args:
            state: Current pipeline state.
            eval_result: The evaluation result for this attempt.

        Returns:
            The updated state.
        """
        state.evaluations.append(eval_result)
        state.touch()
        logger.debug(
            "Evaluation added for '%s': attempt=%d, score=%.2f, passed=%s",
            state.topic,
            eval_result.attempt_number,
            eval_result.overall_score,
            eval_result.passed,
        )
        return state

    def increment_attempt(self, state: PipelineState) -> PipelineState:
        """Increment the current attempt counter.

        Args:
            state: Current pipeline state.

        Returns:
            The updated state with ``current_attempt`` incremented.
        """
        state.current_attempt += 1
        state.touch()
        logger.debug(
            "Attempt incremented for '%s': now %d",
            state.topic,
            state.current_attempt,
        )
        return state

    def update_step_index(
        self, state: PipelineState, index: int
    ) -> PipelineState:
        """Update the current step index.

        Args:
            state: Current pipeline state.
            index: New step index.

        Returns:
            The updated state.
        """
        state.current_step_index = index
        state.touch()
        return state

    def update_cost(
        self,
        state: PipelineState,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> PipelineState:
        """Accumulate token and cost data.

        Args:
            state: Current pipeline state.
            tokens_in: Input tokens to add.
            tokens_out: Output tokens to add.
            cost_usd: Cost in USD to add.

        Returns:
            The updated state.
        """
        state.tokens_used_in += tokens_in
        state.tokens_used_out += tokens_out
        state.cost_accumulated += cost_usd
        state.touch()
        return state

    def mark_in_progress(self, state: PipelineState) -> PipelineState:
        """Set the state status to IN_PROGRESS.

        Args:
            state: Current pipeline state.

        Returns:
            The updated state.
        """
        state.status = PipelineStatus.IN_PROGRESS
        state.touch()
        logger.info("Pipeline IN_PROGRESS for '%s'", state.topic)
        return state

    def mark_completed(self, state: PipelineState) -> PipelineState:
        """Set the state status to COMPLETED.

        Args:
            state: Current pipeline state.

        Returns:
            The updated state.
        """
        state.status = PipelineStatus.COMPLETED
        state.touch()
        logger.info(
            "Pipeline COMPLETED for '%s': score=%.2f, attempts=%d",
            state.topic,
            state.evaluations[-1].overall_score if state.evaluations else 0.0,
            state.current_attempt,
        )
        return state

    def mark_failed(
        self, state: PipelineState, error: str
    ) -> PipelineState:
        """Set the state status to FAILED with an error message.

        Args:
            state: Current pipeline state.
            error: Error description.

        Returns:
            The updated state.
        """
        state.status = PipelineStatus.FAILED
        state.error_message = error
        state.touch()
        logger.error(
            "Pipeline FAILED for '%s': %s",
            state.topic,
            error,
        )
        return state

    # ── Context building ───────────────────────────────────────────────

    def get_context_for_section(
        self, state: PipelineState, section_index: int
    ) -> str:
        """Build context string for generating a specific section.

        The context includes:
        1. Story Bible (JSON summary)
        2. Full outline (JSON)
        3. Previous section's summary (if any)
        4. Last ~500 words of the most recently completed section

        Args:
            state: Current pipeline state (must have story_bible and
                outline set).
            section_index: Zero-based index of the section about to
                be generated.

        Returns:
            A concatenated context string suitable for injection into
            the section prompt.

        Raises:
            StateError: If story_bible or outline is not set.
        """
        if state.story_bible is None:
            raise StateError(
                "Cannot build section context: story_bible is not set"
            )
        if state.outline is None:
            raise StateError(
                "Cannot build section context: outline is not set"
            )

        parts: list[str] = []

        # 1. Story Bible
        bible_json = state.story_bible.model_dump_json(indent=2)
        parts.append(f"=== STORY BIBLE ===\n{bible_json}")

        # 2. Full outline
        outline_json = state.outline.model_dump_json(indent=2)
        parts.append(f"=== OUTLINE ===\n{outline_json}")

        # 3. Previous section summary
        if section_index > 0 and state.section_summaries:
            prev_idx = min(section_index - 1, len(state.section_summaries) - 1)
            if prev_idx >= 0:
                parts.append(
                    f"=== PREVIOUS SECTION SUMMARY ===\n"
                    f"{state.section_summaries[prev_idx]}"
                )

        # 4. Last ~500 words of the most recent section text
        if section_index > 0 and state.sections_completed:
            last_section = state.sections_completed[-1]
            if last_section.text:
                words = last_section.text.split()
                tail_words = words[-_CONTEXT_LAST_WORDS:]
                tail_text = " ".join(tail_words)
                parts.append(
                    f"=== LAST {_CONTEXT_LAST_WORDS} WORDS ===\n{tail_text}"
                )

        context = "\n\n".join(parts)
        logger.debug(
            "Built context for section %d of '%s': %d chars",
            section_index,
            state.topic,
            len(context),
        )
        return context

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _generate_story_id(topic: str, language: str) -> str:
        """Generate a unique story ID from topic, language, and UUID.

        Args:
            topic: The story topic.
            language: Two-letter language code.

        Returns:
            A string like ``"ancient_temple_en_a1b2c3d4"``.
        """
        slug = topic.lower().replace(" ", "_")[:40]
        short_uuid = uuid.uuid4().hex[:8]
        return f"{slug}_{language}_{short_uuid}"
