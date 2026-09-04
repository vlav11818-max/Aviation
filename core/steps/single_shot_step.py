"""Single-shot story generation step.

Generates the entire story in a single API call. Used for short stories
(< 2000 words) and as the generation engine for the two-pass strategy
(2001-4000 words) where an outline is injected as context.

FIX 1: Injects word-count beat targets (15/35/35/15 split) and hard min/max
guardrails into the prompt so the LLM cannot stop early. max_tokens raised
to target*3 (was target*2) to prevent truncation mid-story.

FIX 2: NOW INJECTS story_bible and outline (when available) into the prompt.
Without these, the LLM had only a short topic phrase and could not fill
3000 words with meaningful content — resulting in 1000-1500 word outputs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.events import EventBus, EventType
from core.exceptions import StepError
from core.state_manager import StateManager
from core.steps.base_step import BaseStep
from models.state import PipelineState

if TYPE_CHECKING:
    from core.api_client import APIClient
    from core.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class SingleShotStep(BaseStep):
    """Generate the entire story in a single API call.

    Used for short stories (under 2000 words) and as the generation
    engine for the two-pass strategy (2001-4000 words). The prompt
    includes all style parameters, cultural instructions, voiceover
    rules, the full Story Bible, the outline (if present for two-pass),
    and explicit word-count beat targets to ensure the LLM produces
    the requested length.
    """

    @property
    def name(self) -> str:
        """Return the step name."""
        return "single_shot"

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Generate full story in one call"

    async def execute(
        self,
        state: PipelineState,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
    ) -> PipelineState:
        """Run single-shot generation.

        Injects the Story Bible (from ConceptStep) and the outline
        (from OutlineStep, if the two-pass strategy produced one) into
        the prompt so the LLM has enough detail to write the full
        target word count.

        Args:
            state: Current pipeline state.
            api_client: Unified API client.
            prompt_manager: Template loader/renderer.
            event_bus: Event bus for GUI communication.

        Returns:
            Updated state with drafts[0] populated.

        Raises:
            StepError: If the API call fails.
        """
        logger.info("SingleShotStep: generating full story for '%s'", state.topic)
        gen = state.generation_config
        target = gen.target_words

        # Compute word-count guardrails injected into the prompt.
        target_words_min = int(target * 0.90)
        target_words_max = int(target * 1.10)

        # Beat word targets: 15% / 35% / 35% / 15% split.
        beat_1 = int(target * 0.15)   # opening hook
        beat_2 = int(target * 0.35)   # escalation
        beat_3 = int(target * 0.35)   # climax
        beat_4 = target - beat_1 - beat_2 - beat_3  # resolution (remainder)

        logger.debug(
            "SingleShotStep: target=%d words, min=%d, max=%d, beats=[%d, %d, %d, %d]",
            target, target_words_min, target_words_max, beat_1, beat_2, beat_3, beat_4,
        )

        voiceover_rules = self._build_voiceover_rules(gen)

        # ── Build Story Bible context ────────────────────────────────
        # The Story Bible is the detailed creative brief produced by
        # ConceptStep.  Without it the LLM has only a short topic phrase
        # and cannot generate enough meaningful content to hit the
        # target word count.  This was the ROOT CAUSE of 3000 → 1200.
        story_bible_text = ""
        if state.story_bible is not None:
            try:
                story_bible_text = state.story_bible.model_dump_json(indent=2)
                logger.debug(
                    "SingleShotStep: injecting story_bible (%d chars) for '%s'",
                    len(story_bible_text),
                    state.topic,
                )
            except Exception as exc:
                logger.warning(
                    "SingleShotStep: failed to serialise story_bible for '%s': %s",
                    state.topic,
                    exc,
                )
        else:
            logger.warning(
                "SingleShotStep: no story_bible available for '%s' — "
                "word count may fall short of target",
                state.topic,
            )

        # ── Build Outline context (for two-pass strategy) ────────────
        # If OutlineStep ran before us (two-pass), inject the outline so
        # the LLM has structural guidance for a longer story.
        outline_context = ""
        if state.outline is not None:
            try:
                outline_context = state.outline.model_dump_json(indent=2)
                logger.debug(
                    "SingleShotStep: injecting outline (%d chars) for '%s'",
                    len(outline_context),
                    state.topic,
                )
            except Exception as exc:
                logger.warning(
                    "SingleShotStep: failed to serialise outline for '%s': %s",
                    state.topic,
                    exc,
                )

        prompt = prompt_manager.render(
            "single_shot",
            language=state.language,
            topic=state.topic,
            target_words=str(target),
            target_words_min=str(target_words_min),
            target_words_max=str(target_words_max),
            beat_1=str(beat_1),
            beat_2=str(beat_2),
            beat_3=str(beat_3),
            beat_4=str(beat_4),
            tone=str(gen.tone),
            perspective=str(gen.perspective),
            register=str(gen.register),
            pacing=str(gen.pacing),
            audience=str(gen.audience),
            genres=", ".join(gen.genres),
            dialog_density=str(gen.dialog_density),
            voiceover_rules=voiceover_rules,
            story_bible=story_bible_text,
            outline_context=outline_context,
        )

        # Use x3 multiplier to ensure the model is never truncated mid-story.
        max_tokens = max(8192, target * 3)

        logger.debug("SingleShotStep: prompt %d chars, max_tokens=%d", len(prompt), max_tokens)

        try:
            draft_text = await api_client.send(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise StepError(
                f"SingleShot API call failed for '{state.topic}': {exc}",
                step_name=self.name, recoverable=False,
            ) from exc

        if not isinstance(draft_text, str):
            draft_text = getattr(draft_text, "text", str(draft_text))

        self._save_artifact(state, "draft_v1.txt", draft_text)

        state_mgr = StateManager()
        state = state_mgr.add_draft(state, draft_text, version=1)

        word_count = len(draft_text.split())
        logger.info("SingleShotStep complete for '%s': %d words (target %d)", state.topic, word_count, target)

        if word_count < target_words_min:
            logger.warning(
                "SingleShotStep: output %d words below minimum %d for '%s' — evaluation will flag this",
                word_count, target_words_min, state.topic,
            )
        return state

    @staticmethod
    def _build_voiceover_rules(gen: "object") -> str:
        """Build voiceover rules string from generation config.

        Args:
            gen: GenerationConfig instance.

        Returns:
            Voiceover instructions string.
        """
        rules: list[str] = []
        if getattr(gen, "voiceover_optimized", True):
            rules.append("Write text optimized for voiceover/TTS narration.")
        if getattr(gen, "avoid_complex_sentences", True):
            rules.append("Use simple, clear sentences. Avoid nested clauses.")
        if getattr(gen, "no_headers", True):
            rules.append("Do NOT include chapter headers, section markers, or titles anywhere in the story text.")
        if getattr(gen, "no_meta_comments", True):
            rules.append("Do NOT include meta-comments, author's notes, or AI commentary.")
        if getattr(gen, "pause_markers", True):
            rules.append("Use natural punctuation to create pauses: commas, periods, ellipses, em dashes.")
        return "\n".join(rules) if rules else "Write naturally for voiceover."
