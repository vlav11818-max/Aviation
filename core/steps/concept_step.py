"""Concept generation step.

Renders the ``concept.txt`` template with topic, language, style
parameters, and cultural instructions.  Sends the prompt to the LLM
API, parses the JSON response into a ``StoryBible``, saves
``concept.json`` to the output directory, and updates the pipeline
state with the Story Bible.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from core.events import EventBus, EventType
from core.exceptions import StepError
from core.state_manager import StateManager
from core.steps.base_step import BaseStep
from models.state import PipelineState
from models.story_bible import StoryBible

if TYPE_CHECKING:
    from core.api_client import APIClient
    from core.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class ConceptStep(BaseStep):
    """Generate a story concept and Story Bible from the topic.

    Produces a JSON object containing premise, setting, characters,
    themes, tone description, narrative voice, and key rules.  The
    parsed result is stored as a ``StoryBible`` on the pipeline state.
    """

    @property
    def name(self) -> str:
        return "concept"

    @property
    def description(self) -> str:
        return "Generate story concept and Story Bible"

    async def execute(
        self,
        state: PipelineState,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
    ) -> PipelineState:
        """Run concept generation.

        Args:
            state: Current pipeline state.
            api_client: Unified API client.
            prompt_manager: Template loader/renderer.
            event_bus: Event bus for GUI communication.

        Returns:
            Updated state with ``story_bible`` populated.

        Raises:
            StepError: If the API call fails or the response cannot
                be parsed into a valid ``StoryBible``.
        """
        logger.info("ConceptStep: generating concept for '%s'", state.topic)
        # Build prompt from template.
        gen = state.generation_config
        prompt = prompt_manager.render(
            "concept",
            language=state.language,
            topic=state.topic,
            tone=gen.tone,
            perspective=gen.perspective,
            register=gen.register,
            pacing=gen.pacing,
            audience=gen.audience,
            genres=", ".join(gen.genres),
            dialog_density=gen.dialog_density,
            target_words=str(gen.target_words),
            native_language_name=self._get_native_name(state.language),
        )

        # Call LLM.
        try:
            response_text = await api_client.send(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=4096,
            )
        except Exception as exc:
            raise StepError(
                f"Concept API call failed for '{state.topic}': {exc}",
                step_name=self.name,
                recoverable=False,
            ) from exc

        # Parse JSON from response.
        story_bible = self._parse_response(response_text, state.topic)

        # Save artifact.
        concept_data = story_bible.model_dump()
        self._save_artifact(
            state, "concept.json", json.dumps(concept_data, indent=2, ensure_ascii=False)
        )

        # Update state.
        state_mgr = StateManager()
        state = state_mgr.update_story_bible(state, story_bible)
        logger.info(
            "ConceptStep complete for '%s': premise='%s...'",
            state.topic,
            story_bible.premise[:60] if story_bible.premise else "",
        )
        return state

    def _parse_response(self, response_text: str, topic: str) -> StoryBible:
        """Extract and validate a StoryBible from the API response.

        The LLM may wrap JSON in markdown fences; this method strips
        them before parsing.

        Args:
            response_text: Raw text returned by the LLM.
            topic: Topic string for error messages.

        Returns:
            A validated ``StoryBible`` instance.

        Raises:
            StepError: If the response is not valid JSON or does not
                conform to the ``StoryBible`` schema.
        """
        cleaned = _extract_json(response_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise StepError(
                f"Concept response for '{topic}' is not valid JSON: {exc}",
                step_name=self.name,
                recoverable=False,
            ) from exc

        try:
            story_bible = StoryBible.model_validate(data)
        except Exception as exc:
            raise StepError(
                f"Concept response for '{topic}' does not match StoryBible schema: {exc}",
                step_name=self.name,
                recoverable=False,
            ) from exc

        return story_bible

    @staticmethod
    def _get_native_name(language: str) -> str:
        """Return the native name for a language code.

        Args:
            language: Two-letter language code.

        Returns:
            Native name string, or the code itself as fallback.
        """
        from models.config import LANGUAGES

        lang_info = LANGUAGES.get(language, {})
        return lang_info.get("native", language)


def _extract_json(text: str) -> str:
    """Strip markdown code fences from LLM output if present.

    Handles ``\\`\\`\\`json ... \\`\\`\\`` and ``\\`\\`\\` ... \\`\\`\\``.

    Args:
        text: Raw LLM response text.

    Returns:
        Cleaned text likely to be valid JSON.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (possibly with language tag).
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        # Remove closing fence.
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()
