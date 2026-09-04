"""Outline generation step.

Renders the outline.txt template with the concept JSON, the selected
structure template, and target word count. Sends the prompt to the LLM,
parses the JSON response into an Outline with sections, saves outline.json,
and updates the pipeline state.

FIX: StructureType is a str-enum. str(StructureType.THREE_ACT) returns
"StructureType.THREE_ACT", breaking file lookup. Use .value instead to
get the plain string "three_act" correctly.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from core.events import EventBus, EventType
from core.exceptions import StepError
from core.state_manager import StateManager
from core.steps.base_step import BaseStep
from models.outline import Outline
from models.state import PipelineState

if TYPE_CHECKING:
    from core.api_client import APIClient
    from core.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class OutlineStep(BaseStep):
    """Generate a structural outline for the story.

    Uses the Story Bible from the concept step and the selected
    structure template to produce an ordered list of sections with
    act labels, target words, key events, and transitions.
    """

    @property
    def name(self) -> str:
        return "outline"

    @property
    def description(self) -> str:
        return "Generate structural outline with section plan"

    async def execute(
        self,
        state: PipelineState,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
    ) -> PipelineState:
        """Run outline generation.

        Args:
            state: Current pipeline state (must have story_bible).
            api_client: Unified API client.
            prompt_manager: Template loader/renderer.
            event_bus: Event bus for GUI communication.

        Returns:
            Updated state with outline populated.

        Raises:
            StepError: If Story Bible is missing, the API call fails,
                or the response cannot be parsed.
        """
        logger.info("OutlineStep: generating outline for '%s'", state.topic)

        if state.story_bible is None:
            raise StepError(
                f"OutlineStep requires a story_bible, but it is not set for '{state.topic}'",
                step_name=self.name,
                recoverable=False,
            )
        gen = state.generation_config

        # FIX: StructureType is a str-enum. Use .value to get the plain
        # string (e.g. "three_act"), NOT str() which returns
        # "StructureType.THREE_ACT" and breaks the file lookup.
        raw_structure = gen.structure if hasattr(gen, "structure") else "three_act"
        structure_name: str = (
            raw_structure.value
            if hasattr(raw_structure, "value")
            else str(raw_structure)
        )

        try:
            structure_template = prompt_manager.get_structure_template(structure_name)
            logger.debug("OutlineStep: loaded structure template '%s'", structure_name)
        except Exception as exc:
            logger.warning(
                "OutlineStep: failed to load structure template '%s', "
                "continuing without: %s",
                structure_name, exc,
            )
            structure_template = {}

        concept_json = state.story_bible.model_dump_json(indent=2)
        structure_json = json.dumps(structure_template, indent=2, ensure_ascii=False)
        estimated_sections = max(3, gen.target_words // 800)

        prompt = prompt_manager.render(
            "outline",
            language=state.language,
            concept_json=concept_json,
            structure_template=structure_json,
            structure_name=structure_name,
            target_words=str(gen.target_words),
            num_sections=str(estimated_sections),
        )

        try:
            response_text = await api_client.send(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4096,
            )
        except Exception as exc:
            raise StepError(
                f"Outline API call failed for '{state.topic}': {exc}",
                step_name=self.name,
                recoverable=False,
            ) from exc

        if not isinstance(response_text, str):
            response_text = getattr(response_text, "text", str(response_text))

        outline = self._parse_response(response_text, state.topic)

        outline_data = outline.model_dump()
        self._save_artifact(
            state, "outline.json",
            json.dumps(outline_data, indent=2, ensure_ascii=False),
        )

        state_mgr = StateManager()
        state = state_mgr.update_outline(state, outline)
        logger.info(
            "OutlineStep complete for '%s': %d sections, structure=%s",
            state.topic, len(outline.sections), outline.structure_type,
        )
        return state

    def _parse_response(self, response_text: str, topic: str) -> Outline:
        """Extract and validate an Outline from the API response.

        Args:
            response_text: Raw text returned by the LLM.
            topic: Topic string for error messages.

        Returns:
            A validated Outline instance.

        Raises:
            StepError: If the response is not valid JSON or does not
                match the Outline schema.
        """
        cleaned = _extract_json(response_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise StepError(
                f"Outline response for '{topic}' is not valid JSON: {exc}",
                step_name=self.name, recoverable=False,
            ) from exc

        try:
            outline = Outline.model_validate(data)
        except Exception as exc:
            raise StepError(
                f"Outline response for '{topic}' does not match Outline schema: {exc}",
                step_name=self.name, recoverable=False,
            ) from exc

        if not outline.sections:
            raise StepError(
                f"Outline for '{topic}' has no sections",
                step_name=self.name, recoverable=False,
            )
        return outline


def _extract_json(text: str) -> str:
    """Strip markdown code fences from LLM output if present.

    Args:
        text: Raw LLM response text.

    Returns:
        Cleaned text likely to be valid JSON.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()
