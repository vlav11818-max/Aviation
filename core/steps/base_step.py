"""Abstract base class for pipeline steps.

Every pipeline step inherits ``BaseStep`` and implements the async
``execute()`` method.  The base class provides concrete helper methods
for saving artefacts and emitting progress events.

Typical subclass::

    class ConceptStep(BaseStep):
        @property
        def name(self) -> str:
            return "concept"

        @property
        def description(self) -> str:
            return "Generate story concept and Story Bible"

        async def execute(
            self,
            state: PipelineState,
            api_client: APIClient,
            prompt_manager: PromptManager,
            event_bus: EventBus,
        ) -> PipelineState:
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from core.events import EventBus, EventType
from core.exceptions import StepError
from models.state import PipelineState
from utils.file_handler import write_file

if TYPE_CHECKING:
    from core.api_client import APIClient
    from core.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """Abstract base class for all pipeline steps.

    Subclasses must implement:
    - ``name`` property: short identifier (e.g. ``"concept"``).
    - ``description`` property: human-readable description.
    - ``execute()`` coroutine: performs the step's work.

    Concrete helper methods:
    - ``_save_artifact()``: persist an intermediate file.
    - ``_emit_progress()``: emit a progress event via the event bus.
    """

    # ── Abstract interface ─────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this step (e.g. ``"concept"``, ``"outline"``)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this step does."""

    @abstractmethod
    async def execute(
        self,
        state: PipelineState,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
    ) -> PipelineState:
        """Run the step's logic and return the updated state.

        Each step receives the current pipeline state, the unified API
        client, the prompt manager, and the event bus.  It should
        perform its work, update the state via the appropriate
        mutations, and return the updated state.

        Args:
            state: Current pipeline state (mutable).
            api_client: Unified API client for LLM calls.
            prompt_manager: Template loader and renderer.
            event_bus: Event bus for GUI communication.

        Returns:
            The updated ``PipelineState`` after this step completes.

        Raises:
            StepError: If the step fails.  Set ``recoverable=True``
                if the pipeline can continue past this failure.
        """

    # ── Concrete helpers ───────────────────────────────────────────────

    def _save_artifact(
        self,
        state: PipelineState,
        filename: str,
        content: str,
    ) -> Path:
        """Save an intermediate artefact to the story's output directory.

        Args:
            state: Current pipeline state (provides ``output_dir``).
            filename: Name of the file to create (e.g. ``"concept.json"``).
            content: Text content to write.

        Returns:
            Path to the written file.

        Raises:
            StepError: If no output directory is set or the write fails.
        """
        if not state.output_dir:
            raise StepError(
                f"Cannot save artifact '{filename}': no output_dir on state",
                step_name=self.name,
                recoverable=False,
            )

        artifact_path = Path(state.output_dir) / filename
        try:
            result_path = write_file(artifact_path, content)
        except OSError as exc:
            raise StepError(
                f"Failed to write artifact '{filename}': {exc}",
                step_name=self.name,
                recoverable=True,
            ) from exc

        logger.debug(
            "Step '%s' saved artifact: %s (%d chars)",
            self.name,
            result_path,
            len(content),
        )
        return result_path

    def _emit_progress(
        self,
        event_bus: EventBus,
        state: PipelineState,
        message: str,
    ) -> None:
        """Emit a progress event for this step.

        Args:
            event_bus: Event bus to emit on.
            state: Current pipeline state (provides topic and step info).
            message: Human-readable progress message.
        """
        event_bus.emit(
            EventType.STEP_COMPLETED,
            step=self.name,
            topic=state.topic,
            story_id=state.story_id,
            message=message,
        )
        logger.debug(
            "Step '%s' progress for '%s': %s",
            self.name,
            state.topic,
            message,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(name={self.name!r})>"
