"""Section stitching step.

Concatenates all completed section texts into a single document.
Identifies seam points (the boundary between consecutive sections)
and uses the LLM to smooth each transition.  The smoothed result is
saved as ``draft_v1.txt`` and stored as the first draft on the state.
"""

from __future__ import annotations

import logging
import re
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

# Number of words to extract from each side of a seam.
_SEAM_WORDS = 200


class StitchStep(BaseStep):
    """Stitch sections together and smooth seam points.

    After section-by-section generation, adjacent sections may have
    abrupt transitions.  This step concatenates all sections, identifies
    each seam, and asks the LLM to produce a smoother transition in the
    overlap zone.  Only the seam area is replaced — the bulk of each
    section is preserved.
    """

    @property
    def name(self) -> str:
        return "stitch"

    @property
    def description(self) -> str:
        return "Concatenate sections and smooth seam transitions"

    async def execute(
        self,
        state: PipelineState,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        event_bus: EventBus,
    ) -> PipelineState:
        """Run stitching.

        Args:
            state: Current pipeline state (must have completed sections).
            api_client: Unified API client.
            prompt_manager: Template loader/renderer.
            event_bus: Event bus for GUI communication.

        Returns:
            Updated state with ``drafts[0]`` containing the stitched text.

        Raises:
            StepError: If no sections are available or stitching fails.
        """
        logger.info("StitchStep: stitching sections for '%s'", state.topic)

        if not state.sections_completed:
            raise StepError(
                f"StitchStep: no completed sections for '{state.topic}'",
                step_name=self.name,
                recoverable=False,
            )
        # Sort sections by index.
        sections = sorted(state.sections_completed, key=lambda s: s.index)

        # If only one section, no stitching needed.
        if len(sections) == 1:
            draft_text = sections[0].text
        else:
            # Build list of section texts.
            section_texts = [s.text for s in sections]

            # Smooth each seam.
            smoothed_texts = list(section_texts)
            for i in range(len(section_texts) - 1):
                end_text = section_texts[i]
                start_text = section_texts[i + 1]

                smoothed = await self._smooth_seam(
                    api_client=api_client,
                    prompt_manager=prompt_manager,
                    end_text=end_text,
                    start_text=start_text,
                    language=state.language,
                    story_bible_summary=self._get_bible_summary(state),
                    seam_index=i,
                )

                if smoothed is not None:
                    # Replace the tail of section i and head of section i+1.
                    smoothed_texts[i] = self._replace_tail(
                        smoothed_texts[i], smoothed, "end"
                    )
                    smoothed_texts[i + 1] = self._replace_head(
                        smoothed_texts[i + 1], smoothed, "start"
                    )

            # Concatenate with double newlines between sections.
            draft_text = "\n\n".join(smoothed_texts)

        # Save artifact.
        # Save with topic-based filename for easy identification.
        topic_filename = self._sanitize_filename(state.topic) + ".txt"
        self._save_artifact(state, topic_filename, draft_text)
        # Also save as draft_v1.txt for internal pipeline tracking.
        self._save_artifact(state, "draft_v1.txt", draft_text)

        # Update state.
        state_mgr = StateManager()
        state = state_mgr.add_draft(state, draft_text, version=1)

        word_count = len(draft_text.split())
        logger.info(
            "StitchStep complete for '%s': %d sections, %d words",
            state.topic,
            len(sections),
            word_count,
        )
        return state

    @staticmethod
    def _sanitize_filename(topic: str) -> str:
        """Convert a topic string into a safe filename.

        Takes the part before ' — ' or ' - ' (the short title) if present,
        removes unsafe characters, and replaces spaces with underscores.

        Args:
            topic: The raw topic string.

        Returns:
            A filesystem-safe filename (without extension).
        """
        # Take the short title before any dash separator.
        for sep in (" — ", " – ", " - ", "\u2014", "\u2013"):
            if sep in topic:
                name = topic.split(sep)[0].strip()
                break
        else:
            name = topic[:80]
        # Remove filesystem-unsafe characters.
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = re.sub(r'\s+', '_', name)
        name = name.strip('._')
        return name[:120] if name else "story"

    async def _smooth_seam(
        self,
        api_client: "APIClient",
        prompt_manager: "PromptManager",
        end_text: str,
        start_text: str,
        language: str,
        story_bible_summary: str,
        seam_index: int,
    ) -> str | None:
        """Ask the LLM to smooth a single seam between two sections.

        Extracts the last ~200 words of the ending section and the first
        ~200 words of the starting section, then asks the LLM to rewrite
        just that overlap for a smoother transition.

        Args:
            api_client: Unified API client.
            prompt_manager: Template loader/renderer.
            end_text: Full text of the section before the seam.
            start_text: Full text of the section after the seam.
            language: Two-letter language code.
            story_bible_summary: Brief story bible summary for context.
            seam_index: Zero-based index of the seam (for logging).

        Returns:
            The smoothed text for the overlap zone, or ``None`` on failure.
        """
        end_words = end_text.split()
        start_words = start_text.split()

        end_of_previous = " ".join(end_words[-_SEAM_WORDS:])
        start_of_next = " ".join(start_words[:_SEAM_WORDS])

        prompt = prompt_manager.render(
            "stitching",
            language=language,
            end_of_previous_section=end_of_previous,
            start_of_next_section=start_of_next,
            story_bible_summary=story_bible_summary,
        )

        try:
            smoothed = await api_client.send(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
            )
            logger.debug("Seam %d smoothed successfully", seam_index)
            return smoothed.strip()
        except Exception as exc:
            logger.warning(
                "StitchStep: seam %d smoothing failed, keeping original: %s",
                seam_index,
                exc,
            )
            return None

    @staticmethod
    def _get_bible_summary(state: PipelineState) -> str:
        """Build a brief Story Bible summary for seam context.

        Args:
            state: Current pipeline state.

        Returns:
            A short summary string.
        """
        if state.story_bible is None:
            return ""
        sb = state.story_bible
        parts = []
        if sb.premise:
            parts.append(f"Premise: {sb.premise}")
        if sb.tone_description:
            parts.append(f"Tone: {sb.tone_description}")
        if sb.narrative_voice:
            parts.append(f"Voice: {sb.narrative_voice}")
        return " | ".join(parts) if parts else ""

    @staticmethod
    def _replace_tail(text: str, smoothed: str, _marker: str) -> str:
        """Replace the trailing ~200 words of *text* with the first half
        of the smoothed text.

        In practice, seam smoothing is advisory — we keep the smoothed
        overlap as a whole rather than surgically splitting it.  The
        simple approach: return original text up to the seam area, then
        trust the LLM's version.

        Args:
            text: Original section text.
            smoothed: Smoothed seam text from the LLM.
            _marker: Unused marker for API symmetry.

        Returns:
            Section text with the tail potentially improved.
        """
        words = text.split()
        if len(words) <= _SEAM_WORDS:
            return text
        # Keep everything before the seam zone.
        kept = " ".join(words[:-_SEAM_WORDS])
        # Use the first half of the smoothed text as the new tail.
        smoothed_words = smoothed.split()
        half = len(smoothed_words) // 2
        new_tail = " ".join(smoothed_words[:half]) if half > 0 else smoothed
        return kept + " " + new_tail

    @staticmethod
    def _replace_head(text: str, smoothed: str, _marker: str) -> str:
        """Replace the leading ~200 words of *text* with the second half
        of the smoothed text.

        Args:
            text: Original section text.
            smoothed: Smoothed seam text from the LLM.
            _marker: Unused marker for API symmetry.

        Returns:
            Section text with the head potentially improved.
        """
        words = text.split()
        if len(words) <= _SEAM_WORDS:
            return text
        # Keep everything after the seam zone.
        kept = " ".join(words[_SEAM_WORDS:])
        # Use the second half of the smoothed text as the new head.
        smoothed_words = smoothed.split()
        half = len(smoothed_words) // 2
        new_head = " ".join(smoothed_words[half:]) if half > 0 else smoothed
        return new_head + " " + kept
