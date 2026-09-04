"""Plain-text exporter for AI Story Generator Pro.

``TxtExporter`` writes story text as a clean UTF-8 ``.txt`` file with
normalised line endings, no BOM, and a trailing newline.

Typical usage::

    exporter = TxtExporter()
    path = exporter.export(
        text="The story text ...",
        output_path=Path("output/en/story.txt"),
        language="en",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.exceptions import ExportError
from utils.file_handler import write_file

logger = logging.getLogger(__name__)

# Unicode BOM that should be stripped.
_BOM = "\ufeff"


class TxtExporter:
    """Exports story text as a clean UTF-8 ``.txt`` file.

    Ensures:
    - No BOM (byte-order mark).
    - Normalised line endings (``\\n``).
    - Trailing newline at the end of the file.
    - Non-empty output.
    """

    def export(
        self,
        text: str,
        output_path: str | Path,
        language: str,
    ) -> Path:
        """Write text to a ``.txt`` file.

        Args:
            text: The story text to export.
            output_path: Target file path.
            language: Two-letter language code (for logging/metadata).

        Returns:
            The ``Path`` of the written file.

        Raises:
            ExportError: If the text is empty or the write fails.
        """
        if not text or not text.strip():
            raise ExportError(
                "Cannot export empty text to TXT",
                export_format="txt",
            )

        cleaned = self._clean_text(text)

        logger.info(
            "TxtExporter: exporting %d chars (%s) to %s",
            len(cleaned),
            language,
            output_path,
        )

        try:
            result_path = write_file(output_path, cleaned)
        except OSError as exc:
            raise ExportError(
                f"Failed to write TXT file {output_path}: {exc}",
                export_format="txt",
            ) from exc

        logger.info(
            "TxtExporter: export complete: %s (%d chars)",
            result_path,
            len(cleaned),
        )
        return result_path

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for TXT export.

        Removes BOM, normalises line endings to ``\\n``, strips trailing
        whitespace from each line, and ensures a single trailing newline.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text ready for file writing.
        """
        result = text

        # Strip BOM if present.
        if result.startswith(_BOM):
            result = result[len(_BOM):]

        # Normalise line endings: \r\n → \n, lone \r → \n.
        result = result.replace("\r\n", "\n").replace("\r", "\n")

        # Strip trailing whitespace from each line.
        lines = result.split("\n")
        lines = [line.rstrip() for line in lines]
        result = "\n".join(lines)

        # Strip leading/trailing blank lines but keep internal structure.
        result = result.strip("\n")

        # Ensure trailing newline.
        if result and not result.endswith("\n"):
            result += "\n"

        return result
