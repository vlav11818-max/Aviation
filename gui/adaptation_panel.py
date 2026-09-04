"""Adaptation settings panel for AI Story Generator Pro GUI.

Replaces the ``StylePanel`` content area when adaptation mode is
selected.  Provides source folder file list with auto-detected
language and confidence per file, target language selection,
adaptation mode radio buttons (literal / cultural / free), parameter
checkboxes, and a post-adaptation evaluation toggle.

The panel exposes ``get_config()`` which returns an ``AdaptationConfig``
dictionary compatible with ``TextAdapter``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import customtkinter as ctk

from gui.styles import Colors, Fonts, Padding, create_section_label, create_separator
from models.config import LANGUAGES, SUPPORTED_LANGUAGE_CODES

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────

_MODE_LITERAL: str = "literal"
_MODE_CULTURAL: str = "cultural"
_MODE_FREE: str = "free"

_MODE_LABELS: dict[str, str] = {
    _MODE_LITERAL: "Literal (faithful translation)",
    _MODE_CULTURAL: "Cultural (localised adaptation)",
    _MODE_FREE: "Free (creative reimagining)",
}


class AdaptationPanel(ctk.CTkFrame):
    """Adaptation configuration panel.

    Shown instead of the style panel when the user selects
    "Adapt" mode.  Contains source file information, target
    language selection, adaptation mode, and parameter toggles.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(parent, fg_color=Colors.SURFACE, **kwargs)

        # State
        self._mode_var = ctk.StringVar(value=_MODE_CULTURAL)
        self._target_lang_vars: dict[str, ctk.BooleanVar] = {}
        self._param_vars: dict[str, ctk.BooleanVar] = {}
        self._eval_var = ctk.BooleanVar(value=True)

        # Source file data (populated externally)
        self._source_files: list[dict[str, Any]] = []

        self._build_ui()
        logger.debug("AdaptationPanel initialised")

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all child widgets."""
        self.columnconfigure(0, weight=1)
        row = 0

        # Section header
        header = create_section_label(self, "ADAPTATION", icon="\U0001F310")
        header.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.PANEL, Padding.WIDGET_Y))
        row += 1

        sep = create_separator(self)
        sep.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        # ── Source files display ────────────────────────────────────
        source_label = ctk.CTkLabel(
            self, text="Source Files",
            font=Fonts.get(bold=True), text_color=Colors.TEXT, anchor="w",
        )
        source_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        self._source_scroll = ctk.CTkScrollableFrame(
            self, fg_color=Colors.SURFACE_LIGHT, height=100,
        )
        self._source_scroll.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        self._source_scroll.columnconfigure(0, weight=1)

        self._source_empty_label = ctk.CTkLabel(
            self._source_scroll,
            text="No source folder selected",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED,
        )
        self._source_empty_label.grid(row=0, column=0)
        row += 1

        # ── Adaptation mode ─────────────────────────────────────────
        sep2 = create_separator(self)
        sep2.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        row += 1

        mode_label = ctk.CTkLabel(
            self, text="Adaptation Mode",
            font=Fonts.get(bold=True), text_color=Colors.TEXT, anchor="w",
        )
        mode_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(0, Padding.LABEL_Y))
        row += 1

        mode_frame = ctk.CTkFrame(self, fg_color="transparent")
        mode_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)

        for idx, (mode_val, mode_text) in enumerate(_MODE_LABELS.items()):
            rb = ctk.CTkRadioButton(
                mode_frame,
                text=mode_text,
                variable=self._mode_var,
                value=mode_val,
                font=Fonts.get(size=Fonts.SMALL),
                text_color=Colors.TEXT,
                fg_color=Colors.PRIMARY,
                hover_color=Colors.PRIMARY_HOVER,
            )
            rb.grid(row=idx, column=0, sticky="w", pady=2)
        row += 1

        # ── Parameter checkboxes ────────────────────────────────────
        sep3 = create_separator(self)
        sep3.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        row += 1

        params_label = ctk.CTkLabel(
            self, text="Adaptation Parameters",
            font=Fonts.get(bold=True), text_color=Colors.TEXT, anchor="w",
        )
        params_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(0, Padding.LABEL_Y))
        row += 1

        param_defs = [
            ("adapt_names", "Adapt character names", True),
            ("adapt_references", "Adapt cultural references", True),
            ("adapt_units", "Convert units of measurement", True),
            ("adapt_setting", "Relocate setting to target culture", False),
            ("preserve_length", "Preserve approximate length", True),
            ("voiceover_optimize", "Optimise for voiceover (TTS)", True),
        ]

        params_frame = ctk.CTkFrame(self, fg_color="transparent")
        params_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)

        for pidx, (key, label, default) in enumerate(param_defs):
            var = ctk.BooleanVar(value=default)
            self._param_vars[key] = var
            cb = ctk.CTkCheckBox(
                params_frame,
                text=label,
                variable=var,
                font=Fonts.get(size=Fonts.SMALL),
                text_color=Colors.TEXT,
                fg_color=Colors.PRIMARY,
                hover_color=Colors.PRIMARY_HOVER,
                checkmark_color=Colors.TEXT,
            )
            cb.grid(row=pidx, column=0, sticky="w", pady=2)
        row += 1

        # ── Post-adaptation evaluation ──────────────────────────────
        sep4 = create_separator(self)
        sep4.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        row += 1

        self._eval_cb = ctk.CTkCheckBox(
            self,
            text="Run evaluation after adaptation",
            variable=self._eval_var,
            font=Fonts.get(size=Fonts.SMALL, bold=True),
            text_color=Colors.TEXT,
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            checkmark_color=Colors.TEXT,
        )
        self._eval_cb.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(0, Padding.PANEL))

    # ── Source files display ────────────────────────────────────────

    def set_source_files(self, files: list[dict[str, Any]]) -> None:
        """Populate the source files list.

        Each entry should have at minimum: ``filename``, ``detected_language``,
        ``confidence``, ``is_supported``, ``char_count``.

        Args:
            files: List of file information dictionaries (from
                ``InputValidator.validate_folder``).
        """
        self._source_files = files

        # Clear existing content
        for widget in self._source_scroll.winfo_children():
            widget.destroy()

        if not files:
            empty = ctk.CTkLabel(
                self._source_scroll,
                text="No files found",
                font=Fonts.get(size=Fonts.SMALL),
                text_color=Colors.MUTED,
            )
            empty.grid(row=0, column=0)
            return

        # Header
        headers = ["File", "Language", "Conf.", "Chars"]
        for col, h in enumerate(headers):
            self._source_scroll.columnconfigure(col, weight=1 if col == 0 else 0)
            ctk.CTkLabel(
                self._source_scroll, text=h,
                font=Fonts.get(size=Fonts.TINY, bold=True),
                text_color=Colors.MUTED,
                anchor="w" if col == 0 else "center",
            ).grid(row=0, column=col, sticky="w" if col == 0 else "", padx=3, pady=1)

        for idx, finfo in enumerate(files):
            frow = idx + 1
            filename = finfo.get("filename", "?")
            lang = finfo.get("detected_language", "?")
            confidence = finfo.get("confidence", 0.0)
            is_supported = finfo.get("is_supported", False)
            chars = finfo.get("char_count", 0)
            error = finfo.get("error", "")

            name_colour = Colors.TEXT if not error else Colors.ERROR
            lang_display = lang.upper() if lang else "?"
            lang_colour = Colors.SUCCESS if is_supported else Colors.WARNING
            conf_text = f"{confidence:.0%}" if confidence > 0 else "—"

            ctk.CTkLabel(
                self._source_scroll, text=filename,
                font=Fonts.get(size=Fonts.TINY), text_color=name_colour, anchor="w",
            ).grid(row=frow, column=0, sticky="w", padx=3, pady=1)

            ctk.CTkLabel(
                self._source_scroll, text=lang_display,
                font=Fonts.get(size=Fonts.TINY), text_color=lang_colour, anchor="center",
            ).grid(row=frow, column=1, padx=3, pady=1)

            ctk.CTkLabel(
                self._source_scroll, text=conf_text,
                font=Fonts.get(size=Fonts.TINY), text_color=Colors.MUTED, anchor="center",
            ).grid(row=frow, column=2, padx=3, pady=1)

            ctk.CTkLabel(
                self._source_scroll, text=f"{chars:,}",
                font=Fonts.get(size=Fonts.TINY), text_color=Colors.MUTED, anchor="center",
            ).grid(row=frow, column=3, padx=3, pady=1)

        logger.debug("AdaptationPanel: displayed %d source files", len(files))

    # ── Public API ──────────────────────────────────────────────────

    def get_mode(self) -> str:
        """Return the selected adaptation mode.

        Returns:
            One of ``"literal"``, ``"cultural"``, ``"free"``.
        """
        return self._mode_var.get()

    def get_config(self) -> dict[str, Any]:
        """Return the adaptation configuration as a dictionary.

        The dictionary is compatible with ``TextAdapter`` and
        ``AdaptationParams``.

        Returns:
            Dictionary with keys: ``mode``, ``adapt_names``,
            ``adapt_references``, ``adapt_units``, ``adapt_setting``,
            ``preserve_length``, ``voiceover_optimize``,
            ``run_evaluation``.
        """
        config = {
            "mode": self._mode_var.get(),
            "run_evaluation": self._eval_var.get(),
        }
        for key, var in self._param_vars.items():
            config[key] = var.get()

        logger.debug("AdaptationPanel config: %s", config)
        return config

    def reset(self) -> None:
        """Reset all controls to default values."""
        self._mode_var.set(_MODE_CULTURAL)
        self._eval_var.set(True)

        defaults = {
            "adapt_names": True,
            "adapt_references": True,
            "adapt_units": True,
            "adapt_setting": False,
            "preserve_length": True,
            "voiceover_optimize": True,
        }
        for key, var in self._param_vars.items():
            var.set(defaults.get(key, True))

        # Clear source files
        for widget in self._source_scroll.winfo_children():
            widget.destroy()
        ctk.CTkLabel(
            self._source_scroll,
            text="No source folder selected",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED,
        ).grid(row=0, column=0)

        self._source_files = []
        logger.debug("AdaptationPanel reset to defaults")
