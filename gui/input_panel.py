"""Input panel for AI Story Generator Pro GUI.

Provides mode selection (Generate / Adapt), file and folder pickers,
language selection (11 languages with flags), and multi-language target
checkboxes for adaptation mode.

Adaptation mode supports both single-file and folder selection, with
automatic source language detection via ``LanguageDetector``.

The panel exposes ``get_config()`` which returns a dictionary of the
current user selections.
"""

from __future__ import annotations

import logging
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable

import customtkinter as ctk

from gui.styles import Colors, Fonts, Padding, create_section_label, create_separator
from models.config import LANGUAGES, SUPPORTED_LANGUAGE_CODES

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────

_MODE_GENERATE: str = "generate"
_MODE_ADAPT: str = "adapt"


class InputPanel(ctk.CTkFrame):
    """Input configuration panel.

    Contains mode radio buttons (Generate / Adapt), file picker for
    topic files, folder picker for adaptation source, single-file picker
    for adaptation, primary language dropdown, auto-detected source
    language display, and multi-language checkboxes for adaptation targets.

    Args:
        parent: Parent widget.
        on_mode_change: Optional callback invoked when mode changes.
            Receives the new mode string (``"generate"`` or ``"adapt"``).
    """

    def __init__(
        self,
        parent: Any,
        on_mode_change: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, fg_color=Colors.SURFACE, **kwargs)
        self._on_mode_change = on_mode_change

        # ── State variables ─────────────────────────────────────────
        self._mode_var = ctk.StringVar(value=_MODE_GENERATE)
        self._topics_file_path: str = ""
        self._adapt_folder_path: str = ""
        self._adapt_file_path: str = ""
        self._adapt_source_files: list[Path] = []
        self._detected_source_lang: str = ""
        self._language_var = ctk.StringVar(value="en")
        self._target_lang_vars: dict[str, ctk.BooleanVar] = {}

        self._build_ui()
        logger.debug("InputPanel initialised")

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all child widgets."""
        self.columnconfigure(0, weight=1)
        row = 0

        # Section header
        header = create_section_label(self, "INPUT", icon="\U0001F4C1")
        header.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.PANEL, Padding.WIDGET_Y))
        row += 1

        sep = create_separator(self)
        sep.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        # ── Mode selection ──────────────────────────────────────────
        mode_label = ctk.CTkLabel(
            self, text="Mode", font=Fonts.get(bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        mode_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        mode_frame = ctk.CTkFrame(self, fg_color="transparent")
        mode_frame.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=Padding.LABEL_Y)

        self._radio_generate = ctk.CTkRadioButton(
            mode_frame,
            text="Generate",
            variable=self._mode_var,
            value=_MODE_GENERATE,
            command=self._on_mode_changed,
            font=Fonts.get(),
            text_color=Colors.TEXT,
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
        )
        self._radio_generate.grid(row=0, column=0, padx=(0, Padding.GROUP_X))

        self._radio_adapt = ctk.CTkRadioButton(
            mode_frame,
            text="Adapt",
            variable=self._mode_var,
            value=_MODE_ADAPT,
            command=self._on_mode_changed,
            font=Fonts.get(),
            text_color=Colors.TEXT,
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
        )
        self._radio_adapt.grid(row=0, column=1)
        row += 1

        # ── File picker (topics file for Generate mode) ─────────────
        self._file_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._file_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.WIDGET_Y)
        self._file_frame.columnconfigure(1, weight=1)

        self._file_btn = ctk.CTkButton(
            self._file_frame,
            text="\U0001F4C4 Topics File",
            width=Padding.MIN_BUTTON_WIDTH,
            command=self._pick_topics_file,
            font=Fonts.get(),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
        )
        self._file_btn.grid(row=0, column=0, padx=(0, Padding.WIDGET_X))

        self._file_label = ctk.CTkLabel(
            self._file_frame,
            text="No file selected",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED,
            anchor="w",
        )
        self._file_label.grid(row=0, column=1, sticky="w")
        row += 1

        # ── Adapt: single-file picker ───────────────────────────────
        self._adapt_file_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._adapt_file_frame.columnconfigure(1, weight=1)

        self._adapt_file_btn = ctk.CTkButton(
            self._adapt_file_frame,
            text="\U0001F4C4 Source File",
            width=Padding.MIN_BUTTON_WIDTH,
            command=self._pick_adapt_file,
            font=Fonts.get(),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
        )
        self._adapt_file_btn.grid(row=0, column=0, padx=(0, Padding.WIDGET_X))

        self._adapt_file_label = ctk.CTkLabel(
            self._adapt_file_frame,
            text="No file selected",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED,
            anchor="w",
        )
        self._adapt_file_label.grid(row=0, column=1, sticky="w")

        self._adapt_file_row = row
        row += 1

        # ── Folder picker (adaptation source) ───────────────────────
        self._folder_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._folder_frame.columnconfigure(1, weight=1)

        self._folder_btn = ctk.CTkButton(
            self._folder_frame,
            text="\U0001F4C2 Source Folder",
            width=Padding.MIN_BUTTON_WIDTH,
            command=self._pick_adapt_folder,
            font=Fonts.get(),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
        )
        self._folder_btn.grid(row=0, column=0, padx=(0, Padding.WIDGET_X))

        self._folder_label = ctk.CTkLabel(
            self._folder_frame,
            text="No folder selected",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED,
            anchor="w",
        )
        self._folder_label.grid(row=0, column=1, sticky="w")

        # Initially hidden (shown only in Adapt mode)
        self._folder_row = row
        row += 1

        # ── Detected source language display (adapt mode only) ──────
        self._source_lang_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._source_lang_frame.columnconfigure(1, weight=1)

        source_lang_label = ctk.CTkLabel(
            self._source_lang_frame,
            text="Source language:",
            font=Fonts.get(size=Fonts.SMALL, bold=True),
            text_color=Colors.TEXT,
            anchor="w",
        )
        source_lang_label.grid(row=0, column=0, padx=(0, Padding.WIDGET_X))

        self._source_lang_value = ctk.CTkLabel(
            self._source_lang_frame,
            text="auto-detect on file selection",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED,
            anchor="w",
        )
        self._source_lang_value.grid(row=0, column=1, sticky="w")

        self._source_lang_row = row
        row += 1

        # ── Separator ───────────────────────────────────────────────
        sep2 = create_separator(self)
        sep2.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        row += 1

        # ── Language dropdown ───────────────────────────────────────
        self._lang_label = ctk.CTkLabel(
            self, text="Language", font=Fonts.get(bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        self._lang_label_row = row
        self._lang_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        lang_options = self._build_language_options()
        self._language_menu = ctk.CTkOptionMenu(
            self,
            variable=self._language_var,
            values=lang_options,
            command=self._on_language_changed,
            font=Fonts.get(),
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
            dropdown_fg_color=Colors.SURFACE,
            dropdown_hover_color=Colors.SURFACE_HOVER,
            dropdown_text_color=Colors.TEXT,
        )
        self._lang_menu_row = row
        self._language_menu.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        # Set initial display value
        self._language_var.set(self._format_language_option("en"))
        row += 1

        # ── Multi-language checkboxes (adaptation targets) ──────────
        self._target_langs_label = ctk.CTkLabel(
            self, text="Target Languages", font=Fonts.get(bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        self._target_langs_row = row
        row += 1

        self._target_langs_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=Colors.SURFACE_LIGHT,
            height=120,
        )
        self._target_langs_frame_row = row
        row += 1

        # Build checkboxes for all 11 languages
        for idx, code in enumerate(SUPPORTED_LANGUAGE_CODES):
            var = ctk.BooleanVar(value=False)
            self._target_lang_vars[code] = var
            info = LANGUAGES[code]
            display = f"{info['flag']} {info['name']}"
            cb = ctk.CTkCheckBox(
                self._target_langs_frame,
                text=display,
                variable=var,
                font=Fonts.get(size=Fonts.SMALL),
                text_color=Colors.TEXT,
                fg_color=Colors.PRIMARY,
                hover_color=Colors.PRIMARY_HOVER,
                checkmark_color=Colors.TEXT,
            )
            cb.grid(row=idx // 2, column=idx % 2, sticky="w", padx=Padding.WIDGET_X, pady=2)

        # Initially hidden (shown only in Adapt mode)
        self._update_mode_visibility()

    # ── Language helpers ─────────────────────────────────────────────

    @staticmethod
    def _format_language_option(code: str) -> str:
        """Format a language code as a display string with flag.

        Args:
            code: Two-letter language code.

        Returns:
            Display string like ``"🇬🇧 English (en)"``.
        """
        info = LANGUAGES.get(code, {})
        flag = info.get("flag", "")
        name = info.get("name", code)
        return f"{flag} {name} ({code})"

    @staticmethod
    def _parse_language_option(display: str) -> str:
        """Extract the language code from a display string.

        Args:
            display: Display string like ``"🇬🇧 English (en)"``.

        Returns:
            Two-letter language code.
        """
        # Extract code from the parenthesised suffix
        if "(" in display and display.endswith(")"):
            code = display.rsplit("(", 1)[1].rstrip(")")
            if code in SUPPORTED_LANGUAGE_CODES:
                return code
        # Fallback: try matching name
        for code, info in LANGUAGES.items():
            if info["name"] in display:
                return code
        return "en"

    def _build_language_options(self) -> list[str]:
        """Build the list of display strings for the language dropdown.

        Returns:
            List of formatted language option strings.
        """
        return [self._format_language_option(c) for c in SUPPORTED_LANGUAGE_CODES]

    # ── Callbacks ───────────────────────────────────────────────────

    def _on_mode_changed(self) -> None:
        """Handle mode radio button change."""
        mode = self._mode_var.get()
        logger.info("Input mode changed to: %s", mode)
        self._update_mode_visibility()
        if self._on_mode_change is not None:
            self._on_mode_change(mode)

    def _on_language_changed(self, value: str) -> None:
        """Handle language dropdown change.

        Args:
            value: The selected display string.
        """
        code = self._parse_language_option(value)
        logger.info("Primary language changed to: %s", code)

    def _update_mode_visibility(self) -> None:
        """Show or hide widgets based on current mode."""
        is_adapt = self._mode_var.get() == _MODE_ADAPT

        if is_adapt:
            # Hide topics file picker
            self._file_frame.grid_remove()
            # Show adapt file picker
            self._adapt_file_frame.grid(
                row=self._adapt_file_row, column=0, sticky="ew",
                padx=Padding.PANEL, pady=Padding.WIDGET_Y,
            )
            # Show folder picker
            self._folder_frame.grid(
                row=self._folder_row, column=0, sticky="ew",
                padx=Padding.PANEL, pady=Padding.WIDGET_Y,
            )
            # Show detected source language
            self._source_lang_frame.grid(
                row=self._source_lang_row, column=0, sticky="ew",
                padx=Padding.PANEL, pady=Padding.WIDGET_Y,
            )
            # Hide language dropdown (source lang is auto-detected)
            self._lang_label.grid_remove()
            self._language_menu.grid_remove()
            # Show target languages
            self._target_langs_label.grid(
                row=self._target_langs_row, column=0, sticky="w",
                padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y),
            )
            self._target_langs_frame.grid(
                row=self._target_langs_frame_row, column=0, sticky="ew",
                padx=Padding.PANEL, pady=Padding.LABEL_Y,
            )
        else:
            # Show topics file; hide adapt controls
            self._file_frame.grid(
                row=self._adapt_file_row, column=0, sticky="ew",
                padx=Padding.PANEL, pady=Padding.WIDGET_Y,
            )
            self._adapt_file_frame.grid_remove()
            self._folder_frame.grid_remove()
            self._source_lang_frame.grid_remove()
            # Show language dropdown
            self._lang_label.grid(
                row=self._lang_label_row, column=0, sticky="w",
                padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y),
            )
            self._language_menu.grid(
                row=self._lang_menu_row, column=0, sticky="ew",
                padx=Padding.PANEL, pady=Padding.LABEL_Y,
            )
            # Hide target languages
            self._target_langs_label.grid_remove()
            self._target_langs_frame.grid_remove()

    def _pick_topics_file(self) -> None:
        """Open a file dialog to select a topics .txt file."""
        path = filedialog.askopenfilename(
            title="Select Topics File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self._topics_file_path = path
            display_name = Path(path).name
            self._file_label.configure(text=display_name, text_color=Colors.TEXT)
            logger.info("Topics file selected: %s", path)

    def _pick_adapt_file(self) -> None:
        """Open a file dialog to select a single adaptation source file."""
        path = filedialog.askopenfilename(
            title="Select Source Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            file_path = Path(path)
            self._adapt_file_path = path
            # Clear folder selection (file takes priority)
            self._adapt_folder_path = ""
            self._folder_label.configure(
                text="No folder selected", text_color=Colors.MUTED,
            )
            # Update source files list
            self._adapt_source_files = [file_path]
            display_name = file_path.name
            self._adapt_file_label.configure(
                text=display_name, text_color=Colors.TEXT,
            )
            logger.info("Adaptation file selected: %s", path)
            # Auto-detect source language
            self._detect_source_language(self._adapt_source_files)

    def _pick_adapt_folder(self) -> None:
        """Open a folder dialog to select an adaptation source folder."""
        path = filedialog.askdirectory(
            title="Select Adaptation Source Folder",
        )
        if path:
            self._adapt_folder_path = path
            # Clear single file selection (folder takes priority)
            self._adapt_file_path = ""
            self._adapt_file_label.configure(
                text="No file selected", text_color=Colors.MUTED,
            )
            # Collect .txt files
            folder = Path(path)
            txt_files = sorted(folder.glob("*.txt"))
            self._adapt_source_files = txt_files
            file_count = len(txt_files)
            display = f"{folder.name} ({file_count} .txt files)"
            self._folder_label.configure(text=display, text_color=Colors.TEXT)
            logger.info(
                "Adaptation folder selected: %s (%d files)", path, file_count,
            )
            # Auto-detect source language from first file
            if txt_files:
                self._detect_source_language(txt_files)
            else:
                self._detected_source_lang = ""
                self._source_lang_value.configure(
                    text="no .txt files found",
                    text_color=Colors.ERROR,
                )

    def _detect_source_language(self, files: list[Path]) -> None:
        """Auto-detect source language from the provided files.

        Reads the first file (or the single selected file) and uses
        ``LanguageDetector`` to identify the language.  Updates the
        source language display label.

        Args:
            files: List of source file paths.
        """
        if not files:
            return

        target_file = files[0]
        try:
            # Read a sample of text for detection (first 5000 chars)
            text = target_file.read_text(encoding="utf-8")[:5000]
            if not text.strip():
                self._detected_source_lang = ""
                self._source_lang_value.configure(
                    text="file is empty",
                    text_color=Colors.WARNING,
                )
                return

            from core.language_detector import LanguageDetector

            detector = LanguageDetector()
            result = detector.detect(text)
            detected_code = result.lang_code
            confidence = result.confidence

            self._detected_source_lang = detected_code

            if detected_code in SUPPORTED_LANGUAGE_CODES:
                info = LANGUAGES[detected_code]
                display = (
                    f"{info['flag']} {info['name']} ({detected_code}) "
                    f"— {confidence:.0%} confidence"
                )
                self._source_lang_value.configure(
                    text=display, text_color=Colors.SUCCESS,
                )
                logger.info(
                    "Source language detected: %s (%.0f%% confidence) from %s",
                    detected_code,
                    confidence * 100,
                    target_file.name,
                )
            else:
                display = (
                    f"{detected_code} (unsupported) "
                    f"— {confidence:.0%} confidence"
                )
                self._source_lang_value.configure(
                    text=display, text_color=Colors.WARNING,
                )
                logger.warning(
                    "Detected unsupported language: %s from %s",
                    detected_code,
                    target_file.name,
                )

        except Exception as exc:
            self._detected_source_lang = ""
            self._source_lang_value.configure(
                text=f"detection failed: {exc}",
                text_color=Colors.ERROR,
            )
            logger.warning(
                "Source language detection failed for %s: %s",
                target_file.name,
                exc,
            )

    # ── Public API ──────────────────────────────────────────────────

    def get_mode(self) -> str:
        """Return the current mode.

        Returns:
            ``"generate"`` or ``"adapt"``.
        """
        return self._mode_var.get()

    def get_language(self) -> str:
        """Return the currently selected primary language code.

        In generate mode, returns the language dropdown selection.
        In adapt mode, returns the auto-detected source language.

        Returns:
            Two-letter language code (e.g., ``"en"``, ``"de"``).
        """
        if self._mode_var.get() == _MODE_ADAPT and self._detected_source_lang:
            return self._detected_source_lang
        return self._parse_language_option(self._language_var.get())

    def get_target_languages(self) -> list[str]:
        """Return the list of selected target language codes.

        Only meaningful in adaptation mode.

        Returns:
            List of two-letter language codes.
        """
        return [
            code
            for code, var in self._target_lang_vars.items()
            if var.get()
        ]

    def get_topics_file(self) -> str:
        """Return the path to the selected topics file.

        Returns:
            Absolute path string, or empty string if none selected.
        """
        return self._topics_file_path

    def get_adapt_folder(self) -> str:
        """Return the path to the selected adaptation source folder.

        Returns:
            Absolute path string, or empty string if none selected.
        """
        return self._adapt_folder_path

    def get_adapt_file(self) -> str:
        """Return the path to the selected single adaptation source file.

        Returns:
            Absolute path string, or empty string if none selected.
        """
        return self._adapt_file_path

    def get_adapt_source_files(self) -> list[Path]:
        """Return the resolved list of adaptation source files.

        This is the canonical list regardless of whether the user
        selected a single file or a folder.

        Returns:
            List of ``Path`` objects for each source ``.txt`` file.
        """
        return list(self._adapt_source_files)

    def get_detected_source_lang(self) -> str:
        """Return the auto-detected source language code.

        Returns:
            Two-letter language code, or empty string if not detected.
        """
        return self._detected_source_lang

    def get_config(self) -> dict[str, Any]:
        """Return current input panel configuration as a dictionary.

        Returns:
            Dictionary with keys: ``mode``, ``language``,
            ``topics_file``, ``adapt_folder``, ``adapt_file``,
            ``adapt_source_files``, ``detected_source_lang``,
            ``target_languages``.
        """
        return {
            "mode": self.get_mode(),
            "language": self.get_language(),
            "topics_file": self.get_topics_file(),
            "adapt_folder": self.get_adapt_folder(),
            "adapt_file": self.get_adapt_file(),
            "adapt_source_files": self.get_adapt_source_files(),
            "detected_source_lang": self.get_detected_source_lang(),
            "target_languages": self.get_target_languages(),
        }

    def set_language(self, code: str) -> None:
        """Programmatically set the language dropdown.

        Args:
            code: Two-letter language code.
        """
        if code in SUPPORTED_LANGUAGE_CODES:
            display = self._format_language_option(code)
            self._language_var.set(display)
            self._language_menu.set(display)
            logger.debug("Language set programmatically: %s", code)

    def reset(self) -> None:
        """Reset all input fields to defaults."""
        self._mode_var.set(_MODE_GENERATE)
        self._topics_file_path = ""
        self._adapt_folder_path = ""
        self._adapt_file_path = ""
        self._adapt_source_files = []
        self._detected_source_lang = ""
        self._file_label.configure(text="No file selected", text_color=Colors.MUTED)
        self._adapt_file_label.configure(
            text="No file selected", text_color=Colors.MUTED,
        )
        self._folder_label.configure(
            text="No folder selected", text_color=Colors.MUTED,
        )
        self._source_lang_value.configure(
            text="auto-detect on file selection", text_color=Colors.MUTED,
        )
        self.set_language("en")
        for var in self._target_lang_vars.values():
            var.set(False)
        self._update_mode_visibility()
        logger.debug("InputPanel reset to defaults")
