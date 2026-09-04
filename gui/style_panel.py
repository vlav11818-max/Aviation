"""Style panel for AI Story Generator Pro GUI.

Provides controls for all creative / structural generation parameters:
tone, perspective, register, pacing, dialog density, target length,
story structure, genre multi-select, and audience.  Advanced settings
are in a collapsible section.

The panel exposes ``get_config()`` which returns a fully populated
``GenerationConfig`` model.
"""

from __future__ import annotations

import logging
from typing import Any

import customtkinter as ctk

from gui.styles import Colors, Fonts, Padding, create_section_label, create_separator
from models.config import (
    Audience,
    DialogDensity,
    GenerationConfig,
    Pacing,
    Perspective,
    Register,
    StructureType,
    Tone,
)

logger = logging.getLogger(__name__)


# ── Display mappings ────────────────────────────────────────────────────

_TONE_LABELS: dict[str, str] = {
    Tone.DRAMATIC_CINEMATIC.value: "Dramatic / Cinematic",
    Tone.SUSPENSEFUL.value: "Suspenseful",
    Tone.WARM_EMOTIONAL.value: "Warm / Emotional",
    Tone.DARK_GOTHIC.value: "Dark / Gothic",
    Tone.WHIMSICAL.value: "Whimsical",
    Tone.INSPIRATIONAL.value: "Inspirational",
}

_PERSPECTIVE_LABELS: dict[str, str] = {
    Perspective.FIRST_PERSON.value: "First Person",
    Perspective.SECOND_PERSON.value: "Second Person",
    Perspective.THIRD_PERSON.value: "Third Person",
    Perspective.OMNISCIENT.value: "Omniscient",
}

_REGISTER_LABELS: dict[str, str] = {
    Register.FORMAL.value: "Formal",
    Register.CONVERSATIONAL.value: "Conversational",
    Register.LITERARY.value: "Literary",
    Register.POETIC.value: "Poetic",
}

_PACING_LABELS: dict[str, str] = {
    Pacing.SLOW.value: "Slow",
    Pacing.MEDIUM.value: "Medium",
    Pacing.FAST.value: "Fast",
}

_DIALOG_LABELS: dict[str, str] = {
    DialogDensity.LOW.value: "Low",
    DialogDensity.MEDIUM.value: "Medium",
    DialogDensity.HIGH.value: "High",
}

_STRUCTURE_LABELS: dict[str, str] = {
    StructureType.THREE_ACT.value: "Three Act",
    StructureType.HERO_JOURNEY.value: "Hero's Journey",
    StructureType.IN_MEDIAS_RES.value: "In Medias Res",
    StructureType.EPISODIC.value: "Episodic",
    StructureType.CIRCULAR.value: "Circular",
}

_AUDIENCE_LABELS: dict[str, str] = {
    Audience.CHILDREN.value: "Children",
    Audience.YOUNG_ADULT.value: "Young Adult",
    Audience.ALL_AGES.value: "All Ages",
    Audience.MATURE.value: "Mature",
}

_GENRE_OPTIONS: list[str] = [
    "fantasy",
    "mystery",
    "sci-fi",
    "horror",
    "romance",
    "adventure",
    "drama",
    "thriller",
    "historical",
    "comedy",
]

# Length constraints
_LENGTH_MIN: int = 500
_LENGTH_MAX: int = 10000
_LENGTH_DEFAULT: int = 3000
_LENGTH_STEP: int = 100


def _label_to_value(labels: dict[str, str], display: str) -> str:
    """Find the enum value for a display label.

    Args:
        labels: Mapping of enum-value → display-label.
        display: The display label selected by the user.

    Returns:
        The corresponding enum value string.
    """
    for value, label in labels.items():
        if label == display:
            return value
    return list(labels.keys())[0]


class StylePanel(ctk.CTkFrame):
    """Style configuration panel with all creative generation parameters.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(parent, fg_color=Colors.SURFACE, **kwargs)
        self._genre_vars: dict[str, ctk.BooleanVar] = {}
        self._advanced_visible: bool = False
        self._build_ui()
        logger.debug("StylePanel initialised")

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all child widgets."""
        self.columnconfigure(0, weight=1)
        row = 0

        # Section header
        header = create_section_label(self, "STYLE", icon="\u2728")
        header.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.PANEL, Padding.WIDGET_Y))
        row += 1

        sep = create_separator(self)
        sep.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        # ── Tone ────────────────────────────────────────────────────
        row = self._add_dropdown(
            row, "Tone", "\U0001F3AD",
            list(_TONE_LABELS.values()),
            _TONE_LABELS[Tone.DRAMATIC_CINEMATIC.value],
            "_tone_var",
        )

        # ── Perspective ─────────────────────────────────────────────
        row = self._add_dropdown(
            row, "Perspective", "\U0001F464",
            list(_PERSPECTIVE_LABELS.values()),
            _PERSPECTIVE_LABELS[Perspective.THIRD_PERSON.value],
            "_perspective_var",
        )

        # ── Register ────────────────────────────────────────────────
        row = self._add_dropdown(
            row, "Register", "\U0001F4DD",
            list(_REGISTER_LABELS.values()),
            _REGISTER_LABELS[Register.CONVERSATIONAL.value],
            "_register_var",
        )

        # ── Length (entry + slider) ─────────────────────────────────
        row = self._add_length_control(row)

        # ── Advanced settings toggle ────────────────────────────────
        self._advanced_btn = ctk.CTkButton(
            self,
            text="\u25B6 Advanced Settings",
            command=self._toggle_advanced,
            font=Fonts.get(size=Fonts.SMALL),
            fg_color="transparent",
            hover_color=Colors.SURFACE_HOVER,
            text_color=Colors.INFO,
            anchor="w",
            width=0,
        )
        self._advanced_btn.grid(
            row=row, column=0, sticky="w",
            padx=Padding.PANEL, pady=(Padding.GROUP_Y, Padding.LABEL_Y),
        )
        row += 1

        # ── Advanced section (collapsible) ──────────────────────────
        self._advanced_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._advanced_frame.columnconfigure(0, weight=1)
        self._advanced_row = row
        row += 1

        adv_row = 0

        # Pacing slider
        adv_row = self._add_slider_option(
            self._advanced_frame, adv_row, "Pacing",
            list(_PACING_LABELS.values()),
            _PACING_LABELS[Pacing.MEDIUM.value],
            "_pacing_var",
        )

        # Dialog density slider
        adv_row = self._add_slider_option(
            self._advanced_frame, adv_row, "Dialog Density",
            list(_DIALOG_LABELS.values()),
            _DIALOG_LABELS[DialogDensity.MEDIUM.value],
            "_dialog_var",
        )

        # Structure dropdown
        adv_row = self._add_dropdown_in(
            self._advanced_frame, adv_row, "Structure", "\U0001F4D6",
            list(_STRUCTURE_LABELS.values()),
            _STRUCTURE_LABELS[StructureType.THREE_ACT.value],
            "_structure_var",
        )

        # Audience dropdown
        adv_row = self._add_dropdown_in(
            self._advanced_frame, adv_row, "Audience", "\U0001F3AF",
            list(_AUDIENCE_LABELS.values()),
            _AUDIENCE_LABELS[Audience.ALL_AGES.value],
            "_audience_var",
        )

        # Genre multi-select
        genre_label = ctk.CTkLabel(
            self._advanced_frame,
            text="\U0001F3AA Genre",
            font=Fonts.get(bold=True),
            text_color=Colors.TEXT,
            anchor="w",
        )
        genre_label.grid(
            row=adv_row, column=0, sticky="w",
            padx=0, pady=(Padding.SECTION, Padding.LABEL_Y),
        )
        adv_row += 1

        genre_grid = ctk.CTkFrame(self._advanced_frame, fg_color="transparent")
        genre_grid.grid(row=adv_row, column=0, sticky="ew", pady=Padding.LABEL_Y)
        adv_row += 1

        for idx, genre in enumerate(_GENRE_OPTIONS):
            var = ctk.BooleanVar(value=False)
            self._genre_vars[genre] = var
            display_name = genre.replace("-", " ").replace("_", " ").title()
            cb = ctk.CTkCheckBox(
                genre_grid,
                text=display_name,
                variable=var,
                font=Fonts.get(size=Fonts.SMALL),
                text_color=Colors.TEXT,
                fg_color=Colors.PRIMARY,
                hover_color=Colors.PRIMARY_HOVER,
                checkmark_color=Colors.TEXT,
                width=100,
            )
            cb.grid(row=idx // 3, column=idx % 3, sticky="w", padx=Padding.WIDGET_X, pady=2)

        # Advanced section hidden by default
        # (do not grid the frame until toggled)

    # ── Widget helpers ──────────────────────────────────────────────

    def _add_dropdown(
        self,
        row: int,
        label_text: str,
        icon: str,
        values: list[str],
        default: str,
        var_attr: str,
    ) -> int:
        """Add a labelled dropdown to ``self``.

        Args:
            row: Current grid row.
            label_text: Display label.
            icon: Emoji icon for the label.
            values: Dropdown options.
            default: Default value.
            var_attr: Name of the attribute to store the ``StringVar``.

        Returns:
            Next available grid row.
        """
        var = ctk.StringVar(value=default)
        setattr(self, var_attr, var)

        lbl = ctk.CTkLabel(
            self,
            text=f"{icon} {label_text}",
            font=Fonts.get(bold=True),
            text_color=Colors.TEXT,
            anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        menu = ctk.CTkOptionMenu(
            self,
            variable=var,
            values=values,
            font=Fonts.get(),
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
            dropdown_fg_color=Colors.SURFACE,
            dropdown_hover_color=Colors.SURFACE_HOVER,
            dropdown_text_color=Colors.TEXT,
        )
        menu.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        return row

    def _add_dropdown_in(
        self,
        parent: ctk.CTkFrame,
        row: int,
        label_text: str,
        icon: str,
        values: list[str],
        default: str,
        var_attr: str,
    ) -> int:
        """Add a labelled dropdown to a specific parent frame.

        Args:
            parent: The parent frame.
            row: Current grid row.
            label_text: Display label.
            icon: Emoji icon for the label.
            values: Dropdown options.
            default: Default value.
            var_attr: Name of the attribute to store the ``StringVar``.

        Returns:
            Next available grid row.
        """
        var = ctk.StringVar(value=default)
        setattr(self, var_attr, var)

        lbl = ctk.CTkLabel(
            parent,
            text=f"{icon} {label_text}",
            font=Fonts.get(bold=True),
            text_color=Colors.TEXT,
            anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        menu = ctk.CTkOptionMenu(
            parent,
            variable=var,
            values=values,
            font=Fonts.get(),
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
            dropdown_fg_color=Colors.SURFACE,
            dropdown_hover_color=Colors.SURFACE_HOVER,
            dropdown_text_color=Colors.TEXT,
        )
        menu.grid(row=row, column=0, sticky="ew", pady=Padding.LABEL_Y)
        row += 1

        return row

    def _add_slider_option(
        self,
        parent: ctk.CTkFrame,
        row: int,
        label_text: str,
        labels: list[str],
        default: str,
        var_attr: str,
    ) -> int:
        """Add a labelled segmented button (acts like a discrete slider).

        Args:
            parent: The parent frame.
            row: Current grid row.
            label_text: Display label.
            labels: Option labels (e.g., ``["Slow", "Medium", "Fast"]``).
            default: Default selected value.
            var_attr: Name of the attribute to store the ``StringVar``.

        Returns:
            Next available grid row.
        """
        var = ctk.StringVar(value=default)
        setattr(self, var_attr, var)

        lbl = ctk.CTkLabel(
            parent,
            text=label_text,
            font=Fonts.get(bold=True),
            text_color=Colors.TEXT,
            anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        seg = ctk.CTkSegmentedButton(
            parent,
            values=labels,
            variable=var,
            font=Fonts.get(size=Fonts.SMALL),
            selected_color=Colors.PRIMARY,
            selected_hover_color=Colors.PRIMARY_HOVER,
            unselected_color=Colors.ENTRY_BG,
            unselected_hover_color=Colors.SURFACE_HOVER,
            text_color=Colors.TEXT,
        )
        seg.grid(row=row, column=0, sticky="ew", pady=Padding.LABEL_Y)
        row += 1

        return row

    def _add_length_control(self, row: int) -> int:
        """Add the length entry + slider combo.

        Args:
            row: Current grid row.

        Returns:
            Next available grid row.
        """
        lbl = ctk.CTkLabel(
            self,
            text="\u270F\uFE0F Length (words)",
            font=Fonts.get(bold=True),
            text_color=Colors.TEXT,
            anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        length_frame = ctk.CTkFrame(self, fg_color="transparent")
        length_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        length_frame.columnconfigure(1, weight=1)

        self._length_var = ctk.IntVar(value=_LENGTH_DEFAULT)

        self._length_entry = ctk.CTkEntry(
            length_frame,
            textvariable=self._length_var,
            width=80,
            font=Fonts.get(),
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            border_color=Colors.ENTRY_BORDER,
            justify="center",
        )
        self._length_entry.grid(row=0, column=0, padx=(0, Padding.WIDGET_X))

        self._length_slider = ctk.CTkSlider(
            length_frame,
            from_=_LENGTH_MIN,
            to=_LENGTH_MAX,
            number_of_steps=(_LENGTH_MAX - _LENGTH_MIN) // _LENGTH_STEP,
            variable=self._length_var,
            command=self._on_length_slider_changed,
            fg_color=Colors.ENTRY_BG,
            progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
        )
        self._length_slider.grid(row=0, column=1, sticky="ew")

        self._length_range_label = ctk.CTkLabel(
            self,
            text=f"{_LENGTH_MIN} – {_LENGTH_MAX} words",
            font=Fonts.get(size=Fonts.TINY),
            text_color=Colors.MUTED,
            anchor="w",
        )
        self._length_range_label.grid(row=row + 1, column=0, sticky="w", padx=Padding.PANEL, pady=0)
        row += 2

        # Bind entry validation
        self._length_entry.bind("<FocusOut>", self._on_length_entry_changed)
        self._length_entry.bind("<Return>", self._on_length_entry_changed)

        return row

    # ── Callbacks ───────────────────────────────────────────────────

    def _on_length_slider_changed(self, value: float) -> None:
        """Handle slider movement — snap to step.

        Args:
            value: Raw slider value.
        """
        snapped = int(round(value / _LENGTH_STEP) * _LENGTH_STEP)
        snapped = max(_LENGTH_MIN, min(_LENGTH_MAX, snapped))
        self._length_var.set(snapped)

    def _on_length_entry_changed(self, event: Any = None) -> None:
        """Validate and clamp the length entry value.

        Args:
            event: Tkinter event (unused).
        """
        try:
            raw = self._length_var.get()
        except (ValueError, ctk.TclError):
            raw = _LENGTH_DEFAULT

        clamped = max(_LENGTH_MIN, min(_LENGTH_MAX, raw))
        self._length_var.set(clamped)
        self._length_slider.set(clamped)

    def _toggle_advanced(self) -> None:
        """Show or hide the advanced settings section."""
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            self._advanced_btn.configure(text="\u25BC Advanced Settings")
            self._advanced_frame.grid(
                row=self._advanced_row, column=0, sticky="ew",
                padx=Padding.PANEL, pady=Padding.LABEL_Y,
            )
        else:
            self._advanced_btn.configure(text="\u25B6 Advanced Settings")
            self._advanced_frame.grid_remove()
        logger.debug("Advanced settings toggled: visible=%s", self._advanced_visible)

    # ── Public API ──────────────────────────────────────────────────

    def get_config(self) -> GenerationConfig:
        """Build a ``GenerationConfig`` from the current panel state.

        Returns:
            A populated ``GenerationConfig`` model.
        """
        # Resolve enum values from display labels
        tone_val = _label_to_value(_TONE_LABELS, self._tone_var.get())
        perspective_val = _label_to_value(_PERSPECTIVE_LABELS, self._perspective_var.get())
        register_val = _label_to_value(_REGISTER_LABELS, self._register_var.get())
        pacing_val = _label_to_value(_PACING_LABELS, self._pacing_var.get())
        dialog_val = _label_to_value(_DIALOG_LABELS, self._dialog_var.get())
        structure_val = _label_to_value(_STRUCTURE_LABELS, self._structure_var.get())
        audience_val = _label_to_value(_AUDIENCE_LABELS, self._audience_var.get())

        # Collect selected genres
        genres = [g for g, var in self._genre_vars.items() if var.get()]

        # Clamp length
        try:
            target_words = self._length_var.get()
        except (ValueError, ctk.TclError):
            target_words = _LENGTH_DEFAULT
        target_words = max(_LENGTH_MIN, min(_LENGTH_MAX, target_words))

        config = GenerationConfig(
            tone=Tone(tone_val),
            perspective=Perspective(perspective_val),
            register=Register(register_val),
            pacing=Pacing(pacing_val),
            dialog_density=DialogDensity(dialog_val),
            target_words=target_words,
            structure=StructureType(structure_val),
            audience=Audience(audience_val),
            genres=genres,
        )

        logger.debug(
            "StylePanel config: tone=%s, words=%d, structure=%s, genres=%s",
            config.tone.value, config.target_words,
            config.structure.value, config.genres,
        )
        return config

    def load_from_config(self, config: GenerationConfig) -> None:
        """Populate panel controls from a ``GenerationConfig``.

        Args:
            config: The configuration to load.
        """
        self._tone_var.set(_TONE_LABELS.get(config.tone.value, config.tone.value))
        self._perspective_var.set(_PERSPECTIVE_LABELS.get(config.perspective.value, config.perspective.value))
        self._register_var.set(_REGISTER_LABELS.get(config.register.value, config.register.value))
        self._pacing_var.set(_PACING_LABELS.get(config.pacing.value, config.pacing.value))
        self._dialog_var.set(_DIALOG_LABELS.get(config.dialog_density.value, config.dialog_density.value))
        self._structure_var.set(_STRUCTURE_LABELS.get(config.structure.value, config.structure.value))
        self._audience_var.set(_AUDIENCE_LABELS.get(config.audience.value, config.audience.value))

        self._length_var.set(config.target_words)
        self._length_slider.set(config.target_words)

        # Genres
        for genre, var in self._genre_vars.items():
            var.set(genre in config.genres)

        logger.debug("StylePanel loaded from config")

    def reset(self) -> None:
        """Reset all controls to default values."""
        self.load_from_config(GenerationConfig())
        logger.debug("StylePanel reset to defaults")
