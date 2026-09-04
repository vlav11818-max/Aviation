"""Styles and theme definitions for the GUI.

Provides dark-theme color constants, font size definitions, padding /
margin constants, and a helper ``apply_theme()`` function that
configures a customtkinter root window.

Typical usage::

    from gui.styles import Colors, Fonts, Padding, apply_theme

    root = ctk.CTk()
    apply_theme(root)
"""

from __future__ import annotations

import logging
from typing import Any

import customtkinter as ctk

logger = logging.getLogger(__name__)


# ── Color palette (dark theme) ──────────────────────────────────────────


class Colors:
    """Dark-theme color constants used throughout the GUI."""

    # Base surfaces
    BACKGROUND: str = "#1a1a2e"
    SURFACE: str = "#16213e"
    SURFACE_LIGHT: str = "#1f2b47"
    SURFACE_HOVER: str = "#253350"

    # Primary accent
    PRIMARY: str = "#0f3460"
    PRIMARY_HOVER: str = "#154785"
    PRIMARY_LIGHT: str = "#1a5276"

    # Secondary accent
    SECONDARY: str = "#533483"
    SECONDARY_HOVER: str = "#6a42a8"

    # Text
    TEXT: str = "#e0e0e0"
    TEXT_DIM: str = "#b0b0c0"
    MUTED: str = "#6c7293"
    PLACEHOLDER: str = "#4a4e6a"

    # Semantic
    SUCCESS: str = "#27ae60"
    SUCCESS_HOVER: str = "#2ecc71"
    WARNING: str = "#f39c12"
    WARNING_HOVER: str = "#f1c40f"
    ERROR: str = "#e74c3c"
    ERROR_HOVER: str = "#ff6b6b"
    INFO: str = "#3498db"
    INFO_HOVER: str = "#5dade2"

    # Status indicators
    STATUS_DONE: str = "#27ae60"
    STATUS_RUNNING: str = "#f39c12"
    STATUS_FAILED: str = "#e74c3c"
    STATUS_QUEUED: str = "#6c7293"
    STATUS_PAUSED: str = "#9b59b6"

    # Borders and separators
    BORDER: str = "#2c3e6b"
    SEPARATOR: str = "#2c3e6b"

    # Input fields
    ENTRY_BG: str = "#1f2b47"
    ENTRY_BORDER: str = "#2c3e6b"

    # Scrollbar
    SCROLLBAR: str = "#2c3e6b"
    SCROLLBAR_HOVER: str = "#3d5289"


# ── Font definitions ────────────────────────────────────────────────────


class Fonts:
    """Font family and size constants.

    Note: Actual ``CTkFont`` instances should be created at widget-build
    time because customtkinter requires the display to be initialised
    first.  Use the ``get()`` class method to obtain a tuple suitable
    for the ``font`` parameter.
    """

    FAMILY: str = "Segoe UI"
    FAMILY_MONO: str = "Consolas"

    # Sizes
    HEADING: int = 18
    SUBHEADING: int = 15
    BODY: int = 13
    SMALL: int = 11
    TINY: int = 10
    MONO: int = 12

    @classmethod
    def get(
        cls,
        size: int | None = None,
        bold: bool = False,
        mono: bool = False,
    ) -> tuple[str, int, str]:
        """Return a font tuple ``(family, size, weight)``.

        Args:
            size: Font size in points.  Defaults to ``BODY``.
            bold: Whether to use bold weight.
            mono: Whether to use the monospace family.

        Returns:
            A 3-tuple suitable for customtkinter ``font`` parameter.
        """
        family = cls.FAMILY_MONO if mono else cls.FAMILY
        point = size if size is not None else cls.BODY
        weight = "bold" if bold else "normal"
        return (family, point, weight)


# ── Padding / margin constants ──────────────────────────────────────────


class Padding:
    """Standard padding and margin values in pixels."""

    # Outer frame padding
    WINDOW: int = 10
    PANEL: int = 12
    SECTION: int = 8

    # Inner widget spacing
    WIDGET_X: int = 8
    WIDGET_Y: int = 6
    LABEL_Y: int = 4

    # Between groups
    GROUP_Y: int = 16
    GROUP_X: int = 12

    # Button padding
    BUTTON_X: int = 16
    BUTTON_Y: int = 8

    # Minimum sizes
    MIN_ENTRY_WIDTH: int = 200
    MIN_BUTTON_WIDTH: int = 120

    # Column proportions (used by MainWindow grid)
    LEFT_COL_WEIGHT: int = 1
    RIGHT_COL_WEIGHT: int = 2


# ── Theme application ──────────────────────────────────────────────────


def apply_theme(root: ctk.CTk) -> None:
    """Apply the dark theme to a customtkinter root window.

    Sets the appearance mode to dark, configures default colours, and
    applies the standard colour scheme.

    Args:
        root: The root ``CTk`` window to theme.
    """
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root.configure(fg_color=Colors.BACKGROUND)

    logger.debug("Dark theme applied to root window")


def create_section_label(
    parent: Any,
    text: str,
    icon: str = "",
) -> ctk.CTkLabel:
    """Create a styled section header label.

    Args:
        parent: Parent widget.
        text: Label text.
        icon: Optional emoji/icon prefix.

    Returns:
        A configured ``CTkLabel`` widget.
    """
    display = f"{icon} {text}" if icon else text
    label = ctk.CTkLabel(
        parent,
        text=display,
        font=Fonts.get(size=Fonts.SUBHEADING, bold=True),
        text_color=Colors.TEXT,
        anchor="w",
    )
    return label


def create_separator(parent: Any) -> ctk.CTkFrame:
    """Create a thin horizontal separator line.

    Args:
        parent: Parent widget.

    Returns:
        A 1-pixel-high frame acting as a visual separator.
    """
    sep = ctk.CTkFrame(
        parent,
        height=1,
        fg_color=Colors.SEPARATOR,
    )
    return sep
