"""Cost estimation dialog for AI Story Generator Pro GUI.

Modal dialog shown before generation starts.  Displays the number of
topics, estimated tokens, estimated cost (total and per-story),
model / provider information, and a per-step cost breakdown.
The user can confirm ("Proceed") or cancel.

Typical usage::

    dialog = CostDialog(parent, estimate=estimate)
    dialog.wait_window()
    if dialog.confirmed:
        # start generation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from gui.styles import Colors, Fonts, Padding

if TYPE_CHECKING:
    from core.cost_estimator import CostEstimate

logger = logging.getLogger(__name__)


class CostDialog(ctk.CTkToplevel):
    """Modal cost-estimation dialog.

    Args:
        parent: Parent window.
        estimate: A ``CostEstimate`` instance with pre-computed values.
    """

    def __init__(
        self,
        parent: Any,
        estimate: "CostEstimate",
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._estimate = estimate
        self.confirmed: bool = False

        # Window config
        self.title("Cost Estimate")
        self.geometry("440x520")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.configure(fg_color=Colors.SURFACE)

        self._build_ui()
        logger.debug("CostDialog opened")

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all child widgets."""
        self.columnconfigure(0, weight=1)

        row = 0

        # Title
        title = ctk.CTkLabel(
            self,
            text="\U0001F4B0 Cost Estimate",
            font=Fonts.get(size=Fonts.HEADING, bold=True),
            text_color=Colors.TEXT,
        )
        title.grid(row=row, column=0, pady=(Padding.GROUP_Y, Padding.WIDGET_Y))
        row += 1

        # Summary card
        summary_frame = ctk.CTkFrame(self, fg_color=Colors.SURFACE_LIGHT, corner_radius=8)
        summary_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.WIDGET_Y)
        summary_frame.columnconfigure(0, weight=1)

        est = self._estimate
        summary_items = [
            ("Topics", str(est.topics_count)),
            ("Target Words / Story", f"{est.target_words:,}"),
            ("Strategy", est.strategy_name.replace("_", " ").title()),
            ("Model", est.model),
            ("Est. Input Tokens", f"{est.estimated_tokens_in:,}"),
            ("Est. Output Tokens", f"{est.estimated_tokens_out:,}"),
        ]

        for srow, (label, value) in enumerate(summary_items):
            lbl = ctk.CTkLabel(
                summary_frame, text=label,
                font=Fonts.get(size=Fonts.SMALL),
                text_color=Colors.MUTED, anchor="w",
            )
            lbl.grid(row=srow, column=0, sticky="w", padx=Padding.WIDGET_X, pady=1)
            val = ctk.CTkLabel(
                summary_frame, text=value,
                font=Fonts.get(size=Fonts.SMALL, bold=True),
                text_color=Colors.TEXT, anchor="e",
            )
            val.grid(row=srow, column=1, sticky="e", padx=Padding.WIDGET_X, pady=1)
        row += 1

        # Cost highlight
        cost_frame = ctk.CTkFrame(self, fg_color=Colors.PRIMARY, corner_radius=8)
        cost_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        cost_frame.columnconfigure(0, weight=1)

        total_lbl = ctk.CTkLabel(
            cost_frame, text="Estimated Total Cost",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.TEXT_DIM,
        )
        total_lbl.grid(row=0, column=0, pady=(Padding.WIDGET_Y, 0))

        total_val = ctk.CTkLabel(
            cost_frame,
            text=f"${est.total_usd:.4f}",
            font=Fonts.get(size=Fonts.HEADING, bold=True),
            text_color=Colors.TEXT,
        )
        total_val.grid(row=1, column=0, pady=(0, 2))

        per_story_lbl = ctk.CTkLabel(
            cost_frame,
            text=f"(${est.per_story_usd:.4f} per story)",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.TEXT_DIM,
        )
        per_story_lbl.grid(row=2, column=0, pady=(0, Padding.WIDGET_Y))
        row += 1

        # Per-step breakdown
        if est.breakdown_by_step:
            breakdown_lbl = ctk.CTkLabel(
                self, text="Per-Step Breakdown (per story)",
                font=Fonts.get(size=Fonts.SMALL, bold=True),
                text_color=Colors.TEXT, anchor="w",
            )
            breakdown_lbl.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y))
            row += 1

            bd_frame = ctk.CTkFrame(self, fg_color=Colors.SURFACE_LIGHT, corner_radius=6)
            bd_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
            bd_frame.columnconfigure(1, weight=1)

            max_step_cost = max(est.breakdown_by_step.values(), default=1.0)

            for brow, (step_name, step_cost) in enumerate(est.breakdown_by_step.items()):
                display_name = step_name.replace("_", " ").title()
                ctk.CTkLabel(
                    bd_frame, text=display_name,
                    font=Fonts.get(size=Fonts.TINY),
                    text_color=Colors.TEXT_DIM, anchor="w",
                ).grid(row=brow, column=0, sticky="w", padx=Padding.WIDGET_X, pady=1)

                bar = ctk.CTkProgressBar(
                    bd_frame, height=10,
                    fg_color=Colors.ENTRY_BG,
                    progress_color=Colors.INFO,
                )
                bar.grid(row=brow, column=1, sticky="ew", padx=4, pady=1)
                bar.set(step_cost / max_step_cost if max_step_cost > 0 else 0)

                ctk.CTkLabel(
                    bd_frame, text=f"${step_cost:.4f}",
                    font=Fonts.get(size=Fonts.TINY),
                    text_color=Colors.MUTED, width=60, anchor="e",
                ).grid(row=brow, column=2, sticky="e", padx=Padding.WIDGET_X, pady=1)
            row += 1

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.GROUP_Y)
        btn_frame.columnconfigure((0, 1), weight=1)

        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self._on_cancel,
            font=Fonts.get(),
            fg_color=Colors.SURFACE_LIGHT,
            hover_color=Colors.SURFACE_HOVER,
            text_color=Colors.TEXT,
        )
        cancel_btn.grid(row=0, column=0, sticky="ew", padx=(0, Padding.WIDGET_X))

        proceed_text = f"Proceed — ~${est.total_usd:.2f}"
        proceed_btn = ctk.CTkButton(
            btn_frame,
            text=proceed_text,
            command=self._on_proceed,
            font=Fonts.get(bold=True),
            fg_color=Colors.SUCCESS,
            hover_color=Colors.SUCCESS_HOVER,
            text_color=Colors.TEXT,
        )
        proceed_btn.grid(row=0, column=1, sticky="ew")

    # ── Callbacks ───────────────────────────────────────────────────

    def _on_proceed(self) -> None:
        """Handle the Proceed button — confirm and close."""
        self.confirmed = True
        logger.info("Cost dialog: user confirmed ($%.4f)", self._estimate.total_usd)
        self.destroy()

    def _on_cancel(self) -> None:
        """Handle the Cancel button — decline and close."""
        self.confirmed = False
        logger.info("Cost dialog: user cancelled")
        self.destroy()
