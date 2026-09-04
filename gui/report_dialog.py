"""Report dialog for AI Story Generator Pro GUI.

Modal dialog shown after a batch generation completes.  Displays a
summary (completed / failed count, average score, total cost, total
time, throughput), a scrollable per-story results table, and buttons
to export the report (JSON / CSV) and open the output folder.

Typical usage::

    dialog = ReportDialog(parent, batch_result=result)
    dialog.wait_window()
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from tkinter import filedialog
from typing import Any

import customtkinter as ctk

from gui.styles import Colors, Fonts, Padding

logger = logging.getLogger(__name__)


class ReportDialog(ctk.CTkToplevel):
    """Post-generation report dialog.

    Args:
        parent: Parent window.
        batch_result: A ``BatchResult`` dataclass with per-topic results.
        output_dir: Path to the output directory (for "Open folder").
    """

    def __init__(
        self,
        parent: Any,
        batch_result: Any,
        output_dir: str = "output",
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._result = batch_result
        self._output_dir = output_dir

        self.title("Generation Report")
        self.geometry("600x650")
        self.resizable(True, True)
        self.minsize(500, 500)
        self.transient(parent)
        self.grab_set()
        self.configure(fg_color=Colors.SURFACE)

        self._build_ui()
        logger.debug("ReportDialog opened")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all child widgets."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        row = 0

        # Title
        title = ctk.CTkLabel(
            self,
            text="\U0001F4CB Generation Report",
            font=Fonts.get(size=Fonts.HEADING, bold=True),
            text_color=Colors.TEXT,
        )
        title.grid(row=row, column=0, pady=(Padding.GROUP_Y, Padding.WIDGET_Y))
        row += 1

        # ── Summary card ──────────────────────────────────────────────────────
        summary_frame = ctk.CTkFrame(
            self, fg_color=Colors.SURFACE_LIGHT, corner_radius=8,
        )
        summary_frame.grid(
            row=row, column=0, sticky="ew",
            padx=Padding.PANEL, pady=Padding.WIDGET_Y,
        )

        result = self._result
        total = getattr(result, "total", 0)
        completed = getattr(result, "completed", 0)
        failed = getattr(result, "failed", 0)
        cached = getattr(result, "cached", 0)
        elapsed = getattr(result, "elapsed_seconds", 0.0)

        results_list = getattr(result, "results", [])
        scores = [
            r.score for r in results_list if getattr(r, "score", 0.0) > 0
        ]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        total_cost = sum(
            getattr(r, "cost", 0.0) for r in results_list
        )

        metrics = [
            ("Total", str(total)),
            ("✅ Done", str(completed)),
            ("❌ Failed", str(failed)),
            ("ðŸ’¾ Cached", str(cached)),
            ("Avg Score", f"{avg_score:.1f}"),
            ("Cost", f"${total_cost:.4f}"),
            ("Time", f"{elapsed:.0f}s"),
        ]

        for c in range(len(metrics)):
            summary_frame.columnconfigure(c, weight=1)

        for col, (label, value) in enumerate(metrics):
            cell = ctk.CTkFrame(summary_frame, fg_color="transparent")
            cell.grid(
                row=0, column=col,
                padx=Padding.WIDGET_X, pady=Padding.WIDGET_Y,
            )
            ctk.CTkLabel(
                cell, text=label,
                font=Fonts.get(size=Fonts.TINY), text_color=Colors.MUTED,
            ).pack()
            ctk.CTkLabel(
                cell, text=value,
                font=Fonts.get(size=Fonts.SMALL, bold=True),
                text_color=Colors.TEXT,
            ).pack()

        row += 1

        # ── Per-story table ───────────────────────────────────────────────────
        self._build_story_table(row)
        row += 1

        # ── Action buttons ────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(
            row=row, column=0, sticky="ew",
            padx=Padding.PANEL, pady=(Padding.SECTION, Padding.PANEL),
        )

        ctk.CTkButton(
            btn_frame,
            text="ðŸ“‚ Open Folder",
            command=self._open_output_folder,
            font=Fonts.get(size=Fonts.SMALL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
            width=110,
        ).pack(side="left", padx=(0, Padding.WIDGET_X))

        ctk.CTkButton(
            btn_frame,
            text="ðŸ’¾ JSON",
            command=self._export_json,
            font=Fonts.get(size=Fonts.SMALL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
            width=80,
        ).pack(side="left", padx=(0, Padding.WIDGET_X))

        ctk.CTkButton(
            btn_frame,
            text="ðŸ’¾ CSV",
            command=self._export_csv,
            font=Fonts.get(size=Fonts.SMALL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
            width=80,
        ).pack(side="left", padx=(0, Padding.WIDGET_X))

        ctk.CTkButton(
            btn_frame,
            text="Close",
            command=self.destroy,
            font=Fonts.get(size=Fonts.SMALL),
            fg_color=Colors.SURFACE_LIGHT,
            hover_color=Colors.SURFACE_HOVER,
            text_color=Colors.TEXT,
            width=80,
        ).pack(side="right")

    def _build_story_table(self, row: int) -> None:
        """Build the scrollable per-story results table.

        Args:
            row: Grid row to place the table in.
        """
        scroll = ctk.CTkScrollableFrame(self, fg_color=Colors.SURFACE_LIGHT)
        scroll.grid(
            row=row, column=0, sticky="nsew",
            padx=Padding.PANEL, pady=Padding.WIDGET_Y,
        )
        self.rowconfigure(row, weight=1)

        headers = ["#", "Topic", "Status", "Score", "Attempts", "Time", "Cost"]
        col_widths = [30, 0, 60, 55, 65, 55, 60]  # 0 = stretches
        for c, (h, w) in enumerate(zip(headers, col_widths)):
            if w == 0:
                scroll.columnconfigure(c, weight=1)
            ctk.CTkLabel(
                scroll, text=h,
                font=Fonts.get(size=Fonts.TINY, bold=True),
                text_color=Colors.MUTED,
                anchor="w" if c == 1 else "center",
                width=w if w else 0,
            ).grid(row=0, column=c, padx=2, pady=(2, 4), sticky="w" if c == 1 else "")

        results_list = getattr(self._result, "results", [])
        for idx, r in enumerate(results_list, start=1):
            topic = getattr(r, "topic", "")[:40]
            success = getattr(r, "success", False)
            score = getattr(r, "score", 0.0)
            attempts = getattr(r, "attempts", 0)
            elapsed = getattr(r, "elapsed_seconds", 0.0)
            cost = getattr(r, "cost", 0.0)

            status_icon = "✅" if success else "❌"
            score_str = f"{score:.1f}" if score > 0 else "—"
            score_colour = (
                Colors.SUCCESS if score >= 9.0
                else Colors.WARNING if score >= 7.0
                else Colors.ERROR if score > 0
                else Colors.MUTED
            )

            row_vals = [
                (str(idx), Colors.MUTED, "center"),
                (topic, Colors.TEXT, "w"),
                (status_icon, Colors.SUCCESS if success else Colors.ERROR, "center"),
                (score_str, score_colour, "center"),
                (str(attempts), Colors.TEXT_DIM, "center"),
                (f"{elapsed:.1f}s", Colors.TEXT_DIM, "center"),
                (f"${cost:.4f}", Colors.TEXT_DIM, "center"),
            ]

            for c, (text, colour, anchor) in enumerate(row_vals):
                ctk.CTkLabel(
                    scroll, text=text,
                    font=Fonts.get(size=Fonts.TINY),
                    text_color=colour,
                    anchor=anchor,
                ).grid(
                    row=idx, column=c, padx=2, pady=1,
                    sticky="w" if anchor == "w" else "",
                )

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _build_report_data(self) -> dict[str, Any]:
        """Build a serialisable report dictionary from the batch result.

        Returns:
            Dictionary with summary and per-story results.
        """
        results_list = getattr(self._result, "results", [])
        stories = [
            {
                "topic": getattr(r, "topic", ""),
                "success": getattr(r, "success", False),
                "score": getattr(r, "score", 0.0),
                "attempts": getattr(r, "attempts", 0),
                "cached": getattr(r, "cached", False),
                "elapsed_seconds": getattr(r, "elapsed_seconds", 0.0),
                "error": getattr(r, "error", ""),
                "output_dir": getattr(r, "output_dir", ""),
            }
            for r in results_list
        ]

        return {
            "total": getattr(self._result, "total", 0),
            "completed": getattr(self._result, "completed", 0),
            "failed": getattr(self._result, "failed", 0),
            "cached": getattr(self._result, "cached", 0),
            "elapsed_seconds": getattr(self._result, "elapsed_seconds", 0.0),
            "stories": stories,
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_json(self) -> None:
        """Export the report as JSON."""
        path = filedialog.asksaveasfilename(
            title="Save Report (JSON)",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return

        data = self._build_report_data()
        try:
            Path(path).write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Report exported to JSON: %s", path)
        except OSError as exc:
            logger.error("Failed to export JSON report: %s", exc)

    def _export_csv(self) -> None:
        """Export per-story results as CSV.

        Uses ``csv.writer`` via ``io.StringIO`` so all quoting, escaping,
        and line endings are handled correctly — including fields that
        contain commas, double-quotes, newlines, carriage returns, or
        tabs.
        """
        path = filedialog.asksaveasfilename(
            title="Save Report (CSV)",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        data = self._build_report_data()
        stories = data.get("stories", [])
        if not stories:
            logger.warning("No stories to export as CSV")
            return

        columns = list(stories[0].keys())
        output = io.StringIO()
        writer = csv.writer(
            output,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\r\n",  # RFC 4180
        )
        writer.writerow(columns)

        for s in stories:
            row_vals = []
            for col in columns:
                val = s.get(col, "")
                # Serialise list values to semicolon strings.
                if isinstance(val, list):
                    val = "; ".join(str(v) for v in val)
                row_vals.append(val)
            writer.writerow(row_vals)

        try:
            Path(path).write_text(output.getvalue(), encoding="utf-8")
            logger.info("Report exported to CSV: %s", path)
        except OSError as exc:
            logger.error("Failed to export CSV report: %s", exc)

    # ── Output folder ─────────────────────────────────────────────────────────

    def _open_output_folder(self) -> None:
        """Open the output directory in the system file manager."""
        folder = Path(self._output_dir)
        if not folder.exists():
            logger.warning("Output folder does not exist: %s", folder)
            return

        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
            logger.info("Opened output folder: %s", folder)
        except Exception as exc:
            logger.error("Failed to open output folder: %s", exc)
