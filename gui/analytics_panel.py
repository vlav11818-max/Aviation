"""Analytics panel for AI Story Generator Pro GUI.

Displays accumulated analytics data loaded from
``data/analytics/stats.json``.  Includes an all-time summary,
per-language and per-provider tables, a score distribution visualisation,
a common-issues list, and export buttons (CSV / JSON).

Handles empty or missing analytics data gracefully.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from tkinter import filedialog
from typing import Any

import customtkinter as ctk

from core.analytics_collector import AnalyticsCollector
from gui.styles import Colors, Fonts, Padding, create_section_label, create_separator

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_ANALYTICS_PATH = Path("data/analytics/stats.json")


class AnalyticsPanel(ctk.CTkFrame):
    """Analytics display panel.

    Loads and renders historical generation statistics.

    Args:
        parent: Parent widget.
        analytics_path: Path to the stats JSON file.
    """

    def __init__(
        self,
        parent: Any,
        analytics_path: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, fg_color=Colors.SURFACE, **kwargs)
        self._analytics_path = analytics_path or _DEFAULT_ANALYTICS_PATH
        self._stats: dict[str, Any] = {}

        # SQLite-backed analytics collector.
        analytics_dir = str(self._analytics_path.parent)
        self._collector: AnalyticsCollector | None = None
        try:
            self._collector = AnalyticsCollector(analytics_dir=analytics_dir)
        except Exception as exc:
            logger.warning("AnalyticsPanel: could not init AnalyticsCollector: %s", exc)

        self._build_ui()
        self.refresh()
        logger.debug("AnalyticsPanel initialised")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_stats(self) -> dict[str, Any]:
        """Load analytics data via AnalyticsCollector (SQLite-backed).

        Falls back to reading stats.json if the collector is not
        available.

        Returns:
            Parsed dictionary with stories, summary, and updated_at.
        """
        # Primary path: read from SQLite via AnalyticsCollector.
        if self._collector is not None:
            try:
                stats = self._collector.get_stats()
                data = stats.model_dump()
                logger.debug(
                    "Analytics loaded from SQLite: %d story records",
                    len(data.get("stories", [])),
                )
                return data
            except Exception as exc:
                logger.warning(
                    "AnalyticsPanel: SQLite read failed, trying JSON fallback: %s",
                    exc,
                )

        # Fallback: read legacy stats.json.
        if not self._analytics_path.exists():
            logger.debug("Analytics file not found: %s", self._analytics_path)
            return {"stories": [], "summary": {}, "updated_at": None}

        try:
            text = self._analytics_path.read_text(encoding="utf-8")
            data = json.loads(text)
            logger.debug(
                "Analytics loaded from JSON: %d story records",
                len(data.get("stories", [])),
            )
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load analytics: %s", exc)
            return {"stories": [], "summary": {}, "updated_at": None}

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all child widgets."""
        self.columnconfigure(0, weight=1)
        row = 0

        # Header + refresh button
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(
            row=row, column=0, sticky="ew",
            padx=Padding.PANEL, pady=(Padding.PANEL, Padding.WIDGET_Y),
        )
        header_frame.columnconfigure(0, weight=1)

        header = create_section_label(header_frame, "ANALYTICS", icon="\U0001F4CA")
        header.grid(row=0, column=0, sticky="w")

        self._refresh_btn = ctk.CTkButton(
            header_frame,
            text="\U0001F504 Refresh",
            command=self.refresh,
            font=Fonts.get(size=Fonts.SMALL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
            width=80,
        )
        self._refresh_btn.grid(row=0, column=1)
        row += 1

        sep = create_separator(self)
        sep.grid(
            row=row, column=0, sticky="ew",
            padx=Padding.PANEL, pady=Padding.LABEL_Y,
        )
        row += 1

        # Scrollable content area
        self._content = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._content.grid(
            row=row, column=0, sticky="nsew",
            padx=Padding.PANEL, pady=Padding.LABEL_Y,
        )
        self._content.columnconfigure(0, weight=1)
        self.grid_rowconfigure(row, weight=1)
        row += 1

        # Export buttons
        export_frame = ctk.CTkFrame(self, fg_color="transparent")
        export_frame.grid(
            row=row, column=0, sticky="ew",
            padx=Padding.PANEL, pady=(Padding.SECTION, Padding.PANEL),
        )

        self._export_json_btn = ctk.CTkButton(
            export_frame,
            text="\U0001F4BE JSON",
            command=self._export_json,
            font=Fonts.get(size=Fonts.SMALL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
            width=80,
        )
        self._export_json_btn.pack(side="left", padx=(0, Padding.WIDGET_X))

        self._export_csv_btn = ctk.CTkButton(
            export_frame,
            text="\U0001F4BE CSV",
            command=self._export_csv,
            font=Fonts.get(size=Fonts.SMALL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
            width=80,
        )
        self._export_csv_btn.pack(side="left")

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _clear_content(self) -> None:
        """Remove all widgets from the scrollable content area."""
        for widget in self._content.winfo_children():
            widget.destroy()

    def _render_stats(self) -> None:
        """Render all analytics sections into the content area."""
        self._clear_content()

        stories = self._stats.get("stories", [])
        summary = self._stats.get("summary", {})
        content_row = 0

        if not stories and not summary:
            empty_lbl = ctk.CTkLabel(
                self._content,
                text="No analytics data yet.\nGenerate some stories to see statistics here.",
                font=Fonts.get(),
                text_color=Colors.MUTED,
            )
            empty_lbl.grid(row=0, column=0, pady=Padding.GROUP_Y)
            return

        content_row = self._render_summary(content_row, summary, stories)
        content_row = self._render_per_language(content_row, stories)
        content_row = self._render_per_provider(content_row, stories)
        content_row = self._render_score_distribution(content_row, stories)
        content_row = self._render_common_issues(content_row, stories)

    def _render_summary(
        self,
        row: int,
        summary: dict[str, Any],
        stories: list[dict[str, Any]],
    ) -> int:
        """Render the all-time summary section.

        Args:
            row: Current grid row.
            summary: Summary dictionary from stats.
            stories: Raw story records.

        Returns:
            Next grid row.
        """
        lbl = ctk.CTkLabel(
            self._content,
            text="All-Time Summary",
            font=Fonts.get(size=Fonts.SUBHEADING, bold=True),
            text_color=Colors.TEXT,
            anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", pady=(0, Padding.LABEL_Y))
        row += 1

        total_stories = summary.get("total_stories", len(stories))
        total_words = summary.get("total_words", 0)
        total_cost = summary.get("total_cost", 0.0)
        avg_cost = total_cost / total_stories if total_stories > 0 else 0.0

        if not total_words and stories:
            total_words = sum(s.get("word_count", 0) for s in stories)
        if not total_cost and stories:
            total_cost = sum(s.get("cost", 0.0) for s in stories)
            avg_cost = total_cost / len(stories) if stories else 0.0

        metrics = [
            ("Total Stories", str(total_stories)),
            ("Total Words", f"{total_words:,}"),
            ("Total Cost", f"${total_cost:.2f}"),
            ("Avg Cost/Story", f"${avg_cost:.4f}"),
        ]

        grid = ctk.CTkFrame(
            self._content, fg_color=Colors.SURFACE_LIGHT, corner_radius=6,
        )
        grid.grid(row=row, column=0, sticky="ew", pady=Padding.LABEL_Y)
        for c in range(len(metrics)):
            grid.columnconfigure(c, weight=1)

        for col, (label, value) in enumerate(metrics):
            cell = ctk.CTkFrame(grid, fg_color="transparent")
            cell.grid(row=0, column=col, padx=Padding.WIDGET_X, pady=Padding.WIDGET_Y)
            ctk.CTkLabel(
                cell, text=label,
                font=Fonts.get(size=Fonts.TINY), text_color=Colors.MUTED,
            ).pack()
            ctk.CTkLabel(
                cell, text=value,
                font=Fonts.get(size=Fonts.BODY, bold=True), text_color=Colors.TEXT,
            ).pack()
        row += 1
        return row

    def _render_per_language(
        self, row: int, stories: list[dict[str, Any]]
    ) -> int:
        """Render the per-language statistics table.

        Args:
            row: Current grid row.
            stories: Raw story records.

        Returns:
            Next grid row.
        """
        lbl = ctk.CTkLabel(
            self._content, text="By Language",
            font=Fonts.get(bold=True), text_color=Colors.TEXT, anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        lang_data: dict[str, dict[str, Any]] = {}
        for s in stories:
            lang = s.get("language", "??")
            entry = lang_data.setdefault(
                lang, {"count": 0, "score_sum": 0.0, "cost": 0.0},
            )
            entry["count"] += 1
            entry["score_sum"] += s.get("score", 0.0)
            entry["cost"] += s.get("cost", 0.0)

        if not lang_data:
            ctk.CTkLabel(
                self._content, text="No data",
                font=Fonts.get(size=Fonts.SMALL), text_color=Colors.MUTED,
            ).grid(row=row, column=0, sticky="w")
            return row + 1

        table = ctk.CTkFrame(
            self._content, fg_color=Colors.SURFACE_LIGHT, corner_radius=6,
        )
        table.grid(row=row, column=0, sticky="ew", pady=Padding.LABEL_Y)
        headers = ["Lang", "Stories", "Avg Score", "Cost"]
        for c, h in enumerate(headers):
            table.columnconfigure(c, weight=1)
            ctk.CTkLabel(
                table, text=h, font=Fonts.get(size=Fonts.TINY, bold=True),
                text_color=Colors.MUTED, anchor="center",
            ).grid(row=0, column=c, padx=4, pady=2)

        for trow, (lang, data) in enumerate(sorted(lang_data.items()), start=1):
            avg = data["score_sum"] / data["count"] if data["count"] else 0.0
            vals = [
                lang.upper(), str(data["count"]),
                f"{avg:.1f}", f"${data['cost']:.2f}",
            ]
            for c, v in enumerate(vals):
                ctk.CTkLabel(
                    table, text=v, font=Fonts.get(size=Fonts.TINY),
                    text_color=Colors.TEXT_DIM, anchor="center",
                ).grid(row=trow, column=c, padx=4, pady=1)
        row += 1
        return row

    def _render_per_provider(
        self, row: int, stories: list[dict[str, Any]]
    ) -> int:
        """Render the per-provider statistics table.

        Args:
            row: Current grid row.
            stories: Raw story records.

        Returns:
            Next grid row.
        """
        lbl = ctk.CTkLabel(
            self._content, text="By Provider",
            font=Fonts.get(bold=True), text_color=Colors.TEXT, anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        prov_data: dict[str, dict[str, Any]] = {}
        for s in stories:
            prov = s.get("provider", s.get("model", "unknown"))
            entry = prov_data.setdefault(
                prov, {"count": 0, "score_sum": 0.0, "cost": 0.0},
            )
            entry["count"] += 1
            entry["score_sum"] += s.get("score", 0.0)
            entry["cost"] += s.get("cost", 0.0)

        if not prov_data:
            ctk.CTkLabel(
                self._content, text="No data",
                font=Fonts.get(size=Fonts.SMALL), text_color=Colors.MUTED,
            ).grid(row=row, column=0, sticky="w")
            return row + 1

        table = ctk.CTkFrame(
            self._content, fg_color=Colors.SURFACE_LIGHT, corner_radius=6,
        )
        table.grid(row=row, column=0, sticky="ew", pady=Padding.LABEL_Y)
        headers = ["Provider", "Stories", "Avg Score", "Cost"]
        for c, h in enumerate(headers):
            table.columnconfigure(c, weight=1)
            ctk.CTkLabel(
                table, text=h, font=Fonts.get(size=Fonts.TINY, bold=True),
                text_color=Colors.MUTED, anchor="center",
            ).grid(row=0, column=c, padx=4, pady=2)

        for trow, (prov, data) in enumerate(sorted(prov_data.items()), start=1):
            avg = data["score_sum"] / data["count"] if data["count"] else 0.0
            vals = [
                prov, str(data["count"]),
                f"{avg:.1f}", f"${data['cost']:.2f}",
            ]
            for c, v in enumerate(vals):
                ctk.CTkLabel(
                    table, text=v, font=Fonts.get(size=Fonts.TINY),
                    text_color=Colors.TEXT_DIM, anchor="center",
                ).grid(row=trow, column=c, padx=4, pady=1)
        row += 1
        return row

    def _render_score_distribution(
        self, row: int, stories: list[dict[str, Any]]
    ) -> int:
        """Render score distribution as horizontal progress bars.

        Args:
            row: Current grid row.
            stories: Raw story records.

        Returns:
            Next grid row.
        """
        lbl = ctk.CTkLabel(
            self._content, text="Score Distribution",
            font=Fonts.get(bold=True), text_color=Colors.TEXT, anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        buckets = {"9.0–10.0": 0, "8.0–9.0": 0, "7.0–8.0": 0, "< 7.0": 0}
        for s in stories:
            score = s.get("score", 0.0)
            if score <= 0:
                continue
            if score >= 9.0:
                buckets["9.0–10.0"] += 1
            elif score >= 8.0:
                buckets["8.0–9.0"] += 1
            elif score >= 7.0:
                buckets["7.0–8.0"] += 1
            else:
                buckets["< 7.0"] += 1

        total = sum(buckets.values()) or 1
        colours = {
            "9.0–10.0": Colors.SUCCESS,
            "8.0–9.0": Colors.WARNING,
            "7.0–8.0": Colors.INFO,
            "< 7.0": Colors.ERROR,
        }

        for bucket, count in buckets.items():
            frame = ctk.CTkFrame(self._content, fg_color="transparent")
            frame.grid(row=row, column=0, sticky="ew", pady=1)
            frame.columnconfigure(1, weight=1)

            ctk.CTkLabel(
                frame, text=bucket,
                font=Fonts.get(size=Fonts.TINY), text_color=Colors.TEXT_DIM,
                width=70, anchor="w",
            ).grid(row=0, column=0, sticky="w")

            bar = ctk.CTkProgressBar(
                frame, height=12,
                fg_color=Colors.ENTRY_BG,
                progress_color=colours.get(bucket, Colors.INFO),
            )
            bar.grid(row=0, column=1, sticky="ew", padx=(4, 4))
            bar.set(count / total)

            ctk.CTkLabel(
                frame, text=str(count),
                font=Fonts.get(size=Fonts.TINY), text_color=Colors.MUTED,
                width=30, anchor="e",
            ).grid(row=0, column=2, sticky="e")

            row += 1
        return row

    def _render_common_issues(
        self, row: int, stories: list[dict[str, Any]]
    ) -> int:
        """Render the most common error/issue strings.

        Args:
            row: Current grid row.
            stories: Raw story records.

        Returns:
            Next grid row.
        """
        lbl = ctk.CTkLabel(
            self._content, text="Common Issues",
            font=Fonts.get(bold=True), text_color=Colors.TEXT, anchor="w",
        )
        lbl.grid(row=row, column=0, sticky="w", pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        issue_counts: dict[str, int] = {}
        for s in stories:
            for issue in s.get("errors", s.get("issues", [])):
                issue_str = str(issue)[:80]
                issue_counts[issue_str] = issue_counts.get(issue_str, 0) + 1

        if not issue_counts:
            ctk.CTkLabel(
                self._content, text="No issues recorded",
                font=Fonts.get(size=Fonts.SMALL), text_color=Colors.MUTED,
            ).grid(row=row, column=0, sticky="w")
            return row + 1

        sorted_issues = sorted(
            issue_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for irow, (issue, count) in enumerate(sorted_issues):
            frame = ctk.CTkFrame(self._content, fg_color="transparent")
            frame.grid(row=row + irow, column=0, sticky="ew", pady=1)
            frame.columnconfigure(1, weight=1)

            ctk.CTkLabel(
                frame, text=f"({count})", font=Fonts.get(size=Fonts.TINY),
                text_color=Colors.WARNING, width=30, anchor="w",
            ).grid(row=0, column=0, sticky="w")

            ctk.CTkLabel(
                frame, text=issue, font=Fonts.get(size=Fonts.TINY),
                text_color=Colors.TEXT_DIM, anchor="w",
            ).grid(row=0, column=1, sticky="w", padx=4)

        row += len(sorted_issues)
        return row

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Reload analytics data and re-render."""
        self._stats = self._load_stats()
        self._render_stats()
        logger.debug("AnalyticsPanel refreshed")

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_json(self) -> None:
        """Export analytics as a JSON file."""
        path = filedialog.asksaveasfilename(
            title="Export Analytics (JSON)",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return

        try:
            Path(path).write_text(
                json.dumps(self._stats, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Analytics exported to JSON: %s", path)
        except OSError as exc:
            logger.error("Failed to export JSON: %s", exc)

    def _export_csv(self) -> None:
        """Export per-story analytics as a CSV file.

        Uses ``csv.writer`` via ``io.StringIO`` so all quoting, escaping,
        and line endings are handled correctly — including fields that
        contain commas, double-quotes, newlines, carriage returns, or
        tabs. The ``errors`` list is serialised as a semicolon-separated
        string so each story occupies exactly one row.
        """
        path = filedialog.asksaveasfilename(
            title="Export Analytics (CSV)",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        stories = self._stats.get("stories", [])
        if not stories:
            logger.warning("No story data to export as CSV")
            return

        # Derive a stable, complete column list.
        seen: set[str] = set()
        columns: list[str] = []
        for s in stories:
            for k in s:
                if k not in seen:
                    seen.add(k)
                    columns.append(k)

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
                # Serialise list values (e.g. errors) to semicolon strings.
                if isinstance(val, list):
                    val = "; ".join(str(v) for v in val)
                row_vals.append(val)
            writer.writerow(row_vals)

        try:
            Path(path).write_text(output.getvalue(), encoding="utf-8")
            logger.info("Analytics exported to CSV: %s", path)
        except OSError as exc:
            logger.error("Failed to export CSV: %s", exc)
