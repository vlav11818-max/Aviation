"""Progress panel for AI Story Generator Pro GUI.

Displays overall progress, per-worker status bars, a task queue table
with colour-coded status indicators, and a statistics bar with elapsed
time, cost, average score, error count, and throughput.

The panel is updated by calling ``update_from_snapshot()`` with a
``ProgressSnapshot`` obtained from ``ProgressTracker.get_snapshot()``.
"""

from __future__ import annotations

import logging
from typing import Any

import customtkinter as ctk

from gui.styles import Colors, Fonts, Padding, create_section_label, create_separator

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────

_MAX_QUEUE_DISPLAY: int = 200


class _WorkerBar(ctk.CTkFrame):
    """A single worker status bar showing topic, step, and progress.

    Args:
        parent: Parent widget.
        worker_id: Numeric worker identifier.
    """

    def __init__(self, parent: Any, worker_id: int, **kwargs: Any) -> None:
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._worker_id = worker_id
        self.columnconfigure(1, weight=1)

        self._id_label = ctk.CTkLabel(
            self,
            text=f"W{worker_id}",
            font=Fonts.get(size=Fonts.TINY, bold=True),
            text_color=Colors.MUTED,
            width=30,
            anchor="w",
        )
        self._id_label.grid(row=0, column=0, padx=(0, Padding.WIDGET_X))

        self._bar = ctk.CTkProgressBar(
            self,
            height=14,
            fg_color=Colors.ENTRY_BG,
            progress_color=Colors.WARNING,
        )
        self._bar.grid(row=0, column=1, sticky="ew")
        self._bar.set(0)

        self._detail_label = ctk.CTkLabel(
            self,
            text="idle",
            font=Fonts.get(size=Fonts.TINY),
            text_color=Colors.MUTED,
            anchor="w",
        )
        self._detail_label.grid(row=1, column=0, columnspan=2, sticky="w")

    def update_status(self, topic: str, step: str, detail: str) -> None:
        """Update the worker bar display.

        Args:
            topic: Current topic name (empty if idle).
            step: Current pipeline step name.
            detail: Extra detail (e.g. section index).
        """
        if topic:
            text = f"{topic}"
            if step:
                text += f" — {step}"
            if detail:
                text += f" ({detail})"
            self._detail_label.configure(text=text, text_color=Colors.TEXT_DIM)
            self._bar.set(0.5)  # indeterminate — pulse at 50 %
            self._bar.configure(progress_color=Colors.WARNING)
        else:
            self._detail_label.configure(text="idle", text_color=Colors.MUTED)
            self._bar.set(0)
            self._bar.configure(progress_color=Colors.ENTRY_BG)


class _TaskRow:
    """One row of the task queue table.

    Attributes:
        index_label: Row number label.
        topic_label: Topic name label.
        status_label: Status icon label.
        attempts_label: Attempt count label.
        score_label: Score label.
    """

    def __init__(
        self,
        parent: Any,
        row: int,
        index: int,
        topic: str,
    ) -> None:
        self.index_label = ctk.CTkLabel(
            parent, text=str(index), width=30,
            font=Fonts.get(size=Fonts.TINY), text_color=Colors.MUTED, anchor="w",
        )
        self.index_label.grid(row=row, column=0, sticky="w", padx=2, pady=1)

        self.topic_label = ctk.CTkLabel(
            parent, text=topic,
            font=Fonts.get(size=Fonts.TINY), text_color=Colors.TEXT, anchor="w",
        )
        self.topic_label.grid(row=row, column=1, sticky="w", padx=2, pady=1)

        self.status_label = ctk.CTkLabel(
            parent, text="\u23F3", width=24,
            font=Fonts.get(size=Fonts.TINY), text_color=Colors.STATUS_QUEUED, anchor="center",
        )
        self.status_label.grid(row=row, column=2, padx=2, pady=1)

        self.attempts_label = ctk.CTkLabel(
            parent, text="0", width=40,
            font=Fonts.get(size=Fonts.TINY), text_color=Colors.MUTED, anchor="center",
        )
        self.attempts_label.grid(row=row, column=3, padx=2, pady=1)

        self.score_label = ctk.CTkLabel(
            parent, text="—", width=40,
            font=Fonts.get(size=Fonts.TINY), text_color=Colors.MUTED, anchor="center",
        )
        self.score_label.grid(row=row, column=4, padx=2, pady=1)

    def set_status(
        self,
        status: str,
        attempts: int = 0,
        score: float = 0.0,
    ) -> None:
        """Update the row's status, attempts, and score.

        Args:
            status: One of ``"queued"``, ``"running"``, ``"done"``,
                ``"failed"``.
            attempts: Number of evaluation attempts.
            score: Final score (0.0 if not evaluated).
        """
        icon_map = {
            "queued": ("\u23F3", Colors.STATUS_QUEUED),
            "running": ("\u25B6", Colors.STATUS_RUNNING),
            "done": ("\u2705", Colors.STATUS_DONE),
            "failed": ("\u274C", Colors.STATUS_FAILED),
            "paused": ("\u23F8", Colors.STATUS_PAUSED),
        }
        icon, colour = icon_map.get(status, ("\u2753", Colors.MUTED))
        self.status_label.configure(text=icon, text_color=colour)
        self.attempts_label.configure(text=str(attempts))

        if score > 0:
            score_colour = Colors.SUCCESS if score >= 9.0 else Colors.WARNING
            self.score_label.configure(text=f"{score:.1f}", text_color=score_colour)
        else:
            self.score_label.configure(text="—", text_color=Colors.MUTED)


class ProgressPanel(ctk.CTkFrame):
    """Progress display panel.

    Shows overall progress, per-worker bars, task queue table, and
    a statistics bar.

    Args:
        parent: Parent widget.
        max_workers: Maximum number of worker bars to display.
    """

    def __init__(
        self,
        parent: Any,
        max_workers: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, fg_color=Colors.SURFACE, **kwargs)
        self._max_workers = max_workers
        self._worker_bars: list[_WorkerBar] = []
        self._task_rows: dict[str, _TaskRow] = {}
        self._task_row_counter: int = 0

        self._build_ui()
        logger.debug("ProgressPanel initialised (max_workers=%d)", max_workers)

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all child widgets."""
        self.columnconfigure(0, weight=1)
        row = 0

        # ── Overall progress ────────────────────────────────────────
        header = create_section_label(self, "PROGRESS", icon="\U0001F4CA")
        header.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.PANEL, Padding.WIDGET_Y))
        row += 1

        overall_frame = ctk.CTkFrame(self, fg_color="transparent")
        overall_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.WIDGET_Y)
        overall_frame.columnconfigure(0, weight=1)

        self._overall_bar = ctk.CTkProgressBar(
            overall_frame,
            height=22,
            fg_color=Colors.ENTRY_BG,
            progress_color=Colors.SUCCESS,
        )
        self._overall_bar.grid(row=0, column=0, sticky="ew", padx=(0, Padding.WIDGET_X))
        self._overall_bar.set(0)

        self._overall_label = ctk.CTkLabel(
            overall_frame,
            text="0 / 0 (0%)",
            font=Fonts.get(size=Fonts.SMALL, bold=True),
            text_color=Colors.TEXT,
            width=120,
            anchor="e",
        )
        self._overall_label.grid(row=0, column=1)
        row += 1

        # ── Per-worker bars ─────────────────────────────────────────
        sep1 = create_separator(self)
        sep1.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        row += 1

        workers_label = ctk.CTkLabel(
            self, text="Workers",
            font=Fonts.get(size=Fonts.SMALL, bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        workers_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        self._workers_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._workers_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        self._workers_frame.columnconfigure(0, weight=1)

        for wid in range(self._max_workers):
            bar = _WorkerBar(self._workers_frame, worker_id=wid + 1)
            bar.grid(row=wid, column=0, sticky="ew", pady=2)
            self._worker_bars.append(bar)
        row += 1

        # ── Task queue table ────────────────────────────────────────
        sep2 = create_separator(self)
        sep2.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        row += 1

        queue_label = ctk.CTkLabel(
            self, text="Task Queue",
            font=Fonts.get(size=Fonts.SMALL, bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        queue_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        self._queue_scroll = ctk.CTkScrollableFrame(
            self,
            fg_color=Colors.SURFACE_LIGHT,
            height=180,
        )
        self._queue_scroll.grid(row=row, column=0, sticky="nsew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        self._queue_scroll.columnconfigure(1, weight=1)
        self.grid_rowconfigure(row, weight=1)

        # Table header
        headers = ["#", "Topic", "Status", "Att.", "Score"]
        widths = [30, 0, 24, 40, 40]
        for col, (text, w) in enumerate(zip(headers, widths)):
            lbl = ctk.CTkLabel(
                self._queue_scroll, text=text, width=w if w else 0,
                font=Fonts.get(size=Fonts.TINY, bold=True),
                text_color=Colors.MUTED, anchor="w" if col < 2 else "center",
            )
            lbl.grid(row=0, column=col, sticky="w" if col < 2 else "", padx=2, pady=(0, 2))
        row += 1

        # ── Stats bar ──────────────────────────────────────────────
        sep3 = create_separator(self)
        sep3.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        row += 1

        self._stats_frame = ctk.CTkFrame(self, fg_color=Colors.SURFACE_LIGHT, corner_radius=6, height=32)
        self._stats_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=(0, Padding.PANEL))
        self._stats_frame.grid_propagate(False)
        for c in range(5):
            self._stats_frame.columnconfigure(c, weight=1)

        self._elapsed_label = self._stat_cell(self._stats_frame, 0, "\u23F1 00:00:00")
        self._cost_stat_label = self._stat_cell(self._stats_frame, 1, "\U0001F4B0 $0.0000")
        self._avg_score_label = self._stat_cell(self._stats_frame, 2, "\U0001F4C8 Avg: 0.0")
        self._error_label = self._stat_cell(self._stats_frame, 3, "\u274C Err: 0")
        self._speed_label = self._stat_cell(self._stats_frame, 4, "\u26A1 0.0/h")

    @staticmethod
    def _stat_cell(parent: ctk.CTkFrame, col: int, text: str) -> ctk.CTkLabel:
        """Create a statistics cell label.

        Args:
            parent: Parent frame.
            col: Column index.
            text: Initial text.

        Returns:
            The created label.
        """
        lbl = ctk.CTkLabel(
            parent, text=text,
            font=Fonts.get(size=Fonts.TINY),
            text_color=Colors.TEXT_DIM, anchor="center",
        )
        lbl.grid(row=0, column=col, sticky="ew", padx=4, pady=4)
        return lbl

    # ── Public API ──────────────────────────────────────────────────

    def set_topics(self, topics: list[str]) -> None:
        """Populate the task queue table with initial topics.

        Args:
            topics: List of topic strings in processing order.
        """
        # Clear existing rows
        for widget in self._queue_scroll.winfo_children():
            if not isinstance(widget, ctk.CTkLabel):
                continue
            # Keep header row (row 0)
            info = widget.grid_info()
            if info and int(info.get("row", 0)) > 0:
                widget.destroy()

        self._task_rows.clear()
        self._task_row_counter = 0

        for idx, topic in enumerate(topics[:_MAX_QUEUE_DISPLAY]):
            grid_row = idx + 1  # row 0 is header
            task_row = _TaskRow(
                self._queue_scroll, grid_row,
                index=idx + 1, topic=topic,
            )
            self._task_rows[topic] = task_row
            self._task_row_counter = idx + 1

        logger.debug("ProgressPanel: loaded %d topics into queue", len(topics))

    def update_topic_status(
        self,
        topic: str,
        status: str,
        attempts: int = 0,
        score: float = 0.0,
    ) -> None:
        """Update a single topic's status in the task queue.

        Args:
            topic: Topic string.
            status: One of ``"queued"``, ``"running"``, ``"done"``,
                ``"failed"``, ``"paused"``.
            attempts: Number of evaluation attempts.
            score: Final score.
        """
        row = self._task_rows.get(topic)
        if row is not None:
            row.set_status(status, attempts=attempts, score=score)

    def update_from_snapshot(self, snapshot: Any) -> None:
        """Update all display elements from a ``ProgressSnapshot``.

        Args:
            snapshot: A ``ProgressSnapshot`` instance from
                ``ProgressTracker.get_snapshot()``.
        """
        # Overall progress bar
        total = snapshot.overall_total
        completed = snapshot.overall_completed
        pct = snapshot.overall_percent

        progress_value = pct / 100.0 if total > 0 else 0.0
        self._overall_bar.set(progress_value)
        self._overall_label.configure(text=f"{completed} / {total} ({pct:.0f}%)")

        # Worker bars
        for idx, bar in enumerate(self._worker_bars):
            if idx < len(snapshot.workers):
                ws = snapshot.workers[idx]
                bar.update_status(ws.topic, ws.step, ws.detail)
            else:
                bar.update_status("", "", "")

        # Stats bar
        elapsed_secs = snapshot.elapsed_time
        hours = int(elapsed_secs // 3600)
        minutes = int((elapsed_secs % 3600) // 60)
        seconds = int(elapsed_secs % 60)
        self._elapsed_label.configure(text=f"\u23F1 {hours:02d}:{minutes:02d}:{seconds:02d}")
        self._cost_stat_label.configure(text=f"\U0001F4B0 ${snapshot.total_cost:.4f}")
        self._avg_score_label.configure(text=f"\U0001F4C8 Avg: {snapshot.avg_score:.1f}")
        self._error_label.configure(text=f"\u274C Err: {snapshot.error_count}")
        self._speed_label.configure(text=f"\u26A1 {snapshot.speed_stories_per_hour:.1f}/h")

    def reset(self) -> None:
        """Clear all progress state for a new batch."""
        self._overall_bar.set(0)
        self._overall_label.configure(text="0 / 0 (0%)")
        for bar in self._worker_bars:
            bar.update_status("", "", "")
        self._elapsed_label.configure(text="\u23F1 00:00:00")
        self._cost_stat_label.configure(text="\U0001F4B0 $0.0000")
        self._avg_score_label.configure(text="\U0001F4C8 Avg: 0.0")
        self._error_label.configure(text="\u274C Err: 0")
        self._speed_label.configure(text="\u26A1 0.0/h")

        # Clear task rows
        for widget in list(self._queue_scroll.winfo_children()):
            info = widget.grid_info()
            if info and int(info.get("row", 0)) > 0:
                widget.destroy()
        self._task_rows.clear()
        self._task_row_counter = 0
        logger.debug("ProgressPanel reset")
