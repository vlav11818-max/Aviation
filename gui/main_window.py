"""Main application window for AI Story Generator Pro.

Provides a two-column layout with a left sidebar (input, style, action
buttons) and a right content area (tabbed notebook with progress, log,
and analytics panels).  Orchestrates generation by collecting parameters
from panels, validating input, showing cost estimation, and dispatching
work to ``ParallelProcessor`` in a background thread.

The window polls the ``EventBus`` every 100 ms via ``self.after()``
to update progress, log entries, and cost counters.

FIX: ProgressPanel and AnalyticsPanel are now instantiated and embedded
     into their respective tabs (were placeholders before).
     _on_close() saves user preferences before destroying.
     _on_generation_finished() refreshes the analytics panel.
     _on_start() loads topics into progress panel and switches to tab.
     _handle_event() updates progress panel on step/story events.

FIX v2: Adaptation mode now correctly uses TextAdapter instead of the
     generation pipeline.  StylePanel is hidden and AdaptationPanel is
     shown when Adapt mode is selected.  Single-file and folder source
     selection are both supported.  Source language is auto-detected.

FIX v3: topics_file_path is passed to ParallelProcessor.process_batch()
     so that successfully completed topics are marked with "OK " prefix
     in the source file for easy tracking and re-run skipping.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from core.events import Event, EventBus, EventType
from gui.adaptation_panel import AdaptationPanel
from gui.api_settings_panel import ApiSettingsPanel
from gui.analytics_panel import AnalyticsPanel
from gui.input_panel import InputPanel
from gui.progress_panel import ProgressPanel
from gui.style_panel import StylePanel
from gui.styles import Colors, Fonts, Padding, apply_theme

if TYPE_CHECKING:
    from core.cost_estimator import CostEstimate, CostEstimator
    from core.input_validator import InputValidator
    from core.parallel_processor import ParallelProcessor
    from core.settings import Settings

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────

_WINDOW_TITLE: str = "AI Story Generator Pro v1.0"
_WINDOW_MIN_WIDTH: int = 1100
_WINDOW_MIN_HEIGHT: int = 700
_DEFAULT_GEOMETRY: str = "1280x800"
_POLL_INTERVAL_MS: int = 100
_LOG_MAX_LINES: int = 5000


class MainWindow(ctk.CTk):
    """Main application window.

    Manages the overall layout, panel wiring, menu bar, hotkeys,
    EventBus polling, and generation orchestration.

    Args:
        event_bus: Thread-safe event bus for core-GUI communication.
        settings: Application settings.
        parallel_processor: Batch processing engine (may be ``None``
            during early startup; wired later via ``set_processor``).
        input_validator: Input validation utility.
        cost_estimator: Cost estimation utility.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: "Settings",
        parallel_processor: "ParallelProcessor | None" = None,
        input_validator: "InputValidator | None" = None,
        cost_estimator: "CostEstimator | None" = None,
    ) -> None:
        super().__init__()
        self._event_bus = event_bus
        self._settings = settings
        self._processor = parallel_processor
        self._validator = input_validator
        self._cost_estimator = cost_estimator

        # Generation state
        self._is_generating: bool = False
        self._is_paused: bool = False
        self._generation_thread: threading.Thread | None = None

        # Apply theme
        apply_theme(self)

        # Window config
        self.title(_WINDOW_TITLE)
        self.geometry(_DEFAULT_GEOMETRY)
        self.minsize(_WINDOW_MIN_WIDTH, _WINDOW_MIN_HEIGHT)

        # Build UI
        self._build_layout()
        self._build_menu()
        self._bind_hotkeys()

        # Start event polling
        self.after(_POLL_INTERVAL_MS, self._poll_events)

        logger.info("MainWindow initialised")

    # ── Dependency injection (late binding) ──────────────────────────

    def set_processor(self, processor: "ParallelProcessor") -> None:
        """Wire the parallel processor after construction.

        Args:
            processor: The ``ParallelProcessor`` instance.
        """
        self._processor = processor
        logger.debug("ParallelProcessor wired to MainWindow")

    def set_validator(self, validator: "InputValidator") -> None:
        """Wire the input validator after construction.

        Args:
            validator: The ``InputValidator`` instance.
        """
        self._validator = validator
        logger.debug("InputValidator wired to MainWindow")

    def set_cost_estimator(self, estimator: "CostEstimator") -> None:
        """Wire the cost estimator after construction.

        Args:
            estimator: The ``CostEstimator`` instance.
        """
        self._cost_estimator = estimator
        logger.debug("CostEstimator wired to MainWindow")

    # ── Layout ──────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        """Build the two-column layout with panels."""
        # Root grid: 1 row, 2 columns
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=Padding.LEFT_COL_WEIGHT, minsize=320)
        self.grid_columnconfigure(1, weight=Padding.RIGHT_COL_WEIGHT)

        # ── Left column (scrollable sidebar) ────────────────────────
        self._left_scroll = ctk.CTkScrollableFrame(
            self,
            fg_color=Colors.BACKGROUND,
            scrollbar_fg_color=Colors.SCROLLBAR,
            scrollbar_button_color=Colors.SCROLLBAR,
            scrollbar_button_hover_color=Colors.SCROLLBAR_HOVER,
        )
        self._left_scroll.grid(
            row=0, column=0, sticky="nsew",
            padx=(Padding.WINDOW, 0), pady=Padding.WINDOW,
        )
        self._left_scroll.columnconfigure(0, weight=1)

        left_row = 0

        # Input panel
        self._input_panel = InputPanel(
            self._left_scroll,
            on_mode_change=self._on_mode_changed,
        )
        self._input_panel.grid(row=left_row, column=0, sticky="ew", pady=(0, Padding.SECTION))
        left_row += 1

        # Style panel (shown in Generate mode)
        self._style_panel = StylePanel(self._left_scroll)
        self._style_panel.grid(row=left_row, column=0, sticky="ew", pady=Padding.SECTION)

        # Adaptation panel (shown in Adapt mode, initially hidden)
        self._adaptation_panel = AdaptationPanel(self._left_scroll)
        # Same grid row as style_panel — they occupy the same slot
        self._style_adapt_row = left_row
        left_row += 1

        # API settings panel
        self._api_panel = ApiSettingsPanel(self._left_scroll, event_bus=self._event_bus)
        self._api_panel.grid(row=left_row, column=0, sticky="ew", pady=Padding.SECTION)
        left_row += 1

        # ── Action buttons ──────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self._left_scroll, fg_color="transparent")
        btn_frame.grid(row=left_row, column=0, sticky="ew", pady=Padding.SECTION)
        btn_frame.columnconfigure(0, weight=1)

        self._start_btn = ctk.CTkButton(
            btn_frame,
            text="\U0001F680 START",
            command=self._on_start,
            font=Fonts.get(size=Fonts.SUBHEADING, bold=True),
            fg_color=Colors.SUCCESS,
            hover_color=Colors.SUCCESS_HOVER,
            text_color=Colors.TEXT,
            height=42,
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", pady=Padding.LABEL_Y)

        self._pause_btn = ctk.CTkButton(
            btn_frame,
            text="\u23F8 PAUSE",
            command=self._on_pause,
            font=Fonts.get(bold=True),
            fg_color=Colors.WARNING,
            hover_color=Colors.WARNING_HOVER,
            text_color=Colors.TEXT,
            height=36,
            state="disabled",
        )
        self._pause_btn.grid(row=1, column=0, sticky="ew", pady=Padding.LABEL_Y)

        self._stop_btn = ctk.CTkButton(
            btn_frame,
            text="\u23F9 STOP",
            command=self._on_stop,
            font=Fonts.get(bold=True),
            fg_color=Colors.ERROR,
            hover_color=Colors.ERROR_HOVER,
            text_color=Colors.TEXT,
            height=36,
            state="disabled",
        )
        self._stop_btn.grid(row=2, column=0, sticky="ew", pady=Padding.LABEL_Y)

        self._cost_label = ctk.CTkLabel(
            btn_frame,
            text="",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED,
            anchor="w",
        )
        self._cost_label.grid(row=3, column=0, sticky="w", pady=Padding.LABEL_Y)

        left_row += 1

        # ── Right column (tabbed notebook) ──────────────────────────
        self._right_frame = ctk.CTkFrame(self, fg_color=Colors.BACKGROUND)
        self._right_frame.grid(
            row=0, column=1, sticky="nsew",
            padx=Padding.WINDOW, pady=Padding.WINDOW,
        )
        self._right_frame.grid_rowconfigure(0, weight=1)
        self._right_frame.grid_columnconfigure(0, weight=1)

        self._notebook = ctk.CTkTabview(
            self._right_frame,
            fg_color=Colors.SURFACE,
            segmented_button_fg_color=Colors.PRIMARY,
            segmented_button_selected_color=Colors.PRIMARY_HOVER,
            segmented_button_selected_hover_color=Colors.PRIMARY_LIGHT,
            segmented_button_unselected_color=Colors.SURFACE_LIGHT,
            segmented_button_unselected_hover_color=Colors.SURFACE_HOVER,
            text_color=Colors.TEXT,
        )
        self._notebook.grid(row=0, column=0, sticky="nsew")

        # Create tabs
        self._tab_progress = self._notebook.add("Progress")
        self._tab_log = self._notebook.add("Log")
        self._tab_analytics = self._notebook.add("Analytics")

        # ── Progress tab — real ProgressPanel ───────────────────────
        self._tab_progress.grid_rowconfigure(0, weight=1)
        self._tab_progress.grid_columnconfigure(0, weight=1)

        max_workers = getattr(
            getattr(self._settings, "parallelism", None),
            "max_workers",
            3,
        )
        self._progress_panel = ProgressPanel(
            self._tab_progress,
            max_workers=max_workers,
        )
        self._progress_panel.grid(row=0, column=0, sticky="nsew")

        # ── Log tab ─────────────────────────────────────────────────
        self._tab_log.grid_rowconfigure(0, weight=1)
        self._tab_log.grid_columnconfigure(0, weight=1)

        self._log_text = ctk.CTkTextbox(
            self._tab_log,
            font=Fonts.get(size=Fonts.MONO, mono=True),
            text_color=Colors.TEXT,
            fg_color=Colors.SURFACE_LIGHT,
            wrap="word",
            state="disabled",
        )
        self._log_text.grid(row=0, column=0, sticky="nsew", padx=Padding.WIDGET_X, pady=Padding.WIDGET_Y)

        # Log footer buttons
        log_footer = ctk.CTkFrame(self._tab_log, fg_color="transparent")
        log_footer.grid(row=1, column=0, sticky="ew", padx=Padding.WIDGET_X, pady=Padding.LABEL_Y)

        self._clear_log_btn = ctk.CTkButton(
            log_footer,
            text="\U0001F5D1 Clear Log",
            command=self._clear_log,
            font=Fonts.get(size=Fonts.SMALL),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
            width=100,
        )
        self._clear_log_btn.pack(side="right")

        # ── Analytics tab — real AnalyticsPanel ─────────────────────
        self._tab_analytics.grid_rowconfigure(0, weight=1)
        self._tab_analytics.grid_columnconfigure(0, weight=1)

        analytics_path = Path(
            getattr(
                getattr(self._settings, "paths", None),
                "analytics_dir",
                "data/analytics",
            )
        ) / "stats.json"
        self._analytics_panel = AnalyticsPanel(
            self._tab_analytics,
            analytics_path=analytics_path,
        )
        self._analytics_panel.grid(row=0, column=0, sticky="nsew")

        # ── Status bar ──────────────────────────────────────────────
        self._status_bar = ctk.CTkFrame(self, fg_color=Colors.SURFACE, height=28)
        self._status_bar.grid(
            row=1, column=0, columnspan=2, sticky="ew",
            padx=0, pady=0,
        )
        self._status_bar.grid_propagate(False)
        self._status_bar.columnconfigure(0, weight=1)

        self._status_label = ctk.CTkLabel(
            self._status_bar,
            text="Ready",
            font=Fonts.get(size=Fonts.TINY),
            text_color=Colors.MUTED,
            anchor="w",
        )
        self._status_label.grid(row=0, column=0, sticky="w", padx=Padding.WIDGET_X, pady=2)

        self._cost_status = ctk.CTkLabel(
            self._status_bar,
            text="",
            font=Fonts.get(size=Fonts.TINY),
            text_color=Colors.MUTED,
            anchor="e",
        )
        self._cost_status.grid(row=0, column=1, sticky="e", padx=Padding.WIDGET_X, pady=2)

    # ── Menu bar ────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        """Build the application menu bar.

        Note: customtkinter does not have a native menu widget, so we
        use tkinter's Menu attached to the root.
        """
        import tkinter as tk

        menubar = tk.Menu(self, tearoff=False)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open Topics File  (Ctrl+O)", command=self._hotkey_open_file)
        file_menu.add_command(label="Select Folder  (Ctrl+D)", command=self._hotkey_select_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export Report  (Ctrl+E)", command=self._hotkey_export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=False)
        settings_menu.add_command(label="Reset All Settings", command=self._reset_all)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.configure(menu=menubar)

    # ── Hotkeys ─────────────────────────────────────────────────────

    def _bind_hotkeys(self) -> None:
        """Bind keyboard shortcuts."""
        self.bind("<Control-o>", lambda e: self._hotkey_open_file())
        self.bind("<Control-r>", lambda e: self._on_start())
        self.bind("<Control-p>", lambda e: self._on_pause())
        self.bind("<Control-d>", lambda e: self._hotkey_select_folder())
        self.bind("<Control-e>", lambda e: self._hotkey_export_report())
        self.bind("<Control-l>", lambda e: self._clear_log())
        self.bind("<Control-a>", lambda e: self._hotkey_analytics())
        self.bind("<Escape>", lambda e: self._on_stop())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _hotkey_open_file(self) -> None:
        """Trigger the topics file picker from a hotkey."""
        self._input_panel._pick_topics_file()

    def _hotkey_select_folder(self) -> None:
        """Trigger the folder picker from a hotkey."""
        self._input_panel._pick_adapt_folder()

    def _hotkey_export_report(self) -> None:
        """Export report via ReportDialog."""
        try:
            from gui.report_dialog import ReportDialog
            ReportDialog(self)
        except Exception as exc:
            logger.warning("Could not open report dialog: %s", exc)
            self._append_log("Export report requested")

    def _hotkey_analytics(self) -> None:
        """Switch to the analytics tab."""
        self._notebook.set("Analytics")

    # ── Event bus polling ───────────────────────────────────────────

    def _poll_events(self) -> None:
        """Drain all pending events from the EventBus and handle them.

        Scheduled to run every ``_POLL_INTERVAL_MS`` milliseconds.
        """
        events = self._event_bus.poll_all()
        for event in events:
            self._handle_event(event)

        # Reschedule
        self.after(_POLL_INTERVAL_MS, self._poll_events)

    def _handle_event(self, event: Event) -> None:
        """Process a single event from the EventBus.

        Args:
            event: The event to process.
        """
        etype = event.type
        data = event.data

        if etype == EventType.LOG_MESSAGE:
            msg = data.get("message", "")
            level = data.get("level", "INFO")
            # Check for connection-test result signal from APISettingsPanel.
            if msg.startswith("__CONNECTION_TEST__|"):
                parts = msg.split("|", 2)
                if len(parts) == 3 and hasattr(self, "_api_panel"):
                    success = parts[1] == "True"
                    result_msg = parts[2]
                    self._api_panel._handle_connection_test_event(
                        success, result_msg
                    )
                return
            self._append_log(msg, level=level)

        elif etype == EventType.STEP_STARTED:
            step = data.get("step", "?")
            topic = data.get("topic", "?")
            self._append_log(f"Step '{step}' started for: {topic}")
            self._status_label.configure(text=f"Running: {step} — {topic}")
            # Update progress panel
            self._progress_panel.update_topic_status(topic, "running")

        elif etype == EventType.STEP_COMPLETED:
            step = data.get("step", "?")
            topic = data.get("topic", "?")
            self._append_log(f"Step '{step}' completed for: {topic}")

        elif etype == EventType.SECTION_COMPLETED:
            section_idx = data.get("section_index", "?")
            total = data.get("total_sections", "?")
            topic = data.get("topic", "?")
            self._append_log(f"Section {section_idx}/{total} done for: {topic}")

        elif etype == EventType.EVALUATION_RESULT:
            score = data.get("score", 0.0)
            attempt = data.get("attempt", 1)
            topic = data.get("topic", "?")
            passed = data.get("passed", False)
            icon = "\u2705" if passed else "\u26A0\uFE0F"
            self._append_log(
                f"{icon} Evaluation: score={score:.1f}, attempt={attempt} for: {topic}"
            )
            status = "done" if passed else "running"
            self._progress_panel.update_topic_status(
                topic, status, attempts=attempt, score=score,
            )

        elif etype == EventType.REVISION_STARTED:
            attempt = data.get("attempt", 1)
            topic = data.get("topic", "?")
            self._append_log(f"Revision attempt {attempt} for: {topic}")

        elif etype == EventType.STORY_COMPLETED:
            topic = data.get("topic", "?")
            score = data.get("score", 0.0)
            attempts = data.get("attempts", 1)
            self._append_log(f"\u2705 Story completed: {topic} (score: {score:.1f})")
            self._progress_panel.update_topic_status(
                topic, "done", attempts=attempts, score=score,
            )

        elif etype == EventType.BATCH_COMPLETED:
            total = data.get("total", 0)
            completed = data.get("completed", 0)
            failed = data.get("failed", 0)
            elapsed = data.get("elapsed_seconds", 0.0)
            self._append_log(
                f"\U0001F3C1 Batch complete: {completed}/{total} done, "
                f"{failed} failed, {elapsed:.1f}s"
            )
            self._on_generation_finished()

        elif etype == EventType.COST_UPDATE:
            cost = data.get("total_cost_usd", 0.0)
            self._cost_status.configure(text=f"\U0001F4B0 ${cost:.4f}")

        elif etype == EventType.API_ERROR:
            error = data.get("error", "Unknown")
            provider = data.get("provider", "?")
            self._append_log(f"\u274C API error ({provider}): {error}", level="ERROR")

        elif etype == EventType.API_FALLBACK:
            from_prov = data.get("from_provider", "?")
            to_prov = data.get("to_provider", "?")
            self._append_log(f"\u26A1 Fallback: {from_prov} \u2192 {to_prov}", level="WARNING")

        elif etype == EventType.STEP_FAILED:
            step = data.get("step", "?")
            topic = data.get("topic", "?")
            error = data.get("error", "Unknown")
            self._append_log(f"\u274C Step '{step}' failed for {topic}: {error}", level="ERROR")
            self._progress_panel.update_topic_status(topic, "failed")

    # ── Log helpers ─────────────────────────────────────────────────

    def _append_log(self, message: str, level: str = "INFO") -> None:
        """Append a timestamped line to the log panel.

        Args:
            message: The text to display.
            level: Severity level for optional formatting.
        """
        now = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = ""
        if level == "ERROR":
            prefix = "[ERROR] "
        elif level == "WARNING":
            prefix = "[WARN]  "

        line = f"[{now}] {prefix}{message}\n"

        self._log_text.configure(state="normal")
        self._log_text.insert("end", line)

        # Trim if too long
        line_count = int(self._log_text.index("end-1c").split(".")[0])
        if line_count > _LOG_MAX_LINES:
            self._log_text.delete("1.0", f"{line_count - _LOG_MAX_LINES}.0")

        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        """Clear all log text."""
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        logger.debug("Log panel cleared")

    # ── Generation orchestration ────────────────────────────────────

    def _on_start(self) -> None:
        """Handle the START button click.

        Collects parameters, validates input, optionally shows cost
        estimation, and starts the generation or adaptation in a
        background thread.
        """
        if self._is_generating:
            self._append_log("Generation already in progress")
            return

        # Collect configuration
        input_cfg = self._input_panel.get_config()
        mode = input_cfg["mode"]
        language = input_cfg["language"]

        if mode == "generate":
            self._start_generation(input_cfg, language)
        elif mode == "adapt":
            self._start_adaptation(input_cfg)
        else:
            self._append_log(f"\u274C Unknown mode: {mode}", level="ERROR")

    def _start_generation(self, input_cfg: dict[str, Any], language: str) -> None:
        """Validate inputs and launch the generation pipeline.

        Args:
            input_cfg: Configuration dictionary from InputPanel.
            language: Two-letter language code.
        """
        topics_file = input_cfg["topics_file"]
        if not topics_file:
            self._append_log("\u274C No topics file selected", level="ERROR")
            self._status_label.configure(text="Error: no topics file")
            return

        # Validate topics file
        topics = self._validate_topics_file(topics_file)
        if topics is None:
            return

        gen_config = self._style_panel.get_config()
        api_config = self._api_panel.get_config()

        # Cost estimation
        if self._cost_estimator is not None:
            from core.strategies import select_strategy
            strategy = select_strategy(gen_config.target_words)
            strategy_name = getattr(strategy, "name", "full_pipeline")
            if isinstance(strategy_name, property):
                strategy_name = "full_pipeline"
            try:
                estimate = self._cost_estimator.estimate_cost(
                    topics_count=len(topics),
                    target_words=gen_config.target_words,
                    strategy_name=strategy_name,
                    model=api_config.primary_model,
                )
                self._cost_label.configure(
                    text=f"\U0001F4B0 Estimated cost: ~${estimate.total_usd:.2f}"
                )
                self._append_log(
                    f"Cost estimate: ~${estimate.total_usd:.2f} "
                    f"(${estimate.per_story_usd:.4f}/story, "
                    f"{len(topics)} topics)"
                )
            except Exception as exc:
                logger.warning("Cost estimation failed: %s", exc)
                self._cost_label.configure(text="")

        # Load topics into progress panel and switch to it
        self._progress_panel.set_topics(topics)
        self._notebook.set("Progress")

        # Start generation
        self._append_log(
            f"\U0001F680 Starting: {len(topics)} topics, "
            f"language={language}, model={api_config.primary_model}"
        )

        self._set_generating_state(True)

        if self._processor is not None:
            self._processor.reset_signals()
            self._generation_thread = threading.Thread(
                target=self._run_generation,
                args=(topics, gen_config, api_config, language, topics_file),
                daemon=True,
                name="generation-worker",
            )
            self._generation_thread.start()
        else:
            self._append_log(
                "\u26A0\uFE0F ParallelProcessor not available — "
                "generation cannot start",
                level="WARNING",
            )
            self._set_generating_state(False)

    def _start_adaptation(self, input_cfg: dict[str, Any]) -> None:
        """Validate inputs and launch the adaptation pipeline.

        Reads source files, detects source language, and runs
        TextAdapter for each file × target language combination.

        Args:
            input_cfg: Configuration dictionary from InputPanel.
        """
        source_files = input_cfg.get("adapt_source_files", [])
        if not source_files:
            self._append_log(
                "\u274C No source files selected. "
                "Use 'Source File' or 'Source Folder' to select texts.",
                level="ERROR",
            )
            self._status_label.configure(text="Error: no source files")
            return

        target_langs = input_cfg.get("target_languages", [])
        if not target_langs:
            self._append_log(
                "\u274C No target languages selected", level="ERROR",
            )
            self._status_label.configure(text="Error: select target languages")
            return

        source_lang = input_cfg.get("detected_source_lang", "")
        if not source_lang:
            self._append_log(
                "\u274C Source language not detected. "
                "Please select a source file first so the language "
                "can be auto-detected.",
                level="ERROR",
            )
            self._status_label.configure(text="Error: source language unknown")
            return

        # Warn if target includes source language
        if source_lang in target_langs:
            self._append_log(
                f"\u26A0\uFE0F Target languages include the source language "
                f"({source_lang}) — skipping it.",
                level="WARNING",
            )
            target_langs = [lg for lg in target_langs if lg != source_lang]
            if not target_langs:
                self._append_log(
                    "\u274C No target languages remaining after removing "
                    "source language.",
                    level="ERROR",
                )
                return

        # Get adaptation config from AdaptationPanel
        adapt_cfg = self._adaptation_panel.get_config()
        api_config = self._api_panel.get_config()

        # Build task list for progress panel
        task_names = []
        for src_file in source_files:
            for tgt_lang in target_langs:
                task_names.append(f"{src_file.stem} → {tgt_lang}")

        total_tasks = len(task_names)
        self._progress_panel.set_topics(task_names)
        self._notebook.set("Progress")

        self._append_log(
            f"\U0001F310 Starting adaptation: {len(source_files)} file(s), "
            f"{len(target_langs)} target language(s) = {total_tasks} tasks. "
            f"Source: {source_lang}, Targets: {', '.join(target_langs)}, "
            f"Mode: {adapt_cfg['mode']}, Model: {api_config.primary_model}"
        )

        self._set_generating_state(True)

        self._generation_thread = threading.Thread(
            target=self._run_adaptation,
            args=(
                source_files,
                source_lang,
                target_langs,
                adapt_cfg,
                api_config,
            ),
            daemon=True,
            name="adaptation-worker",
        )
        self._generation_thread.start()

    def _validate_topics_file(self, path: str) -> list[str] | None:
        """Validate a topics file and return the topic list.

        Args:
            path: Path to the topics file.

        Returns:
            List of topic strings, or ``None`` on validation failure.
        """
        if self._validator is not None:
            try:
                result = self._validator.validate_topics_file(path)
                for warning in result.warnings:
                    self._append_log(f"\u26A0\uFE0F {warning}", level="WARNING")
                if not result.topics:
                    self._append_log("\u274C Topics file is empty", level="ERROR")
                    return None
                self._append_log(f"Loaded {len(result.topics)} topics from file")
                return result.topics
            except Exception as exc:
                self._append_log(f"\u274C Validation error: {exc}", level="ERROR")
                return None
        else:
            # Fallback: read file directly
            try:
                text = Path(path).read_text(encoding="utf-8")
                topics = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip() and not line.strip().startswith("OK ")
                ]
                if not topics:
                    self._append_log("\u274C Topics file is empty", level="ERROR")
                    return None
                self._append_log(f"Loaded {len(topics)} topics from file")
                return topics
            except Exception as exc:
                self._append_log(f"\u274C Failed to read file: {exc}", level="ERROR")
                return None

    def _run_generation(
        self,
        topics: list[str],
        gen_config: Any,
        api_config: Any,
        language: str,
        topics_file: str,
    ) -> None:
        """Execute the generation batch in a background thread.

        Creates a new asyncio event loop, runs ``process_batch``, and
        emits a ``BATCH_COMPLETED`` event when done.

        Args:
            topics: List of topic strings.
            gen_config: ``GenerationConfig`` instance.
            api_config: ``APIConfig`` instance.
            language: Two-letter language code.
            topics_file: Path to the source topics file (for marking
                completed topics with ``OK `` prefix).
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self._processor.process_batch(
                    topics=topics,
                    gen_config=gen_config,
                    api_config=api_config,
                    language=language,
                    topics_file_path=topics_file,
                )
            )
            self._event_bus.emit(
                EventType.BATCH_COMPLETED,
                total=result.total,
                completed=result.completed,
                failed=result.failed,
                elapsed_seconds=result.elapsed_seconds,
            )
        except Exception as exc:
            logger.error("Generation thread error: %s", exc, exc_info=True)
            self._event_bus.emit(
                EventType.LOG_MESSAGE,
                message=f"Generation failed: {exc}",
                level="ERROR",
            )
            self._event_bus.emit(
                EventType.BATCH_COMPLETED,
                topic="",
                story_id="",
                error=str(exc),
            )
        finally:
            loop.close()

    def _run_adaptation(
        self,
        source_files: list[Path],
        source_lang: str,
        target_langs: list[str],
        adapt_cfg: dict[str, Any],
        api_config: Any,
    ) -> None:
        """Execute adaptation in a background thread.

        For each source file × target language, reads the source text,
        calls ``TextAdapter.adapt()``, and saves the result.

        Args:
            source_files: List of source ``.txt`` file paths.
            source_lang: Detected source language code.
            target_langs: List of target language codes.
            adapt_cfg: Adaptation config from AdaptationPanel.
            api_config: ``APIConfig`` instance from ApiSettingsPanel.
        """
        import time

        from core.adapter import AdaptationMode, AdaptationParams, TextAdapter
        from core.api_client import APIClient
        from core.prompt_manager import PromptManager
        from utils.file_handler import write_file, ensure_dir

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        start_time = time.monotonic()
        completed = 0
        failed = 0
        total = len(source_files) * len(target_langs)

        try:
            # Reconfigure API client with user's chosen provider/model
            api_client = APIClient(
                api_config=api_config,
                settings=self._settings,
                event_bus=self._event_bus,
            )

            resources_dir = Path(self._settings.paths.resources_dir)
            prompt_manager = PromptManager(resources_dir=resources_dir)

            adapter = TextAdapter(settings=self._settings)

            # Map string mode to enum
            mode_str = adapt_cfg.get("mode", "cultural")
            mode = AdaptationMode(mode_str)

            params = AdaptationParams(
                adapt_names=adapt_cfg.get("adapt_names", True),
                adapt_references=adapt_cfg.get("adapt_references", True),
                adapt_units=adapt_cfg.get("adapt_units", True),
                adapt_setting=adapt_cfg.get("adapt_setting", False),
                preserve_length=adapt_cfg.get("preserve_length", True),
                voiceover_optimize=adapt_cfg.get("voiceover_optimize", True),
                run_evaluation=adapt_cfg.get("run_evaluation", True),
            )

            output_base = Path(self._settings.paths.output_dir)

            for src_file in source_files:
                # Read source text
                try:
                    source_text = src_file.read_text(encoding="utf-8")
                except Exception as exc:
                    logger.error(
                        "Failed to read source file %s: %s", src_file, exc,
                    )
                    self._event_bus.emit(
                        EventType.LOG_MESSAGE,
                        message=f"\u274C Cannot read {src_file.name}: {exc}",
                        level="ERROR",
                    )
                    failed += len(target_langs)
                    for tgt_lang in target_langs:
                        task_name = f"{src_file.stem} → {tgt_lang}"
                        self._event_bus.emit(
                            EventType.LOG_MESSAGE,
                            message=f"\u274C {task_name}: source unreadable",
                            level="ERROR",
                        )
                    continue

                if not source_text.strip():
                    logger.warning("Source file %s is empty", src_file)
                    self._event_bus.emit(
                        EventType.LOG_MESSAGE,
                        message=f"\u26A0\uFE0F {src_file.name} is empty — skipping",
                        level="WARNING",
                    )
                    failed += len(target_langs)
                    continue

                for tgt_lang in target_langs:
                    task_name = f"{src_file.stem} → {tgt_lang}"

                    self._event_bus.emit(
                        EventType.LOG_MESSAGE,
                        message=f"\U0001F310 Adapting: {task_name}",
                        level="INFO",
                    )

                    try:
                        # Create output dir BEFORE adapt so evaluation
                        # can save artifacts (eval_v1.json, etc.)
                        out_dir = output_base / tgt_lang / src_file.stem
                        ensure_dir(out_dir)

                        result = loop.run_until_complete(
                            adapter.adapt(
                                source_text=source_text,
                                source_lang=source_lang,
                                target_lang=tgt_lang,
                                mode=mode,
                                params=params,
                                api_client=api_client,
                                prompt_manager=prompt_manager,
                                event_bus=self._event_bus,
                                output_dir=str(out_dir),
                            )
                        )

                        if result.error:
                            logger.error(
                                "Adaptation failed for %s: %s",
                                task_name, result.error,
                            )
                            self._event_bus.emit(
                                EventType.LOG_MESSAGE,
                                message=(
                                    f"\u274C {task_name}: {result.error}"
                                ),
                                level="ERROR",
                            )
                            failed += 1
                            continue

                        # Save adapted text
                        out_path = out_dir / "final.txt"
                        write_file(out_path, result.adapted_text)

                        completed += 1
                        score_info = ""
                        if result.score > 0:
                            score_info = f", score={result.score:.1f}"
                        self._event_bus.emit(
                            EventType.LOG_MESSAGE,
                            message=(
                                f"\u2705 {task_name}: "
                                f"{result.source_word_count}\u2192"
                                f"{result.adapted_word_count} words"
                                f"{score_info}, "
                                f"{result.duration_seconds:.1f}s"
                            ),
                            level="INFO",
                        )

                        logger.info(
                            "Adaptation saved: %s → %s (%d words)",
                            task_name,
                            out_path,
                            result.adapted_word_count,
                        )

                    except Exception as exc:
                        logger.error(
                            "Adaptation error for %s: %s",
                            task_name, exc, exc_info=True,
                        )
                        self._event_bus.emit(
                            EventType.LOG_MESSAGE,
                            message=f"\u274C {task_name}: {exc}",
                            level="ERROR",
                        )
                        failed += 1

            elapsed = time.monotonic() - start_time

            self._event_bus.emit(
                EventType.BATCH_COMPLETED,
                total=total,
                completed=completed,
                failed=failed,
                elapsed_seconds=elapsed,
            )

        except Exception as exc:
            logger.error(
                "Adaptation thread error: %s", exc, exc_info=True,
            )
            self._event_bus.emit(
                EventType.LOG_MESSAGE,
                message=f"Adaptation failed: {exc}",
                level="ERROR",
            )
            self._event_bus.emit(
                EventType.BATCH_COMPLETED,
                total=total,
                completed=completed,
                failed=failed,
                error=str(exc),
            )
        finally:
            try:
                api_client.close_sync()
            except Exception:
                pass
            loop.close()

    def _on_pause(self) -> None:
        """Handle the PAUSE / RESUME button click."""
        if not self._is_generating:
            return

        if self._is_paused:
            # Resume
            if self._processor is not None:
                self._processor.resume()
            self._is_paused = False
            self._pause_btn.configure(text="\u23F8 PAUSE")
            self._status_label.configure(text="Running...")
            self._append_log("\u25B6 Resumed")
        else:
            # Pause
            if self._processor is not None:
                self._processor.request_pause()
            self._is_paused = True
            self._pause_btn.configure(text="\u25B6 RESUME")
            self._status_label.configure(text="Paused")
            self._append_log("\u23F8 Paused")

    def _on_stop(self) -> None:
        """Handle the STOP button click."""
        if not self._is_generating:
            return

        if self._processor is not None:
            self._processor.request_stop()
        self._append_log("\u23F9 Stop requested — finishing current topics...")
        self._status_label.configure(text="Stopping...")
        self._stop_btn.configure(state="disabled")

    def _set_generating_state(self, generating: bool) -> None:
        """Update button states for generation/idle modes.

        Args:
            generating: ``True`` if generation is in progress.
        """
        self._is_generating = generating
        self._is_paused = False

        if generating:
            self._start_btn.configure(state="disabled")
            self._pause_btn.configure(state="normal", text="\u23F8 PAUSE")
            self._stop_btn.configure(state="normal")
            self._status_label.configure(text="Running...")
        else:
            self._start_btn.configure(state="normal")
            self._pause_btn.configure(state="disabled", text="\u23F8 PAUSE")
            self._stop_btn.configure(state="disabled")

    def _on_generation_finished(self) -> None:
        """Reset UI state after generation completes or fails."""
        self._set_generating_state(False)
        self._status_label.configure(text="Complete")
        self._generation_thread = None

        # Refresh analytics panel with new data
        try:
            self._analytics_panel.refresh()
            logger.debug("Analytics panel refreshed after generation")
        except Exception as exc:
            logger.warning("Failed to refresh analytics panel: %s", exc)

    # ── Mode switching ──────────────────────────────────────────────

    def _on_mode_changed(self, mode: str) -> None:
        """Handle mode change from the input panel.

        Switches between StylePanel (generate) and AdaptationPanel
        (adapt) in the left sidebar.

        Args:
            mode: ``"generate"`` or ``"adapt"``.
        """
        logger.info("Application mode changed to: %s", mode)

        if mode == "adapt":
            # Hide StylePanel, show AdaptationPanel
            self._style_panel.grid_remove()
            self._adaptation_panel.grid(
                row=self._style_adapt_row, column=0,
                sticky="ew", pady=Padding.SECTION,
            )
            self._status_label.configure(text="Mode: Adaptation")
        else:
            # Hide AdaptationPanel, show StylePanel
            self._adaptation_panel.grid_remove()
            self._style_panel.grid(
                row=self._style_adapt_row, column=0,
                sticky="ew", pady=Padding.SECTION,
            )
            self._status_label.configure(text="Mode: Generation")

    # ── Menu actions ────────────────────────────────────────────────

    def _reset_all(self) -> None:
        """Reset all panels to their default values."""
        self._input_panel.reset()
        self._style_panel.reset()
        self._adaptation_panel.reset()
        self._api_panel.reset()
        self._progress_panel.reset()
        self._clear_log()
        self._cost_label.configure(text="")
        self._cost_status.configure(text="")
        self._status_label.configure(text="Ready")
        self._append_log("All settings reset to defaults")

    def _show_about(self) -> None:
        """Display an About dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("About")
        dialog.geometry("360x200")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(fg_color=Colors.SURFACE)

        ctk.CTkLabel(
            dialog,
            text="AI Story Generator Pro",
            font=Fonts.get(size=Fonts.HEADING, bold=True),
            text_color=Colors.TEXT,
        ).pack(pady=(Padding.GROUP_Y, Padding.WIDGET_Y))

        ctk.CTkLabel(
            dialog,
            text="Version 1.0",
            font=Fonts.get(),
            text_color=Colors.TEXT_DIM,
        ).pack()

        ctk.CTkLabel(
            dialog,
            text=(
                "Generates evergreen YouTube voiceover stories\n"
                "in 11 languages using LLM APIs."
            ),
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED,
            justify="center",
        ).pack(pady=Padding.WIDGET_Y)

        ctk.CTkButton(
            dialog,
            text="Close",
            command=dialog.destroy,
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
            width=80,
        ).pack(pady=Padding.GROUP_Y)

    def _show_shortcuts(self) -> None:
        """Display a keyboard shortcuts dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Keyboard Shortcuts")
        dialog.geometry("340x320")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(fg_color=Colors.SURFACE)

        title = ctk.CTkLabel(
            dialog,
            text="Keyboard Shortcuts",
            font=Fonts.get(size=Fonts.SUBHEADING, bold=True),
            text_color=Colors.TEXT,
        )
        title.pack(pady=(Padding.GROUP_Y, Padding.WIDGET_Y))

        shortcuts = [
            ("Ctrl+O", "Open topics file"),
            ("Ctrl+D", "Select folder"),
            ("Ctrl+R", "Start generation"),
            ("Ctrl+P", "Pause / Resume"),
            ("Escape", "Stop generation"),
            ("Ctrl+E", "Export report"),
            ("Ctrl+L", "Clear log"),
            ("Ctrl+A", "Analytics tab"),
        ]

        for key, desc in shortcuts:
            row_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            row_frame.pack(fill="x", padx=Padding.PANEL, pady=1)

            key_lbl = ctk.CTkLabel(
                row_frame,
                text=key,
                font=Fonts.get(size=Fonts.SMALL, bold=True, mono=True),
                text_color=Colors.INFO,
                width=80,
                anchor="w",
            )
            key_lbl.pack(side="left")

            desc_lbl = ctk.CTkLabel(
                row_frame,
                text=desc,
                font=Fonts.get(size=Fonts.SMALL),
                text_color=Colors.TEXT_DIM,
                anchor="w",
            )
            desc_lbl.pack(side="left", padx=Padding.WIDGET_X)

        close_btn = ctk.CTkButton(
            dialog,
            text="Close",
            command=dialog.destroy,
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
            width=80,
        )
        close_btn.pack(pady=Padding.GROUP_Y)

    # ── Window close ────────────────────────────────────────────────

    def _on_close(self) -> None:
        """Handle window close — save preferences and stop generation."""
        # Save user preferences (provider, model, etc.)
        try:
            self._api_panel.save_preferences()
            logger.info("User preferences saved on exit")
        except Exception as exc:
            logger.warning("Failed to save preferences on exit: %s", exc)

        if self._is_generating:
            if self._processor is not None:
                self._processor.request_stop()
            self._append_log("Shutting down — stopping generation...")

        logger.info("MainWindow closing")
        self.destroy()
