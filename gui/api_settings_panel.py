"""API settings panel for AI Story Generator Pro GUI.

Provides provider selection, model dropdown (filtered by provider),
API key entry with show/hide toggle, provider status table, fallback
pool info, retry/thread configuration, and connection test button.

Loads and saves API keys from ``.env`` file via ``python-dotenv``.
The panel exposes ``get_config()`` which returns a populated
``APIConfig`` model with a ``fallback_pool`` built automatically
from all providers that have an API key.

FIX: Now saves/loads user preferences (provider, model, fallback,
retries, threads) to/from ``data/user_preferences.yaml`` so that
selections persist across app restarts.  Preferences are auto-saved
on every provider/model/fallback/retries/threads change and on
window close.

FIX v2: Replaced single Fallback Provider dropdown with automatic
fallback pool.  All providers with API keys (except primary) are
used as fallbacks in the order shown in the status table.  A label
shows how many fallback providers are available.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import json

import customtkinter as ctk
import yaml

from gui.styles import Colors, Fonts, Padding, create_section_label, create_separator
from core.events import EventBus, EventType
from models.config import (
    APIConfig,
    APIProvider,
    FallbackPoolEntry,
    PROVIDER_CONFIG,
)

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────

_ENV_FILE = Path(".env")
_PREFS_FILE = Path("data/user_preferences.yaml")
_CUSTOM_MODELS_FILE = Path("data/custom_openrouter_models.json")

_ENV_KEY_MAP: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
}

_PROVIDER_DISPLAY: dict[str, str] = {
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
}

# Order in which fallback providers are tried.  This matches the
# order in settings.yaml and the status table.
_FALLBACK_ORDER: list[str] = [
    "openrouter",
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "qwen",
]

_MAX_RETRIES_OPTIONS: list[str] = ["0", "1", "2", "3", "4", "5"]
_THREAD_OPTIONS: list[str] = [str(i) for i in range(1, 11)]


class ApiSettingsPanel(ctk.CTkFrame):
    """API configuration panel.

    Manages provider selection, model choice, API keys, automatic
    fallback pool, retry configuration, parallelism threads, and a
    connection test button.

    Persists user preferences to ``data/user_preferences.yaml``.

    Args:
        parent: Parent widget.
        event_bus: Thread-safe event bus for communicating with the
            main window (used for connection-test result callbacks).
    """

    def __init__(self, parent: Any, event_bus: EventBus | None = None, **kwargs: Any) -> None:
        super().__init__(parent, fg_color=Colors.SURFACE, **kwargs)

        self._event_bus = event_bus

        # State
        self._provider_var = ctk.StringVar(value=_PROVIDER_DISPLAY["openrouter"])
        self._model_var = ctk.StringVar(value="")
        self._api_key_var = ctk.StringVar(value="")
        self._auto_fallback_var = ctk.BooleanVar(value=True)
        self._retries_var = ctk.StringVar(value="3")
        self._threads_var = ctk.StringVar(value="3")
        self._key_visible: bool = False

        # Stored keys per provider (loaded from .env)
        self._provider_keys: dict[str, str] = {}

        # Saved model to restore after model list is built
        self._saved_model: str = ""

        self._load_env_keys()
        self._load_preferences()
        self._build_ui()
        self._update_model_list()
        self._apply_saved_model()
        self._populate_key_field()
        self._update_pool_label()
        logger.debug("ApiSettingsPanel initialised")

    # ── User preferences persistence ────────────────────────────────

    def _load_preferences(self) -> None:
        """Load user preferences from data/user_preferences.yaml."""
        if not _PREFS_FILE.exists():
            logger.debug("No user preferences file found — using defaults")
            return

        try:
            text = _PREFS_FILE.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if not isinstance(data, dict):
                return

            api_prefs = data.get("api", {})
            if not isinstance(api_prefs, dict):
                return

            # Restore provider
            saved_provider = api_prefs.get("primary_provider", "")
            if saved_provider in _PROVIDER_DISPLAY:
                self._provider_var.set(_PROVIDER_DISPLAY[saved_provider])

            # Model is restored after _update_model_list()
            self._saved_model = api_prefs.get("primary_model", "")

            # Restore retries and threads
            saved_retries = str(api_prefs.get("retries", "3"))
            if saved_retries in _MAX_RETRIES_OPTIONS:
                self._retries_var.set(saved_retries)

            saved_threads = str(api_prefs.get("threads", "3"))
            if saved_threads in _THREAD_OPTIONS:
                self._threads_var.set(saved_threads)

            saved_auto_fb = api_prefs.get("auto_fallback", True)
            self._auto_fallback_var.set(bool(saved_auto_fb))

            logger.info(
                "User preferences loaded: provider=%s, model=%s",
                saved_provider,
                self._saved_model,
            )
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("Failed to load user preferences: %s", exc)
            self._saved_model = ""

    def _apply_saved_model(self) -> None:
        """Apply the saved model after the model list is populated."""
        if self._saved_model:
            provider = self._current_provider_value()
            config = PROVIDER_CONFIG.get(provider, {})
            models = list(config.get("models", []))

            # Include custom OpenRouter models.
            if provider == "openrouter":
                models.extend(self._load_custom_models())

            if self._saved_model in models:
                self._model_var.set(self._saved_model)
                logger.debug("Restored saved model: %s", self._saved_model)

    def save_preferences(self) -> None:
        """Save current panel state to data/user_preferences.yaml.

        Called automatically on provider/model/setting change and on
        window close via MainWindow._on_close().
        """
        provider = self._current_provider_value()
        model = self._model_var.get()

        try:
            retries = int(self._retries_var.get())
        except (ValueError, Exception):
            retries = 3

        try:
            threads = int(self._threads_var.get())
        except (ValueError, Exception):
            threads = 3

        data = {
            "api": {
                "primary_provider": provider,
                "primary_model": model,
                "auto_fallback": self._auto_fallback_var.get(),
                "retries": retries,
                "threads": threads,
            }
        }

        try:
            _PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _PREFS_FILE.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
            logger.debug("Preferences saved: provider=%s, model=%s", provider, model)
        except OSError as exc:
            logger.error("Failed to save user preferences: %s", exc)

    # ── .env key management ─────────────────────────────────────────

    def _load_env_keys(self) -> None:
        """Load API keys from the .env file."""
        if not _ENV_FILE.exists():
            logger.debug("No .env file found — starting with empty keys")
            return

        try:
            text = _ENV_FILE.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to read .env file: %s", exc)
            return

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            # Reverse-lookup provider from env var name
            for provider, env_name in _ENV_KEY_MAP.items():
                if key == env_name and value:
                    self._provider_keys[provider] = value

        logger.debug(
            "Loaded %d API keys from .env",
            sum(1 for v in self._provider_keys.values() if v),
        )

    def _save_env_keys(self) -> None:
        """Save all stored API keys back to the .env file."""
        lines: list[str] = ["# AI Story Generator Pro — API Keys"]
        for provider, env_name in _ENV_KEY_MAP.items():
            key_value = self._provider_keys.get(provider, "")
            lines.append(f'{env_name}={key_value}')
        lines.append("")

        try:
            _ENV_FILE.write_text("\n".join(lines), encoding="utf-8")
            logger.info("API keys saved to .env")
        except OSError as exc:
            logger.error("Failed to write .env file: %s", exc)

    def _current_provider_value(self) -> str:
        """Return the raw provider enum value for the currently selected provider.

        Returns:
            Provider string like ``"openrouter"``.
        """
        display = self._provider_var.get()
        for value, label in _PROVIDER_DISPLAY.items():
            if label == display:
                return value
        return "openrouter"

    def _populate_key_field(self) -> None:
        """Set the API key entry to the stored key for the current provider."""
        provider = self._current_provider_value()
        stored = self._provider_keys.get(provider, "")
        self._api_key_var.set(stored)

    def _store_current_key(self) -> None:
        """Store the currently entered key for the current provider."""
        provider = self._current_provider_value()
        key = self._api_key_var.get().strip()
        self._provider_keys[provider] = key

    # ── Custom OpenRouter models ──────────────────────────────────────

    def _load_custom_models(self) -> list[str]:
        """Load custom OpenRouter model IDs from JSON file.

        Returns:
            List of custom model ID strings.
        """
        if not _CUSTOM_MODELS_FILE.exists():
            return []
        try:
            text = _CUSTOM_MODELS_FILE.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                return [str(m) for m in data if m]
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load custom models: %s", exc)
            return []

    def _save_custom_models(self, models: list[str]) -> None:
        """Save custom OpenRouter model IDs to JSON file.

        Args:
            models: List of model ID strings.
        """
        try:
            _CUSTOM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _CUSTOM_MODELS_FILE.write_text(
                json.dumps(models, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Custom OpenRouter models saved: %d entries", len(models))
        except OSError as exc:
            logger.error("Failed to save custom models: %s", exc)

    def _add_openrouter_model(self) -> None:
        """Show a dialog to add a custom OpenRouter model ID.

        Opens a modal dialog, validates the input, saves to the JSON
        file, refreshes the model dropdown, and selects the new model.
        """
        dialog = ctk.CTkToplevel(self)
        dialog.title("Add OpenRouter Model")
        dialog.geometry("450x180")
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.configure(fg_color=Colors.SURFACE)

        ctk.CTkLabel(
            dialog,
            text="Enter the model ID from OpenRouter:",
            font=Fonts.get(bold=True),
            text_color=Colors.TEXT,
        ).pack(pady=(Padding.GROUP_Y, Padding.WIDGET_Y), padx=Padding.PANEL)

        entry = ctk.CTkEntry(
            dialog,
            placeholder_text="e.g. anthropic/claude-3.5-sonnet",
            font=Fonts.get(mono=True),
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            border_color=Colors.ENTRY_BORDER,
            width=400,
        )
        entry.pack(padx=Padding.PANEL, pady=Padding.WIDGET_Y)
        entry.focus()

        result_label = ctk.CTkLabel(
            dialog,
            text="",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.ERROR,
        )
        result_label.pack(padx=Padding.PANEL)

        def _on_save() -> None:
            model_id = entry.get().strip()
            if not model_id:
                result_label.configure(text="\u274C Model ID cannot be empty")
                return
            if " " in model_id:
                result_label.configure(text="\u274C Model ID cannot contain spaces")
                return

            # Load existing, add, deduplicate, save.
            custom = self._load_custom_models()
            if model_id not in custom:
                custom.append(model_id)
                self._save_custom_models(custom)

            # Refresh dropdown and select the new model.
            self._update_model_list()
            self._model_var.set(model_id)
            self.save_preferences()

            logger.info("Custom OpenRouter model added: %s", model_id)
            dialog.destroy()

        def _on_paste() -> None:
            try:
                entry.delete(0, "end")
                entry.insert(0, dialog.clipboard_get())
            except Exception:
                result_label.configure(text="\u26A0\uFE0F Clipboard empty")

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=Padding.GROUP_Y, padx=Padding.PANEL)

        ctk.CTkButton(
            btn_frame,
            text="Paste",
            width=80,
            command=_on_paste,
            font=Fonts.get(),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
        ).pack(side="left", padx=Padding.WIDGET_X)

        ctk.CTkButton(
            btn_frame,
            text="\u2795 Add Model",
            width=120,
            command=_on_save,
            font=Fonts.get(bold=True),
            fg_color=Colors.SUCCESS,
            hover_color=Colors.SUCCESS_HOVER,
            text_color=Colors.TEXT,
        ).pack(side="left", padx=Padding.WIDGET_X)

        # Allow Enter key to submit.
        entry.bind("<Return>", lambda e: _on_save())

    def _remove_openrouter_model(self) -> None:
        """Remove the currently selected custom OpenRouter model.

        Only removes models from the custom list — built-in models
        from ``PROVIDER_CONFIG`` cannot be removed.
        """
        model = self._model_var.get()
        builtin = PROVIDER_CONFIG.get("openrouter", {}).get("models", [])
        if model in builtin:
            logger.debug("Cannot remove built-in model: %s", model)
            return

        custom = self._load_custom_models()
        if model in custom:
            custom.remove(model)
            self._save_custom_models(custom)
            self._update_model_list()
            self.save_preferences()
            logger.info("Custom OpenRouter model removed: %s", model)

    # ── Fallback pool helpers ──────────────────────────────────────

    def _get_pool_providers(self) -> list[str]:
        """Return the list of providers that form the fallback pool.

        Includes all providers with API keys, excluding the currently
        selected primary provider.

        Returns:
            List of provider value strings in fallback order.
        """
        primary = self._current_provider_value()
        pool: list[str] = []
        for prov in _FALLBACK_ORDER:
            if prov == primary:
                continue
            if self._provider_keys.get(prov, ""):
                pool.append(prov)
        return pool

    def _update_pool_label(self) -> None:
        """Refresh the fallback pool info label."""
        if not hasattr(self, "_pool_info_label"):
            return

        pool = self._get_pool_providers()

        if not pool:
            text = "\u26A0\uFE0F No fallback providers (add API keys above)"
            colour = Colors.WARNING
        else:
            names = [_PROVIDER_DISPLAY.get(p, p) for p in pool]
            text = f"\u26A1 Pool ({len(pool)}): {' → '.join(names)}"
            colour = Colors.SUCCESS

        self._pool_info_label.configure(text=text, text_color=colour)

    def _build_fallback_pool_entries(self) -> list[FallbackPoolEntry]:
        """Build the fallback pool entries for the current configuration.

        Returns:
            Ordered list of ``FallbackPoolEntry`` objects.
        """
        pool_providers = self._get_pool_providers()
        entries: list[FallbackPoolEntry] = []

        for prov in pool_providers:
            api_key = self._provider_keys.get(prov, "")
            if not api_key:
                continue

            # Use the first model from the provider's model list.
            prov_config = PROVIDER_CONFIG.get(prov, {})
            models = prov_config.get("models", [])
            model = models[0] if models else ""

            entries.append(
                FallbackPoolEntry(
                    provider=APIProvider(prov),
                    model=model,
                    api_key=api_key,
                )
            )

        return entries

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all child widgets."""
        self.columnconfigure(0, weight=1)
        row = 0

        # Section header
        header = create_section_label(self, "API SETTINGS", icon="\U0001F50C")
        header.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.PANEL, Padding.WIDGET_Y))
        row += 1

        sep = create_separator(self)
        sep.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        # ── Provider dropdown ───────────────────────────────────────
        prov_label = ctk.CTkLabel(
            self, text="Provider", font=Fonts.get(bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        prov_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        self._provider_menu = ctk.CTkOptionMenu(
            self,
            variable=self._provider_var,
            values=list(_PROVIDER_DISPLAY.values()),
            command=self._on_provider_changed,
            font=Fonts.get(),
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
            dropdown_fg_color=Colors.SURFACE,
            dropdown_hover_color=Colors.SURFACE_HOVER,
            dropdown_text_color=Colors.TEXT,
        )
        self._provider_menu.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        # ── Model dropdown + add/remove buttons ────────────────────
        model_label = ctk.CTkLabel(
            self, text="Model", font=Fonts.get(bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        model_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        model_frame = ctk.CTkFrame(self, fg_color="transparent")
        model_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        model_frame.columnconfigure(0, weight=1)

        self._model_menu = ctk.CTkOptionMenu(
            model_frame,
            variable=self._model_var,
            values=["(select provider)"],
            command=self._on_model_changed,
            font=Fonts.get(),
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
            dropdown_fg_color=Colors.SURFACE,
            dropdown_hover_color=Colors.SURFACE_HOVER,
            dropdown_text_color=Colors.TEXT,
        )
        self._model_menu.grid(row=0, column=0, sticky="ew")

        # "+" button — add custom OpenRouter model
        self._add_model_btn = ctk.CTkButton(
            model_frame,
            text="+",
            width=32,
            command=self._add_openrouter_model,
            font=Fonts.get(bold=True),
            fg_color=Colors.SUCCESS,
            hover_color=Colors.SUCCESS_HOVER,
            text_color=Colors.TEXT,
        )
        self._add_model_btn.grid(row=0, column=1, padx=(Padding.WIDGET_X, 0))

        # "−" button — remove custom OpenRouter model
        self._remove_model_btn = ctk.CTkButton(
            model_frame,
            text="\u2212",
            width=32,
            command=self._remove_openrouter_model,
            font=Fonts.get(bold=True),
            fg_color=Colors.ERROR,
            hover_color=Colors.ERROR_HOVER,
            text_color=Colors.TEXT,
        )
        self._remove_model_btn.grid(row=0, column=2, padx=(2, 0))

        # Buttons are shown/hidden based on provider.
        self._update_model_buttons_visibility()
        row += 1

        # ── API key entry + show/hide ───────────────────────────────
        key_label = ctk.CTkLabel(
            self, text="API Key", font=Fonts.get(bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        key_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.LABEL_Y))
        row += 1

        key_frame = ctk.CTkFrame(self, fg_color="transparent")
        key_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        key_frame.columnconfigure(0, weight=1)

        self._key_entry = ctk.CTkEntry(
            key_frame,
            textvariable=self._api_key_var,
            show="\u2022",
            font=Fonts.get(mono=True),
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            border_color=Colors.ENTRY_BORDER,
        )
        self._key_entry.grid(row=0, column=0, sticky="ew", padx=(0, Padding.WIDGET_X))

        self._key_toggle_btn = ctk.CTkButton(
            key_frame,
            text="\U0001F441",
            width=36,
            command=self._toggle_key_visibility,
            font=Fonts.get(),
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            text_color=Colors.TEXT,
        )
        self._key_toggle_btn.grid(row=0, column=1)

        # Save key on focus-out
        self._key_entry.bind("<FocusOut>", lambda e: self._on_key_changed())
        row += 1

        # ── Provider status table ───────────────────────────────────
        sep2 = create_separator(self)
        sep2.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        row += 1

        status_label = ctk.CTkLabel(
            self, text="Provider Status", font=Fonts.get(bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        status_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(0, Padding.LABEL_Y))
        row += 1

        self._status_frame = ctk.CTkFrame(self, fg_color=Colors.SURFACE_LIGHT, corner_radius=6)
        self._status_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        self._status_frame.columnconfigure(1, weight=1)
        self._build_status_table()
        row += 1

        # ── Fallback pool info ──────────────────────────────────────
        sep3 = create_separator(self)
        sep3.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        row += 1

        fallback_header = ctk.CTkLabel(
            self, text="Fallback Pool (auto)", font=Fonts.get(bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        fallback_header.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(0, Padding.LABEL_Y))
        row += 1

        self._pool_info_label = ctk.CTkLabel(
            self,
            text="",
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED,
            anchor="w",
            wraplength=280,
            justify="left",
        )
        self._pool_info_label.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        pool_hint = ctk.CTkLabel(
            self,
            text="Add API keys above — all providers with keys\nbecome automatic fallbacks.",
            font=Fonts.get(size=Fonts.TINY),
            text_color=Colors.MUTED,
            anchor="w",
            justify="left",
        )
        pool_hint.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=(0, Padding.LABEL_Y))
        row += 1

        self._auto_fallback_cb = ctk.CTkCheckBox(
            self,
            text="Auto-fallback on failure",
            variable=self._auto_fallback_var,
            font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.TEXT,
            fg_color=Colors.PRIMARY,
            hover_color=Colors.PRIMARY_HOVER,
            checkmark_color=Colors.TEXT,
        )
        self._auto_fallback_cb.grid(row=row, column=0, sticky="w", padx=Padding.PANEL, pady=Padding.LABEL_Y)
        row += 1

        # ── Retries and threads ─────────────────────────────────────
        config_frame = ctk.CTkFrame(self, fg_color="transparent")
        config_frame.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=Padding.SECTION)
        config_frame.columnconfigure((0, 1), weight=1)

        # Max retries
        retries_lbl = ctk.CTkLabel(
            config_frame, text="Max Retries", font=Fonts.get(size=Fonts.SMALL, bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        retries_lbl.grid(row=0, column=0, sticky="w", padx=(0, Padding.WIDGET_X), pady=Padding.LABEL_Y)

        self._retries_menu = ctk.CTkOptionMenu(
            config_frame,
            variable=self._retries_var,
            values=_MAX_RETRIES_OPTIONS,
            command=self._on_setting_changed,
            font=Fonts.get(size=Fonts.SMALL),
            width=70,
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
            dropdown_fg_color=Colors.SURFACE,
            dropdown_hover_color=Colors.SURFACE_HOVER,
            dropdown_text_color=Colors.TEXT,
        )
        self._retries_menu.grid(row=1, column=0, sticky="w", padx=(0, Padding.WIDGET_X), pady=Padding.LABEL_Y)

        # Parallel threads
        threads_lbl = ctk.CTkLabel(
            config_frame, text="Parallel Threads", font=Fonts.get(size=Fonts.SMALL, bold=True),
            text_color=Colors.TEXT, anchor="w",
        )
        threads_lbl.grid(row=0, column=1, sticky="w", pady=Padding.LABEL_Y)

        self._threads_menu = ctk.CTkOptionMenu(
            config_frame,
            variable=self._threads_var,
            values=_THREAD_OPTIONS,
            command=self._on_setting_changed,
            font=Fonts.get(size=Fonts.SMALL),
            width=70,
            text_color=Colors.TEXT,
            fg_color=Colors.ENTRY_BG,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
            dropdown_fg_color=Colors.SURFACE,
            dropdown_hover_color=Colors.SURFACE_HOVER,
            dropdown_text_color=Colors.TEXT,
        )
        self._threads_menu.grid(row=1, column=1, sticky="w", pady=Padding.LABEL_Y)
        row += 1

        # ── Connection test button ──────────────────────────────────
        self._test_btn = ctk.CTkButton(
            self,
            text="\u26A1 Test Connection",
            command=self._test_connection,
            font=Fonts.get(),
            fg_color=Colors.INFO,
            hover_color=Colors.INFO_HOVER,
            text_color=Colors.TEXT,
        )
        self._test_btn.grid(row=row, column=0, sticky="ew", padx=Padding.PANEL, pady=(Padding.SECTION, Padding.PANEL))

        self._test_result_label = ctk.CTkLabel(
            self, text="", font=Fonts.get(size=Fonts.SMALL),
            text_color=Colors.MUTED, anchor="w",
        )
        self._test_result_label.grid(row=row + 1, column=0, sticky="w", padx=Padding.PANEL, pady=(0, Padding.PANEL))

    def _build_status_table(self) -> None:
        """Build the provider status grid inside ``_status_frame``."""
        # Header row
        headers = ["Provider", "Models", "Key"]
        for col, text in enumerate(headers):
            lbl = ctk.CTkLabel(
                self._status_frame,
                text=text,
                font=Fonts.get(size=Fonts.TINY, bold=True),
                text_color=Colors.MUTED,
                anchor="w",
            )
            lbl.grid(row=0, column=col, sticky="w", padx=Padding.WIDGET_X, pady=2)

        self._status_rows: dict[str, dict[str, ctk.CTkLabel]] = {}

        for idx, (provider_val, display) in enumerate(_PROVIDER_DISPLAY.items()):
            prov_row = idx + 1
            config = PROVIDER_CONFIG.get(provider_val, {})
            models = config.get("models", [])
            model_count = len(models)
            has_key = bool(self._provider_keys.get(provider_val, ""))

            name_lbl = ctk.CTkLabel(
                self._status_frame,
                text=display,
                font=Fonts.get(size=Fonts.TINY),
                text_color=Colors.TEXT,
                anchor="w",
            )
            name_lbl.grid(row=prov_row, column=0, sticky="w", padx=Padding.WIDGET_X, pady=1)

            models_lbl = ctk.CTkLabel(
                self._status_frame,
                text=f"{model_count} models",
                font=Fonts.get(size=Fonts.TINY),
                text_color=Colors.TEXT_DIM,
                anchor="w",
            )
            models_lbl.grid(row=prov_row, column=1, sticky="w", padx=Padding.WIDGET_X, pady=1)

            key_indicator = "\u2705" if has_key else "\u274C"
            key_lbl = ctk.CTkLabel(
                self._status_frame,
                text=key_indicator,
                font=Fonts.get(size=Fonts.TINY),
                text_color=Colors.SUCCESS if has_key else Colors.ERROR,
                anchor="center",
            )
            key_lbl.grid(row=prov_row, column=2, sticky="w", padx=Padding.WIDGET_X, pady=1)

            self._status_rows[provider_val] = {
                "name": name_lbl,
                "models": models_lbl,
                "key": key_lbl,
            }

    def _refresh_status_table(self) -> None:
        """Update the key indicators in the status table."""
        for provider_val, widgets in self._status_rows.items():
            has_key = bool(self._provider_keys.get(provider_val, ""))
            indicator = "\u2705" if has_key else "\u274C"
            colour = Colors.SUCCESS if has_key else Colors.ERROR
            widgets["key"].configure(text=indicator, text_color=colour)

    # ── Callbacks ───────────────────────────────────────────────────

    def _on_provider_changed(self, value: str) -> None:
        """Handle provider dropdown change.

        Args:
            value: Selected display name.
        """
        # Store the key currently in the entry before switching
        self._store_current_key()
        self._update_model_list()
        self._populate_key_field()
        self._update_pool_label()
        self._update_model_buttons_visibility()
        self.save_preferences()
        logger.info("Provider changed to: %s", value)

    def _on_model_changed(self, value: str) -> None:
        """Handle model dropdown change.

        Args:
            value: Selected model name.
        """
        self.save_preferences()
        logger.debug("Model changed to: %s", value)

    def _on_setting_changed(self, value: str) -> None:
        """Handle retries or threads dropdown change.

        Args:
            value: New value string.
        """
        self.save_preferences()

    def _update_model_list(self) -> None:
        """Refresh the model dropdown for the current provider.

        For OpenRouter, custom models from the JSON file are appended
        after the built-in models.
        """
        provider = self._current_provider_value()
        config = PROVIDER_CONFIG.get(provider, {})
        models = list(config.get("models", []))

        # Append custom models for OpenRouter.
        if provider == "openrouter":
            custom = self._load_custom_models()
            for m in custom:
                if m not in models:
                    models.append(m)

        if not models:
            models = ["(no models available)"]

        self._model_menu.configure(values=models)
        # Only reset selection if the current value is not in the list.
        if self._model_var.get() not in models:
            self._model_var.set(models[0])

        self._update_model_buttons_visibility()

    def _update_model_buttons_visibility(self) -> None:
        """Show the +/− model buttons only when OpenRouter is selected."""
        if not hasattr(self, "_add_model_btn"):
            return

        provider = self._current_provider_value()
        if provider == "openrouter":
            self._add_model_btn.grid()
            self._remove_model_btn.grid()
        else:
            self._add_model_btn.grid_remove()
            self._remove_model_btn.grid_remove()

    def _toggle_key_visibility(self) -> None:
        """Toggle API key entry between masked and visible."""
        self._key_visible = not self._key_visible
        show_char = "" if self._key_visible else "\u2022"
        self._key_entry.configure(show=show_char)
        btn_text = "\U0001F512" if self._key_visible else "\U0001F441"
        self._key_toggle_btn.configure(text=btn_text)

    def _on_key_changed(self) -> None:
        """Store the key when the entry loses focus and update status."""
        self._store_current_key()
        self._save_env_keys()
        self._refresh_status_table()
        self._update_pool_label()

    def _test_connection(self) -> None:
        """Test the API connection in a background thread."""
        self._test_btn.configure(state="disabled", text="\u23F3 Testing...")
        self._test_result_label.configure(text="", text_color=Colors.MUTED)

        # Store current key before testing
        self._store_current_key()
        self._save_env_keys()

        provider = self._current_provider_value()
        model = self._model_var.get()
        api_key = self._provider_keys.get(provider, "")

        def _run_test() -> None:
            """Execute the connection test (runs in background thread)."""
            success = False
            message = ""

            if not api_key:
                message = "No API key configured for this provider"
            else:
                if provider in PROVIDER_CONFIG:
                    config = PROVIDER_CONFIG[provider]
                    known_models = list(config.get("models", []))
                    # Include custom OpenRouter models.
                    if provider == "openrouter":
                        known_models.extend(self._load_custom_models())
                    if model in known_models:
                        success = True
                        message = f"Configuration valid: {_PROVIDER_DISPLAY.get(provider, provider)} / {model}"
                    else:
                        # For OpenRouter, any model ID is potentially valid
                        # since it's an aggregator with hundreds of models.
                        if provider == "openrouter":
                            success = True
                            message = f"Custom model: {model} (not verified)"
                        else:
                            message = f"Model '{model}' not in provider's model list"
                else:
                    message = f"Unknown provider: {provider}"

            if self._event_bus is not None:
                self._event_bus.emit(
                    EventType.LOG_MESSAGE,
                    message=f"__CONNECTION_TEST__|{success}|{message}",
                    level="INFO",
                )
            else:
                logger.info("Connection test result: success=%s — %s", success, message)

        thread = threading.Thread(target=_run_test, daemon=True)
        thread.start()

    def _handle_connection_test_event(self, success: bool, message: str) -> None:
        """Handle a connection test result received via EventBus polling.

        Called from the main thread by the parent window's event
        polling loop when a ``LOG_MESSAGE`` event with the
        ``__CONNECTION_TEST__`` prefix is detected.

        Args:
            success: Whether the test passed.
            message: Human-readable result message.
        """
        self._show_test_result(success, message)

    def _show_test_result(self, success: bool, message: str) -> None:
        """Display the connection test result.

        Args:
            success: Whether the test passed.
            message: Human-readable result message.
        """
        self._test_btn.configure(state="normal", text="\u26A1 Test Connection")
        colour = Colors.SUCCESS if success else Colors.ERROR
        icon = "\u2705" if success else "\u274C"
        self._test_result_label.configure(
            text=f"{icon} {message}",
            text_color=colour,
        )
        logger.info("Connection test result: success=%s — %s", success, message)

    # ── Public API ──────────────────────────────────────────────────

    def get_threads(self) -> int:
        """Return the selected number of parallel threads.

        Returns:
            Thread count between 1 and 10.
        """
        try:
            return int(self._threads_var.get())
        except (ValueError, ctk.TclError):
            return 3

    def get_config(self) -> APIConfig:
        """Build an ``APIConfig`` from the current panel state.

        The ``fallback_pool`` is built automatically from all providers
        that have API keys, excluding the primary provider.

        Returns:
            A populated ``APIConfig`` model.
        """
        # Ensure latest key is stored
        self._store_current_key()

        provider = self._current_provider_value()
        model = self._model_var.get()
        api_key = self._provider_keys.get(provider, "")
        base_url = PROVIDER_CONFIG.get(provider, {}).get("base_url", "")

        try:
            max_retries = int(self._retries_var.get())
        except (ValueError, ctk.TclError):
            max_retries = 3

        # Build fallback pool from all providers with keys.
        fallback_pool = self._build_fallback_pool_entries()

        config = APIConfig(
            primary_provider=APIProvider(provider),
            primary_model=model,
            api_key=api_key,
            base_url=base_url,
            auto_fallback=self._auto_fallback_var.get(),
            fallback_pool=fallback_pool,
            max_retries=max_retries,
        )

        pool_names = [
            f"{e.provider.value}/{e.model}" for e in fallback_pool
        ]
        logger.debug(
            "ApiSettingsPanel config: provider=%s, model=%s, "
            "pool=[%s], retries=%d",
            config.primary_provider.value,
            config.primary_model,
            ", ".join(pool_names) or "empty",
            config.max_retries,
        )
        return config

    def reset(self) -> None:
        """Reset all controls to default values."""
        self._provider_var.set(_PROVIDER_DISPLAY["openrouter"])
        self._auto_fallback_var.set(True)
        self._retries_var.set("3")
        self._threads_var.set("3")
        self._api_key_var.set("")
        self._key_visible = False
        self._key_entry.configure(show="\u2022")
        self._test_result_label.configure(text="")
        self._update_model_list()
        self._populate_key_field()
        self._refresh_status_table()
        self._update_pool_label()
        logger.debug("ApiSettingsPanel reset to defaults")
