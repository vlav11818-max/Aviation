"""Entry point for AI Story Generator Pro v1.0.

Initialises all core components, wires dependencies, checks for
unfinished batch recovery, sets up signal handlers for graceful
shutdown, and launches the GUI.

FIX: _build_api_config() now builds a ``fallback_pool`` from all
providers that have API keys in environment variables, giving
automatic multi-provider fallback.

Usage::

    python main.py
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Application entry point."""
    # ── 1. Load settings ────────────────────────────────────────────
    from core.settings import Settings

    try:
        settings = Settings.load("settings.yaml")
    except Exception as exc:
        # Logging is not yet configured, so use basicConfig as fallback.
        logging.basicConfig(level=logging.ERROR)
        logger.error("Failed to load settings: %s", exc)
        sys.exit(1)

    # ── 2. Configure logging ────────────────────────────────────────
    from utils.logger import setup_logging

    setup_logging(
        log_dir=settings.logging.log_dir,
        level=settings.logging.level,
        max_files=settings.logging.max_files,
        max_file_size_mb=settings.logging.max_file_size_mb,
    )
    logger.info("AI Story Generator Pro v1.0 starting")
    logger.info("Settings loaded successfully")

    # ── 3. Core infrastructure ──────────────────────────────────────
    from core.events import EventBus

    event_bus = EventBus()
    logger.info("EventBus created")

    # ── 4. API layer ────────────────────────────────────────────────
    #
    # APIClient creates its own per-provider RateLimiter instances
    # internally.  It requires an APIConfig (from models) that holds
    # provider, model, keys, and resilience settings.
    from core.api_client import APIClient

    api_config = _build_api_config(settings)

    api_client = APIClient(
        api_config=api_config,
        settings=settings,
        event_bus=event_bus,
    )
    logger.info("APIClient created")

    # ── 5. Prompt and state management ──────────────────────────────
    from core.prompt_manager import PromptManager
    from core.state_manager import StateManager

    resources_dir = Path(settings.paths.resources_dir)
    prompt_manager = PromptManager(resources_dir=resources_dir)
    state_manager = StateManager()
    logger.info("PromptManager and StateManager created")

    # ── 6. Infrastructure services ──────────────────────────────────
    from core.analytics_collector import AnalyticsCollector
    from core.cache_manager import CacheManager
    from core.cost_estimator import CostEstimator
    from core.input_validator import InputValidator
    from core.recovery_manager import RecoveryManager

    cache_manager = CacheManager(settings=settings)
    recovery_manager = RecoveryManager(settings=settings)
    analytics_collector = AnalyticsCollector(
        analytics_dir=settings.paths.analytics_dir,
    )
    cost_estimator = CostEstimator(settings=settings)
    input_validator = InputValidator()
    logger.info(
        "Infrastructure services created "
        "(cache, recovery, analytics, cost, validator)"
    )

    # ── 7. Step runner ──────────────────────────────────────────────
    from core.step_runner import StepRunner

    step_runner = StepRunner(
        state_manager=state_manager,
        api_client=api_client,
        prompt_manager=prompt_manager,
        event_bus=event_bus,
        settings=settings,
        analytics_collector=analytics_collector,
    )
    logger.info("StepRunner created")

    # ── 8. Parallel processor ───────────────────────────────────────
    from core.parallel_processor import ParallelProcessor

    parallel_processor = ParallelProcessor(
        step_runner=step_runner,
        api_client=api_client,
        state_manager=state_manager,
        prompt_manager=prompt_manager,
        cache_manager=cache_manager,
        recovery_manager=recovery_manager,
        event_bus=event_bus,
        settings=settings,
    )
    logger.info("ParallelProcessor created")

    # ── 9. Recovery check ───────────────────────────────────────────
    _check_recovery(recovery_manager)

    # ── 10. Build GUI ───────────────────────────────────────────────
    from gui.main_window import MainWindow

    window = MainWindow(
        event_bus=event_bus,
        settings=settings,
    )

    # Wire late-binding dependencies into the window
    window.set_processor(parallel_processor)
    window.set_validator(input_validator)
    window.set_cost_estimator(cost_estimator)
    logger.info("MainWindow created and dependencies wired")

    # ── 11. Signal handlers for graceful shutdown ───────────────────
    _setup_signals(window, api_client)

    # ── 12. Run ─────────────────────────────────────────────────────
    logger.info("Starting GUI main loop")
    try:
        window.mainloop()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
    finally:
        _shutdown(api_client)

    logger.info("AI Story Generator Pro v1.0 exited")


# ── Helpers ─────────────────────────────────────────────────────────────

# Map provider names to env-var names — shared by _build_api_config
# and _build_api_config_from_args.
_KEY_ENV_MAP: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
}

# Order in which fallback providers are tried (matches settings.yaml
# and the GUI status table).
_FALLBACK_ORDER: list[str] = [
    "openrouter",
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "qwen",
]


def _build_api_config(settings: "Settings") -> "APIConfig":
    """Build an APIConfig from settings and environment variables.

    Reads API keys from environment variables and combines them with
    provider/model settings from the YAML config.  Automatically
    builds a ``fallback_pool`` from all providers that have API keys,
    excluding the primary.

    Args:
        settings: The loaded application settings.

    Returns:
        A fully populated ``APIConfig`` with fallback pool.
    """
    from models.config import (
        APIConfig,
        APIProvider,
        FallbackPoolEntry,
        PROVIDER_CONFIG,
    )

    # Load .env file if python-dotenv is available.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    primary_provider_str = settings.api.primary_provider
    primary_model = settings.api.primary_model

    # Resolve primary API key from environment.
    primary_key = os.environ.get(
        _KEY_ENV_MAP.get(primary_provider_str, ""), ""
    )

    # Resolve base URL from PROVIDER_CONFIG.
    primary_base_url = PROVIDER_CONFIG.get(
        primary_provider_str, {}
    ).get("base_url", "")

    # Build the primary provider enum.
    try:
        primary_provider = APIProvider(primary_provider_str)
    except ValueError:
        primary_provider = APIProvider.OPENROUTER
        logger.warning(
            "Unknown primary provider '%s' — defaulting to openrouter",
            primary_provider_str,
        )

    # Build fallback pool from all providers with env keys.
    fallback_pool: list[FallbackPoolEntry] = []
    for prov_name in _FALLBACK_ORDER:
        if prov_name == primary_provider_str:
            continue

        env_var = _KEY_ENV_MAP.get(prov_name, "")
        prov_key = os.environ.get(env_var, "")
        if not prov_key:
            continue

        try:
            prov_enum = APIProvider(prov_name)
        except ValueError:
            logger.warning("Unknown provider in fallback order: %s", prov_name)
            continue

        # Use the first model from provider config, or the model
        # from settings.api.models if available.
        prov_models_cfg = settings.api.models.get(prov_name)
        if prov_models_cfg and prov_models_cfg.primary:
            prov_model = prov_models_cfg.primary
        else:
            prov_config = PROVIDER_CONFIG.get(prov_name, {})
            prov_model_list = prov_config.get("models", [])
            prov_model = prov_model_list[0] if prov_model_list else ""

        fallback_pool.append(
            FallbackPoolEntry(
                provider=prov_enum,
                model=prov_model,
                api_key=prov_key,
            )
        )

    config = APIConfig(
        primary_provider=primary_provider,
        primary_model=primary_model,
        api_key=primary_key,
        base_url=primary_base_url,
        auto_fallback=settings.api.auto_fallback,
        fallback_pool=fallback_pool,
        max_retries=settings.api.max_retries,
        timeout=settings.api.timeout_seconds,
    )

    if not primary_key:
        logger.warning(
            "No API key found for primary provider '%s' "
            "(env var: %s). API calls will fail.",
            primary_provider_str,
            _KEY_ENV_MAP.get(primary_provider_str, "UNKNOWN"),
        )

    pool_summary = ", ".join(
        f"{e.provider.value}/{e.model}" for e in fallback_pool
    ) or "none"
    logger.info(
        "APIConfig built: primary=%s/%s, fallback_pool=[%s], "
        "auto_fallback=%s, key_present=%s",
        primary_provider.value,
        primary_model,
        pool_summary,
        config.auto_fallback,
        bool(primary_key),
    )

    return config


def _check_recovery(recovery_manager: "RecoveryManager") -> None:
    """Check for unfinished batches from a previous run.

    If a recoverable batch exists, log an informational message.
    The GUI will prompt the user for the actual recovery action.

    Args:
        recovery_manager: The recovery manager to query.
    """
    try:
        if recovery_manager.has_unfinished():
            options = recovery_manager.get_recovery_options()
            logger.info(
                "Recovery: unfinished batch found — "
                "total=%d, completed=%d, failed=%d, queued=%d. "
                "User will be prompted in the GUI.",
                options.get("total", 0),
                options.get("completed", 0),
                options.get("failed", 0),
                options.get("queued", 0),
            )
    except Exception as exc:
        logger.warning("Recovery check failed: %s", exc)


def _setup_signals(window: Any, api_client: Any) -> None:
    """Set up OS signal handlers for graceful shutdown.

    ``SIGINT`` and ``SIGTERM`` trigger window destruction and
    session cleanup.

    Args:
        window: The main GUI window.
        api_client: The API client (for session cleanup).
    """
    def _handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down", sig_name)
        try:
            window.destroy()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _shutdown(api_client: Any) -> None:
    """Clean up resources on exit.

    Args:
        api_client: The API client whose sessions to close.
    """
    try:
        api_client.close_sync()
        logger.info("API client sessions closed")
    except Exception as exc:
        logger.warning("Error closing API client: %s", exc)


# ── CLI argument parser ────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ``ArgumentParser``.
    """
    from models.config import (
        PROVIDER_CONFIG,
        SUPPORTED_LANGUAGE_CODES,
        Tone,
        Perspective,
        Register,
        StructureType,
    )

    parser = argparse.ArgumentParser(
        prog="ai-story-generator",
        description="AI Story Generator Pro v1.0 — Generate evergreen YouTube stories.",
    )
    parser.add_argument(
        "--version", action="version", version="AI Story Generator Pro v1.0"
    )

    # Mode
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (batch processing mode).",
    )

    # Topics and language
    parser.add_argument("--topics", type=str, default=None, help="Path to topics file.")
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        choices=SUPPORTED_LANGUAGE_CODES,
        help="Target language code.",
    )

    # Provider config
    valid_providers = list(PROVIDER_CONFIG.keys())
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        choices=valid_providers,
        help="Primary API provider.",
    )
    parser.add_argument("--model", type=str, default="", help="Model identifier.")
    parser.add_argument("--api-key", type=str, default="", dest="api_key", help="API key.")
    parser.add_argument(
        "--fallback-provider",
        type=str,
        default=None,
        dest="fallback_provider",
        choices=[None] + valid_providers,
        help="Fallback API provider (legacy, ignored if pool is active).",
    )
    parser.add_argument(
        "--fallback-model", type=str, default="", dest="fallback_model", help="Fallback model."
    )
    parser.add_argument(
        "--fallback-api-key", type=str, default="", dest="fallback_api_key", help="Fallback API key."
    )

    # Generation config
    valid_tones = [t.value for t in Tone]
    valid_perspectives = [p.value for p in Perspective]
    valid_registers = [r.value for r in Register]
    valid_structures = [s.value for s in StructureType]

    parser.add_argument(
        "--tone", type=str, default=None, choices=valid_tones, help="Story tone."
    )
    parser.add_argument(
        "--perspective",
        type=str,
        default=None,
        choices=valid_perspectives,
        help="Narration perspective.",
    )
    parser.add_argument(
        "--register",
        type=str,
        default=None,
        choices=valid_registers,
        help="Language register.",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default=None,
        choices=valid_structures,
        help="Story structure.",
    )
    parser.add_argument(
        "--target-words", type=int, default=None, dest="target_words", help="Target word count."
    )
    parser.add_argument(
        "--min-score", type=float, default=None, dest="min_score", help="Minimum score (0–10)."
    )
    parser.add_argument(
        "--max-attempts", type=int, default=None, dest="max_attempts", help="Max attempts."
    )

    # Parallelism
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers."
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=None, dest="output_dir", help="Output directory."
    )

    return parser


def _build_api_config_from_args(
    args: "argparse.Namespace",
    settings: "Settings",
) -> "APIConfig":
    """Build an ``APIConfig`` from CLI arguments.

    CLI arguments take precedence over settings.yaml values.
    API keys are resolved from args first, then environment variables.
    Builds a fallback pool automatically from all available keys.

    Args:
        args: Parsed CLI arguments.
        settings: Application settings (for defaults).

    Returns:
        A fully populated ``APIConfig``.
    """
    from models.config import APIConfig, APIProvider, FallbackPoolEntry, PROVIDER_CONFIG

    provider_str = args.provider
    primary_provider = APIProvider(provider_str)

    # Resolve model — use CLI value, or fall back to first model in config.
    model = args.model
    if not model:
        provider_models = PROVIDER_CONFIG.get(provider_str, {}).get("models", [])
        model = provider_models[0] if provider_models else ""

    # Resolve API key — CLI arg > env var.
    api_key = args.api_key
    if not api_key:
        env_var = _KEY_ENV_MAP.get(provider_str, "")
        api_key = os.environ.get(env_var, "")

    # Base URL from provider config.
    base_url = PROVIDER_CONFIG.get(provider_str, {}).get("base_url", "")

    # Build fallback pool from env keys.
    fallback_pool: list[FallbackPoolEntry] = []
    for prov_name in _FALLBACK_ORDER:
        if prov_name == provider_str:
            continue
        env_var = _KEY_ENV_MAP.get(prov_name, "")
        prov_key = os.environ.get(env_var, "")
        if not prov_key:
            continue

        try:
            prov_enum = APIProvider(prov_name)
        except ValueError:
            continue

        prov_models_cfg = settings.api.models.get(prov_name)
        if prov_models_cfg and prov_models_cfg.primary:
            prov_model = prov_models_cfg.primary
        else:
            prov_config = PROVIDER_CONFIG.get(prov_name, {})
            prov_model_list = prov_config.get("models", [])
            prov_model = prov_model_list[0] if prov_model_list else ""

        fallback_pool.append(
            FallbackPoolEntry(
                provider=prov_enum,
                model=prov_model,
                api_key=prov_key,
            )
        )

    return APIConfig(
        primary_provider=primary_provider,
        primary_model=model,
        api_key=api_key,
        base_url=base_url,
        auto_fallback=bool(fallback_pool),
        fallback_pool=fallback_pool,
        max_retries=settings.retry.max_retries if hasattr(settings, "retry") else 3,
        timeout=settings.api.timeout_seconds if hasattr(settings, "api") else 120,
    )


def _build_generation_config_from_args(
    args: "argparse.Namespace",
    settings: "Settings",
) -> "GenerationConfig":
    """Build a ``GenerationConfig`` from CLI arguments.

    CLI values override settings defaults; ``None`` means "use setting".

    Args:
        args: Parsed CLI arguments.
        settings: Application settings (for defaults).

    Returns:
        A fully populated ``GenerationConfig``.
    """
    from models.config import (
        GenerationConfig,
        Tone,
        Perspective,
        Register,
        StructureType,
    )

    gen = settings.generation

    tone = Tone(args.tone) if args.tone else gen.tone
    perspective = Perspective(args.perspective) if args.perspective else gen.perspective
    register = Register(args.register) if args.register else gen.register
    structure = StructureType(args.structure) if args.structure else gen.structure
    target_words = args.target_words if args.target_words is not None else gen.target_words
    min_score = args.min_score if args.min_score is not None else gen.min_score
    max_attempts = args.max_attempts if args.max_attempts is not None else gen.max_attempts

    return GenerationConfig(
        tone=tone,
        perspective=perspective,
        register=register,
        structure=structure,
        target_words=target_words,
        min_score=min_score,
        max_attempts=max_attempts,
    )


def _load_topics(
    path: str,
    validator: "InputValidator",
) -> list[str]:
    """Load and validate a topics file.

    Args:
        path: Path to the topics text file (one topic per line).
        validator: The ``InputValidator`` to use.

    Returns:
        List of validated topic strings.

    Raises:
        SystemExit: If the file is missing, empty, or validation fails.
    """
    from pathlib import Path as _Path

    file_path = _Path(path)
    if not file_path.exists():
        logger.error("Topics file not found: %s", path)
        sys.exit(1)

    try:
        result = validator.validate_topics_file(path)
    except Exception as exc:
        logger.error("Failed to validate topics file '%s': %s", path, exc)
        sys.exit(1)

    for warning in result.warnings:
        logger.warning("Topics: %s", warning)

    if not result.topics:
        logger.error("No valid topics found in '%s'", path)
        sys.exit(1)

    return result.topics


if __name__ == "__main__":
    main()
