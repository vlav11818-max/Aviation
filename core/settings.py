"""Settings loader for AI Story Generator Pro.

Loads ``settings.yaml``, merges with defaults from
``resources/defaults/settings.yaml``, validates with pydantic, and
exposes nested sections as typed objects.  Runtime overrides are
supported by mutating the loaded instance.

Typical usage::

    settings = Settings.load("settings.yaml")
    settings.generation.min_score = 8.5  # runtime override
    # settings.api.primary_provider  # -> "openrouter"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from core.exceptions import ConfigError

logger = logging.getLogger(__name__)

# ── Default resource path ──────────────────────────────────────────────────

_DEFAULT_SETTINGS_PATH = Path("resources/defaults/settings.yaml")


# ── Nested settings models ─────────────────────────────────────────────────


class GenerationSettings(BaseModel):
    """Generation-related tunable parameters."""

    tone: str = Field(default="dramatic_cinematic", description="Story tone preset.")
    perspective: str = Field(default="third_person", description="Narration perspective.")
    register: str = Field(default="conversational", description="Language register.")
    pacing: str = Field(default="medium", description="Story pacing (slow/medium/fast).")
    audience: str = Field(default="all_ages", description="Target audience.")
    dialog_density: str = Field(default="medium", description="Dialog density level.")
    target_words: int = Field(default=3000, ge=500, le=10000, description="Target word count.")
    structure: str = Field(default="three_act", description="Story structure type.")
    genres: list[str] = Field(
        default_factory=lambda: ["fantasy", "mystery"],
        description="Selected genres.",
    )
    min_score: float = Field(
        default=9.0, ge=0.0, le=10.0, description="Minimum acceptable score."
    )
    max_attempts: int = Field(
        default=5, ge=1, le=20, description="Max evaluation/revision attempts."
    )
    min_score_improvement: float = Field(
        default=0.3,
        ge=0.0,
        le=10.0,
        description=(
            "Minimum score improvement between consecutive evaluations to "
            "justify another revision attempt.  If improvement is below "
            "this threshold the loop stops early (plateau detection)."
        ),
    )
    voiceover_optimized: bool = Field(
        default=True, description="Optimize output for voiceover."
    )
    avoid_complex_sentences: bool = Field(
        default=True, description="Simplify sentence structure."
    )
    pause_markers: bool = Field(
        default=True, description="Include pause markers in text."
    )
    no_headers: bool = Field(
        default=True, description="Suppress chapter/section headers."
    )
    no_meta_comments: bool = Field(
        default=True, description="Remove AI meta-comments."
    )
    # ── Per-section quality gate (Fix #8) ───────────────────────────────────
    section_quality_check: bool = Field(
        default=True,
        description=(
            "Run a lightweight programmatic L1 quality check after each "
            "section is generated.  Sections that fail are regenerated up "
            "to section_max_retries additional times before the best result "
            "is kept.  Disable to skip the gate entirely (not recommended)."
        ),
    )
    section_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Maximum number of additional generation attempts when a section "
            "fails the per-section quality gate.  0 = no retries (generate "
            "once regardless of quality).  Max 5."
        ),
    )


class RateLimitsSettings(BaseModel):
    """Per-provider max concurrent requests."""

    openrouter: int = Field(default=5, ge=1)
    openai: int = Field(default=3, ge=1)
    anthropic: int = Field(default=2, ge=1)
    google: int = Field(default=5, ge=1)
    deepseek: int = Field(default=3, ge=1)
    qwen: int = Field(default=3, ge=1)


class PricingEntry(BaseModel):
    """Per-model pricing (USD per 1M tokens)."""

    input: float = Field(ge=0.0, description="Cost per 1M input tokens in USD.")
    output: float = Field(ge=0.0, description="Cost per 1M output tokens in USD.")



class ProviderModels(BaseModel):
    """Per-provider model role configuration.

    Each provider can define three model roles:
    - ``primary``: main generation model (concept, outline, section, revision)
    - ``fallback``: used when primary fails after retries
    - ``evaluation``: lighter/cheaper model used for evaluation prompts
    """

    primary: str = Field(default="", description="Primary generation model.")
    fallback: str = Field(default="", description="Fallback model.")
    evaluation: str = Field(default="", description="Evaluation model (lighter/cheaper).")


class APISettings(BaseModel):
    """API-related settings."""

    primary_provider: str = Field(default="openrouter", description="Primary API provider.")
    primary_model: str = Field(
        default="anthropic/claude-3.5-sonnet", description="Primary model."
    )
    fallback_provider: str = Field(default="openai", description="Fallback API provider.")
    fallback_model: str = Field(default="gpt-4o", description="Fallback model.")
    auto_fallback: bool = Field(
        default=True, description="Enable automatic fallback on errors."
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Max retries per API call."
    )
    timeout_seconds: int = Field(
        default=120, ge=10, le=600, description="Timeout per API call in seconds."
    )
    models: dict[str, ProviderModels] = Field(
        default_factory=dict,
        description=(
            "Per-provider model configuration. Each key is a provider name "
            "(e.g. 'openrouter') mapping to primary/fallback/evaluation models."
        ),
    )
    rate_limits: RateLimitsSettings = Field(default_factory=RateLimitsSettings)
    pricing: dict[str, PricingEntry] = Field(
        default_factory=dict, description="Per-model pricing table."
    )


class RetrySettings(BaseModel):
    """Retry/backoff configuration."""

    initial_delay_seconds: float = Field(default=1.0, ge=0.1)
    max_delay_seconds: float = Field(default=16.0, ge=1.0)
    exponential_base: int = Field(default=2, ge=2, le=4)
    max_retries: int = Field(default=3, ge=0, le=10)


class ParallelismSettings(BaseModel):
    """Parallelism configuration."""

    max_workers: int = Field(
        default=3, ge=1, le=10, description="Number of parallel workers."
    )
    auto_throttle: bool = Field(
        default=True, description="Auto-reduce on rate limit hits."
    )
    separate_queues: bool = Field(
        default=True, description="Separate task queue per worker."
    )


class SSMLSettings(BaseModel):
    """SSML export rules."""

    paragraph_break: str = Field(default="600ms")
    scene_break: str = Field(default="1000ms")
    dialog_pause: str = Field(default="400ms")
    dramatic_pause: str = Field(default="800ms")
    sentence_pause: str = Field(default="200ms")
    slow_for_dramatic: bool = Field(default=True)
    emphasis_for_key_words: bool = Field(default=False)


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: str = Field(default="DEBUG")
    max_files: int = Field(default=10, ge=1)
    max_file_size_mb: int = Field(default=10, ge=1)
    log_dir: str = Field(default="logs")


class PathsSettings(BaseModel):
    """Filesystem path configuration."""

    output_dir: str = Field(default="output")
    data_dir: str = Field(default="data")
    resources_dir: str = Field(default="resources")
    recovery_dir: str = Field(default="data/recovery")
    cache_dir: str = Field(default="data/cache")
    analytics_dir: str = Field(default="data/analytics")


class CacheSettings(BaseModel):
    """Cache configuration."""

    enabled: bool = Field(default=True)
    skip_processed: bool = Field(default=True)


class StrategySettings(BaseModel):
    """Word-count thresholds for automatic strategy selection."""

    single_shot_max: int = Field(default=2000, ge=500)
    two_pass_max: int = Field(default=4000, ge=1000)


# ── Root settings model ────────────────────────────────────────────────────


class Settings(BaseModel):
    """Root settings model aggregating all configuration sections.

    Loaded from ``settings.yaml`` with fallback to
    ``resources/defaults/settings.yaml``.
    """

    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    api: APISettings = Field(default_factory=APISettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    parallelism: ParallelismSettings = Field(default_factory=ParallelismSettings)
    ssml: SSMLSettings = Field(default_factory=SSMLSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        """Load settings from a YAML file, merged with defaults.

        Args:
            path: Path to the user settings file.  If ``None`` or the file
                does not exist, only defaults are used.

        Returns:
            A fully populated ``Settings`` instance.

        Raises:
            ConfigError: If the YAML is malformed or values fail validation.
        """
        defaults_data = cls._read_yaml(_DEFAULT_SETTINGS_PATH)
        user_data: dict[str, Any] = {}

        if path is not None:
            user_path = Path(path)
            if user_path.exists():
                user_data = cls._read_yaml(user_path)
                logger.info("User settings loaded from %s", user_path)
            else:
                logger.warning(
                    "Settings file %s not found — using defaults only", user_path
                )

        merged = cls._deep_merge(defaults_data, user_data)

        try:
            settings = cls.model_validate(merged)
        except (ValueError, TypeError) as exc:
            raise ConfigError(
                f"Invalid settings: {exc}"
            ) from exc

        logger.info("Settings initialised successfully")
        return settings

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        """Read and parse a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed dictionary (may be empty if the file is empty).

        Raises:
            ConfigError: If the file cannot be read or parsed.
        """
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
                return data if isinstance(data, dict) else {}
        except FileNotFoundError as exc:
            raise ConfigError(f"Settings file not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc

    @staticmethod
    def _deep_merge(
        base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Recursively merge *override* into *base*.

        Args:
            base: Default values dictionary.
            override: User values dictionary (takes precedence).

        Returns:
            New merged dictionary.  Neither input is mutated.
        """
        import copy

        merged: dict[str, Any] = {}
        all_keys = set(base) | set(override)
        for key in all_keys:
            if key in override and key in base:
                base_val = base[key]
                over_val = override[key]
                if isinstance(base_val, dict) and isinstance(over_val, dict):
                    merged[key] = Settings._deep_merge(base_val, over_val)
                else:
                    merged[key] = copy.deepcopy(over_val)
            elif key in override:
                merged[key] = copy.deepcopy(override[key])
            else:
                merged[key] = copy.deepcopy(base[key])
        return merged
