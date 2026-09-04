"""Unit tests for the Settings loader.

Tests: load from file, load with missing file (falls back to defaults),
runtime override, validation of invalid values, nested section access,
deep merge behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.exceptions import ConfigError
from core.settings import Settings


# ════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════


class TestSettingsLoading:
    """Tests for Settings.load() and YAML parsing."""

    def test_load_from_valid_file(self, sample_settings_yaml: Path) -> None:
        """Should load and parse a valid settings file."""
        settings = Settings.load(sample_settings_yaml)
        assert settings.generation.tone == "dramatic_cinematic"
        assert settings.api.primary_provider == "openrouter"
        assert settings.parallelism.max_workers == 3

    def test_load_missing_file_uses_defaults(self, tmp_dir: Path) -> None:
        """Missing file should fall back to defaults without error."""
        settings = Settings.load(tmp_dir / "nonexistent.yaml")
        assert settings.generation.target_words == 3000
        assert settings.generation.min_score == 9.0

    def test_load_none_path_uses_defaults(self) -> None:
        """None path should use defaults only."""
        settings = Settings.load(None)
        assert settings.generation.target_words == 3000

    def test_load_malformed_yaml_raises_config_error(self, tmp_dir: Path) -> None:
        """Malformed YAML should raise ConfigError."""
        bad_file = tmp_dir / "bad.yaml"
        bad_file.write_text(":::invalid yaml:::", encoding="utf-8")
        with pytest.raises(ConfigError):
            Settings._read_yaml(bad_file)

    def test_load_empty_file_uses_defaults(self, tmp_dir: Path) -> None:
        """Empty YAML file should produce default settings."""
        empty_file = tmp_dir / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        settings = Settings.load(empty_file)
        assert settings.generation.target_words == 3000


# ════════════════════════════════════════════════════════════════════════
# Deep Merge
# ════════════════════════════════════════════════════════════════════════


class TestDeepMerge:
    """Tests for the recursive deep-merge logic."""

    def test_override_scalar(self) -> None:
        """User value should override default scalar."""
        base = {"generation": {"target_words": 3000}}
        override = {"generation": {"target_words": 5000}}
        merged = Settings._deep_merge(base, override)
        assert merged["generation"]["target_words"] == 5000

    def test_preserve_unspecified(self) -> None:
        """Keys absent from override should be preserved from base."""
        base = {"generation": {"target_words": 3000, "min_score": 9.0}}
        override = {"generation": {"target_words": 5000}}
        merged = Settings._deep_merge(base, override)
        assert merged["generation"]["min_score"] == 9.0
        assert merged["generation"]["target_words"] == 5000

    def test_add_new_keys(self) -> None:
        """Keys in override that are not in base should be added."""
        base = {"a": 1}
        override = {"b": 2}
        merged = Settings._deep_merge(base, override)
        assert merged == {"a": 1, "b": 2}

    def test_nested_merge(self) -> None:
        """Deeply nested dicts should merge recursively."""
        base = {"api": {"rate_limits": {"openai": 3, "anthropic": 2}}}
        override = {"api": {"rate_limits": {"openai": 5}}}
        merged = Settings._deep_merge(base, override)
        assert merged["api"]["rate_limits"]["openai"] == 5
        assert merged["api"]["rate_limits"]["anthropic"] == 2

    def test_does_not_mutate_inputs(self) -> None:
        """Neither base nor override should be modified."""
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        base_copy = {"a": {"b": 1}}
        override_copy = {"a": {"b": 2}}
        Settings._deep_merge(base, override)
        assert base == base_copy
        assert override == override_copy


# ════════════════════════════════════════════════════════════════════════
# Runtime Overrides
# ════════════════════════════════════════════════════════════════════════


class TestRuntimeOverrides:
    """Tests for mutating settings at runtime."""

    def test_override_min_score(self, sample_settings_yaml: Path) -> None:
        """Should be able to change min_score after loading."""
        settings = Settings.load(sample_settings_yaml)
        assert settings.generation.min_score == 9.0
        settings.generation.min_score = 8.5
        assert settings.generation.min_score == 8.5

    def test_override_max_workers(self, sample_settings_yaml: Path) -> None:
        """Should be able to change max_workers after loading."""
        settings = Settings.load(sample_settings_yaml)
        settings.parallelism.max_workers = 5
        assert settings.parallelism.max_workers == 5

    def test_override_nested_value(self, sample_settings_yaml: Path) -> None:
        """Should be able to change deeply nested values."""
        settings = Settings.load(sample_settings_yaml)
        settings.api.primary_model = "gpt-4o"
        assert settings.api.primary_model == "gpt-4o"


# ════════════════════════════════════════════════════════════════════════
# Nested Sections
# ════════════════════════════════════════════════════════════════════════


class TestNestedSections:
    """Tests for accessing nested configuration sections."""

    def test_generation_section(self, sample_settings_yaml: Path) -> None:
        """Generation section should have all expected fields."""
        settings = Settings.load(sample_settings_yaml)
        gen = settings.generation
        assert gen.tone == "dramatic_cinematic"
        assert gen.voiceover_optimized is True
        assert gen.no_headers is True

    def test_api_section(self, sample_settings_yaml: Path) -> None:
        """API section should have provider and rate limits."""
        settings = Settings.load(sample_settings_yaml)
        api = settings.api
        assert api.primary_provider == "openrouter"
        assert api.rate_limits.openrouter == 5
        assert api.rate_limits.anthropic == 2

    def test_retry_section(self, sample_settings_yaml: Path) -> None:
        """Retry section should have backoff parameters."""
        settings = Settings.load(sample_settings_yaml)
        retry = settings.retry
        assert retry.initial_delay_seconds == 1.0
        assert retry.exponential_base == 2

    def test_ssml_section(self, sample_settings_yaml: Path) -> None:
        """SSML section should have all pause durations."""
        settings = Settings.load(sample_settings_yaml)
        ssml = settings.ssml
        assert ssml.paragraph_break == "600ms"
        assert ssml.scene_break == "1000ms"
        assert ssml.slow_for_dramatic is True

    def test_strategy_section(self, sample_settings_yaml: Path) -> None:
        """Strategy section should have word-count thresholds."""
        settings = Settings.load(sample_settings_yaml)
        strat = settings.strategy
        assert strat.single_shot_max == 2000
        assert strat.two_pass_max == 4000

    def test_cache_section(self, sample_settings_yaml: Path) -> None:
        """Cache section should have enabled and skip_processed."""
        settings = Settings.load(sample_settings_yaml)
        assert settings.cache.enabled is True
        assert settings.cache.skip_processed is True


# ════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════


class TestSettingsValidation:
    """Tests for pydantic validation of settings values."""

    def test_invalid_target_words(self, tmp_dir: Path) -> None:
        """target_words below minimum should raise ConfigError."""
        content = {"generation": {"target_words": 10}}
        path = tmp_dir / "bad_settings.yaml"
        path.write_text(
            yaml.dump(content, default_flow_style=False),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            Settings.load(path)

    def test_invalid_max_workers(self, tmp_dir: Path) -> None:
        """max_workers above maximum should raise ConfigError."""
        content = {"parallelism": {"max_workers": 99}}
        path = tmp_dir / "bad_settings.yaml"
        path.write_text(
            yaml.dump(content, default_flow_style=False),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            Settings.load(path)

    def test_partial_override_valid(self, tmp_dir: Path) -> None:
        """Partial settings file should merge cleanly with defaults."""
        content = {"generation": {"min_score": 8.0}}
        path = tmp_dir / "partial.yaml"
        path.write_text(
            yaml.dump(content, default_flow_style=False),
            encoding="utf-8",
        )
        settings = Settings.load(path)
        assert settings.generation.min_score == 8.0
        # Other fields should come from defaults
        assert settings.generation.target_words == 3000
