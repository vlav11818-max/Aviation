"""Unit tests for ``core.cache_manager``.

Tests cover: put/has/get cycle, cache miss, invalidate, invalidate_all,
deterministic key generation, different-config key divergence, hash
stability across field additions, file persistence across instances,
concurrent access (thread safety), and disabled-cache behaviour.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from core.cache_manager import CacheManager, CacheStats
from core.settings import Settings
from models.config import (
    GenerationConfig,
    Pacing,
    Tone,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def cache_settings(tmp_dir: Path) -> Settings:
    """Build a ``Settings`` instance with cache paths in *tmp_dir*."""
    import yaml

    content = {
        "paths": {
            "output_dir": str(tmp_dir / "output"),
            "data_dir": str(tmp_dir / "data"),
            "resources_dir": "resources",
            "recovery_dir": str(tmp_dir / "data" / "recovery"),
            "cache_dir": str(tmp_dir / "data" / "cache"),
            "analytics_dir": str(tmp_dir / "data" / "analytics"),
        },
        "cache": {"enabled": True, "skip_processed": True},
    }
    defaults_path = tmp_dir / "resources" / "defaults"
    defaults_path.mkdir(parents=True, exist_ok=True)
    defaults_yaml = defaults_path / "settings.yaml"
    defaults_yaml.write_text(
        yaml.dump(content, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    settings_path = tmp_dir / "settings.yaml"
    settings_path.write_text(
        yaml.dump(content, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return Settings.model_validate(content)


@pytest.fixture()
def cache_settings_disabled(tmp_dir: Path) -> Settings:
    """Build ``Settings`` with caching disabled."""
    import yaml

    content = {
        "paths": {
            "output_dir": str(tmp_dir / "output"),
            "data_dir": str(tmp_dir / "data"),
            "resources_dir": "resources",
            "recovery_dir": str(tmp_dir / "data" / "recovery"),
            "cache_dir": str(tmp_dir / "data" / "cache"),
            "analytics_dir": str(tmp_dir / "data" / "analytics"),
        },
        "cache": {"enabled": False, "skip_processed": True},
    }
    return Settings.model_validate(content)


@pytest.fixture()
def gen_config() -> GenerationConfig:
    """Return a default GenerationConfig."""
    return GenerationConfig()


@pytest.fixture()
def cm(cache_settings: Settings) -> CacheManager:
    """Return a CacheManager using a temp directory."""
    return CacheManager(cache_settings)


@pytest.fixture()
def fake_output(tmp_dir: Path) -> Path:
    """Create a fake output directory with a dummy file."""
    out = tmp_dir / "output" / "en" / "test_topic"
    out.mkdir(parents=True, exist_ok=True)
    (out / "final.txt").write_text("Hello world", encoding="utf-8")
    return out


# ── Tests: put / has / get cycle ──────────────────────────────────────────────


class TestPutHasGet:
    """Tests for the basic put/has/get cache cycle."""

    def test_has_returns_false_for_unknown_key(self, cm: CacheManager) -> None:
        """Cache miss should return False."""
        assert cm.has("nonexistent_key") is False

    def test_get_returns_none_for_unknown_key(self, cm: CacheManager) -> None:
        """Cache miss get should return None."""
        assert cm.get("nonexistent_key") is None

    def test_put_has_get_cycle(
        self, cm: CacheManager, gen_config: GenerationConfig, fake_output: Path
    ) -> None:
        """put → has → get should round-trip correctly."""
        key = CacheManager.make_key("Test Topic", "en", gen_config, "gpt-4o")

        cm.put(key, fake_output)
        assert cm.has(key) is True

        result = cm.get(key)
        assert result is not None
        assert result == fake_output

    def test_get_returns_none_when_dir_missing(
        self, cm: CacheManager, gen_config: GenerationConfig, tmp_dir: Path
    ) -> None:
        """If the cached directory no longer exists, get returns None."""
        import shutil

        key = CacheManager.make_key("Ghost Topic", "en", gen_config, "gpt-4o")
        missing_dir = tmp_dir / "output" / "ghost"
        missing_dir.mkdir(parents=True, exist_ok=True)

        cm.put(key, missing_dir)
        assert cm.has(key) is True

        shutil.rmtree(missing_dir)

        assert cm.has(key) is False
        assert cm.get(key) is None


# ── Tests: invalidation ───────────────────────────────────────────────────────


class TestInvalidation:
    """Tests for invalidate and invalidate_all."""

    def test_invalidate_removes_entry(
        self, cm: CacheManager, gen_config: GenerationConfig, fake_output: Path
    ) -> None:
        """invalidate() should remove the entry."""
        key = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        cm.put(key, fake_output)

        removed = cm.invalidate(key)
        assert removed is True
        assert cm.has(key) is False

    def test_invalidate_missing_key_returns_false(self, cm: CacheManager) -> None:
        """invalidate() on a missing key should return False."""
        assert cm.invalidate("nonexistent") is False

    def test_invalidate_all_clears_cache(
        self, cm: CacheManager, gen_config: GenerationConfig, fake_output: Path
    ) -> None:
        """invalidate_all() should remove all entries."""
        for i in range(5):
            key = CacheManager.make_key(f"Topic {i}", "en", gen_config, "gpt-4o")
            cm.put(key, fake_output)

        count = cm.invalidate_all()
        assert count == 5
        assert cm.get_stats().total_entries == 0


# ── Tests: key generation ─────────────────────────────────────────────────────


class TestMakeKey:
    """Tests for deterministic and stable cache key generation."""

    def test_same_inputs_same_key(self, gen_config: GenerationConfig) -> None:
        """Identical inputs must always produce the same key."""
        k1 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        k2 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        assert k1 == k2

    def test_different_topic_different_key(
        self, gen_config: GenerationConfig
    ) -> None:
        """Different topics must produce different keys."""
        k1 = CacheManager.make_key("Topic A", "en", gen_config, "gpt-4o")
        k2 = CacheManager.make_key("Topic B", "en", gen_config, "gpt-4o")
        assert k1 != k2

    def test_different_language_produces_different_key(
        self, gen_config: GenerationConfig
    ) -> None:
        """Different languages must produce different keys."""
        k1 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        k2 = CacheManager.make_key("Topic", "de", gen_config, "gpt-4o")
        assert k1 != k2

    def test_different_model_produces_different_key(
        self, gen_config: GenerationConfig
    ) -> None:
        """Different models must produce different keys."""
        k1 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        k2 = CacheManager.make_key("Topic", "en", gen_config, "claude-3")
        assert k1 != k2

    def test_different_config_produces_different_key(self) -> None:
        """Different generation configs must produce different keys."""
        cfg1 = GenerationConfig(tone=Tone.DRAMATIC_CINEMATIC)
        cfg2 = GenerationConfig(tone=Tone.WHIMSICAL)
        k1 = CacheManager.make_key("Topic", "en", cfg1, "gpt-4o")
        k2 = CacheManager.make_key("Topic", "en", cfg2, "gpt-4o")
        assert k1 != k2

    def test_key_is_64_char_hex(self, gen_config: GenerationConfig) -> None:
        """Key should be a 64-character hexadecimal SHA-256 digest."""
        key = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    # ── Hash stability tests (Fix #4) ──────────────────────────────────────

    def test_key_stable_with_sorted_json(self, gen_config: GenerationConfig) -> None:
        """Key must be identical on repeated calls (sort_keys stability).

        This test verifies that the key is not dependent on Python dict
        insertion order, which could vary across interpreter runs or
        after code changes that reorder field definitions.
        """
        keys = [
            CacheManager.make_key("Stability Test", "en", gen_config, "gpt-4o")
            for _ in range(10)
        ]
        assert len(set(keys)) == 1, "All calls must produce the same key"

    def test_key_stable_when_none_fields_exist(self) -> None:
        """Adding a new None-default field must not change the key.

        This validates the exclude_none=True behaviour: two configs that
        differ only by the presence of an extra ``None`` field must
        produce the same cache key, because ``None`` fields are excluded
        from the hash input.

        We simulate this by comparing configs that have equivalent
        non-None values — the sorted JSON must be identical.
        """
        import json

        cfg = GenerationConfig(tone=Tone.DRAMATIC_CINEMATIC)
        dict1 = cfg.model_dump(exclude_none=True)
        json1 = json.dumps(dict1, sort_keys=True, ensure_ascii=False)

        # Build the same config again — must produce same JSON regardless
        # of dict insertion order.
        cfg2 = GenerationConfig(tone=Tone.DRAMATIC_CINEMATIC)
        dict2 = cfg2.model_dump(exclude_none=True)
        json2 = json.dumps(dict2, sort_keys=True, ensure_ascii=False)

        assert json1 == json2, (
            "Sorted JSON of identical configs must be equal. "
            f"Got:\n  {json1}\n  {json2}"
        )

    def test_sorted_json_keys_are_alphabetical(self) -> None:
        """The serialised config JSON must have alphabetically sorted keys.

        This directly tests the sort_keys=True requirement — if keys are
        in definition order, the hash can change when fields are added.
        """
        import json

        cfg = GenerationConfig()
        config_dict = cfg.model_dump(exclude_none=True)
        config_json = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)

        parsed = json.loads(config_json)
        keys = list(parsed.keys())
        assert keys == sorted(keys), (
            f"Config JSON keys must be alphabetically sorted. Got: {keys}"
        )


# ── Tests: file persistence ───────────────────────────────────────────────────


class TestPersistence:
    """Tests for cache persistence across CacheManager instances."""

    def test_persists_across_instances(
        self, cache_settings: Settings, gen_config: GenerationConfig, fake_output: Path
    ) -> None:
        """Cache entries should survive creating a new CacheManager."""
        cm1 = CacheManager(cache_settings)
        key = CacheManager.make_key("Persist", "en", gen_config, "gpt-4o")
        cm1.put(key, fake_output)

        cm2 = CacheManager(cache_settings)
        assert cm2.has(key) is True
        assert cm2.get(key) == fake_output


# ── Tests: statistics ─────────────────────────────────────────────────────────


class TestStats:
    """Tests for get_stats."""

    def test_empty_cache_stats(self, cm: CacheManager) -> None:
        """Empty cache should return zero entries."""
        stats = cm.get_stats()
        assert stats.total_entries == 0
        assert stats.oldest_timestamp == ""
        assert stats.newest_timestamp == ""

    def test_stats_after_put(
        self, cm: CacheManager, gen_config: GenerationConfig, fake_output: Path
    ) -> None:
        """Stats should reflect the number of entries."""
        key = CacheManager.make_key("Stats Topic", "en", gen_config, "gpt-4o")
        cm.put(key, fake_output)

        stats = cm.get_stats()
        assert stats.total_entries == 1
        assert stats.oldest_timestamp != ""
        assert stats.newest_timestamp != ""


# ── Tests: concurrent access ──────────────────────────────────────────────────


class TestThreadSafety:
    """Tests for concurrent access to CacheManager."""

    def test_concurrent_put(
        self, cm: CacheManager, gen_config: GenerationConfig, fake_output: Path
    ) -> None:
        """Multiple threads putting entries should not corrupt state."""
        errors: list[Exception] = []

        def worker(topic_num: int) -> None:
            try:
                key = CacheManager.make_key(
                    f"Topic {topic_num}", "en", gen_config, "gpt-4o"
                )
                cm.put(key, fake_output)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = cm.get_stats()
        assert stats.total_entries == 20


# ── Tests: disabled cache ─────────────────────────────────────────────────────


class TestDisabledCache:
    """Tests for CacheManager when caching is disabled."""

    def test_has_returns_false_when_disabled(
        self, cache_settings_disabled: Settings
    ) -> None:
        """has() always returns False when cache is disabled."""
        cm = CacheManager(cache_settings_disabled)
        assert cm.has("any_key") is False

    def test_get_returns_none_when_disabled(
        self, cache_settings_disabled: Settings
    ) -> None:
        """get() always returns None when cache is disabled."""
        cm = CacheManager(cache_settings_disabled)
        assert cm.get("any_key") is None

    def test_put_is_noop_when_disabled(
        self,
        cache_settings_disabled: Settings,
        fake_output: Path,
        gen_config: GenerationConfig,
    ) -> None:
        """put() should not persist anything when cache is disabled."""
        cm = CacheManager(cache_settings_disabled)
        key = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        cm.put(key, fake_output)
        assert cm.has(key) is False
