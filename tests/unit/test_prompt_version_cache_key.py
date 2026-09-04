"""Tests for Fix #9 — prompt version in cache key.

Covers:
- PromptManager.get_prompt_version() returns a 12-char hex string
- Version is stable (same templates → same hash)
- Version changes when a template file changes
- Version changes when a template file is added or removed
- Version is cached; clear_cache() invalidates it
- Version returns "unknown" when templates directory is missing/empty
- CacheManager.make_key() with and without prompt_version
- Same topic/config produces different key when prompt_version changes
- Omitting prompt_version produces legacy-compatible key (no version suffix)
- make_key() is backward-compatible: old callers omitting prompt_version
  still get a valid key
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.cache_manager import CacheManager
from core.exceptions import PromptTemplateError
from core.prompt_manager import PromptManager
from models.config import GenerationConfig


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def template_dir(tmp_path: Path) -> Path:
    """Create a temporary templates directory with two stub templates."""
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    (tpl_dir / "concept.txt").write_text(
        "Write a concept about {topic}.", encoding="utf-8"
    )
    (tpl_dir / "outline.txt").write_text(
        "Create an outline for {topic}.", encoding="utf-8"
    )
    return tpl_dir


@pytest.fixture()
def pm(tmp_path: Path, template_dir: Path) -> PromptManager:
    """PromptManager wired to temp directories."""
    cultural_dir = tmp_path / "cultural"
    cultural_dir.mkdir()
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    return PromptManager(
        templates_dir=template_dir,
        cultural_dir=cultural_dir,
        structures_dir=structures_dir,
    )


@pytest.fixture()
def gen_config() -> GenerationConfig:
    """Minimal GenerationConfig for key generation tests."""
    return GenerationConfig()


# ── PromptManager.get_prompt_version() ───────────────────────────────────────


class TestGetPromptVersion:
    """Tests for PromptManager.get_prompt_version()."""

    def test_returns_12_char_hex_string(self, pm: PromptManager) -> None:
        """Version should be a 12-character hexadecimal string."""
        version = pm.get_prompt_version()
        assert isinstance(version, str)
        assert len(version) == 12
        assert all(c in "0123456789abcdef" for c in version)

    def test_stable_across_calls(self, pm: PromptManager) -> None:
        """Same templates → same version on repeated calls."""
        v1 = pm.get_prompt_version()
        v2 = pm.get_prompt_version()
        assert v1 == v2

    def test_cached_after_first_call(self, pm: PromptManager) -> None:
        """Second call should use the cache (no recomputation)."""
        v1 = pm.get_prompt_version()
        # Modify a template on disk — should NOT change the cached version.
        # (This tests that caching works, not that it picks up live changes.)
        template_files = list(pm._templates_dir.glob("*.txt"))
        if template_files:
            template_files[0].write_text("changed content", encoding="utf-8")
        v2 = pm.get_prompt_version()
        assert v1 == v2, "Cached version should not change without clear_cache()"

    def test_changes_after_clear_cache_and_file_edit(
        self, pm: PromptManager, template_dir: Path
    ) -> None:
        """Version should change after clear_cache() when a file is edited."""
        v1 = pm.get_prompt_version()
        # Edit a template.
        (template_dir / "concept.txt").write_text(
            "Completely new concept prompt for {topic}.", encoding="utf-8"
        )
        pm.clear_cache()
        v2 = pm.get_prompt_version()
        assert v1 != v2, "Version must change after template content changes"

    def test_changes_after_clear_cache_and_file_added(
        self, pm: PromptManager, template_dir: Path
    ) -> None:
        """Version should change after clear_cache() when a file is added."""
        v1 = pm.get_prompt_version()
        (template_dir / "new_step.txt").write_text(
            "New step template.", encoding="utf-8"
        )
        pm.clear_cache()
        v2 = pm.get_prompt_version()
        assert v1 != v2, "Version must change when a new template is added"

    def test_changes_after_clear_cache_and_file_removed(
        self, pm: PromptManager, template_dir: Path
    ) -> None:
        """Version should change after clear_cache() when a file is removed."""
        v1 = pm.get_prompt_version()
        (template_dir / "outline.txt").unlink()
        pm.clear_cache()
        v2 = pm.get_prompt_version()
        assert v1 != v2, "Version must change when a template is removed"

    def test_different_content_different_version(self, tmp_path: Path) -> None:
        """Two PromptManagers with different template content → different versions."""
        cultural = tmp_path / "cultural"
        cultural.mkdir()
        structures = tmp_path / "structures"
        structures.mkdir()

        tpl_a = tmp_path / "templates_a"
        tpl_a.mkdir()
        (tpl_a / "concept.txt").write_text("Prompt A", encoding="utf-8")

        tpl_b = tmp_path / "templates_b"
        tpl_b.mkdir()
        (tpl_b / "concept.txt").write_text("Prompt B", encoding="utf-8")

        pm_a = PromptManager(
            templates_dir=tpl_a, cultural_dir=cultural, structures_dir=structures
        )
        pm_b = PromptManager(
            templates_dir=tpl_b, cultural_dir=cultural, structures_dir=structures
        )
        assert pm_a.get_prompt_version() != pm_b.get_prompt_version()

    def test_same_content_same_version(self, tmp_path: Path) -> None:
        """Two PromptManagers with identical content → same version."""
        cultural = tmp_path / "cultural"
        cultural.mkdir()
        structures = tmp_path / "structures"
        structures.mkdir()

        for suffix in ("c", "d"):
            tpl = tmp_path / f"templates_{suffix}"
            tpl.mkdir()
            (tpl / "concept.txt").write_text("Same content", encoding="utf-8")

        pm_c = PromptManager(
            templates_dir=tmp_path / "templates_c",
            cultural_dir=cultural,
            structures_dir=structures,
        )
        pm_d = PromptManager(
            templates_dir=tmp_path / "templates_d",
            cultural_dir=cultural,
            structures_dir=structures,
        )
        assert pm_c.get_prompt_version() == pm_d.get_prompt_version()

    def test_returns_unknown_when_dir_missing(self, tmp_path: Path) -> None:
        """Missing templates directory → version is 'unknown'."""
        cultural = tmp_path / "cultural"
        cultural.mkdir()
        structures = tmp_path / "structures"
        structures.mkdir()
        missing_tpl = tmp_path / "nonexistent_templates"
        pm_missing = PromptManager(
            templates_dir=missing_tpl,
            cultural_dir=cultural,
            structures_dir=structures,
        )
        assert pm_missing.get_prompt_version() == "unknown"

    def test_returns_unknown_when_dir_empty(self, tmp_path: Path) -> None:
        """Empty templates directory (no .txt files) → version is 'unknown'."""
        cultural = tmp_path / "cultural"
        cultural.mkdir()
        structures = tmp_path / "structures"
        structures.mkdir()
        empty_tpl = tmp_path / "empty_templates"
        empty_tpl.mkdir()
        pm_empty = PromptManager(
            templates_dir=empty_tpl,
            cultural_dir=cultural,
            structures_dir=structures,
        )
        assert pm_empty.get_prompt_version() == "unknown"

    def test_clear_cache_resets_version(self, pm: PromptManager) -> None:
        """clear_cache() must reset the cached version so it can be recomputed."""
        v1 = pm.get_prompt_version()
        pm.clear_cache()
        # After clear, _prompt_version_cache should be None.
        assert pm._prompt_version_cache is None
        # Recomputation should produce the same value (files unchanged).
        v2 = pm.get_prompt_version()
        assert v1 == v2


# ── CacheManager.make_key() with prompt_version ───────────────────────────────


class TestMakeKeyWithPromptVersion:
    """Tests for CacheManager.make_key() prompt_version parameter."""

    def test_without_version_returns_valid_key(
        self, gen_config: GenerationConfig
    ) -> None:
        """Calling make_key() without prompt_version should still work (backward compat)."""
        key = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in key)

    def test_with_version_returns_valid_key(
        self, gen_config: GenerationConfig
    ) -> None:
        """make_key() with prompt_version should return a valid 64-char hex key."""
        key = CacheManager.make_key(
            "Topic", "en", gen_config, "gpt-4o", prompt_version="abc123def456"
        )
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_version_produces_same_key(
        self, gen_config: GenerationConfig
    ) -> None:
        """Same inputs + same version → same key."""
        k1 = CacheManager.make_key(
            "Topic", "en", gen_config, "gpt-4o", prompt_version="v1"
        )
        k2 = CacheManager.make_key(
            "Topic", "en", gen_config, "gpt-4o", prompt_version="v1"
        )
        assert k1 == k2

    def test_different_version_produces_different_key(
        self, gen_config: GenerationConfig
    ) -> None:
        """Same topic/config but different prompt_version → different key."""
        k1 = CacheManager.make_key(
            "Topic", "en", gen_config, "gpt-4o", prompt_version="v1"
        )
        k2 = CacheManager.make_key(
            "Topic", "en", gen_config, "gpt-4o", prompt_version="v2"
        )
        assert k1 != k2

    def test_with_vs_without_version_different_keys(
        self, gen_config: GenerationConfig
    ) -> None:
        """key(with version) != key(no version) for the same topic/config."""
        k_no_version = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        k_with_version = CacheManager.make_key(
            "Topic", "en", gen_config, "gpt-4o", prompt_version="abc123"
        )
        assert k_no_version != k_with_version

    def test_empty_string_version_equals_no_version(
        self, gen_config: GenerationConfig
    ) -> None:
        """Passing prompt_version='' should give the same key as omitting it."""
        k_default = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o")
        k_empty = CacheManager.make_key(
            "Topic", "en", gen_config, "gpt-4o", prompt_version=""
        )
        assert k_default == k_empty

    def test_existing_differentiators_still_work(
        self, gen_config: GenerationConfig
    ) -> None:
        """With prompt_version, different topics still produce different keys."""
        k1 = CacheManager.make_key(
            "Topic A", "en", gen_config, "gpt-4o", prompt_version="v1"
        )
        k2 = CacheManager.make_key(
            "Topic B", "en", gen_config, "gpt-4o", prompt_version="v1"
        )
        assert k1 != k2

    def test_different_language_with_version_still_different(
        self, gen_config: GenerationConfig
    ) -> None:
        """Language still differentiates keys even with the same prompt_version."""
        k_en = CacheManager.make_key(
            "Topic", "en", gen_config, "gpt-4o", prompt_version="v1"
        )
        k_de = CacheManager.make_key(
            "Topic", "de", gen_config, "gpt-4o", prompt_version="v1"
        )
        assert k_en != k_de


# ── Integration: prompt version flows into cache key ─────────────────────────


class TestPromptVersionIntegration:
    """Integration tests: PromptManager version → CacheManager key."""

    def test_prompt_version_included_in_cache_key(
        self, pm: PromptManager, gen_config: GenerationConfig
    ) -> None:
        """Key generated with PromptManager version should be reproducible."""
        version = pm.get_prompt_version()
        k1 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o",
                                   prompt_version=version)
        k2 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o",
                                   prompt_version=version)
        assert k1 == k2

    def test_key_changes_after_template_edit(
        self, pm: PromptManager, template_dir: Path, gen_config: GenerationConfig
    ) -> None:
        """Editing a template, clearing cache, and recomputing version → new key."""
        v1 = pm.get_prompt_version()
        k1 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o",
                                   prompt_version=v1)

        # Simulate a prompt update.
        (template_dir / "concept.txt").write_text(
            "Improved concept prompt for {topic}.", encoding="utf-8"
        )
        pm.clear_cache()

        v2 = pm.get_prompt_version()
        k2 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o",
                                   prompt_version=v2)

        assert v1 != v2, "Prompt version must change after template edit"
        assert k1 != k2, "Cache key must change after prompt version changes"

    def test_unchanged_templates_same_key(
        self, pm: PromptManager, gen_config: GenerationConfig
    ) -> None:
        """If templates are not edited, the cache key stays the same."""
        v1 = pm.get_prompt_version()
        k1 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o",
                                   prompt_version=v1)
        pm.clear_cache()
        v2 = pm.get_prompt_version()
        k2 = CacheManager.make_key("Topic", "en", gen_config, "gpt-4o",
                                   prompt_version=v2)
        assert k1 == k2, "Key must be stable when templates are not changed"
