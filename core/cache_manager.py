"""Result caching for AI Story Generator Pro.

``CacheManager`` provides a content-addressable cache for completed
generation results.  The cache key is a SHA-256 hash of the topic,
language, style parameters, model identifier, and (optionally) the
prompt template version so that stale results are automatically
invalidated when prompt files change.

Cache metadata is persisted to ``data/cache/generation_cache.json``
and is loaded on construction so that cache state survives restarts.

Thread safety is achieved via a ``threading.Lock`` that protects all
reads/writes to both the in-memory index and the on-disk JSON file.

Typical usage::

    cm = CacheManager(settings)
    prompt_version = prompt_manager.get_prompt_version()   # Fix #9
    key = cm.make_key("Ancient Temple", "en", gen_config, "gpt-4o",
                      prompt_version=prompt_version)
    if cm.has(key):
        output_dir = cm.get(key)
    else:
        # ... run pipeline ...
        cm.put(key, Path("output/en/ancient_temple"))
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.exceptions import ConfigError
from core.settings import Settings
from models.config import GenerationConfig
from utils.file_handler import ensure_dir, read_file, write_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheStats:
    """Statistics about the current cache state.

    Attributes:
        total_entries: Number of cached results.
        oldest_timestamp: ISO-8601 timestamp of the oldest entry,
            or empty string if cache is empty.
        newest_timestamp: ISO-8601 timestamp of the newest entry,
            or empty string if cache is empty.
    """

    total_entries: int = 0
    oldest_timestamp: str = ""
    newest_timestamp: str = ""


@dataclass
class _CacheEntry:
    """Internal representation of a single cache entry.

    Attributes:
        output_dir: Path to the cached output directory.
        timestamp: ISO-8601 string of when the entry was created.
        config_hash: SHA-256 of the config parameters at cache time.
    """

    output_dir: str
    timestamp: str
    config_hash: str

    def to_dict(self) -> dict[str, str]:
        """Serialise to a plain dictionary.

        Returns:
            Dictionary with ``output_dir``, ``timestamp``, and
            ``config_hash`` keys.
        """
        return {
            "output_dir": self.output_dir,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _CacheEntry:
        """Deserialise from a dictionary.

        Args:
            data: Dictionary with ``output_dir``, ``timestamp``, and
                ``config_hash`` keys.

        Returns:
            A ``_CacheEntry`` instance.
        """
        return cls(
            output_dir=str(data.get("output_dir", "")),
            timestamp=str(data.get("timestamp", "")),
            config_hash=str(data.get("config_hash", "")),
        )


class CacheManager:
    """Content-addressable cache for completed generation results.

    The cache key is::

        SHA-256(topic | lang | style_params_json | model [| prompt_version])

    The optional ``prompt_version`` component (Fix #9) is a hash of the
    current prompt template files, obtained via
    ``PromptManager.get_prompt_version()``.  When prompt files change
    the version changes, so old cache entries for the same topic/config
    are silently bypassed rather than returning stale content.

    Cached data is a mapping from key to ``{output_dir, timestamp,
    config_hash}`` persisted in ``data/cache/generation_cache.json``.

    All public methods are thread-safe.

    Args:
        settings: Application settings (provides ``paths.cache_dir`` and
            ``cache.enabled``).
    """

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.cache.enabled
        self._cache_dir = Path(settings.paths.cache_dir)
        self._cache_file = self._cache_dir / "generation_cache.json"
        self._lock = threading.Lock()
        self._index: dict[str, _CacheEntry] = {}

        if self._enabled:
            ensure_dir(self._cache_dir)
            self._load_index()
        else:
            logger.info("CacheManager: caching is disabled in settings")

    # ── Key generation ────────────────────────────────────────────────────────

    @staticmethod
    def make_key(
        topic: str,
        language: str,
        gen_config: GenerationConfig,
        model: str,
        prompt_version: str = "",
    ) -> str:
        """Create a deterministic cache key.

        The key is a SHA-256 hex digest of the concatenation of the
        topic, language, a sorted JSON representation of the generation
        config, the model identifier, and (optionally) a prompt version
        string.

        Including ``prompt_version`` (obtained from
        ``PromptManager.get_prompt_version()``) ensures that the cache
        key changes whenever prompt template files are modified, so
        stale cached results are automatically bypassed without any
        manual cache invalidation.

        Args:
            topic: Story topic / theme.
            language: Two-letter language code.
            gen_config: Creative parameter snapshot.
            model: Model identifier string.
            prompt_version: Short hash fingerprint of the current prompt
                templates (from ``PromptManager.get_prompt_version()``).
                Pass an empty string (default) to omit prompt versioning
                and reproduce the legacy behaviour.

        Returns:
            A 64-character hexadecimal SHA-256 digest string.
        """
        config_json = gen_config.model_dump_json(exclude_none=True)
        # prompt_version is appended only when non-empty so that keys
        # generated without it remain compatible with existing code that
        # doesn't pass the argument.
        if prompt_version:
            raw = f"{topic}|{language}|{config_json}|{model}|{prompt_version}"
        else:
            raw = f"{topic}|{language}|{config_json}|{model}"

        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        logger.debug(
            "CacheManager: key generated for topic='%s', lang=%s, "
            "model=%s, prompt_version='%s': %s",
            topic,
            language,
            model,
            prompt_version or "(none)",
            digest[:12],
        )
        return digest

    # ── Public API ────────────────────────────────────────────────────────────

    def has(self, key: str) -> bool:
        """Check whether the cache contains a result for *key*.

        Args:
            key: The cache key (from ``make_key``).

        Returns:
            ``True`` if the key is present **and** the referenced
            output directory still exists on disk.
        """
        if not self._enabled:
            return False

        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return False

            # Verify output directory still exists.
            if not Path(entry.output_dir).is_dir():
                logger.warning(
                    "CacheManager: entry %s references missing dir %s — removing",
                    key[:12],
                    entry.output_dir,
                )
                del self._index[key]
                self._persist_index()
                return False

            return True

    def get(self, key: str) -> Path | None:
        """Return the output directory path for a cached result.

        Args:
            key: The cache key (from ``make_key``).

        Returns:
            ``Path`` to the output directory, or ``None`` if the key
            is not in the cache or caching is disabled.
        """
        if not self._enabled:
            return None

        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return None

            output_path = Path(entry.output_dir)
            if not output_path.is_dir():
                logger.warning(
                    "CacheManager: get(%s) — dir %s missing, removing entry",
                    key[:12],
                    entry.output_dir,
                )
                del self._index[key]
                self._persist_index()
                return None

            logger.debug(
                "CacheManager: cache hit for %s → %s",
                key[:12],
                entry.output_dir,
            )
            return output_path

    def put(self, key: str, output_dir: str | Path) -> None:
        """Store a completed result in the cache.

        Args:
            key: The cache key (from ``make_key``).
            output_dir: Path to the output directory to cache.
        """
        if not self._enabled:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        entry = _CacheEntry(
            output_dir=str(output_dir),
            timestamp=timestamp,
            config_hash=key,
        )

        with self._lock:
            self._index[key] = entry
            self._persist_index()

        logger.info(
            "CacheManager: stored key %s → %s",
            key[:12],
            output_dir,
        )

    def invalidate(self, key: str) -> bool:
        """Remove a single entry from the cache.

        Args:
            key: The cache key to remove.

        Returns:
            ``True`` if the key was found and removed, ``False``
            otherwise.
        """
        if not self._enabled:
            return False

        with self._lock:
            if key not in self._index:
                return False
            del self._index[key]
            self._persist_index()

        logger.info("CacheManager: invalidated key %s", key[:12])
        return True

    def invalidate_all(self) -> int:
        """Remove all entries from the cache.

        Returns:
            The number of entries that were removed.
        """
        if not self._enabled:
            return 0

        with self._lock:
            count = len(self._index)
            self._index.clear()
            self._persist_index()

        logger.info("CacheManager: invalidated all %d entries", count)
        return count

    def get_stats(self) -> CacheStats:
        """Return statistics about the current cache state.

        Returns:
            A ``CacheStats`` instance with entry count and
            oldest/newest timestamps.
        """
        with self._lock:
            if not self._index:
                return CacheStats()

            timestamps = [e.timestamp for e in self._index.values()]
            sorted_ts = sorted(timestamps)
            return CacheStats(
                total_entries=len(self._index),
                oldest_timestamp=sorted_ts[0],
                newest_timestamp=sorted_ts[-1],
            )

    @property
    def enabled(self) -> bool:
        """Whether caching is currently enabled."""
        return self._enabled

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_index(self) -> None:
        """Load the cache index from disk.

        Must be called with ``self._lock`` NOT held (used only in
        ``__init__``).
        """
        if not self._cache_file.exists():
            logger.debug("CacheManager: no cache file found, starting empty")
            return

        try:
            raw = read_file(self._cache_file)
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning(
                    "CacheManager: cache file is not a JSON object — ignoring"
                )
                return

            for key, entry_data in data.items():
                if isinstance(entry_data, dict):
                    self._index[key] = _CacheEntry.from_dict(entry_data)

            logger.info(
                "CacheManager: loaded %d entries from %s",
                len(self._index),
                self._cache_file,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "CacheManager: failed to load cache file %s: %s — starting empty",
                self._cache_file,
                exc,
            )

    def _persist_index(self) -> None:
        """Write the cache index to disk.

        Must be called with ``self._lock`` held.
        """
        data = {key: entry.to_dict() for key, entry in self._index.items()}
        serialised = json.dumps(data, indent=2, ensure_ascii=False)

        try:
            write_file(self._cache_file, serialised)
            logger.debug(
                "CacheManager: persisted %d entries to %s",
                len(data),
                self._cache_file,
            )
        except OSError as exc:
            logger.error(
                "CacheManager: failed to persist cache index: %s", exc
            )
