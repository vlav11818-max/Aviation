"""Prompt template manager for AI Story Generator Pro.

Loads prompt templates from ``resources/prompts/templates/``, caches
them in memory, and renders them with variable substitution.  Cultural
instructions are loaded per-language and injected automatically.
Structure templates (JSON) are loaded from
``resources/prompts/structures/``.  Voiceover rules are loaded from
``resources/prompts/voiceover_rules.txt`` and injected into templates
that reference ``{voiceover_rules}``.  Retention techniques are loaded
from ``resources/prompts/retention_techniques.txt`` and injected into
templates that reference ``{retention_techniques}``.

Typical usage::

    pm = PromptManager(
        templates_dir=Path("resources/prompts/templates"),
        cultural_dir=Path("resources/prompts/cultural"),
        structures_dir=Path("resources/prompts/structures"),
        voiceover_rules_path=Path("resources/prompts/voiceover_rules.txt"),
        retention_techniques_path=Path("resources/prompts/retention_techniques.txt"),
    )
    prompt = pm.render("concept", language="de", topic="Ancient Temple", ...)

Convenience form using a single root directory::

    pm = PromptManager(resources_dir=Path("resources"))

Prompt versioning (Fix #9)::

    version = pm.get_prompt_version()
    key = CacheManager.make_key(topic, language, gen_config, model,
                                prompt_version=version)

Pass the version to ``CacheManager.make_key()`` so that the cache key
changes automatically when any template file is modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.exceptions import PromptTemplateError
from utils.file_handler import read_file

logger = logging.getLogger(__name__)


class PromptManager:
    """Loads, caches, and renders prompt templates.

    Templates are plain-text files with ``{variable}`` placeholders.
    Cultural instructions are language-specific text files injected
    automatically when ``language`` is provided.  Voiceover rules are
    loaded from a single shared file and injected into any template
    that contains a ``{voiceover_rules}`` placeholder.  Retention
    techniques are loaded from a single shared file and injected into
    any template that contains a ``{retention_techniques}`` placeholder.
    Structure templates are JSON files describing narrative structures.

    All files are cached on first load and served from memory thereafter.

    The constructor accepts either explicit directory paths **or**
    a single ``resources_dir`` convenience path (which must contain
    ``prompts/templates/``, ``prompts/cultural/``,
    ``prompts/structures/`` subdirectories, and
    ``prompts/voiceover_rules.txt`` and
    ``prompts/retention_techniques.txt``).

    Args:
        templates_dir: Path to the directory containing ``.txt`` templates.
        cultural_dir: Path to the directory containing per-language
            cultural instruction files (``{lang}.txt``).
        structures_dir: Path to the directory containing structure JSON
            files (``{structure}.json``).
        voiceover_rules_path: Path to the voiceover rules text file.
        retention_techniques_path: Path to the retention techniques
            text file.
        resources_dir: Convenience alternative — root resources directory
            from which subdirectories are derived automatically.
    """

    def __init__(
        self,
        templates_dir: Path | None = None,
        cultural_dir: Path | None = None,
        structures_dir: Path | None = None,
        voiceover_rules_path: Path | None = None,
        retention_techniques_path: Path | None = None,
        *,
        resources_dir: Path | None = None,
    ) -> None:
        if resources_dir is not None:
            # Convenience: derive subdirectories from a single root.
            base = Path(resources_dir)
            self._templates_dir = base / "prompts" / "templates"
            self._cultural_dir = base / "prompts" / "cultural"
            self._structures_dir = base / "prompts" / "structures"
            self._voiceover_rules_path = base / "prompts" / "voiceover_rules.txt"
            self._retention_techniques_path = base / "prompts" / "retention_techniques.txt"
        elif (
            templates_dir is not None
            and cultural_dir is not None
            and structures_dir is not None
        ):
            self._templates_dir = Path(templates_dir)
            self._cultural_dir = Path(cultural_dir)
            self._structures_dir = Path(structures_dir)
            self._voiceover_rules_path = (
                Path(voiceover_rules_path)
                if voiceover_rules_path is not None
                else self._templates_dir.parent / "voiceover_rules.txt"
            )
            self._retention_techniques_path = (
                Path(retention_techniques_path)
                if retention_techniques_path is not None
                else self._templates_dir.parent / "retention_techniques.txt"
            )
        else:
            raise PromptTemplateError(
                "PromptManager requires either all three directory paths "
                "(templates_dir, cultural_dir, structures_dir) or a single "
                "resources_dir.",
                template_name="__init__",
            )

        # In-memory caches keyed by filename (no extension).
        self._template_cache: dict[str, str] = {}
        self._cultural_cache: dict[str, str] = {}
        self._structure_cache: dict[str, dict[str, Any]] = {}
        self._voiceover_rules_cache: str | None = None
        self._retention_techniques_cache: str | None = None

        # Cached prompt version hash (computed lazily on first call,
        # then invalidated whenever clear_cache() is called).
        self._prompt_version_cache: str | None = None

        logger.info(
            "PromptManager initialised: templates=%s, cultural=%s, "
            "structures=%s, voiceover_rules=%s, retention_techniques=%s",
            self._templates_dir,
            self._cultural_dir,
            self._structures_dir,
            self._voiceover_rules_path,
            self._retention_techniques_path,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        template_name: str,
        language: str = "en",
        **variables: Any,
    ) -> str:
        """Load a template, inject cultural instructions, and fill variables.

        The template is loaded from ``{templates_dir}/{template_name}.txt``.
        If the template contains a ``{cultural_instructions}`` placeholder
        and one is not explicitly provided in *variables*, the cultural
        instructions for *language* are loaded and injected automatically.
        Similarly, if the template contains ``{voiceover_rules}`` and
        one is not provided, the voiceover rules file is loaded and
        injected.  The same auto-injection applies to
        ``{retention_techniques}``.

        Args:
            template_name: Base name of the template file (without
                extension), e.g. ``"concept"``, ``"outline"``.
            language: Two-letter language code for cultural instruction
                injection.
            **variables: Key-value pairs to substitute into the template.
                Each key corresponds to a ``{key}`` placeholder.

        Returns:
            The fully rendered prompt string.

        Raises:
            PromptTemplateError: If the template file is missing, cannot
                be read, or a required variable has no value.
        """
        template_text = self._load_template(template_name)

        # Inject cultural instructions automatically if not provided.
        if (
            "{cultural_instructions}" in template_text
            and "cultural_instructions" not in variables
        ):
            variables["cultural_instructions"] = self.get_cultural_instructions(
                language
            )

        # Inject voiceover rules automatically if not provided.
        if (
            "{voiceover_rules}" in template_text
            and "voiceover_rules" not in variables
        ):
            variables["voiceover_rules"] = self.get_voiceover_rules()

        # Inject retention techniques automatically if not provided.
        if (
            "{retention_techniques}" in template_text
            and "retention_techniques" not in variables
        ):
            variables["retention_techniques"] = self.get_retention_techniques()

        # Inject language into variables if not already present.
        if "language" not in variables:
            variables["language"] = language

        rendered = self._substitute(template_text, template_name, variables)

        logger.debug(
            "Rendered template '%s' for language '%s' (%d chars)",
            template_name,
            language,
            len(rendered),
        )
        return rendered

    def get_cultural_instructions(self, language: str) -> str:
        """Load cultural instruction text for a given language.

        Args:
            language: Two-letter language code (e.g., ``"de"``).

        Returns:
            The cultural instructions as a string, or an empty string
            if no cultural file exists for the given language.
        """
        if language in self._cultural_cache:
            logger.debug("Cultural instructions cache hit: %s", language)
            return self._cultural_cache[language]

        file_path = self._cultural_dir / f"{language}.txt"
        if not file_path.exists():
            logger.warning(
                "Cultural instructions file not found for language '%s': %s",
                language,
                file_path,
            )
            self._cultural_cache[language] = ""
            return ""

        try:
            content = read_file(file_path)
        except OSError as exc:
            logger.error(
                "Failed to read cultural instructions for '%s': %s",
                language,
                exc,
            )
            self._cultural_cache[language] = ""
            return ""

        self._cultural_cache[language] = content
        logger.info(
            "Loaded cultural instructions for '%s' (%d chars)",
            language,
            len(content),
        )
        return content

    def get_voiceover_rules(self) -> str:
        """Load voiceover rules from the shared resource file.

        Returns:
            The voiceover rules text, or a minimal fallback string
            if the file cannot be loaded.
        """
        if self._voiceover_rules_cache is not None:
            return self._voiceover_rules_cache

        if not self._voiceover_rules_path.exists():
            logger.warning(
                "Voiceover rules file not found: %s — using minimal fallback",
                self._voiceover_rules_path,
            )
            fallback = (
                "Write in simple, clear sentences optimized for voiceover narration. "
                "Use punctuation for natural pauses. Avoid headers, markers, and "
                "meta-commentary. Keep sentences between 8 and 25 words."
            )
            self._voiceover_rules_cache = fallback
            return fallback

        try:
            content = read_file(self._voiceover_rules_path)
        except OSError as exc:
            logger.error(
                "Failed to read voiceover rules: %s",
                exc,
            )
            self._voiceover_rules_cache = ""
            return ""

        self._voiceover_rules_cache = content
        logger.info(
            "Loaded voiceover rules (%d chars)",
            len(content),
        )
        return content

    def get_retention_techniques(self) -> str:
        """Load retention techniques from the shared resource file.

        Returns:
            The retention techniques text, or a minimal fallback string
            if the file cannot be loaded.
        """
        if self._retention_techniques_cache is not None:
            return self._retention_techniques_cache

        if not self._retention_techniques_path.exists():
            logger.warning(
                "Retention techniques file not found: %s — using minimal fallback",
                self._retention_techniques_path,
            )
            fallback = (
                "Use audience retention techniques throughout the story: "
                "open loops (hint at revelations before delivering them), "
                "delayed revelations (build tension before content), "
                "somatic responses (specific physical reactions after shocks), "
                "sensory anchors (recurring sensory details), "
                "echo phrases (callbacks that gain new meaning), "
                "and breather scenes (quiet moments between intensity peaks)."
            )
            self._retention_techniques_cache = fallback
            return fallback

        try:
            content = read_file(self._retention_techniques_path)
        except OSError as exc:
            logger.error(
                "Failed to read retention techniques: %s",
                exc,
            )
            self._retention_techniques_cache = ""
            return ""

        self._retention_techniques_cache = content
        logger.info(
            "Loaded retention techniques (%d chars)",
            len(content),
        )
        return content

    def get_structure_template(self, structure: str) -> dict[str, Any]:
        """Load a story structure template (JSON).

        Args:
            structure: Structure name (e.g., ``"three_act"``).

        Returns:
            Parsed JSON dictionary describing the structure.

        Raises:
            PromptTemplateError: If the structure file is missing,
                cannot be read, or is not valid JSON.
        """
        if structure in self._structure_cache:
            logger.debug("Structure cache hit: %s", structure)
            return self._structure_cache[structure]

        file_path = self._structures_dir / f"{structure}.json"
        if not file_path.exists():
            raise PromptTemplateError(
                f"Structure template file not found: {file_path}",
                template_name=structure,
            )

        try:
            raw = read_file(file_path)
        except OSError as exc:
            raise PromptTemplateError(
                f"Failed to read structure template '{structure}': {exc}",
                template_name=structure,
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PromptTemplateError(
                f"Structure template '{structure}' is not valid JSON: {exc}",
                template_name=structure,
            ) from exc

        if not isinstance(data, dict):
            raise PromptTemplateError(
                f"Structure template '{structure}' must be a JSON object, "
                f"got {type(data).__name__}",
                template_name=structure,
            )

        self._structure_cache[structure] = data
        logger.info("Loaded structure template '%s'", structure)
        return data

    def get_cleanup_patterns(self, language: str) -> list[str]:
        """Load language-specific cleanup patterns from cultural files.

        Cultural files may contain a ``CLEANUP_PATTERNS:`` section at the
        end, listing regex patterns for chapter/section markers in that
        language.  This method extracts those patterns.

        Args:
            language: Two-letter language code.

        Returns:
            List of regex pattern strings, or empty list if none found.
        """
        cultural_text = self.get_cultural_instructions(language)
        if not cultural_text:
            return []

        # Look for CLEANUP_PATTERNS section.
        marker = "CLEANUP_PATTERNS:"
        idx = cultural_text.find(marker)
        if idx == -1:
            return []

        patterns_section = cultural_text[idx + len(marker):]
        patterns: list[str] = []
        for line in patterns_section.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Stop if we hit another section header (all caps + colon).
            if line.isupper() and line.endswith(":"):
                break
            patterns.append(line)

        logger.debug(
            "Loaded %d cleanup patterns for language '%s'",
            len(patterns),
            language,
        )
        return patterns

    def get_prompt_version(self) -> str:
        """Return a short hash fingerprint of all template and resource files.

        The hash is computed over the sorted, concatenated contents of
        every ``.txt`` file in the templates directory, plus the
        voiceover rules and retention techniques files.  It changes
        whenever any of these files is added, removed, or modified.

        The result is cached on first call and cleared by
        ``clear_cache()``.  This allows callers to include the version
        in the cache key so that stale cached results are automatically
        invalidated when prompts are updated::

            version = pm.get_prompt_version()
            key = CacheManager.make_key(
                topic, language, gen_config, model,
                prompt_version=version,
            )

        Returns:
            A 12-character hexadecimal prefix of the SHA-256 hash of all
            template and resource contents, or ``"unknown"`` if the
            templates directory does not exist or is empty.
        """
        if self._prompt_version_cache is not None:
            return self._prompt_version_cache

        if not self._templates_dir.exists():
            logger.warning(
                "PromptManager.get_prompt_version: templates directory "
                "does not exist: %s",
                self._templates_dir,
            )
            self._prompt_version_cache = "unknown"
            return self._prompt_version_cache

        # Collect all .txt files, sorted for determinism.
        template_files = sorted(self._templates_dir.glob("*.txt"))

        if not template_files:
            logger.warning(
                "PromptManager.get_prompt_version: no .txt files found in %s",
                self._templates_dir,
            )
            self._prompt_version_cache = "unknown"
            return self._prompt_version_cache

        hasher = hashlib.sha256()
        for file_path in template_files:
            # Include the filename so renames also change the hash.
            hasher.update(file_path.name.encode("utf-8"))
            try:
                content = read_file(file_path)
                hasher.update(content.encode("utf-8"))
            except OSError as exc:
                logger.warning(
                    "PromptManager.get_prompt_version: could not read %s: %s",
                    file_path,
                    exc,
                )
                # Include the error marker in the hash so failures are
                # not silently ignored.
                hasher.update(b"__READ_ERROR__")

        # Include shared resource files in the version hash so that
        # changes to voiceover rules or retention techniques also
        # invalidate cached results.
        for resource_path in [
            self._voiceover_rules_path,
            self._retention_techniques_path,
        ]:
            if resource_path.exists():
                hasher.update(resource_path.name.encode("utf-8"))
                try:
                    content = read_file(resource_path)
                    hasher.update(content.encode("utf-8"))
                except OSError as exc:
                    logger.warning(
                        "PromptManager.get_prompt_version: could not read "
                        "resource %s: %s",
                        resource_path,
                        exc,
                    )
                    hasher.update(b"__READ_ERROR__")

        # Use a 12-char prefix — enough to distinguish versions without
        # bloating the cache key.
        version = hasher.hexdigest()[:12]
        self._prompt_version_cache = version

        logger.debug(
            "PromptManager: computed prompt version=%s from %d template(s) "
            "+ resource files",
            version,
            len(template_files),
        )
        return version

    def clear_cache(self) -> None:
        """Clear all in-memory caches.

        Useful for testing or when resource files have been modified
        at runtime.  Also invalidates the cached prompt version so
        ``get_prompt_version()`` recomputes it on the next call.
        """
        self._template_cache.clear()
        self._cultural_cache.clear()
        self._structure_cache.clear()
        self._voiceover_rules_cache = None
        self._retention_techniques_cache = None
        self._prompt_version_cache = None
        logger.info("PromptManager caches cleared")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_template(self, template_name: str) -> str:
        """Load a template file from disk (or cache).

        Args:
            template_name: Base name (no extension).

        Returns:
            Raw template text.

        Raises:
            PromptTemplateError: If file not found or unreadable.
        """
        if template_name in self._template_cache:
            logger.debug("Template cache hit: %s", template_name)
            return self._template_cache[template_name]

        file_path = self._templates_dir / f"{template_name}.txt"
        if not file_path.exists():
            raise PromptTemplateError(
                f"Template file not found: {file_path}",
                template_name=template_name,
            )

        try:
            content = read_file(file_path)
        except OSError as exc:
            raise PromptTemplateError(
                f"Failed to read template '{template_name}': {exc}",
                template_name=template_name,
            ) from exc

        self._template_cache[template_name] = content
        logger.info(
            "Loaded template '%s' (%d chars)",
            template_name,
            len(content),
        )
        return content

    def _substitute(
        self,
        template: str,
        template_name: str,
        variables: dict[str, Any],
    ) -> str:
        """Substitute ``{key}`` placeholders in a template.

        Uses a safe formatter that leaves unmatched placeholders as-is
        rather than raising ``KeyError``.  This allows partial rendering
        when not all variables are known yet.

        Args:
            template: The raw template text.
            template_name: Name (for error messages).
            variables: Key-value mapping.

        Returns:
            Template with substituted values.
        """
        safe_vars = _SafeDict(variables)
        try:
            return template.format_map(safe_vars)
        except (ValueError, IndexError) as exc:
            raise PromptTemplateError(
                f"Variable substitution failed for template "
                f"'{template_name}': {exc}",
                template_name=template_name,
            ) from exc


class _SafeDict(dict):
    """Dict subclass that returns ``{key}`` for missing keys.

    Used with ``str.format_map()`` so that placeholders without
    corresponding values are left intact rather than raising
    ``KeyError``.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
