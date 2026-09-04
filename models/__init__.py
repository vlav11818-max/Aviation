"""Data models for the Aviation Content Factory.

Only three modules survive after the generic-story cleanup:

* :mod:`models.aviation_bible` — the aviation-flavoured StoryBible
  and the RAG-extraction schema.
* :mod:`models.story_bible` — the small ``Character`` / ``Setting`` /
  ``StoryBible`` primitives the aviation bible reuses.
* :mod:`models.config` — API-configuration and (legacy) generation-
  config models used by the LiteLLM-backed ``core.api_client``.
"""

from models.aviation_bible import (
    Aircraft,
    AviationStoryBible,
    CausalLink,
    CrewMember,
    ExtractedFacts,
    Mode,
    NarrativeStructure,
    ROTATION_ORDER,
    Route,
    TimelineEvent,
)
from models.config import (
    APIConfig,
    APIFormat,
    APIProvider,
    FallbackPoolEntry,
    LANGUAGES,
    PROVIDER_CONFIG,
    SUPPORTED_LANGUAGE_CODES,
)
from models.story_bible import Character, Setting, StoryBible

__all__ = [
    # config
    "APIConfig",
    "APIFormat",
    "APIProvider",
    "FallbackPoolEntry",
    "LANGUAGES",
    "PROVIDER_CONFIG",
    "SUPPORTED_LANGUAGE_CODES",
    # story primitives
    "Character",
    "Setting",
    "StoryBible",
    # aviation
    "Aircraft",
    "AviationStoryBible",
    "CausalLink",
    "CrewMember",
    "ExtractedFacts",
    "Mode",
    "NarrativeStructure",
    "ROTATION_ORDER",
    "Route",
    "TimelineEvent",
]
