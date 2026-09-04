"""Token counting utilities.

Provides two functions:

- ``count_tokens(text, model)`` — exact count via ``tiktoken`` for
  OpenAI-compatible models, character-based approximation for others.
- ``estimate_tokens(word_count)`` — rough estimate from a target word
  count (useful for pre-generation cost estimation).

The module also exports ``MODEL_ENCODING_MAP`` which maps known model
identifiers to their tiktoken encoding names.
"""

from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)

# ── Model → tiktoken encoding mapping ──────────────────────────────────
#
# Models that use the same encoding are grouped together.  Any model not
# in this map falls back to character-based approximation.

MODEL_ENCODING_MAP: dict[str, str] = {
    # OpenAI GPT-4o / GPT-4-turbo family → cl100k_base
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # OpenRouter model paths (still cl100k_base compatible for counting)
    "openai/gpt-4o": "cl100k_base",
    "openai/gpt-4-turbo": "cl100k_base",
    "openai/gpt-4o-mini": "cl100k_base",
    # Anthropic models via OpenRouter (cl100k_base is a reasonable proxy)
    "anthropic/claude-3.5-sonnet": "cl100k_base",
    "anthropic/claude-3-opus": "cl100k_base",
    # Anthropic native model identifiers
    "claude-3-5-sonnet-20241022": "cl100k_base",
    "claude-3-opus-20240229": "cl100k_base",
}

# Average characters per token for non-tiktoken models.
# English averages ~4 chars/token; multilingual text is closer to 3–3.5.
_CHARS_PER_TOKEN_APPROX: float = 3.5

# Average tokens per word (across languages, conservative estimate).
_TOKENS_PER_WORD_APPROX: float = 1.35

# Cache for tiktoken encodings (keyed by encoding name).
_encoding_cache: dict[str, tiktoken.Encoding] = {}


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in *text* for *model*.

    For models in ``MODEL_ENCODING_MAP``, uses ``tiktoken`` for an exact
    count.  For all other models, falls back to a character-based
    approximation: ``len(text) / 3.5``.

    Args:
        text: The text to tokenise.
        model: Model identifier (e.g. ``"gpt-4o"``,
            ``"anthropic/claude-3.5-sonnet"``).

    Returns:
        Token count (always ≥ 0).
    """
    if not text:
        return 0

    encoding_name = MODEL_ENCODING_MAP.get(model)

    if encoding_name is not None:
        return _count_tiktoken(text, encoding_name, model)

    return _count_approximate(text)


def estimate_tokens(word_count: int) -> int:
    """Estimate token count from a target word count.

    Useful for pre-generation cost estimation when the actual text does
    not yet exist.  Uses the approximation
    ``tokens ≈ word_count × 1.35``.

    Args:
        word_count: Target number of words.

    Returns:
        Estimated token count (always ≥ 0).
    """
    if word_count <= 0:
        return 0
    return max(1, int(word_count * _TOKENS_PER_WORD_APPROX))


# ── internal helpers ────────────────────────────────────────────────────


def _count_tiktoken(text: str, encoding_name: str, model: str) -> int:
    """Count tokens using tiktoken.

    Args:
        text: Text to tokenise.
        encoding_name: Tiktoken encoding name (e.g. ``"cl100k_base"``).
        model: Model identifier (for logging only).

    Returns:
        Exact token count.
    """
    encoding = _get_encoding(encoding_name)
    if encoding is None:
        logger.warning(
            "tiktoken encoding '%s' not available for model '%s', "
            "falling back to approximation",
            encoding_name,
            model,
        )
        return _count_approximate(text)

    tokens = encoding.encode(text)
    return len(tokens)


def _count_approximate(text: str) -> int:
    """Approximate token count from character length.

    Args:
        text: Text to estimate.

    Returns:
        Approximate token count.
    """
    return max(1, int(len(text) / _CHARS_PER_TOKEN_APPROX))


def _get_encoding(encoding_name: str) -> tiktoken.Encoding | None:
    """Get a tiktoken encoding, using a module-level cache.

    Args:
        encoding_name: Tiktoken encoding name.

    Returns:
        The encoding, or ``None`` if not available.
    """
    if encoding_name in _encoding_cache:
        return _encoding_cache[encoding_name]

    try:
        enc = tiktoken.get_encoding(encoding_name)
        _encoding_cache[encoding_name] = enc
        logger.debug("tiktoken encoding '%s' loaded and cached", encoding_name)
        return enc
    except Exception as exc:
        logger.warning(
            "Failed to load tiktoken encoding '%s': %s", encoding_name, exc
        )
        return None
