"""Unit tests for utils.token_counter.

Tests exact token counting via tiktoken for OpenAI models, approximate
counting for non-OpenAI models, the estimate_tokens word-count helper,
edge cases (empty string, very long text), and the model encoding map.
"""

from __future__ import annotations

import pytest

from utils.token_counter import (
    MODEL_ENCODING_MAP,
    count_tokens,
    estimate_tokens,
)


# ═══════════════════════════════════════════════════════════════════════
# count_tokens — tiktoken path
# ═══════════════════════════════════════════════════════════════════════


class TestCountTokensTiktoken:
    """Tests for count_tokens using tiktoken (OpenAI models)."""

    def test_known_short_string_gpt4o(self) -> None:
        """A short English sentence should return a positive int for gpt-4o."""
        text = "Hello, how are you today?"
        result = count_tokens(text, "gpt-4o")
        assert isinstance(result, int)
        assert result > 0
        # "Hello, how are you today?" is typically 7 tokens with cl100k_base.
        assert 4 <= result <= 12

    def test_known_longer_string_gpt4o(self) -> None:
        """A longer text should produce more tokens than a short one."""
        short = "Hello"
        long_text = "The quick brown fox jumps over the lazy dog. " * 10
        short_count = count_tokens(short, "gpt-4o")
        long_count = count_tokens(long_text, "gpt-4o")
        assert long_count > short_count

    def test_openrouter_model_path(self) -> None:
        """Models prefixed with 'openai/' should also use tiktoken."""
        text = "Testing OpenRouter model path"
        result = count_tokens(text, "openai/gpt-4o")
        assert isinstance(result, int)
        assert result > 0

    def test_anthropic_openrouter_model(self) -> None:
        """Anthropic models via OpenRouter should use the cl100k_base proxy."""
        text = "This tests the Anthropic model via OpenRouter"
        result = count_tokens(text, "anthropic/claude-3.5-sonnet")
        assert isinstance(result, int)
        assert result > 0

    def test_anthropic_native_model(self) -> None:
        """Anthropic native model identifiers should use tiktoken."""
        text = "Testing native Anthropic model"
        result = count_tokens(text, "claude-3-5-sonnet-20241022")
        assert isinstance(result, int)
        assert result > 0

    def test_multiple_calls_consistent(self) -> None:
        """Repeated calls with the same input should return the same count."""
        text = "Consistency check for token counting"
        model = "gpt-4o"
        counts = [count_tokens(text, model) for _ in range(5)]
        assert len(set(counts)) == 1

    def test_unicode_text(self) -> None:
        """Unicode / multilingual text should be handled correctly."""
        text = "Привет, как дела? Всё хорошо."
        result = count_tokens(text, "gpt-4o")
        assert isinstance(result, int)
        assert result > 0


# ═══════════════════════════════════════════════════════════════════════
# count_tokens — approximation path
# ═══════════════════════════════════════════════════════════════════════


class TestCountTokensApprox:
    """Tests for count_tokens using character approximation (non-OpenAI)."""

    def test_unknown_model_uses_approximation(self) -> None:
        """A model not in MODEL_ENCODING_MAP should use char approximation."""
        text = "A" * 350  # 350 chars / 3.5 = 100 tokens
        result = count_tokens(text, "gemini-1.5-pro")
        assert isinstance(result, int)
        assert result == 100

    def test_deepseek_model(self) -> None:
        """DeepSeek model should fall back to approximation."""
        text = "Hello world"
        result = count_tokens(text, "deepseek-chat")
        assert isinstance(result, int)
        assert result > 0

    def test_qwen_model(self) -> None:
        """Qwen model should fall back to approximation."""
        text = "Testing qwen model token counting"
        result = count_tokens(text, "qwen-max")
        assert isinstance(result, int)
        assert result > 0

    def test_longer_text_produces_more_tokens(self) -> None:
        """Longer text should produce proportionally more tokens."""
        short = "Hello"
        long_text = "Hello " * 200  # ~1200 chars
        short_count = count_tokens(short, "unknown-model")
        long_count = count_tokens(long_text, "unknown-model")
        assert long_count > short_count


# ═══════════════════════════════════════════════════════════════════════
# count_tokens — edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestCountTokensEdgeCases:
    """Edge case tests for count_tokens."""

    def test_empty_string_returns_zero(self) -> None:
        """An empty string should return 0 tokens."""
        assert count_tokens("", "gpt-4o") == 0
        assert count_tokens("", "unknown-model") == 0

    def test_whitespace_only(self) -> None:
        """Whitespace-only text should still return a positive count."""
        result = count_tokens("   \n\t  ", "gpt-4o")
        assert isinstance(result, int)
        assert result >= 0

    def test_very_long_text(self) -> None:
        """Very long text should not crash and should return a large count."""
        text = "word " * 50_000  # ~250,000 chars
        result = count_tokens(text, "gpt-4o")
        assert isinstance(result, int)
        assert result > 10_000

    def test_single_character(self) -> None:
        """A single character should return at least 1 token."""
        result = count_tokens("a", "gpt-4o")
        assert result >= 1

    def test_special_characters(self) -> None:
        """Special characters and emojis should be tokenised."""
        result = count_tokens("ðŸŽ‰ðŸš€ðŸ’¡", "gpt-4o")
        assert isinstance(result, int)
        assert result > 0


# ═══════════════════════════════════════════════════════════════════════
# estimate_tokens
# ═══════════════════════════════════════════════════════════════════════


class TestEstimateTokens:
    """Tests for estimate_tokens (word count → token estimate)."""

    def test_typical_value(self) -> None:
        """3000 words should produce roughly 4050 tokens (3000 × 1.35)."""
        result = estimate_tokens(3000)
        assert isinstance(result, int)
        assert result == 4050

    def test_small_value(self) -> None:
        """500 words should produce a reasonable estimate."""
        result = estimate_tokens(500)
        assert result == 675  # 500 × 1.35

    def test_zero_words(self) -> None:
        """Zero word count should return 0."""
        assert estimate_tokens(0) == 0

    def test_negative_words(self) -> None:
        """Negative word count should return 0."""
        assert estimate_tokens(-100) == 0

    def test_large_value(self) -> None:
        """10000 words should produce a proportionally large estimate."""
        result = estimate_tokens(10000)
        assert result == 13500  # 10000 × 1.35

    def test_one_word(self) -> None:
        """One word should return at least 1 token."""
        result = estimate_tokens(1)
        assert result >= 1


# ═══════════════════════════════════════════════════════════════════════
# MODEL_ENCODING_MAP
# ═══════════════════════════════════════════════════════════════════════


class TestModelEncodingMap:
    """Tests for the MODEL_ENCODING_MAP constant."""

    def test_map_is_not_empty(self) -> None:
        """The encoding map should have entries."""
        assert len(MODEL_ENCODING_MAP) > 0

    def test_gpt4o_in_map(self) -> None:
        """gpt-4o should be in the encoding map."""
        assert "gpt-4o" in MODEL_ENCODING_MAP

    def test_all_values_are_strings(self) -> None:
        """All encoding names should be strings."""
        for model, encoding in MODEL_ENCODING_MAP.items():
            assert isinstance(model, str), f"Key {model!r} is not a string"
            assert isinstance(encoding, str), f"Value for {model!r} is not a string"

    def test_openrouter_paths_present(self) -> None:
        """OpenRouter-style model paths should be in the map."""
        assert "openai/gpt-4o" in MODEL_ENCODING_MAP
        assert "anthropic/claude-3.5-sonnet" in MODEL_ENCODING_MAP
