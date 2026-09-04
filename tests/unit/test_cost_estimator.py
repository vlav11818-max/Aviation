"""Unit tests for core.cost_estimator.

Tests pre-generation estimation for different strategies and models,
actual cost calculation, pricing table completeness, edge cases
(0 topics, unknown model), and the CostEstimate model itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from core.cost_estimator import CostEstimate, CostEstimator
from core.settings import Settings


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def settings_with_pricing(tmp_path: Path) -> Settings:
    """Return Settings with a known pricing table for deterministic tests."""
    content: dict[str, Any] = {
        "generation": {
            "tone": "dramatic_cinematic",
            "target_words": 3000,
            "min_score": 9.0,
            "max_attempts": 5,
        },
        "api": {
            "primary_provider": "openrouter",
            "primary_model": "anthropic/claude-3.5-sonnet",
            "fallback_provider": "openai",
            "fallback_model": "gpt-4o",
            "auto_fallback": True,
            "max_retries": 3,
            "timeout_seconds": 120,
            "rate_limits": {
                "openrouter": 5,
                "openai": 3,
                "anthropic": 2,
                "google": 5,
                "deepseek": 3,
                "qwen": 3,
            },
            "pricing": {
                "anthropic/claude-3.5-sonnet": {
                    "input": 3.0,
                    "output": 15.0,
                },
                "gpt-4o": {
                    "input": 2.5,
                    "output": 10.0,
                },
                "deepseek-chat": {
                    "input": 0.14,
                    "output": 0.28,
                },
                "gemini-1.5-pro": {
                    "input": 1.25,
                    "output": 5.0,
                },
                "qwen-max": {
                    "input": 1.6,
                    "output": 6.4,
                },
            },
        },
        "retry": {
            "initial_delay_seconds": 1.0,
            "max_delay_seconds": 16.0,
            "exponential_base": 2,
            "max_retries": 3,
        },
        "parallelism": {"max_workers": 3},
        "ssml": {"paragraph_break": "600ms"},
        "logging": {"level": "DEBUG", "log_dir": str(tmp_path / "logs")},
        "paths": {
            "output_dir": str(tmp_path / "output"),
            "data_dir": str(tmp_path / "data"),
            "resources_dir": "resources",
            "recovery_dir": str(tmp_path / "data" / "recovery"),
            "cache_dir": str(tmp_path / "data" / "cache"),
            "analytics_dir": str(tmp_path / "data" / "analytics"),
        },
        "cache": {"enabled": True, "skip_processed": True},
        "strategy": {"single_shot_max": 2000, "two_pass_max": 4000},
    }
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        yaml.dump(content, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    # Write a matching defaults file so Settings.load merges properly.
    defaults_dir = tmp_path / "resources" / "defaults"
    defaults_dir.mkdir(parents=True, exist_ok=True)
    defaults_path = defaults_dir / "settings.yaml"
    defaults_path.write_text(
        yaml.dump(content, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    return Settings.model_validate(content)


@pytest.fixture()
def estimator(settings_with_pricing: Settings) -> CostEstimator:
    """Return a CostEstimator with known pricing."""
    return CostEstimator(settings_with_pricing)


# ═══════════════════════════════════════════════════════════════════════
# estimate_cost — different strategies
# ═══════════════════════════════════════════════════════════════════════


class TestEstimateCostStrategies:
    """Tests for estimate_cost with different strategies."""

    def test_single_shot_strategy(self, estimator: CostEstimator) -> None:
        """Single shot should produce a valid estimate with 4 steps."""
        result = estimator.estimate_cost(
            topics_count=10,
            target_words=1500,
            strategy_name="single_shot",
            model="anthropic/claude-3.5-sonnet",
        )
        assert isinstance(result, CostEstimate)
        assert result.total_usd > 0
        assert result.per_story_usd > 0
        assert result.total_usd == pytest.approx(
            result.per_story_usd * 10, abs=1e-4
        )
        assert result.topics_count == 10
        assert result.strategy_name == "single_shot"
        # single_shot strategy has 4 steps: concept, single_shot, clean, evaluate
        assert len(result.breakdown_by_step) == 4
        assert "concept" in result.breakdown_by_step
        assert "single_shot" in result.breakdown_by_step
        assert "clean" in result.breakdown_by_step
        assert "evaluate" in result.breakdown_by_step

    def test_two_pass_strategy(self, estimator: CostEstimator) -> None:
        """Two pass should have 5 steps and a higher cost than single shot."""
        result = estimator.estimate_cost(
            topics_count=10,
            target_words=3000,
            strategy_name="two_pass",
            model="anthropic/claude-3.5-sonnet",
        )
        assert len(result.breakdown_by_step) == 5
        assert "outline" in result.breakdown_by_step
        assert result.total_usd > 0

    def test_full_pipeline_strategy(self, estimator: CostEstimator) -> None:
        """Full pipeline should have 6 steps and the highest cost."""
        result = estimator.estimate_cost(
            topics_count=10,
            target_words=5000,
            strategy_name="full_pipeline",
            model="anthropic/claude-3.5-sonnet",
        )
        assert len(result.breakdown_by_step) == 6
        assert "section" in result.breakdown_by_step
        assert "stitch" in result.breakdown_by_step
        assert result.total_usd > 0

    def test_full_costs_more_than_single(
        self, estimator: CostEstimator
    ) -> None:
        """Full pipeline with the same word count should cost more."""
        single = estimator.estimate_cost(
            topics_count=1,
            target_words=3000,
            strategy_name="single_shot",
            model="anthropic/claude-3.5-sonnet",
        )
        full = estimator.estimate_cost(
            topics_count=1,
            target_words=3000,
            strategy_name="full_pipeline",
            model="anthropic/claude-3.5-sonnet",
        )
        assert full.per_story_usd > single.per_story_usd

    def test_unknown_strategy_defaults_to_full(
        self, estimator: CostEstimator
    ) -> None:
        """An unrecognised strategy name should fall back to full_pipeline."""
        result = estimator.estimate_cost(
            topics_count=1,
            target_words=3000,
            strategy_name="nonexistent_strategy",
            model="anthropic/claude-3.5-sonnet",
        )
        assert len(result.breakdown_by_step) == 6


# ═══════════════════════════════════════════════════════════════════════
# estimate_cost — different models
# ═══════════════════════════════════════════════════════════════════════


class TestEstimateCostModels:
    """Tests for estimate_cost with different models / pricing."""

    def test_cheaper_model_costs_less(self, estimator: CostEstimator) -> None:
        """DeepSeek (cheap) should cost less than Claude (expensive)."""
        expensive = estimator.estimate_cost(
            topics_count=10,
            target_words=3000,
            strategy_name="full_pipeline",
            model="anthropic/claude-3.5-sonnet",
        )
        cheap = estimator.estimate_cost(
            topics_count=10,
            target_words=3000,
            strategy_name="full_pipeline",
            model="deepseek-chat",
        )
        assert cheap.total_usd < expensive.total_usd

    def test_more_topics_costs_more(self, estimator: CostEstimator) -> None:
        """50 topics should cost exactly 5× more than 10 topics."""
        ten = estimator.estimate_cost(
            topics_count=10,
            target_words=3000,
            strategy_name="full_pipeline",
            model="gpt-4o",
        )
        fifty = estimator.estimate_cost(
            topics_count=50,
            target_words=3000,
            strategy_name="full_pipeline",
            model="gpt-4o",
        )
        assert fifty.total_usd == pytest.approx(ten.total_usd * 5, abs=1e-4)

    def test_more_words_costs_more(self, estimator: CostEstimator) -> None:
        """Higher word count should produce a higher estimate."""
        small = estimator.estimate_cost(
            topics_count=1,
            target_words=1000,
            strategy_name="full_pipeline",
            model="gpt-4o",
        )
        large = estimator.estimate_cost(
            topics_count=1,
            target_words=5000,
            strategy_name="full_pipeline",
            model="gpt-4o",
        )
        assert large.per_story_usd > small.per_story_usd


# ═══════════════════════════════════════════════════════════════════════
# estimate_cost — edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEstimateCostEdgeCases:
    """Edge case tests for estimate_cost."""

    def test_zero_topics(self, estimator: CostEstimator) -> None:
        """Zero topics should return a zero estimate."""
        result = estimator.estimate_cost(
            topics_count=0,
            target_words=3000,
            strategy_name="full_pipeline",
            model="gpt-4o",
        )
        assert result.total_usd == 0.0
        assert result.per_story_usd == 0.0
        assert result.topics_count == 0

    def test_unknown_model_returns_zero_cost(
        self, estimator: CostEstimator
    ) -> None:
        """An unknown model (no pricing entry) should produce $0.00."""
        result = estimator.estimate_cost(
            topics_count=10,
            target_words=3000,
            strategy_name="full_pipeline",
            model="totally-unknown-model",
        )
        assert result.total_usd == 0.0
        assert result.per_story_usd == 0.0
        # Breakdown should still have step keys, just $0 each.
        assert len(result.breakdown_by_step) == 6
        for step_cost in result.breakdown_by_step.values():
            assert step_cost == 0.0

    def test_one_topic(self, estimator: CostEstimator) -> None:
        """One topic should have total == per_story."""
        result = estimator.estimate_cost(
            topics_count=1,
            target_words=3000,
            strategy_name="full_pipeline",
            model="gpt-4o",
        )
        assert result.total_usd == pytest.approx(result.per_story_usd, abs=1e-6)

    def test_estimated_tokens_populated(self, estimator: CostEstimator) -> None:
        """Estimated token fields should be populated for non-zero batches."""
        result = estimator.estimate_cost(
            topics_count=5,
            target_words=3000,
            strategy_name="full_pipeline",
            model="gpt-4o",
        )
        assert result.estimated_tokens_in > 0
        assert result.estimated_tokens_out > 0


# ═══════════════════════════════════════════════════════════════════════
# calculate_actual_cost
# ═══════════════════════════════════════════════════════════════════════


class TestCalculateActualCost:
    """Tests for calculate_actual_cost (runtime tracking)."""

    def test_known_model(self, estimator: CostEstimator) -> None:
        """Known model should return a positive cost."""
        cost = estimator.calculate_actual_cost(
            tokens_in=10_000,
            tokens_out=5_000,
            model="anthropic/claude-3.5-sonnet",
        )
        assert cost > 0
        # 10000 in * 3.0/1M + 5000 out * 15.0/1M = 0.03 + 0.075 = 0.105
        assert cost == pytest.approx(0.105, abs=1e-6)

    def test_gpt4o_cost(self, estimator: CostEstimator) -> None:
        """GPT-4o pricing should calculate correctly."""
        cost = estimator.calculate_actual_cost(
            tokens_in=1_000_000,
            tokens_out=1_000_000,
            model="gpt-4o",
        )
        # 1M * 2.5/1M + 1M * 10.0/1M = 2.5 + 10.0 = 12.5
        assert cost == pytest.approx(12.5, abs=1e-4)

    def test_unknown_model_returns_zero(
        self, estimator: CostEstimator
    ) -> None:
        """An unknown model should return 0.0."""
        cost = estimator.calculate_actual_cost(
            tokens_in=10_000,
            tokens_out=5_000,
            model="unknown-model-xyz",
        )
        assert cost == 0.0

    def test_zero_tokens(self, estimator: CostEstimator) -> None:
        """Zero tokens should return $0.00 even for a known model."""
        cost = estimator.calculate_actual_cost(
            tokens_in=0,
            tokens_out=0,
            model="gpt-4o",
        )
        assert cost == 0.0

    def test_deepseek_cheap(self, estimator: CostEstimator) -> None:
        """DeepSeek should be very cheap."""
        cost = estimator.calculate_actual_cost(
            tokens_in=10_000,
            tokens_out=5_000,
            model="deepseek-chat",
        )
        # 10000 * 0.14/1M + 5000 * 0.28/1M = 0.0014 + 0.0014 = 0.0028
        assert cost == pytest.approx(0.0028, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# get_pricing_table
# ═══════════════════════════════════════════════════════════════════════


class TestGetPricingTable:
    """Tests for get_pricing_table."""

    def test_returns_dict(self, estimator: CostEstimator) -> None:
        """Should return a non-empty dict."""
        table = estimator.get_pricing_table()
        assert isinstance(table, dict)
        assert len(table) > 0

    def test_entries_have_input_output(self, estimator: CostEstimator) -> None:
        """Each entry should have 'input' and 'output' keys."""
        table = estimator.get_pricing_table()
        for model, entry in table.items():
            assert "input" in entry, f"Missing 'input' for {model}"
            assert "output" in entry, f"Missing 'output' for {model}"
            assert isinstance(entry["input"], float)
            assert isinstance(entry["output"], float)


# ═══════════════════════════════════════════════════════════════════════
# CostEstimate model
# ═══════════════════════════════════════════════════════════════════════


class TestCostEstimateModel:
    """Tests for the CostEstimate pydantic model."""

    def test_defaults(self) -> None:
        """Default CostEstimate should have all zeros."""
        ce = CostEstimate()
        assert ce.total_usd == 0.0
        assert ce.per_story_usd == 0.0
        assert ce.breakdown_by_step == {}
        assert ce.topics_count == 0

    def test_round_trip(self) -> None:
        """CostEstimate should round-trip through model_dump/model_validate."""
        ce = CostEstimate(
            total_usd=10.5,
            per_story_usd=1.05,
            breakdown_by_step={"concept": 0.3, "outline": 0.5},
            topics_count=10,
            target_words=3000,
            strategy_name="full_pipeline",
            model="gpt-4o",
            estimated_tokens_in=50000,
            estimated_tokens_out=30000,
        )
        data = ce.model_dump()
        restored = CostEstimate.model_validate(data)
        assert restored == ce

    def test_serialization_json(self) -> None:
        """CostEstimate should serialise to JSON and back."""
        ce = CostEstimate(total_usd=5.0, per_story_usd=0.5)
        json_str = ce.model_dump_json()
        restored = CostEstimate.model_validate_json(json_str)
        assert restored.total_usd == 5.0
