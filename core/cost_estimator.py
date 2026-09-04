"""API cost estimation for story generation.

Provides two modes:

- **Pre-generation** — ``estimate_cost()`` predicts total USD cost for
  a batch of topics *before* generation starts (shown in the cost
  confirmation dialog).
- **Runtime** — ``calculate_actual_cost()`` computes the exact cost for
  a single API call from real token counts (used by ``APIClient``).

Token estimates are derived from target word counts via
``utils.token_counter.estimate_tokens``.  Pricing is read from
``settings.api.pricing`` (per-model, USD per 1 M tokens).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from core.settings import Settings
from utils.token_counter import estimate_tokens

logger = logging.getLogger(__name__)


# ── Token multipliers per pipeline step ─────────────────────────────────
#
# These represent *approximate* ratios of tokens consumed per step
# relative to the target output word count (converted to tokens).
# The multipliers are empirical: prompt overhead + response tokens.
#
# For example, a concept step sends a prompt (~500 tokens of instructions
# + topic) and receives a JSON (~800 tokens), so the multiplier is
# roughly 1.0× the target-output token estimate for both in and out
# combined.

_STEP_MULTIPLIERS: dict[str, dict[str, float]] = {
    "concept": {"input": 0.5, "output": 0.4},
    "outline": {"input": 0.8, "output": 0.5},
    "section": {"input": 1.5, "output": 1.0},
    "single_shot": {"input": 0.8, "output": 1.0},
    "stitch": {"input": 1.2, "output": 1.0},
    "clean": {"input": 0.0, "output": 0.0},  # programmatic, no API call
    "evaluate": {"input": 1.2, "output": 0.3},
    "revise": {"input": 1.5, "output": 1.0},
}

# Mapping of strategy name → list of step names involved.
_STRATEGY_STEPS: dict[str, list[str]] = {
    "single_shot": ["concept", "single_shot", "clean", "evaluate"],
    "two_pass": ["concept", "outline", "single_shot", "clean", "evaluate"],
    "full_pipeline": [
        "concept",
        "outline",
        "section",
        "stitch",
        "clean",
        "evaluate",
    ],
}


# ── CostEstimate model ─────────────────────────────────────────────────


class CostEstimate(BaseModel):
    """Result of a pre-generation cost estimation.

    Attributes:
        total_usd: Estimated total cost for all topics in USD.
        per_story_usd: Estimated cost per single story in USD.
        breakdown_by_step: Per-step cost breakdown for one story
            (step name → estimated USD).
        topics_count: Number of topics in the batch.
        target_words: Target words per story.
        strategy_name: Strategy used for estimation.
        model: Model used for pricing lookup.
        estimated_tokens_in: Estimated total input tokens (all stories).
        estimated_tokens_out: Estimated total output tokens (all stories).
    """

    total_usd: float = Field(
        default=0.0, ge=0.0, description="Total estimated cost in USD."
    )
    per_story_usd: float = Field(
        default=0.0, ge=0.0, description="Estimated cost per story in USD."
    )
    breakdown_by_step: dict[str, float] = Field(
        default_factory=dict,
        description="Per-step cost breakdown for one story (step → USD).",
    )
    topics_count: int = Field(
        default=0, ge=0, description="Number of topics."
    )
    target_words: int = Field(
        default=0, ge=0, description="Target words per story."
    )
    strategy_name: str = Field(
        default="", description="Strategy used for estimation."
    )
    model: str = Field(
        default="", description="Model used for pricing lookup."
    )
    estimated_tokens_in: int = Field(
        default=0, ge=0, description="Total estimated input tokens."
    )
    estimated_tokens_out: int = Field(
        default=0, ge=0, description="Total estimated output tokens."
    )


# ── CostEstimator ──────────────────────────────────────────────────────


class CostEstimator:
    """Estimates and calculates API costs.

    Args:
        settings: Application settings (provides pricing table).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def estimate_cost(
        self,
        topics_count: int,
        target_words: int,
        strategy_name: str,
        model: str,
    ) -> CostEstimate:
        """Estimate total cost for a batch before generation starts.

        Uses per-step token multipliers and the pricing table from
        settings to produce a rough cost projection.

        Args:
            topics_count: Number of topics to generate.
            target_words: Target word count per story.
            strategy_name: Strategy name (``"single_shot"``,
                ``"two_pass"``, or ``"full_pipeline"``).
            model: Model identifier for pricing lookup.

        Returns:
            A ``CostEstimate`` with totals and per-step breakdown.
        """
        if topics_count <= 0:
            logger.debug("estimate_cost called with 0 topics, returning zero estimate")
            return CostEstimate(
                topics_count=0,
                target_words=target_words,
                strategy_name=strategy_name,
                model=model,
            )

        base_tokens = estimate_tokens(target_words)
        steps = _STRATEGY_STEPS.get(strategy_name, _STRATEGY_STEPS["full_pipeline"])

        breakdown: dict[str, float] = {}
        total_tokens_in_per_story = 0
        total_tokens_out_per_story = 0

        for step_name in steps:
            multipliers = _STEP_MULTIPLIERS.get(
                step_name, {"input": 0.5, "output": 0.5}
            )
            step_tokens_in = int(base_tokens * multipliers["input"])
            step_tokens_out = int(base_tokens * multipliers["output"])
            step_cost = self._price_tokens(
                step_tokens_in, step_tokens_out, model
            )
            breakdown[step_name] = round(step_cost, 6)
            total_tokens_in_per_story += step_tokens_in
            total_tokens_out_per_story += step_tokens_out

        per_story = sum(breakdown.values())
        total = per_story * topics_count

        logger.info(
            "Cost estimate: %d topics × %d words (%s, %s) = $%.4f total, "
            "$%.4f/story, ~%d tok_in + ~%d tok_out per story",
            topics_count,
            target_words,
            strategy_name,
            model,
            total,
            per_story,
            total_tokens_in_per_story,
            total_tokens_out_per_story,
        )

        return CostEstimate(
            total_usd=round(total, 6),
            per_story_usd=round(per_story, 6),
            breakdown_by_step=breakdown,
            topics_count=topics_count,
            target_words=target_words,
            strategy_name=strategy_name,
            model=model,
            estimated_tokens_in=total_tokens_in_per_story * topics_count,
            estimated_tokens_out=total_tokens_out_per_story * topics_count,
        )

    def calculate_actual_cost(
        self,
        tokens_in: int,
        tokens_out: int,
        model: str,
    ) -> float:
        """Calculate the actual cost of a single API call.

        Used at runtime to track real spend.

        Args:
            tokens_in: Actual input token count.
            tokens_out: Actual output token count.
            model: Model identifier for pricing lookup.

        Returns:
            Cost in USD.  Returns 0.0 if the model is not in the
            pricing table.
        """
        cost = self._price_tokens(tokens_in, tokens_out, model)
        logger.debug(
            "Actual cost: model=%s, tokens_in=%d, tokens_out=%d, cost=$%.6f",
            model,
            tokens_in,
            tokens_out,
            cost,
        )
        return cost

    def get_pricing_table(self) -> dict[str, dict[str, float]]:
        """Return the current pricing table as a plain dict.

        Returns:
            Dict of ``{model: {"input": float, "output": float}}``.
        """
        result: dict[str, dict[str, float]] = {}
        for model_name, entry in self._settings.api.pricing.items():
            result[model_name] = {
                "input": entry.input,
                "output": entry.output,
            }
        return result

    # ── private helpers ─────────────────────────────────────────────

    def _price_tokens(
        self, tokens_in: int, tokens_out: int, model: str
    ) -> float:
        """Look up pricing and compute cost.

        Args:
            tokens_in: Input token count.
            tokens_out: Output token count.
            model: Model identifier.

        Returns:
            Cost in USD.  Returns 0.0 with a warning if the model
            has no pricing entry.
        """
        pricing = self._settings.api.pricing.get(model)
        if pricing is None:
            logger.warning(
                "CostEstimator: no pricing for model '%s', returning $0.00",
                model,
            )
            return 0.0

        cost_in = (tokens_in / 1_000_000) * pricing.input
        cost_out = (tokens_out / 1_000_000) * pricing.output
        return cost_in + cost_out
