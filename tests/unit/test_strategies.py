"""Unit tests for core.strategies.

Tests: select_strategy returns correct strategy for various word counts,
boundary values (exactly 2000, exactly 4000), all predefined strategies
contain expected steps in correct order.

Strategy definitions:
- single_shot  (<= 2000 words): Concept -> SingleShot -> Clean -> Evaluate
- two_pass     (2001-4000 words): Concept -> Outline -> SingleShot -> Clean -> Evaluate
- full_pipeline (> 4000 words): Concept -> Outline -> Section -> Stitch -> Clean -> Evaluate

The two_pass strategy uses SingleShotStep with the outline injected as
context — NOT SectionStep/StitchStep.  Those belong exclusively to
full_pipeline.
"""

from __future__ import annotations

import pytest

from core.settings import Settings
from core.steps.base_step import BaseStep
from core.steps.clean_step import CleanStep
from core.steps.concept_step import ConceptStep
from core.steps.evaluate_step import EvaluateStep
from core.steps.outline_step import OutlineStep
from core.steps.section_step import SectionStep
from core.steps.single_shot_step import SingleShotStep
from core.steps.stitch_step import StitchStep
from core.strategies import (
    FULL_PIPELINE_STRATEGY_NAME,
    SINGLE_SHOT_STRATEGY_NAME,
    TWO_PASS_STRATEGY_NAME,
    get_strategy,
    list_strategies,
    select_strategy,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    """Default settings (uses built-in defaults)."""
    return Settings()


# ── Tests: select_strategy ────────────────────────────────────────────────────


class TestSelectStrategy:
    """Tests for automatic strategy selection based on word count."""

    def test_short_story_single_shot(self, settings: Settings) -> None:
        """<= 2000 words should select single_shot."""
        name, steps = select_strategy(1000, settings)
        assert name == SINGLE_SHOT_STRATEGY_NAME

    def test_exactly_2000_single_shot(self, settings: Settings) -> None:
        """Exactly 2000 words should select single_shot."""
        name, steps = select_strategy(2000, settings)
        assert name == SINGLE_SHOT_STRATEGY_NAME

    def test_medium_story_two_pass(self, settings: Settings) -> None:
        """2001-4000 words should select two_pass."""
        name, steps = select_strategy(3000, settings)
        assert name == TWO_PASS_STRATEGY_NAME

    def test_exactly_4000_two_pass(self, settings: Settings) -> None:
        """Exactly 4000 words should select two_pass."""
        name, steps = select_strategy(4000, settings)
        assert name == TWO_PASS_STRATEGY_NAME

    def test_long_story_full_pipeline(self, settings: Settings) -> None:
        """> 4000 words should select full_pipeline."""
        name, steps = select_strategy(5000, settings)
        assert name == FULL_PIPELINE_STRATEGY_NAME

    def test_very_long_story_full_pipeline(self, settings: Settings) -> None:
        """10000 words should select full_pipeline."""
        name, steps = select_strategy(10000, settings)
        assert name == FULL_PIPELINE_STRATEGY_NAME

    def test_minimum_words_single_shot(self, settings: Settings) -> None:
        """500 words (minimum) should select single_shot."""
        name, steps = select_strategy(500, settings)
        assert name == SINGLE_SHOT_STRATEGY_NAME

    def test_returns_list_of_base_step_subclasses(self, settings: Settings) -> None:
        """Returned steps should all be BaseStep subclasses."""
        _name, steps = select_strategy(3000, settings)
        for step_cls in steps:
            assert issubclass(step_cls, BaseStep)

    def test_two_pass_boundary_2001(self, settings: Settings) -> None:
        """2001 words should select two_pass, not single_shot."""
        name, _steps = select_strategy(2001, settings)
        assert name == TWO_PASS_STRATEGY_NAME

    def test_full_pipeline_boundary_4001(self, settings: Settings) -> None:
        """4001 words should select full_pipeline, not two_pass."""
        name, _steps = select_strategy(4001, settings)
        assert name == FULL_PIPELINE_STRATEGY_NAME


# ── Tests: get_strategy ───────────────────────────────────────────────────────


class TestGetStrategy:
    """Tests for explicit strategy lookup by name."""

    def test_single_shot_by_name(self) -> None:
        """get_strategy('single_shot') should return a valid list."""
        steps = get_strategy(SINGLE_SHOT_STRATEGY_NAME)
        assert len(steps) > 0

    def test_two_pass_by_name(self) -> None:
        """get_strategy('two_pass') should return a valid list."""
        steps = get_strategy(TWO_PASS_STRATEGY_NAME)
        assert len(steps) > 0

    def test_full_pipeline_by_name(self) -> None:
        """get_strategy('full_pipeline') should return a valid list."""
        steps = get_strategy(FULL_PIPELINE_STRATEGY_NAME)
        assert len(steps) > 0

    def test_unknown_strategy_raises_value_error(self) -> None:
        """get_strategy with unknown name should raise ValueError."""
        with pytest.raises(ValueError):
            get_strategy("nonexistent_strategy")


# ── Tests: strategy contents ──────────────────────────────────────────────────


class TestStrategyContents:
    """Tests that each strategy contains the expected step classes."""

    # ── Single Shot ──

    def test_single_shot_has_concept(self) -> None:
        """Single shot should start with ConceptStep."""
        steps = get_strategy(SINGLE_SHOT_STRATEGY_NAME)
        assert steps[0] is ConceptStep

    def test_single_shot_has_single_shot_step(self) -> None:
        """Single shot should include SingleShotStep."""
        steps = get_strategy(SINGLE_SHOT_STRATEGY_NAME)
        assert SingleShotStep in steps

    def test_single_shot_has_clean(self) -> None:
        """Single shot should include CleanStep."""
        steps = get_strategy(SINGLE_SHOT_STRATEGY_NAME)
        assert CleanStep in steps

    def test_single_shot_has_evaluate(self) -> None:
        """Single shot should include EvaluateStep."""
        steps = get_strategy(SINGLE_SHOT_STRATEGY_NAME)
        assert EvaluateStep in steps

    def test_single_shot_no_section_step(self) -> None:
        """Single shot should NOT include SectionStep."""
        steps = get_strategy(SINGLE_SHOT_STRATEGY_NAME)
        assert SectionStep not in steps

    def test_single_shot_no_outline(self) -> None:
        """Single shot should NOT include OutlineStep."""
        steps = get_strategy(SINGLE_SHOT_STRATEGY_NAME)
        assert OutlineStep not in steps

    def test_single_shot_no_stitch(self) -> None:
        """Single shot should NOT include StitchStep."""
        steps = get_strategy(SINGLE_SHOT_STRATEGY_NAME)
        assert StitchStep not in steps

    # ── Two Pass ──

    def test_two_pass_has_concept(self) -> None:
        """Two pass should start with ConceptStep."""
        steps = get_strategy(TWO_PASS_STRATEGY_NAME)
        assert steps[0] is ConceptStep

    def test_two_pass_has_outline(self) -> None:
        """Two pass should include OutlineStep."""
        steps = get_strategy(TWO_PASS_STRATEGY_NAME)
        assert OutlineStep in steps

    def test_two_pass_has_single_shot_step(self) -> None:
        """Two pass should include SingleShotStep (outline injected as context)."""
        steps = get_strategy(TWO_PASS_STRATEGY_NAME)
        assert SingleShotStep in steps

    def test_two_pass_has_clean(self) -> None:
        """Two pass should include CleanStep."""
        steps = get_strategy(TWO_PASS_STRATEGY_NAME)
        assert CleanStep in steps

    def test_two_pass_has_evaluate(self) -> None:
        """Two pass should include EvaluateStep."""
        steps = get_strategy(TWO_PASS_STRATEGY_NAME)
        assert EvaluateStep in steps

    def test_two_pass_no_section_step(self) -> None:
        """Two pass should NOT include SectionStep (that is full pipeline only)."""
        steps = get_strategy(TWO_PASS_STRATEGY_NAME)
        assert SectionStep not in steps

    def test_two_pass_no_stitch_step(self) -> None:
        """Two pass should NOT include StitchStep (that is full pipeline only)."""
        steps = get_strategy(TWO_PASS_STRATEGY_NAME)
        assert StitchStep not in steps

    def test_two_pass_outline_before_single_shot(self) -> None:
        """OutlineStep must come before SingleShotStep in two_pass."""
        steps = get_strategy(TWO_PASS_STRATEGY_NAME)
        outline_idx = steps.index(OutlineStep)
        single_shot_idx = steps.index(SingleShotStep)
        assert outline_idx < single_shot_idx

    def test_two_pass_differs_from_full_pipeline(self) -> None:
        """Two pass and full pipeline must be distinct step lists."""
        two_pass = get_strategy(TWO_PASS_STRATEGY_NAME)
        full = get_strategy(FULL_PIPELINE_STRATEGY_NAME)
        assert two_pass != full

    # ── Full Pipeline ──

    def test_full_pipeline_has_concept(self) -> None:
        """Full pipeline should start with ConceptStep."""
        steps = get_strategy(FULL_PIPELINE_STRATEGY_NAME)
        assert steps[0] is ConceptStep

    def test_full_pipeline_has_all_stages(self) -> None:
        """Full pipeline should include all core steps."""
        steps = get_strategy(FULL_PIPELINE_STRATEGY_NAME)
        assert ConceptStep in steps
        assert OutlineStep in steps
        assert SectionStep in steps
        assert StitchStep in steps
        assert CleanStep in steps
        assert EvaluateStep in steps

    def test_full_pipeline_no_single_shot(self) -> None:
        """Full pipeline should NOT include SingleShotStep."""
        steps = get_strategy(FULL_PIPELINE_STRATEGY_NAME)
        assert SingleShotStep not in steps

    def test_full_pipeline_outline_before_section(self) -> None:
        """OutlineStep must come before SectionStep in full_pipeline."""
        steps = get_strategy(FULL_PIPELINE_STRATEGY_NAME)
        outline_idx = steps.index(OutlineStep)
        section_idx = steps.index(SectionStep)
        assert outline_idx < section_idx

    def test_full_pipeline_section_before_stitch(self) -> None:
        """SectionStep must come before StitchStep in full_pipeline."""
        steps = get_strategy(FULL_PIPELINE_STRATEGY_NAME)
        section_idx = steps.index(SectionStep)
        stitch_idx = steps.index(StitchStep)
        assert section_idx < stitch_idx

    # ── Cross-strategy checks ──

    def test_evaluate_is_last_in_all_strategies(self) -> None:
        """EvaluateStep should be the last step in every strategy."""
        for name in list_strategies():
            steps = get_strategy(name)
            assert steps[-1] is EvaluateStep, (
                f"Strategy '{name}' does not end with EvaluateStep"
            )

    def test_clean_before_evaluate_in_all_strategies(self) -> None:
        """CleanStep must come directly before EvaluateStep in every strategy."""
        for name in list_strategies():
            steps = get_strategy(name)
            eval_idx = steps.index(EvaluateStep)
            clean_idx = steps.index(CleanStep)
            assert clean_idx == eval_idx - 1, (
                f"Strategy '{name}': CleanStep must be immediately before EvaluateStep"
            )

    def test_all_strategies_start_with_concept(self) -> None:
        """Every strategy must start with ConceptStep."""
        for name in list_strategies():
            steps = get_strategy(name)
            assert steps[0] is ConceptStep, (
                f"Strategy '{name}' does not start with ConceptStep"
            )


# ── Tests: list_strategies ────────────────────────────────────────────────────


class TestListStrategies:
    """Tests for strategy listing."""

    def test_returns_three_strategies(self) -> None:
        """Should list exactly 3 predefined strategies."""
        names = list_strategies()
        assert len(names) == 3

    def test_contains_all_names(self) -> None:
        """Should contain all expected strategy names."""
        names = list_strategies()
        assert SINGLE_SHOT_STRATEGY_NAME in names
        assert TWO_PASS_STRATEGY_NAME in names
        assert FULL_PIPELINE_STRATEGY_NAME in names

    def test_returns_sorted_list(self) -> None:
        """list_strategies should return names in sorted order."""
        names = list_strategies()
        assert names == sorted(names)
