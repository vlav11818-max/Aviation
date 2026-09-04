"""Tests for the 6-narrative-structure library."""

from __future__ import annotations

from aviation.axes import NarrativeStructureV2 as Structure
from aviation.structures import (
    SPECS,
    all_structures,
    next_in_rotation,
    prompt_for,
    suggest_structure,
)


class TestSpecs:
    def test_six_structures_present(self):
        assert set(SPECS.keys()) == set(Structure)
        assert len(list(all_structures())) == 6

    def test_each_spec_has_all_fields(self):
        for spec in all_structures():
            assert spec.display_name
            assert spec.tagline
            assert spec.beat_summary
            assert spec.scene_count == 13
            assert spec.quarterly_quota >= 1
            assert isinstance(spec.when_to_use, list)
            assert isinstance(spec.when_not_to_use, list)


class TestPrompts:
    def test_each_structure_produces_distinct_prompt(self):
        bible_json = '{"working_title": "Sample"}'
        prompts = [prompt_for(s, bible_json) for s in Structure]
        # All non-empty, all mention the shared "13" scene requirement,
        # and no two structures produce the same output.
        for p in prompts:
            assert "13" in p
            assert "STRUCTURE:" in p
            assert bible_json in p
        assert len({p for p in prompts}) == 6

    def test_kishotenketsu_prompt_bans_foreshadowing_words(self):
        p = prompt_for(Structure.KISHOTENKETSU, "{}")
        for word in ("danger", "wrong", "unusual", "sensed", "little did"):
            assert word in p.lower()

    def test_investigation_prompt_names_the_investigator(self):
        p = prompt_for(Structure.INVESTIGATION, "{}")
        assert "investigator" in p.lower()
        assert "flashback" in p.lower()


class TestDecisionTree:
    def test_historical_parallel_prefers_braid(self):
        assert suggest_structure(has_historical_parallel=True) == Structure.BRAID

    def test_multiple_povs_picks_rashomon(self):
        assert suggest_structure(multiple_povs_available=True) == Structure.RASHOMON

    def test_mystery_picks_investigation(self):
        assert suggest_structure(cause_is_mystery=True) == Structure.INVESTIGATION

    def test_iconic_moment_picks_in_media_res(self):
        assert suggest_structure(has_iconic_critical_moment=True) == Structure.IN_MEDIA_RES

    def test_routine_pro_picks_kishotenketsu(self):
        assert suggest_structure(
            has_iconic_critical_moment=False, routine_pro_with_surprise=True,
        ) == Structure.KISHOTENKETSU

    def test_default_is_three_act(self):
        assert suggest_structure(has_iconic_critical_moment=False) == Structure.THREE_ACT


class TestRotation:
    def test_none_returns_first(self):
        assert next_in_rotation(None) == Structure.THREE_ACT

    def test_walks_forward(self):
        s = None
        seen = []
        for _ in range(10):
            s = next_in_rotation(s)
            seen.append(s)
        # 10 picks should not all be the same structure.
        assert len(set(seen)) >= 3
