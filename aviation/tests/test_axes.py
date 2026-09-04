"""Tests for the eight rotation axes + per-axis cooldowns."""

from __future__ import annotations

from pathlib import Path

import pytest

from aviation.axes import (
    AXES,
    HOOK_MAX_USES,
    HOOK_MAX_USES_WINDOW,
    HookPattern,
    SubGenre,
    AircraftClass,
    Setting,
    IncidentType,
    TwistType,
    Resolution,
    EmotionalBeat,
    NarrativeStructureV2,
    PROTAGONIST_ARCHETYPES,
    STRUCTURE_QUARTERLY_QUOTAS,
    SUBGENRE_QUARTERLY_QUOTAS,
)
from core.history import HistoryStore
from models.aviation_bible import (
    Aircraft,
    AviationStoryBible,
    CrewMember,
    Mode,
    NarrativeStructure,
    Route,
)


@pytest.fixture()
def store(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path / "hist.db")


class TestAxesRegistry:
    def test_all_documented_axes_present(self):
        # 8 primary axes + 3 tracked-but-free-form.
        for axis in (
            "sub_genre", "aircraft_type", "setting",
            "protagonist_archetype", "inciting_incident",
            "twist_type", "resolution", "hook_pattern",
            "narrative_structure", "narrator_voice",
            "character_first_name", "fictional_airline",
        ):
            assert axis in AXES, axis

    def test_cooldowns_match_brief(self):
        assert AXES["sub_genre"].cooldown == 2
        assert AXES["aircraft_type"].cooldown == 5
        assert AXES["setting"].cooldown == 4
        assert AXES["protagonist_archetype"].cooldown == 6
        assert AXES["inciting_incident"].cooldown == 8
        assert AXES["twist_type"].cooldown == 10
        assert AXES["resolution"].cooldown == 6
        assert AXES["hook_pattern"].cooldown == 3
        assert AXES["character_first_name"].cooldown == 15
        assert AXES["fictional_airline"].cooldown == 8

    def test_enum_populations_match_brief(self):
        assert len(list(SubGenre)) == 11
        assert len(list(AircraftClass)) == 12
        assert len(list(Setting)) == 14
        assert len(list(IncidentType)) == 27
        assert len(list(TwistType)) == 15
        assert len(list(Resolution)) == 12
        assert len(list(EmotionalBeat)) == 8
        assert len(list(HookPattern)) == 7
        assert len(list(NarrativeStructureV2)) == 6
        assert len(PROTAGONIST_ARCHETYPES) == 35

    def test_quarterly_quotas_sum_reasonably(self):
        # Sub-genres: 12 videos / month × 3 months = ~36 videos
        assert sum(SUBGENRE_QUARTERLY_QUOTAS.values()) >= 36
        # Structures: 36 videos / quarter total
        assert sum(STRUCTURE_QUARTERLY_QUOTAS.values()) == 36


class TestAxisCooldown:
    def _bible(self, tag: str) -> AviationStoryBible:
        return AviationStoryBible(
            mode=Mode.FICTIONAL,
            working_title=tag,
            narrative_structure=NarrativeStructure.THREE_ACT,
            aircraft=Aircraft(
                type="A320",
                operator=f"Operator {tag}",
                registration=f"N{tag}AB",
                flight_number=f"{tag}1",
            ),
            route=Route(origin=f"Origin{tag}", destination=f"Dest{tag}"),
            crew=[CrewMember(name=f"Captain {tag}", role="captain")],
        )

    def test_cooldown_blocks_reuse_within_window(self, store: HistoryStore):
        store.record_bible("a", self._bible("A"), axes={"sub_genre": SubGenre.MIRACLE_LANDING.value})
        # Cooldown for sub_genre is 2 — the very next call must block.
        assert not store.cooldown_ok("sub_genre", SubGenre.MIRACLE_LANDING.value)
        assert store.cooldown_ok("sub_genre", SubGenre.WEATHER.value)

    def test_cooldown_releases_after_window(self, store: HistoryStore):
        store.record_bible("a", self._bible("A"), axes={"sub_genre": SubGenre.MIRACLE_LANDING.value})
        store.record_bible("b", self._bible("B"), axes={"sub_genre": SubGenre.WEATHER.value})
        store.record_bible("c", self._bible("C"), axes={"sub_genre": SubGenre.MECHANICAL.value})
        # Cooldown = 2 → after 2 other stories, Miracle should be re-usable.
        assert store.cooldown_ok("sub_genre", SubGenre.MIRACLE_LANDING.value)

    def test_hook_max_uses_window(self, store: HistoryStore):
        # Insert 3 uses of pattern A in a row — the 4th should fail.
        for i in range(3):
            store.record_bible(
                f"h{i}",
                self._bible(f"H{i}"),
                axes={"hook_pattern": HookPattern.A_SPECIFIC_DETAIL.value},
            )
        assert not store.hook_pattern_allowed(HookPattern.A_SPECIFIC_DETAIL.value)
        assert store.hook_pattern_allowed(HookPattern.B_TIME_COMPRESSION.value)

    def test_parameter_tuple_uniqueness(self, store: HistoryStore):
        store.record_bible(
            "id1",
            self._bible("X"),
            axes={
                "sub_genre": SubGenre.MIRACLE_LANDING.value,
                "aircraft_type": AircraftClass.NARROW_BODY.value,
                "setting": Setting.NORTH_ATLANTIC.value,
            },
        )
        # Same triple must fail.
        assert not store.parameter_tuple_unique(
            SubGenre.MIRACLE_LANDING.value,
            AircraftClass.NARROW_BODY.value,
            Setting.NORTH_ATLANTIC.value,
        )
        # A different triple is fine.
        assert store.parameter_tuple_unique(
            SubGenre.MIRACLE_LANDING.value,
            AircraftClass.WIDE_BODY.value,
            Setting.NORTH_ATLANTIC.value,
        )


class TestSeedIncidentTracking:
    def test_used_seed_returns_true_after_record(self, tmp_path: Path):
        store = HistoryStore(tmp_path / "h.db")
        bible = AviationStoryBible(
            mode=Mode.REAL,
            working_title="Sioux City retelling",
            narrative_structure=NarrativeStructure.THREE_ACT,
            aircraft=Aircraft(type="DC-10", operator="United", registration="N1819U", flight_number="UA232"),
        )
        assert not store.seed_used("United 232 — Sioux City")
        store.record_bible("job1", bible, seed_incident_name="United 232 — Sioux City")
        assert store.seed_used("United 232 — Sioux City")
        assert store.seed_used("united 232 — sioux city")  # case-insensitive
