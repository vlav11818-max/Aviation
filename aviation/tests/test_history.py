"""Unit tests for the Global History Manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.history import HistoryStore, resolve_violations
from models.aviation_bible import (
    Aircraft,
    AviationStoryBible,
    CrewMember,
    Mode,
    NarrativeStructure,
    ROTATION_ORDER,
    Route,
)


@pytest.fixture()
def store(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path / "hist.db")


def _bible(
    *,
    mode: Mode = Mode.FICTIONAL,
    operator: str = "TransOcean Airways",
    registration: str = "N123AB",
    flight_number: str = "TO447",
    crew_name: str = "Captain Alpha",
    origin: str = "JFK",
    destination: str = "LHR",
    structure: NarrativeStructure = NarrativeStructure.THREE_ACT,
    title: str = "Test bible",
) -> AviationStoryBible:
    return AviationStoryBible(
        mode=mode,
        working_title=title,
        narrative_structure=structure,
        aircraft=Aircraft(
            type="Airbus A320",
            operator=operator,
            registration=registration,
            flight_number=flight_number,
        ),
        route=Route(origin=origin, destination=destination),
        crew=[CrewMember(name=crew_name, role="captain")],
    )


class TestRotation:
    def test_first_call_returns_first_of_order(self, store: HistoryStore):
        assert store.next_structure() == ROTATION_ORDER[0]

    def test_next_walks_the_order(self, store: HistoryStore):
        store.record_bible("id1", _bible(structure=NarrativeStructure.IN_MEDIA_RES))
        assert store.next_structure() == NarrativeStructure.THREE_ACT
        store.record_bible("id2", _bible(structure=NarrativeStructure.THREE_ACT))
        assert store.next_structure() == NarrativeStructure.RASHOMON

    def test_wraps_after_last(self, store: HistoryStore):
        for s in ROTATION_ORDER:
            store.record_bible(f"id-{s.value}", _bible(structure=s, operator=f"Op {s.value}", registration=f"N{hash(s.value)%9999:04d}A", flight_number=f"FL{hash(s.value)%9999:04d}"))
        assert store.next_structure() == ROTATION_ORDER[0]


class TestUniqueness:
    def test_fictional_reuse_flagged(self, store: HistoryStore):
        store.record_bible("id1", _bible(operator="TransOcean Airways"))
        b2 = _bible(operator="TransOcean Airways", registration="N999XY", flight_number="TO888", crew_name="Captain Beta")
        violations = store.check_bible(b2)
        assert any("airline" in v for v in violations)

    def test_real_mode_never_violates(self, store: HistoryStore):
        store.record_bible("id1", _bible(operator="Air France", registration="F-GZCP"))
        b2 = _bible(mode=Mode.REAL, operator="Air France", registration="F-GZCP")
        assert store.check_bible(b2) == []

    def test_normalization_matches_variants(self, store: HistoryStore):
        store.record_bible("id1", _bible(operator="Sky-High Air"))
        b2 = _bible(operator="SKY HIGH AIR", registration="N1XY", flight_number="SH1", crew_name="Beta")
        assert any("airline" in v for v in store.check_bible(b2))

    def test_aircraft_type_can_repeat(self, store: HistoryStore):
        store.record_bible("id1", _bible())
        b2 = _bible(operator="Other Ops", registration="N9XY", flight_number="OO9", crew_name="Beta", origin="ORD", destination="MAD")
        # No violation: aircraft type A320 is shared but not enforced.
        assert store.check_bible(b2) == []

    def test_resolve_violations_renames(self):
        bible = _bible(operator="TransOcean Airways")
        resolve_violations(bible, attempt=0)
        assert "Nova" in bible.aircraft.operator
        # Registration last char bumped.
        assert bible.aircraft.registration != "N123AB"


class TestSummary:
    def test_summary_lists_registered_elements(self, store: HistoryStore):
        store.record_bible("id1", _bible(operator="Alpha Air", registration="N100AA", flight_number="AA1"))
        s = store.summary()
        assert s.completed_incidents == 1
        assert "Alpha Air" in s.elements_by_kind["airline"]
        assert "N100AA" in s.elements_by_kind["registration"]
