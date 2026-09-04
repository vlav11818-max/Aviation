"""Tests for the seed catalog / name / airline loaders."""

from __future__ import annotations

import pytest

from aviation.resources import (
    filter_incidents,
    load_airlines,
    load_incidents,
    load_names,
    sample_airline,
    sample_names,
    reload_all,
)


def setup_function(_):
    reload_all()


class TestLoaders:
    def test_incident_catalog_populated(self):
        inc = load_incidents()
        assert len(inc) >= 40, "expected at least 40 seeded incidents"
        assert inc[0].name

    def test_names_pool_populated(self):
        names = load_names()
        assert len(names) >= 100, "expected a healthy pool of pre-vetted names"
        e = names[0]
        assert e.first_name and e.last_name

    def test_airlines_pool_populated(self):
        airlines = load_airlines()
        assert len(airlines) >= 30
        assert airlines[0].name


class TestFiltering:
    def test_filter_by_sub_genre(self):
        pool = filter_incidents(sub_genre="Miracle emergency landing")
        assert len(pool) >= 5
        for inc in pool:
            assert "miracle emergency landing" in (
                inc.sub_genre_primary + " " + inc.sub_genre_secondary
            ).lower()

    def test_max_risk_low_excludes_high(self):
        low_only = filter_incidents(max_risk="LOW")
        for inc in low_only:
            assert inc.monetization_risk.upper() == "LOW"

    def test_excluded_names_are_dropped(self):
        first = load_incidents()[0]
        assert not any(
            i.name == first.name
            for i in filter_incidents(excluded_names=[first.name])
        )


class TestSampling:
    def test_sample_names_returns_distinct(self):
        n = sample_names(5)
        firsts = {p.first_name for p in n}
        assert len(firsts) == 5

    def test_sample_names_excludes(self):
        first = load_names()[0].first_name
        n = sample_names(5, excluded_first_names=[first])
        assert first not in {p.first_name for p in n}

    def test_sample_airline_returns_pool_entry(self):
        a = sample_airline()
        assert a is not None and a.name


class TestPromptSummary:
    def test_summary_for_prompt_contains_key_fields(self):
        inc = load_incidents()[0]
        s = inc.summary_for_prompt()
        assert "SEED:" in s
        assert inc.name in s
        # Sources / dramatic details / risk are in there if the incident carries them.
        if inc.sources:
            assert inc.sources[0] in s
