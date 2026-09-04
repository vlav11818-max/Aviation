"""Tests for the similarity checker."""

from __future__ import annotations

import os

import pytest

from aviation.similarity import (
    NOUN_MAX_OVERLAP,
    TITLE_MAX_COSINE,
    check_opening,
    check_title,
    embed,
    noun_overlap,
)


@pytest.fixture(autouse=True)
def _force_mock():
    prev = os.environ.get("AVIATION_FORCE_MOCK")
    os.environ["AVIATION_FORCE_MOCK"] = "1"
    yield
    if prev is None:
        os.environ.pop("AVIATION_FORCE_MOCK", None)
    else:
        os.environ["AVIATION_FORCE_MOCK"] = prev


class TestNounOverlap:
    def test_zero_when_no_shared_nouns(self):
        assert noun_overlap("Alpha flight one", "Bravo runway two") < 0.5

    def test_high_overlap_when_words_repeat(self):
        a = "The Miracle on the Hudson: Sully's Landing"
        b = "The Miracle Landing on the Hudson River"
        assert noun_overlap(a, b) > 0.3


class TestEmbed:
    def test_mock_embedding_is_deterministic(self):
        v1 = embed("cascade at flight level 370")
        v2 = embed("cascade at flight level 370")
        assert v1 == v2

    def test_different_text_different_vector(self):
        assert embed("cockpit fire mayday") != embed("routine cruise sunny weather")


class TestCheckTitle:
    def test_passes_with_no_priors(self):
        r = check_title("Any Title", priors=[])
        assert r.passed

    def test_fails_on_near_duplicate(self):
        title = "The Miracle on the Hudson: Autopilot Failed at Seat 7C"
        r = check_title(title, priors=[title])
        assert not r.passed
        assert any(f.check.startswith("title_") for f in r.findings)

    def test_passes_on_distinct_titles(self):
        r = check_title(
            "Rookie First Officer's First Emergency Landing",
            priors=["Fuel Exhaustion at Cruise: The Longest Glide"],
        )
        # Mock embeddings shouldn't flag two very different titles as similar.
        assert r.passed


class TestCheckOpening:
    def test_passes_with_no_priors(self):
        r = check_opening("some opening text", priors=[])
        assert r.passed
