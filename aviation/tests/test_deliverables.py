"""Unit tests for deliverables (SSML dialect, YouTube markdown, storyboard CSV)."""

from __future__ import annotations

import re

import pytest

from aviation.deliverables import (
    _default_notice,
    _pad_tags,
    chapter_starts,
    render_storyboard_csv,
    render_storyboard_pipe,
    render_youtube_metadata,
    to_elevenlabs_ssml,
    total_runtime_seconds,
)
from models.aviation_bible import (
    Aircraft,
    AviationStoryBible,
    Mode,
    Route,
)


class TestSsml:
    def test_paragraph_break_inserted(self):
        text = "First paragraph.\n\nSecond paragraph."
        ssml = to_elevenlabs_ssml(text, paragraph_ms=800, cliff_ms=1500)
        assert '<break time="0.8s" />' in ssml
        assert "First paragraph." in ssml
        assert "Second paragraph." in ssml

    def test_cliffhanger_uses_longer_break(self):
        text = "The captain called mayday. Then silence."
        ssml = to_elevenlabs_ssml(text, paragraph_ms=800, cliff_ms=1500)
        assert '<break time="1.5s" />' in ssml

    def test_strips_headings(self):
        text = "# Chapter 4\n\nThe descent began."
        ssml = to_elevenlabs_ssml(text)
        assert "Chapter 4" not in ssml
        assert "The descent began." in ssml


class TestYoutubeMetadata:
    def _bible(self, mode: Mode = Mode.FICTIONAL) -> AviationStoryBible:
        return AviationStoryBible(
            mode=mode,
            working_title="Cascade at Flight Level 370",
            logline="A cascading hydraulic failure at cruise.",
            aircraft=Aircraft(type="A340-300", operator="TransOcean Airways", registration="N123AB", flight_number="TO447"),
            route=Route(origin="JFK", destination="LHR"),
            fictionalization_notice="Fictional composite.",
        )

    def test_renders_all_required_sections(self):
        bible = self._bible()
        md = render_youtube_metadata(
            metadata={
                "titles": ["Title A", "Title B", "Title C"],
                "hook": "Hook goes here.",
                "personal_note": "A short note.",
                "tags": ["a"] * 25,
                "sources": ["src 1"],
            },
            bible=bible,
            chapter_titles_and_starts=[("Ch1", 0.0), ("Ch2", 300.0)],
            total_seconds=600.0,
            mode=Mode.FICTIONAL,
        )
        assert "# Cascade at Flight Level 370" in md
        assert "## Title options" in md
        assert "- Title A" in md
        assert "⚠️ **Fictionalization notice:**" in md
        assert "⏱ **CHAPTERS**" in md
        assert "0:00 – Ch1" in md
        assert "5:00 – Ch2" in md
        assert "📚 **SOURCES**" in md
        assert "🖋 **NOTE FROM ME**" in md
        assert "A short note." in md
        assert "**TAGS**" in md

    def test_pad_tags_when_short(self):
        bible = self._bible()
        tags = _pad_tags(["aviation"], bible)
        assert len(tags) >= 20
        assert "A340-300" in tags
        assert "TransOcean Airways" in tags

    def test_notice_defaults_per_mode(self):
        assert "fictional" in _default_notice(Mode.FICTIONAL).lower()
        assert "report" in _default_notice(Mode.REAL).lower()


class TestStoryboard:
    def test_csv_header_and_row(self):
        rows = [
            {"timestamp": "0:00", "seconds": 0.0, "text": "hello, world", "image_prompt": 'a "quoted" prompt'},
        ]
        csv = render_storyboard_csv(rows)
        assert csv.splitlines()[0] == "timestamp,seconds,text_segment,image_prompt"
        # comma in text is quoted; quotes in prompt are doubled.
        assert '"hello, world"' in csv
        assert '"a ""quoted"" prompt"' in csv

    def test_pipe_layout(self):
        rows = [{"timestamp": "0:00", "text": "a | b", "image_prompt": "prompt | x"}]
        pipe = render_storyboard_pipe(rows)
        assert pipe.splitlines()[0] == "Timestamp | Text Segment | Image Generation Prompt"
        # pipes in fields are replaced with slashes.
        assert "a / b" in pipe
        assert "prompt / x" in pipe


class TestRuntimeMath:
    def test_chapter_starts_are_cumulative(self):
        texts = ["one " * 300, "two " * 150, "three " * 450]
        titles = ["A", "B", "C"]
        starts = chapter_starts(texts, wpm=150, titles=titles)
        assert starts[0] == ("A", 0.0)
        # 300 words at 150 wpm = 120s
        assert abs(starts[1][1] - 120.0) < 0.5
        # + 150 words / 150 wpm = 60s
        assert abs(starts[2][1] - 180.0) < 0.5

    def test_total_runtime(self):
        secs = total_runtime_seconds(["one " * 300], wpm=150)
        assert abs(secs - 120.0) < 0.5
