"""Unit tests for aviation.text (segmenter, TTS strip, ledger merge)."""

from __future__ import annotations

import pytest

from aviation.text import (
    count_words,
    csv_escape,
    last_words,
    merge_ledger,
    segment_script,
    segment_with_timestamps,
    strip_for_tts,
    words_to_seconds,
)


class TestCounters:
    def test_count_words(self):
        assert count_words("") == 0
        assert count_words("   ") == 0
        assert count_words("one two three") == 3
        assert count_words("multi   spaced\ntext") == 3

    def test_words_to_seconds(self):
        assert words_to_seconds(150, wpm=150) == pytest.approx(60.0)
        assert words_to_seconds(0, wpm=150) == 0.0
        assert words_to_seconds(300, wpm=0) == 0.0

    def test_last_words(self):
        assert last_words("a b c d e", 3) == "c d e"
        assert last_words("", 5) == ""
        assert last_words("only", 0) == ""


class TestStripForTts:
    def test_removes_chapter_markers(self):
        text = "Chapter 3 — The Descent\n\nHer hand moved.\n"
        cleaned = strip_for_tts(text)
        assert "Chapter 3" not in cleaned
        assert "Her hand moved." in cleaned

    def test_removes_markdown_headings(self):
        text = "# Big heading\n\nSome narration.\n\n## Smaller\n"
        cleaned = strip_for_tts(text)
        assert "Big heading" not in cleaned
        assert "Some narration." in cleaned

    def test_removes_stage_directions(self):
        text = "The alarm sounded. [pause 1s] Then quiet.\n"
        cleaned = strip_for_tts(text)
        assert "[pause" not in cleaned

    def test_collapses_blank_lines(self):
        text = "A.\n\n\n\nB.\n"
        assert strip_for_tts(text).count("\n\n") == 1


class TestSegmenter:
    def test_short_sentence_stays_whole(self):
        segs = segment_script("The captain nodded.")
        assert segs == ["The captain nodded."]

    def test_long_sentence_splits_on_commas(self):
        long_sentence = (
            "The captain nodded, then reached for the throttles, watched "
            "the engine parameters roll back, checked the yaw damper, and "
            "trimmed the nose down while first officer read the checklist."
        )
        segs = segment_script(long_sentence, target_words=8)
        assert len(segs) > 1
        assert all(len(s.split()) <= 12 for s in segs)

    def test_timestamps_monotonic(self):
        rows = segment_with_timestamps(
            "First. Second sentence. Third short one.", wpm=150
        )
        seconds = [r["seconds"] for r in rows]
        assert seconds == sorted(seconds)
        assert all("timestamp" in r for r in rows)


class TestCsvEscape:
    def test_plain_value_unquoted(self):
        assert csv_escape("hello") == "hello"

    def test_comma_gets_quoted(self):
        assert csv_escape("a, b") == '"a, b"'

    def test_quote_doubled_inside(self):
        assert csv_escape('say "hi"') == '"say ""hi"""'


class TestLedger:
    def test_dedup_across_batches(self):
        merged = merge_ledger(
            ["Captain deferred APU write-up."],
            ["  captain deferred apu write-up.  "],
            ["First officer's daughter is nine."],
        )
        assert len(merged) == 2

    def test_preserves_first_seen_wording(self):
        merged = merge_ledger(
            ["A distinctive wording."],
            ["a Distinctive Wording."],
        )
        assert merged == ["A distinctive wording."]
