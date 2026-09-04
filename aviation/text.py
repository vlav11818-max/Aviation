"""Text helpers used by the aviation orchestrator, exporters, and tests."""

from __future__ import annotations

import re
from typing import Iterable

# One "word" for word count = a whitespace-separated token.
_WORD_RE = re.compile(r"\S+")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def words_to_seconds(words: int, wpm: int = 150) -> float:
    if wpm <= 0:
        return 0.0
    return (words / wpm) * 60.0


def last_words(text: str, n: int) -> str:
    if n <= 0 or not text:
        return ""
    tokens = text.split()
    return " ".join(tokens[-n:])


# ── TTS strip ─────────────────────────────────────────────────────────
#
# The clean TTS script must be pure narration: no markdown headings, no
# bracketed stage directions, no chapter labels, no meta-comments.

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.M)
_STAGE_DIR = re.compile(r"\[(?:pause|sfx|scene|beat|music|silence|break)[^\]]*\]", re.I)
_CHAPTER_MARK = re.compile(r"^\s*(?:chapter|part|act|section)\s+[\w\dIVXLC]+.*$", re.I | re.M)
_META = re.compile(r"^\s*(?:note|editor|writer|author|the end|end of chapter)[:\s].*$", re.I | re.M)
_HR = re.compile(r"^\s*(?:\*\s*){3,}\s*$|^\s*(?:-\s*){3,}\s*$", re.M)
_MULTI_BLANK = re.compile(r"\n{3,}")


def strip_for_tts(text: str) -> str:
    """Return a clean-narration version of ``text`` for the TTS script."""
    if not text:
        return ""
    t = text
    t = _MD_HEADING.sub("", t)
    t = _CHAPTER_MARK.sub("", t)
    t = _STAGE_DIR.sub("", t)
    t = _META.sub("", t)
    t = _HR.sub("", t)
    t = _MULTI_BLANK.sub("\n\n", t).strip()
    return t


# ── CSV escape ────────────────────────────────────────────────────────

def csv_escape(value) -> str:
    """RFC-4180-style CSV field escape."""
    s = "" if value is None else str(value)
    needs_quote = any(ch in s for ch in (",", "\n", "\r", '"'))
    if needs_quote:
        s = s.replace('"', '""')
        return f'"{s}"'
    return s


# ── Segmenting for storyboard ─────────────────────────────────────────

# Aim ~5 seconds of speech per segment at 150 wpm = 12.5 words.
_SEG_WORDS_TARGET = 13


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _split_sentences(text: str) -> list[str]:
    """Split text on end-punctuation + whitespace + capital/digit/quote."""
    return [p.strip() for p in _SENTENCE_SPLIT.split(text or "") if p.strip()]


def segment_script(text: str, target_words: int = _SEG_WORDS_TARGET) -> list[str]:
    """Split ``text`` into segments of roughly ``target_words`` each.

    Segments preferentially break on sentence boundaries; a long
    sentence is split at commas or word boundaries when necessary.
    """
    out: list[str] = []
    for sentence in _split_sentences(text):
        words = sentence.split()
        if len(words) <= target_words + 4:
            out.append(sentence)
            continue
        # Long sentence: try commas first.
        parts = _split_on_commas(sentence, target_words)
        for p in parts:
            words = p.split()
            if len(words) <= target_words + 4:
                out.append(p)
            else:
                # Chunk by target words.
                for i in range(0, len(words), target_words):
                    out.append(" ".join(words[i : i + target_words]))
    return out


def _split_on_commas(sentence: str, target: int) -> list[str]:
    parts = re.split(r"(?<=,)\s+", sentence)
    grouped: list[str] = []
    buf: list[str] = []
    count = 0
    for p in parts:
        pw = p.split()
        if count and count + len(pw) > target + 4:
            grouped.append(" ".join(buf))
            buf = pw
            count = len(pw)
        else:
            buf.extend(pw)
            count += len(pw)
    if buf:
        grouped.append(" ".join(buf))
    return grouped


def segment_with_timestamps(
    text: str,
    wpm: int = 150,
    target_words: int = _SEG_WORDS_TARGET,
) -> list[dict]:
    """Return [{index, text, seconds, timestamp}, …] for the storyboard."""
    segments = segment_script(text, target_words=target_words)
    rows: list[dict] = []
    running = 0.0
    for idx, seg in enumerate(segments):
        dur = words_to_seconds(count_words(seg), wpm)
        row = {
            "index": idx,
            "text": seg,
            "seconds": round(running, 2),
            "duration": round(dur, 2),
            "timestamp": _format_timestamp(running),
        }
        rows.append(row)
        running += dur
    return rows


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


# ── Merge continuity ledger ───────────────────────────────────────────

def merge_ledger(existing: Iterable[str], *new_batches: Iterable[str]) -> list[str]:
    """Merge one or more new fact lists into an existing ledger, dedup."""
    seen: set[str] = set()
    out: list[str] = []

    def _push(f: str) -> None:
        key = re.sub(r"\s+", " ", f.strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(f.strip())

    for f in existing:
        _push(f)
    for batch in new_batches:
        for f in batch:
            _push(f)
    return out
