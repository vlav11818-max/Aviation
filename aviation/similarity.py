"""Similarity checks that run before a script is considered final.

Three checks, all guardrails from Layer 5:

* **Title embedding similarity** — cosine between the candidate title
  and every recent title. Fail if any exceeds ``TITLE_MAX_COSINE``.
* **Opening embedding similarity** — same, but against the first ~500
  words of each recent manuscript. Fail on ``OPENING_MAX_COSINE``.
* **Noun-token overlap** — Jaccard-like overlap of capitalised /
  quoted nouns in the candidate title against each recent title. Fail
  on ``NOUN_MAX_OVERLAP``.

Embedding uses LiteLLM (whichever provider the caller configures). If
no embedding provider is configured, or ``AVIATION_FORCE_MOCK=1``, the
embedding steps fall back to a deterministic bag-of-words vector so
the pipeline still runs offline — with the important caveat that the
mock's numbers are indicative, not calibrated for real semantic
similarity. The noun-overlap check is always meaningful.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


TITLE_MAX_COSINE = 0.75
OPENING_MAX_COSINE = 0.80
NOUN_MAX_OVERLAP = 0.30
OPENING_WINDOW_WORDS = 500


# ── noun-token extraction (spaCy-free) ────────────────────────────────

_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-'']+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "with", "by", "from", "as", "is", "was", "were", "be", "been",
    "into", "over", "under", "after", "before", "during", "how",
    "why", "what", "who", "when", "where", "this", "that", "these",
    "those", "it", "its", "him", "his", "her", "them", "they",
    "our", "your", "my", "we", "you", "i",
    "so", "if", "yet", "than", "then", "not", "no",
}


def _noun_tokens(text: str) -> set[str]:
    """Rough 'proper-noun / content-noun' extraction.

    Keeps: capitalised tokens (proper nouns), digits inside numbers
    (flight 447 → 447), and any content word ≥ 5 chars that isn't a
    stopword.
    """
    out: set[str] = set()
    for m in _TOKEN.finditer(text or ""):
        w = m.group(0)
        if w[0].isupper() or (len(w) >= 5 and w.lower() not in _STOP):
            out.add(w.lower())
    for num in re.findall(r"\b\d{2,}\b", text or ""):
        out.add(num)
    return out


def noun_overlap(candidate: str, prior: str) -> float:
    """Jaccard-like overlap: |A ∩ B| / |A|, symmetric-friendly."""
    a = _noun_tokens(candidate)
    b = _noun_tokens(prior)
    if not a:
        return 0.0
    return len(a & b) / len(a)


# ── embeddings ────────────────────────────────────────────────────────


def _mock_embed(text: str, dim: int = 128) -> list[float]:
    """Deterministic bag-of-words hash embedding.

    Same word → same vector position. Coarse but good enough for
    smoke-testing and offline runs.
    """
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z]+", (text or "").lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if (h >> 31) & 1 else -1.0
        vec[idx] += sign
    # L2 normalise.
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def embed(text: str, model: str | None = None) -> list[float]:
    """Return an embedding vector for ``text``.

    * Uses LiteLLM's ``embedding`` when a real model is configured
      (default OpenAI ``text-embedding-3-small`` when
      ``OPENAI_API_KEY`` is set; provider prefix respected otherwise).
    * Falls back to a deterministic hash embedding when
      ``AVIATION_FORCE_MOCK=1`` or no key is configured for the model.
    """
    force_mock = os.environ.get("AVIATION_FORCE_MOCK", "").strip() == "1"
    if force_mock:
        return _mock_embed(text)
    picked = model or "text-embedding-3-small"
    # If we don't have any relevant key, mock instead of failing.
    env_key = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "text-embedding-3-small": "OPENAI_API_KEY",
        "text-embedding-3-large": "OPENAI_API_KEY",
    }.get(picked.split("/", 1)[0], "OPENAI_API_KEY")
    if not os.environ.get(env_key):
        return _mock_embed(text)
    try:
        import litellm  # type: ignore

        resp = litellm.embedding(model=picked, input=[text or ""])
        return list(resp["data"][0]["embedding"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding call failed (%s); falling back to mock: %s", picked, exc)
        return _mock_embed(text)


# ── checks ────────────────────────────────────────────────────────────


@dataclass
class SimilarityFinding:
    check: str
    status: str            # "pass" | "fail" | "warn"
    score: float
    against: str           # short identifier of the prior item
    detail: str = ""


@dataclass
class SimilarityReport:
    findings: list[SimilarityFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(f.status != "fail" for f in self.findings)

    def as_summary(self) -> str:
        if not self.findings:
            return "no similarity checks were run (no prior stories on record)"
        lines = []
        for f in self.findings:
            lines.append(f"[{f.status:4}] {f.check}: {f.score:.2f} vs {f.against!r} — {f.detail}")
        return "\n".join(lines)


def _first_words(text: str, n: int = OPENING_WINDOW_WORDS) -> str:
    tokens = (text or "").split()
    return " ".join(tokens[:n])


def check_title(
    candidate: str,
    priors: Iterable[str],
    *,
    max_cosine: float = TITLE_MAX_COSINE,
    max_noun_overlap: float = NOUN_MAX_OVERLAP,
    embed_model: str | None = None,
) -> SimilarityReport:
    report = SimilarityReport()
    priors = list(priors)
    if not priors:
        return report
    cand_emb = embed(candidate, model=embed_model)
    for prior in priors:
        cos = _cosine(cand_emb, embed(prior, model=embed_model))
        overlap = noun_overlap(candidate, prior)
        if cos >= max_cosine:
            report.findings.append(SimilarityFinding(
                check="title_embedding",
                status="fail",
                score=cos,
                against=prior[:60],
                detail=f"cosine ≥ {max_cosine}",
            ))
        if overlap >= max_noun_overlap:
            report.findings.append(SimilarityFinding(
                check="title_noun_overlap",
                status="fail",
                score=overlap,
                against=prior[:60],
                detail=f"noun overlap ≥ {int(max_noun_overlap * 100)} %",
            ))
    if not report.findings:
        report.findings.append(SimilarityFinding(
            check="title", status="pass", score=0.0, against="all recent", detail="clean",
        ))
    return report


def check_opening(
    manuscript: str,
    priors: Iterable[str],
    *,
    max_cosine: float = OPENING_MAX_COSINE,
    embed_model: str | None = None,
) -> SimilarityReport:
    report = SimilarityReport()
    priors = list(priors)
    if not priors:
        return report
    head = _first_words(manuscript)
    cand_emb = embed(head, model=embed_model)
    for prior in priors:
        cos = _cosine(cand_emb, embed(_first_words(prior), model=embed_model))
        if cos >= max_cosine:
            report.findings.append(SimilarityFinding(
                check="opening_embedding",
                status="fail",
                score=cos,
                against=prior[:60],
                detail=f"cosine ≥ {max_cosine}",
            ))
    if not report.findings:
        report.findings.append(SimilarityFinding(
            check="opening", status="pass", score=0.0, against="all recent", detail="clean",
        ))
    return report
