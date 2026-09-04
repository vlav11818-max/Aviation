"""PDF ingestion for the aviation content factory.

Reads NTSB / AAIB / BEA / TSB accident-report PDFs (or plain-text
fallbacks) and returns a cleaned text stream, a per-page map, and
convenient chunk windows for the LLM ingest step.

PyMuPDF (``pymupdf``) is the primary parser — fast, handles most
layouts well including multi-column reports. ``pdfplumber`` is used
as a fallback for scans that PyMuPDF fails on. If neither can
extract usable text (e.g. a fully rasterised scan), the caller is
told to run OCR upstream — this module does not embed an OCR engine
on purpose (keeps the dependency surface small).

Usage::

    doc = load_pdf(Path("AF447_final_report.pdf"))
    for chunk in doc.chunks(max_chars=18_000):
        facts = call_llm_extract(chunk)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Approximate chunk size for the LLM ingest call. Big enough to cover
# a chapter of a typical report; small enough that a 4k-context model
# still has room for the extraction prompt + JSON output.
DEFAULT_CHUNK_CHARS = 18_000


# ── Simple text-cleaning helpers ──────────────────────────────────────

_MULTI_WS = re.compile(r"[ \t]+")
_NUL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HEADER_RUN = re.compile(r"\n(?:[A-Z][A-Z0-9\- ]{4,}\n){3,}", re.M)  # repeating shouty headers
_SOFT_HYPHEN = re.compile(r"­")
_END_HYPHEN = re.compile(r"([A-Za-z])-\n([a-z])")


def _clean(text: str) -> str:
    """Normalise whitespace and de-hyphenate line-broken words."""
    text = _NUL.sub("", text)
    text = _SOFT_HYPHEN.sub("", text)
    text = _END_HYPHEN.sub(r"\1\2", text)
    text = _MULTI_WS.sub(" ", text)
    # Collapse >2 blank lines to 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Data holders ──────────────────────────────────────────────────────


@dataclass
class Page:
    """One extracted page."""

    index: int
    text: str


@dataclass
class PDFDocument:
    """The parsed report ready to be sliced into ingest chunks."""

    path: Path
    filename: str
    pages: list[Page] = field(default_factory=list)
    parser: str = ""  # 'pymupdf' | 'pdfplumber' | 'text'

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def chunks(self, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
        """Split the document into ordered ~``max_chars`` chunks.

        Chunks respect paragraph boundaries when possible and never
        split mid-page unless a single page is bigger than ``max_chars``.
        """
        buf: list[str] = []
        current = 0
        chunks: list[str] = []
        for page in self.pages:
            page_text = page.text.strip()
            if not page_text:
                continue
            if current + len(page_text) + 2 > max_chars and buf:
                chunks.append("\n\n".join(buf))
                buf = []
                current = 0
            if len(page_text) > max_chars:
                # Big page: split at paragraph boundaries.
                paras = page_text.split("\n\n")
                acc: list[str] = []
                for para in paras:
                    if current + len(para) + 2 > max_chars and acc:
                        buf.extend(acc)
                        chunks.append("\n\n".join(buf))
                        buf = []
                        current = 0
                        acc = []
                    acc.append(para)
                    current += len(para) + 2
                if acc:
                    buf.extend(acc)
            else:
                buf.append(page_text)
                current += len(page_text) + 2
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks


# ── Parsers ───────────────────────────────────────────────────────────


def _read_pymupdf(path: Path) -> list[Page]:
    import pymupdf  # type: ignore

    pages: list[Page] = []
    doc = pymupdf.open(path)
    try:
        for i in range(len(doc)):
            raw = doc[i].get_text("text") or ""
            pages.append(Page(index=i, text=_clean(raw)))
    finally:
        doc.close()
    return pages


def _read_pdfplumber(path: Path) -> list[Page]:
    import pdfplumber  # type: ignore

    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            pages.append(Page(index=i, text=_clean(raw)))
    return pages


def _read_text(path: Path) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [Page(index=0, text=_clean(raw))]


# ── Public API ────────────────────────────────────────────────────────


def load_pdf(path: Path | str) -> PDFDocument:
    """Load a PDF (or a plain-text fallback) and return a parsed document.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If both parsers produce empty text (likely a scan
            that needs OCR — we don't ship OCR).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    suffix = p.suffix.lower()

    if suffix in {".txt", ".md"}:
        pages = _read_text(p)
        return PDFDocument(path=p, filename=p.name, pages=pages, parser="text")

    parser = ""
    pages: list[Page] = []
    try:
        pages = _read_pymupdf(p)
        parser = "pymupdf"
    except Exception as exc:
        logger.warning("PyMuPDF failed on %s: %s. Falling back to pdfplumber.", p.name, exc)

    total = sum(len(pg.text) for pg in pages)
    if total < 200:  # basically nothing extracted — try the other parser
        try:
            pages = _read_pdfplumber(p)
            parser = "pdfplumber"
        except Exception as exc:
            logger.warning("pdfplumber also failed on %s: %s.", p.name, exc)

    total = sum(len(pg.text) for pg in pages)
    if total < 200:
        raise ValueError(
            f"Could not extract meaningful text from {p.name}. It is likely a "
            "rasterised scan; run OCR upstream (e.g. `ocrmypdf`) and try again."
        )

    return PDFDocument(path=p, filename=p.name, pages=pages, parser=parser)


def load_many(paths: Iterable[Path | str]) -> list[PDFDocument]:
    """Load a list of PDF paths, skipping ones that fail parsing."""
    out: list[PDFDocument] = []
    for path in paths:
        try:
            out.append(load_pdf(path))
        except Exception as exc:
            logger.error("Skipping %s: %s", path, exc)
    return out
