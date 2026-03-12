"""Phase 1 — Document Parser. See phases/phase_1_ingestion.md §1.1"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)

_SUPPORTED = {".pdf", ".md", ".txt"}

# Patterns that look like page headers/footers — stripped from text
_HEADER_FOOTER_RE = re.compile(
    r"^\s*(page\s+\d+\s*(of\s*\d+)?|©.+|\d+\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
# Heading detection in Markdown
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class RawChunk:
    text: str
    source: str       # filename only
    page: int         # 1-based
    section: str | None = None


def parse_document(path: Path) -> list[RawChunk]:
    """Parse a document and return one RawChunk per page/section.

    Supports: .pdf, .md, .txt
    """
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {_SUPPORTED}")

    if suffix == ".pdf":
        return _parse_pdf(path)
    return _parse_text(path)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _parse_pdf(path: Path) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    source = path.name

    with open(path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = _clean(raw)
            if not text:
                logger.debug("Empty page %d in %s — skipping", page_num, source)
                continue
            section = _detect_section_pdf(text)
            chunks.append(RawChunk(text=text, source=source, page=page_num, section=section))

    logger.info("Parsed %d pages from %s", len(chunks), source)
    return chunks


def _detect_section_pdf(text: str) -> str | None:
    """Return the first line that looks like a heading (short, title-case or all-caps)."""
    for line in text.splitlines():
        line = line.strip()
        if 3 < len(line) < 80 and (line.istitle() or line.isupper()):
            return line
    return None


# ---------------------------------------------------------------------------
# Markdown / plain text
# ---------------------------------------------------------------------------

def _parse_text(path: Path) -> list[RawChunk]:
    source = path.name
    content = path.read_text(encoding="utf-8", errors="replace")

    headings = list(_MD_HEADING_RE.finditer(content))
    if not headings:
        # No headings — treat the whole file as one chunk (page=1)
        return [RawChunk(text=_clean(content), source=source, page=1)]

    chunks: list[RawChunk] = []
    for i, match in enumerate(headings):
        section_title = match.group(2).strip()
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        text = _clean(content[start:end])
        if text:
            chunks.append(RawChunk(text=text, source=source, page=i + 1, section=section_title))

    logger.info("Parsed %d sections from %s", len(chunks), source)
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Normalize whitespace and strip common header/footer patterns."""
    text = _HEADER_FOOTER_RE.sub("", text)
    # collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
