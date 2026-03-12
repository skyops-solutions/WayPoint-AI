"""Phase 1 — Text Chunker. See phases/phase_1_ingestion.md §1.2"""
from __future__ import annotations

import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingestion.parser import RawChunk
from src.models import ChunkMetadata, DocumentChunk

# Filename prefix → doc_type mapping
_DOC_TYPE_PREFIXES: dict[str, str] = {
    "faq_": "faq",
    "policy_": "policy",
    "guide_": "guide",
    "dest_": "destination",
}
_DEFAULT_DOC_TYPE = "general"

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def infer_doc_type(filename: str) -> str:
    """Infer doc_type from filename prefix convention."""
    name = Path(filename).name.lower()
    for prefix, doc_type in _DOC_TYPE_PREFIXES.items():
        if name.startswith(prefix):
            return doc_type
    return _DEFAULT_DOC_TYPE


def chunk_document(raw_chunks: list[RawChunk], doc_type: str) -> list[DocumentChunk]:
    """Split RawChunks into DocumentChunks with full metadata.

    Each RawChunk (one per page/section) is further split into token-sized pieces.
    The source doc_id is derived from the source filename.
    """
    result: list[DocumentChunk] = []

    for raw in raw_chunks:
        doc_id = Path(raw.source).stem
        sub_texts = _splitter.split_text(raw.text)

        for sub_text in sub_texts:
            text = sub_text.strip()
            if not text:
                continue
            result.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=text,
                    metadata=ChunkMetadata(
                        source=raw.source,
                        doc_type=doc_type,
                        page=raw.page,
                        section=raw.section,
                    ),
                )
            )

    return result
