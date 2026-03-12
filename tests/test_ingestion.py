"""Tests for Phase 1 — ingestion pipeline."""
import uuid
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.chunker import chunk_document, infer_doc_type
from src.ingestion.indexer import FAISSIndex
from src.ingestion.parser import RawChunk, _clean, parse_document
from src.models import ChunkMetadata, DocumentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(content: str = "hello world", doc_id: str = "test_doc") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        content=content,
        metadata=ChunkMetadata(source="test.pdf", doc_type="faq", page=1),
        embedding=list(np.random.rand(768).astype(float)),
    )


def _make_raw(text: str = "Sample text.", source: str = "doc.pdf", page: int = 1) -> RawChunk:
    return RawChunk(text=text, source=source, page=page)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestClean:
    def test_strips_whitespace(self):
        assert _clean("  hello   world  ") == "hello world"

    def test_collapses_blank_lines(self):
        result = _clean("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strips_page_number_footer(self):
        text = "Some content\nPage 1 of 5"
        result = _clean(text)
        assert "Page 1 of 5" not in result

    def test_empty_string(self):
        assert _clean("") == ""


class TestParseDocument:
    def test_unsupported_extension_raises(self, tmp_path: Path):
        f = tmp_path / "file.docx"
        f.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            parse_document(f)

    def test_parses_markdown_no_headings(self, tmp_path: Path):
        f = tmp_path / "test.md"
        f.write_text("Hello world. This is a travel guide.")
        chunks = parse_document(f)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0].text
        assert chunks[0].page == 1

    def test_parses_markdown_with_headings(self, tmp_path: Path):
        f = tmp_path / "faq_test.md"
        f.write_text(
            "# Cancellations\nYou can cancel within 24h.\n\n## Refunds\nRefunds take 5 days."
        )
        chunks = parse_document(f)
        assert len(chunks) == 2
        assert chunks[0].section == "Cancellations"
        assert chunks[1].section == "Refunds"

    def test_parses_txt_file(self, tmp_path: Path):
        f = tmp_path / "info.txt"
        f.write_text("Travel info here.")
        chunks = parse_document(f)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TestInferDocType:
    @pytest.mark.parametrize("filename,expected", [
        ("faq_general.pdf", "faq"),
        ("policy_cancellation.pdf", "policy"),
        ("guide_visa.pdf", "guide"),
        ("dest_bali.pdf", "destination"),
        ("random_document.pdf", "general"),
        ("FAQ_UPPER.pdf", "faq"),  # case insensitive
    ])
    def test_infer(self, filename: str, expected: str):
        assert infer_doc_type(filename) == expected


class TestChunkDocument:
    def test_produces_chunks_with_metadata(self):
        raw = [_make_raw("This is a sentence. " * 50)]
        chunks = chunk_document(raw, "faq")
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata.doc_type == "faq"
            assert c.metadata.source == "doc.pdf"
            assert c.chunk_id

    def test_empty_raw_chunks(self):
        assert chunk_document([], "policy") == []

    def test_blank_text_skipped(self):
        raw = [_make_raw("   \n\n   ")]
        chunks = chunk_document(raw, "faq")
        assert chunks == []

    def test_section_propagated(self):
        raw = [RawChunk(
            text="Refund policy details here.",
            source="policy.pdf",
            page=2,
            section="Refunds",
        )]
        chunks = chunk_document(raw, "policy")
        assert all(c.metadata.section == "Refunds" for c in chunks)


# ---------------------------------------------------------------------------
# FAISS Indexer
# ---------------------------------------------------------------------------

class TestFAISSIndex:
    def test_add_and_search(self):
        index = FAISSIndex()
        chunk = _make_chunk("cancellation policy details")
        index.add([chunk])
        assert index.total_vectors == 1

        results = index.search(chunk.embedding, top_k=1)
        assert len(results) == 1
        assert results[0][0].chunk_id == chunk.chunk_id
        assert results[0][1] > 0.99  # near-perfect self-similarity

    def test_incremental_add_skips_existing_doc(self):
        index = FAISSIndex()
        c1 = _make_chunk("first doc", doc_id="doc_a")
        c2 = _make_chunk("second doc", doc_id="doc_a")  # same doc_id
        index.add([c1])
        index.add([c2])
        assert index.total_vectors == 1  # c2 skipped

    def test_incremental_add_new_doc(self):
        index = FAISSIndex()
        index.add([_make_chunk(doc_id="doc_a")])
        index.add([_make_chunk(doc_id="doc_b")])
        assert index.total_vectors == 2

    def test_save_and_load(self, tmp_path: Path):
        index = FAISSIndex()
        chunk = _make_chunk("booking information")
        index.add([chunk])
        index.save(tmp_path)

        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "index_meta.json").exists()

        loaded = FAISSIndex()
        loaded.load(tmp_path)
        assert loaded.total_vectors == 1
        assert "test_doc" in loaded.indexed_doc_ids

    def test_search_returns_empty_on_empty_index(self):
        index = FAISSIndex()
        results = index.search(list(np.random.rand(768)), top_k=5)
        assert results == []

    def test_add_raises_if_no_embedding(self):
        index = FAISSIndex()
        chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id="no_emb",
            content="text",
            metadata=ChunkMetadata(source="x.pdf", doc_type="faq", page=1),
            embedding=[],
        )
        with pytest.raises(ValueError, match="no embedding"):
            index.add([chunk])

    def test_load_missing_files_raises(self, tmp_path: Path):
        index = FAISSIndex()
        with pytest.raises(FileNotFoundError):
            index.load(tmp_path)
