"""Tests for Phase 2 — semantic retrieval layer.

Uses a real FAISS index with random embeddings (as per CLAUDE.md conventions).
No API calls — query_embedder is patched to return a deterministic vector.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.ingestion.indexer import FAISSIndex
from src.models import ChunkMetadata, DocumentChunk, RetrievalResult
from src.retrieval.reranker import _jaccard, _is_duplicate, rerank
from src.retrieval.vector_search import VectorSearch

DIM = 768


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = DIM) -> list[float]:
    v = np.random.rand(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_chunk(
    content: str = "travel info",
    doc_id: str = "doc_a",
    doc_type: str = "faq",
    page: int = 1,
    source: str = "faq_test.pdf",
    embedding: list[float] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        content=content,
        metadata=ChunkMetadata(source=source, doc_type=doc_type, page=page),
        embedding=embedding or _unit_vec(),
    )


def _build_index(chunks: list[DocumentChunk]) -> FAISSIndex:
    index = FAISSIndex()
    index.add(chunks)
    return index


# ---------------------------------------------------------------------------
# VectorSearch
# ---------------------------------------------------------------------------

class TestVectorSearch:
    def test_returns_results_above_threshold(self):
        chunk = _make_chunk()
        index = _build_index([chunk])
        search = VectorSearch(index)

        results = search.search(chunk.embedding, top_k=5)
        assert len(results) == 1
        assert results[0][1] > 0.99  # self-similarity ≈ 1.0

    def test_filters_below_threshold(self):
        chunk = _make_chunk()
        index = _build_index([chunk])
        search = VectorSearch(index)

        # Orthogonal vector → near-zero similarity
        orthogonal = _unit_vec()
        results = search.search(orthogonal, top_k=5)
        # May or may not return; if returned, score must be >= threshold
        for _, score in results:
            assert score >= 0.35

    def test_empty_index_returns_empty(self):
        index = FAISSIndex()
        search = VectorSearch(index)
        assert search.search(_unit_vec(), top_k=5) == []

    def test_doc_type_filter(self):
        faq_chunk = _make_chunk(doc_type="faq", doc_id="faq_doc")
        policy_chunk = _make_chunk(doc_type="policy", doc_id="policy_doc")
        index = _build_index([faq_chunk, policy_chunk])
        search = VectorSearch(index)

        results = search.search(faq_chunk.embedding, top_k=10, doc_type_filter="policy")
        assert all(c.metadata.doc_type == "policy" for c, _ in results)

    def test_no_filter_returns_all_types(self):
        chunks = [
            _make_chunk(doc_type="faq", doc_id="a"),
            _make_chunk(doc_type="policy", doc_id="b"),
            _make_chunk(doc_type="destination", doc_id="c"),
        ]
        index = _build_index(chunks)
        search = VectorSearch(index)
        query = _unit_vec()
        results = search.search(query, top_k=10)
        doc_types = {c.metadata.doc_type for c, _ in results}
        assert len(doc_types) > 1

    def test_respects_top_k(self):
        chunks = [_make_chunk(doc_id=f"doc_{i}") for i in range(10)]
        index = _build_index(chunks)
        search = VectorSearch(index)
        results = search.search(_unit_vec(), top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical(self):
        assert _jaccard("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert _jaccard("hello world", "foo bar baz") == 0.0

    def test_partial(self):
        score = _jaccard("hello world foo", "hello world bar")
        assert 0.0 < score < 1.0

    def test_empty_string(self):
        assert _jaccard("", "hello") == 0.0


class TestIsDuplicate:
    def test_same_page_high_overlap_is_duplicate(self):
        base = _make_chunk(
            content="cancellation policy refund details here",
            source="policy.pdf", page=3,
        )
        duplicate = _make_chunk(
            content="cancellation policy refund details here",   # identical
            source="policy.pdf", page=3,
        )
        assert _is_duplicate(duplicate, [base])

    def test_different_page_not_duplicate(self):
        a = _make_chunk(content="same content same content", source="policy.pdf", page=1)
        b = _make_chunk(content="same content same content", source="policy.pdf", page=2)
        assert not _is_duplicate(b, [a])

    def test_different_source_not_duplicate(self):
        a = _make_chunk(content="same content same content", source="a.pdf", page=1)
        b = _make_chunk(content="same content same content", source="b.pdf", page=1)
        assert not _is_duplicate(b, [a])


class TestRerank:
    def test_returns_top_n(self):
        chunks = [(_make_chunk(doc_id=f"d{i}"), 0.9 - i * 0.05) for i in range(10)]
        result = rerank(chunks, top_n=5)
        assert len(result) <= 5

    def test_empty_input(self):
        assert rerank([], top_n=5) == []

    def test_policy_ranked_above_destination_at_equal_sim(self):
        policy = _make_chunk(doc_type="policy", doc_id="p")
        destination = _make_chunk(doc_type="destination", doc_id="d")
        # Both with same vector similarity
        result = rerank([(policy, 0.8), (destination, 0.8)], top_n=2)
        assert result[0].metadata.doc_type == "policy"

    def test_deduplication_removes_near_duplicate(self):
        text = "visa requirements documents passport needed " * 5
        c1 = _make_chunk(content=text, source="guide.pdf", page=1, doc_id="g1")
        c2 = _make_chunk(content=text, source="guide.pdf", page=1, doc_id="g2")
        result = rerank([(c1, 0.9), (c2, 0.85)], top_n=5)
        assert len(result) == 1  # duplicate removed

    def test_ordering_descending(self):
        chunks = [
            (_make_chunk(doc_type="faq", doc_id="a"), 0.5),
            (_make_chunk(doc_type="faq", doc_id="b"), 0.9),
            (_make_chunk(doc_type="faq", doc_id="c"), 0.7),
        ]
        result = rerank(chunks, top_n=3)
        assert result[0].doc_id == "b"


# ---------------------------------------------------------------------------
# Retriever (integration — query_embedder patched)
# ---------------------------------------------------------------------------

class TestRetriever:
    @pytest.mark.anyio
    async def test_retrieve_returns_result(self):
        from src.retrieval.retriever import Retriever

        chunk = _make_chunk(content="flight booking cancellation policy")
        index = _build_index([chunk])

        with patch(
            "src.retrieval.retriever.embed_query",
            new=AsyncMock(return_value=chunk.embedding),
        ):
            retriever = Retriever(index)
            result = await retriever.retrieve("how do I cancel my flight?")

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) >= 1
        assert result.max_similarity > 0.0
        assert len(result.query_embedding) == DIM

    @pytest.mark.anyio
    async def test_retrieve_empty_index_returns_zero_sim(self):
        from src.retrieval.retriever import Retriever

        index = FAISSIndex()
        with patch(
            "src.retrieval.retriever.embed_query",
            new=AsyncMock(return_value=_unit_vec()),
        ):
            retriever = Retriever(index)
            result = await retriever.retrieve("anything")

        assert result.chunks == []
        assert result.max_similarity == 0.0

    @pytest.mark.anyio
    async def test_retrieve_with_intent_hint_filters(self):
        from src.retrieval.retriever import Retriever

        policy_chunk = _make_chunk(doc_type="policy", doc_id="pol", content="refund policy details")
        faq_chunk = _make_chunk(doc_type="faq", doc_id="faq", content="refund policy details")
        index = _build_index([policy_chunk, faq_chunk])

        with patch(
            "src.retrieval.retriever.embed_query",
            new=AsyncMock(return_value=policy_chunk.embedding),
        ):
            retriever = Retriever(index)
            result = await retriever.retrieve("refund?", doc_type_hint="cancellation")

        assert all(c.metadata.doc_type == "policy" for c in result.chunks)
