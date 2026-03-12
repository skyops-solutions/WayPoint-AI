"""Phase 2 — Vector Search. See phases/phase_2_retrieval.md §2.2

Wraps FAISSIndex.search() with:
  - similarity threshold filtering (drops score < 0.35)
  - optional doc_type post-filter
  - fetches extra candidates when filtering is active so top_k results survive
"""
from __future__ import annotations

import logging

from src.config import settings
from src.ingestion.indexer import FAISSIndex
from src.models import DocumentChunk

logger = logging.getLogger(__name__)

# How many extra candidates to fetch when doc_type filter is active,
# to compensate for chunks dropped by the filter.
_FILTER_OVERSAMPLE = 4


class VectorSearch:
    def __init__(self, index: FAISSIndex) -> None:
        self._index = index

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        doc_type_filter: str | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search the FAISS index and return (chunk, score) pairs.

        Steps:
          1. Fetch top_k candidates (more if filter is active).
          2. Drop candidates below similarity_threshold.
          3. Apply doc_type post-filter if requested.
          4. Return up to top_k results ordered by descending score.
        """
        fetch_k = top_k * _FILTER_OVERSAMPLE if doc_type_filter else top_k
        candidates = self._index.search(query_vector, top_k=fetch_k)

        # 1. Threshold filter
        candidates = [
            (chunk, score)
            for chunk, score in candidates
            if score >= settings.similarity_threshold
        ]

        # 2. doc_type filter
        if doc_type_filter:
            candidates = [
                (chunk, score)
                for chunk, score in candidates
                if chunk.metadata.doc_type == doc_type_filter
            ]

        result = candidates[:top_k]
        logger.debug(
            "VectorSearch: fetch_k=%d → after threshold+filter=%d → returned=%d",
            fetch_k,
            len(candidates),
            len(result),
        )
        return result
