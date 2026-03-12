"""Phase 2 — Retriever Orchestrator. See phases/phase_2_retrieval.md §2.4 & §2.5"""
from __future__ import annotations

import logging

from src.config import settings
from src.ingestion.indexer import FAISSIndex
from src.models import RetrievalResult
from src.retrieval.query_embedder import embed_query
from src.retrieval.reranker import rerank
from src.retrieval.vector_search import VectorSearch

logger = logging.getLogger(__name__)

# §2.5 — Intent → doc_type filter mapping
_INTENT_TO_DOC_TYPE: dict[str, str] = {
    "cancellation": "policy",
    "destination_info": "destination",
    "booking_help": "faq",
    "policy_question": "policy",
}


class Retriever:
    def __init__(self, index: FAISSIndex) -> None:
        self._search = VectorSearch(index)

    async def retrieve(
        self,
        query: str,
        doc_type_hint: str | None = None,
    ) -> RetrievalResult:
        """Full retrieval pipeline: embed → search → rerank → return top-N.

        Args:
            query:          Raw user message.
            doc_type_hint:  Intent label from routing layer (optional).
                            Mapped to a doc_type filter when provided.

        Returns:
            RetrievalResult with top-N ranked chunks, max_similarity, and query embedding.
        """
        query_vector = await embed_query(query)

        doc_type_filter = _resolve_doc_type_filter(doc_type_hint)

        candidates = self._search.search(
            query_vector=query_vector,
            top_k=settings.retrieval_top_k,
            doc_type_filter=doc_type_filter,
        )

        max_similarity = candidates[0][1] if candidates else 0.0

        ranked_chunks = rerank(candidates, top_n=settings.retrieval_top_n)

        logger.info(
            "Retrieved %d chunks (max_sim=%.3f, filter=%s) for query: %.60s",
            len(ranked_chunks),
            max_similarity,
            doc_type_filter or "none",
            query,
        )

        return RetrievalResult(
            chunks=ranked_chunks,
            max_similarity=max_similarity,
            query_embedding=query_vector,
        )


def _resolve_doc_type_filter(doc_type_hint: str | None) -> str | None:
    """Map an intent hint to a doc_type filter string, or None to search all."""
    if doc_type_hint is None:
        return None
    return _INTENT_TO_DOC_TYPE.get(doc_type_hint)
