"""Phase 2 — Reranker. See phases/phase_2_retrieval.md §2.3

Reranking strategy:
  final_score = 0.7 * vector_sim + 0.3 * recency_boost

recency_boost by doc_type (higher = preferred):
  policy / faq  → 1.0   (authoritative, frequently updated)
  guide         → 0.7
  destination   → 0.5
  general       → 0.8

Deduplication: if two chunks share the same source+page AND have
Jaccard word-overlap > 0.8, keep only the higher-scored one.
"""
from __future__ import annotations

from src.models import DocumentChunk

# Weights
_W_SIM = 0.7
_W_BOOST = 0.3

# Recency boost per doc_type (all values in [0, 1])
_RECENCY_BOOST: dict[str, float] = {
    "policy": 1.0,
    "faq": 1.0,
    "guide": 0.7,
    "destination": 0.5,
    "general": 0.8,
}
_DEFAULT_BOOST = 0.8

_DEDUP_OVERLAP_THRESHOLD = 0.8


def rerank(
    candidates: list[tuple[DocumentChunk, float]],
    top_n: int = 5,
) -> list[DocumentChunk]:
    """Rerank candidates and return top_n deduplicated chunks.

    Args:
        candidates: (chunk, vector_similarity_score) pairs from VectorSearch.
        top_n: Maximum number of chunks to return.

    Returns:
        List of chunks in descending relevance order.
    """
    if not candidates:
        return []

    # Compute final scores
    scored: list[tuple[DocumentChunk, float]] = []
    for chunk, sim in candidates:
        boost = _RECENCY_BOOST.get(chunk.metadata.doc_type, _DEFAULT_BOOST)
        final = _W_SIM * sim + _W_BOOST * boost
        scored.append((chunk, final))

    # Sort descending by final score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate near-duplicate chunks from the same page
    kept: list[DocumentChunk] = []
    for chunk, _ in scored:
        if not _is_duplicate(chunk, kept):
            kept.append(chunk)
        if len(kept) >= top_n:
            break

    return kept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union


def _is_duplicate(chunk: DocumentChunk, kept: list[DocumentChunk]) -> bool:
    """Return True if chunk is near-duplicate of any already-kept chunk."""
    for existing in kept:
        same_page = (
            existing.metadata.source == chunk.metadata.source
            and existing.metadata.page == chunk.metadata.page
        )
        if same_page and _jaccard(chunk.content, existing.content) > _DEDUP_OVERLAP_THRESHOLD:
            return True
    return False
