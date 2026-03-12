"""Phase 2 — Query Embedder. See phases/phase_2_retrieval.md §2.1

Embeds a user query via google-genai and L2-normalises the vector.
Maintains an in-process cache so the same query is never embedded twice
within the lifetime of the server process.
"""
from __future__ import annotations

import logging

import numpy as np
from google import genai

from src.config import settings

logger = logging.getLogger(__name__)

# In-process cache: query text → normalised vector
_cache: dict[str, list[float]] = {}

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


async def embed_query(query: str) -> list[float]:
    """Embed a single query string and return an L2-normalised vector.

    Uses an in-process cache to avoid re-embedding identical queries.
    """
    if query in _cache:
        logger.debug("Query embedding cache hit.")
        return _cache[query]

    client = _get_client()
    response = await client.aio.models.embed_content(
        model=settings.embedding_model,
        contents=[query],
    )
    vector = np.array(response.embeddings[0].values, dtype=np.float32)
    # L2-normalise so cosine similarity == inner product (required for IndexFlatIP)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    result = vector.tolist()
    _cache[query] = result
    return result
