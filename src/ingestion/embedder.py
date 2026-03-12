"""Phase 1 — Embedder. See phases/phase_1_ingestion.md §1.3

Uses google-genai >= 1.0.0 async client.
Batches 100 chunks per call, retries on rate limits with exponential backoff,
and caches embeddings to disk by content hash to avoid re-embedding unchanged docs.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from google import genai

from src.config import settings
from src.models import DocumentChunk

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0          # seconds
_CACHE_FILE = Path("./data/embedding_cache.json")

# Lazy-initialised client
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, list[float]]:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            logger.warning("Embedding cache corrupt — starting fresh.")
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(cache))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

async def embed_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Embed all chunks in batches of 100. Fills chunk.embedding in place.

    Already-cached chunks (same content hash) are skipped.
    Returns the same list with embeddings filled.
    """
    cache = _load_cache()
    client = _get_client()

    # Separate cached vs. needs embedding
    to_embed: list[tuple[int, DocumentChunk]] = []
    for i, chunk in enumerate(chunks):
        h = _content_hash(chunk.content)
        if h in cache:
            chunk.embedding = cache[h]
        else:
            to_embed.append((i, chunk))

    logger.info(
        "%d chunks cached, %d to embed", len(chunks) - len(to_embed), len(to_embed)
    )

    # Process in batches
    for batch_start in range(0, len(to_embed), _BATCH_SIZE):
        batch = to_embed[batch_start : batch_start + _BATCH_SIZE]
        texts = [c.content for _, c in batch]

        embeddings = await _embed_with_retry(client, texts)

        for (_, chunk), vector in zip(batch, embeddings):
            chunk.embedding = vector
            cache[_content_hash(chunk.content)] = vector

        logger.info(
            "Embedded batch %d-%d / %d",
            batch_start + 1,
            batch_start + len(batch),
            len(to_embed),
        )

    _save_cache(cache)
    return chunks


async def _embed_with_retry(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Call embed_content with exponential backoff on rate-limit errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.aio.models.embed_content(
                model=settings.embedding_model,
                contents=texts,
            )
            return [emb.values for emb in response.embeddings]
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "quota" in str(exc).lower()
            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE ** attempt
                logger.warning("Rate limit hit, retrying in %.1fs (attempt %d)", wait, attempt + 1)
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("embed_with_retry exhausted retries")  # unreachable
