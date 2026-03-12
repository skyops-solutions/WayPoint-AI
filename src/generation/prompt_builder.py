"""Phase 3 — Prompt Builder. See phases/phase_3_generation.md §3.1"""
from __future__ import annotations

from pathlib import Path

from src.models import DocumentChunk, Turn

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"
_SYSTEM_PROMPT: str | None = None


def _load_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return _SYSTEM_PROMPT


def build_prompt(
    query: str,
    retrieved_chunks: list[DocumentChunk],
    history: list[Turn],
    max_history_turns: int = 5,
) -> str:
    """Assemble the full prompt: system + context + history + user query."""
    parts: list[str] = []

    parts.append(_load_system_prompt())
    parts.append("")

    # Context block
    if retrieved_chunks:
        parts.append("[CONTEXT]")
        for chunk in retrieved_chunks:
            m = chunk.metadata
            parts.append(f"[Source: {m.source} | Page {m.page} | {m.doc_type}]")
            parts.append(chunk.content)
            parts.append("")
    else:
        parts.append("[CONTEXT]\nNo relevant documents were found.\n")

    # History block — last N turns (user + assistant interleaved)
    recent = history[-(max_history_turns * 2):]
    if recent:
        parts.append("[CONVERSATION HISTORY]")
        for turn in recent:
            label = "User" if turn.role == "user" else "Assistant"
            parts.append(f"{label}: {turn.content}")
        parts.append("")

    # User query
    parts.append("[USER QUESTION]")
    parts.append(query)

    return "\n".join(parts)
