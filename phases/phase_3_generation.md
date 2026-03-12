# Phase 3 — LLM Response Generation

## Goal
Use retrieved context chunks and conversation history to generate structured, grounded responses via Gemini 2.5 Flash. Enforce the response schema and implement guardrails against hallucination.

## Deliverables
- `src/generation/prompt_builder.py` — builds final prompt from context + history
- `src/generation/llm_client.py` — async Gemini API wrapper
- `src/generation/response_parser.py` — validates and parses structured output
- `src/generation/generator.py` — orchestrates generation pipeline
- `tests/test_generation.py`

## Tasks

### 3.1 Prompt Builder
Assembles the full prompt passed to the LLM.

**System prompt** (static, loaded from `src/generation/prompts/system.txt`):
```
You are a helpful travel agency assistant. Answer customer questions using ONLY the provided context.
If the context does not support the answer, set confidence below 0.6 and escalate_to_human to true.
Never fabricate booking links, prices, or policy details.
Always respond with valid JSON matching the required schema.
```

**Context block**: format each chunk as:
```
[Source: {doc} | Page {page} | {doc_type}]
{content}
```

**History block**: last N turns formatted as:
```
User: {message}
Assistant: {answer}
```

**User message block**: current query.

```python
def build_prompt(
    query: str,
    retrieved_chunks: list[DocumentChunk],
    history: list[Turn],
    max_history_turns: int = 5
) -> str:
```

### 3.2 LLM Client
Async wrapper around `google-genai`:

```python
class GeminiClient:
    model: str = "gemini-2.5-flash"

    async def generate(
        self,
        prompt: str,
        response_schema: dict,     # JSON schema for structured output
        temperature: float = 0.2   # low temp for factual accuracy
    ) -> str:
```

- Set `temperature=0.2` for factual consistency
- Use Gemini's native structured output (`response_mime_type="application/json"`) where available
- Timeout: 10s; retry once on timeout before returning escalation fallback
- Log token usage per call

### 3.3 Response Schema (enforced)
```json
{
  "type": "object",
  "required": ["answer", "confidence", "escalate_to_human"],
  "properties": {
    "answer": { "type": "string" },
    "booking_link": { "type": ["string", "null"] },
    "related_services": {
      "type": "array",
      "items": { "type": "string" },
      "maxItems": 3
    },
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "doc": { "type": "string" },
          "page": { "type": "integer" }
        }
      }
    },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "escalate_to_human": { "type": "boolean" }
  }
}
```

### 3.4 Response Parser
- Parse JSON response from LLM
- Validate against schema using `jsonschema`
- If parsing fails:
  1. Retry generation once with added instruction: "Your previous response was not valid JSON. Respond only with JSON."
  2. If still fails → return hardcoded escalation fallback:
```python
FALLBACK_RESPONSE = BotResponse(
    answer="I'm having trouble processing your request right now. Let me connect you with our support team.",
    booking_link=None,
    related_services=[],
    sources=[],
    confidence=0.0,
    escalate_to_human=True
)
```

### 3.5 Generator Orchestrator
```python
class Generator:
    async def generate_response(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        session: ChatSession
    ) -> BotResponse:
```

Confidence override rule: if `retrieval_result.max_similarity < 0.4`, override `confidence` to min(llm_confidence, 0.4) and set `escalate_to_human=True`.

## Guardrails
- **No context, no answer**: if 0 chunks retrieved → escalate immediately
- **Source grounding check**: if `sources` in response is empty but `answer` is non-trivial → log as potential hallucination, cap confidence at 0.5
- **Booking link validation**: booking links must match pattern `^https?://` — strip or nullify invalid values

## Acceptance Criteria
- [ ] 100% of responses parse to valid `BotResponse` (with fallback on failure)
- [ ] `temperature=0.2` verified in all Gemini calls
- [ ] Escalation override fires correctly when `max_similarity < 0.4`
- [ ] Booking link guardrail strips non-URL values
- [ ] Unit tests: valid response, malformed JSON retry, timeout fallback, empty retrieval

## Dependencies
```
google-genai>=1.0.0
jsonschema>=4.0
pydantic>=2.0
```
