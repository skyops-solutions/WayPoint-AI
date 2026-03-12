# Phase 4 — Routing & Escalation Logic

## Goal
Detect user intent, compute confidence, and automatically route complex or ambiguous cases to human customer support. Log all escalations with full context.

## Deliverables
- `src/routing/intent_detector.py` — classify user message intent
- `src/routing/escalation.py` — escalation decision logic + webhook
- `src/routing/router.py` — orchestrates routing decisions
- `tests/test_routing.py`

## Tasks

### 4.1 Intent Detector
Classify each user message into one of the supported intents.

**Intent taxonomy:**
| Intent | Examples |
|--------|---------|
| `booking_inquiry` | "How do I book a flight to Rome?" |
| `cancellation` | "I want to cancel my trip" |
| `itinerary_change` | "Can I change my hotel dates?" |
| `destination_info` | "What's the weather like in Bali?" |
| `policy_question` | "What is your refund policy?" |
| `complaint` | "I'm very unhappy with my experience" |
| `refund_request` | "I want a refund for my cancelled tour" |
| `legal_dispute` | "I'm considering legal action" |
| `accessibility_need` | "I need wheelchair accessible options" |
| `human_request` | "Let me speak to a person" |
| `general` | Anything else |

**Implementation (MVP):** Use Gemini for intent classification. Single lightweight call with prompt:
```
Classify the following customer message into one of these intents: [list].
Respond with only the intent label.
Message: {message}
```

```python
async def detect_intent(message: str) -> str:
    """Returns intent label string."""
```

### 4.2 Escalation Decision Logic
Centralize all escalation rules:

```python
ALWAYS_ESCALATE_INTENTS = {"complaint", "legal_dispute", "human_request"}
CONDITIONAL_ESCALATE_INTENTS = {"refund_request", "accessibility_need"}

def should_escalate(
    intent: str,
    confidence: float,
    session: ChatSession,
    confidence_threshold: float = 0.6
) -> tuple[bool, str]:
    """
    Returns (should_escalate, reason).
    reason: 'low_confidence' | 'intent' | 'repeated_low_confidence' | 'user_request'
    """
```

Rules (in priority order):
1. If `intent` in `ALWAYS_ESCALATE_INTENTS` → escalate, reason=`intent`
2. If user message contains explicit request for human → escalate, reason=`user_request`
3. If `confidence < threshold` → escalate, reason=`low_confidence`
4. If last 3 bot turns in session all had `confidence < threshold` → escalate, reason=`repeated_low_confidence`
5. If `intent` in `CONDITIONAL_ESCALATE_INTENTS` AND `confidence < 0.75` → escalate, reason=`intent`

### 4.3 Escalation Webhook
When escalation is triggered, fire a POST to `HUMAN_SUPPORT_WEBHOOK`:

```python
async def fire_escalation(session: ChatSession, reason: str, intent: str) -> None:
```

Payload:
```json
{
  "session_id": "string",
  "timestamp": "ISO8601",
  "reason": "low_confidence | intent | ...",
  "detected_intent": "string",
  "transcript": [
    {"role": "user|assistant", "content": "string", "timestamp": "ISO8601"}
  ],
  "last_user_message": "string"
}
```

- Fire-and-forget (don't block response on webhook success)
- Log webhook failures — don't surface to user
- Set `escalate_to_human=true` in bot response regardless of webhook result

### 4.4 Router Orchestrator
```python
class Router:
    async def route(
        self,
        message: str,
        bot_response: BotResponse,
        session: ChatSession
    ) -> BotResponse:
        """
        Runs intent detection, applies escalation rules,
        mutates bot_response.escalate_to_human if needed,
        fires webhook if escalating.
        Returns updated BotResponse.
        """
```

The router runs **after** generation — it can override the LLM's own `escalate_to_human` field upward (never downward: if LLM sets it true, keep it true).

### 4.5 Escalation Response Message
When `escalate_to_human=true`, append to `answer`:
```
I'll connect you with one of our travel specialists who can assist you further.
You'll be reached shortly. Reference ID: {session_id[:8]}
```

## Acceptance Criteria
- [ ] All `ALWAYS_ESCALATE_INTENTS` trigger escalation 100% of the time
- [ ] `confidence < 0.6` triggers escalation
- [ ] 3-turn consecutive low-confidence rule fires correctly
- [ ] Webhook payload includes full transcript
- [ ] Webhook failure does not affect user-facing response
- [ ] Unit tests cover all 5 escalation rules + webhook failure handling

## Dependencies
```
google-genai>=1.0.0
httpx>=0.27          # async HTTP for webhook
```
