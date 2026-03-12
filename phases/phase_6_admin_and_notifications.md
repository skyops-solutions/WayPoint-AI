# Phase 6 — Admin Dashboard & Telegram Escalation Notifications

## Goal

Two independent additions that complete the production loop:

1. **Admin Dashboard** — a protected `/admin` page in the React frontend where staff can monitor all conversations, filter escalations, and re-index documents via UI button.
2. **Telegram Notifications** — when `escalate_to_human: true`, the backend sends an instant Telegram message to a staff group/chat with full context so a human can follow up.

---

## 6.1 Telegram Escalation Notifications

### Why
When a complaint, legal dispute, or low-confidence case is detected, a human agent must be notified immediately. Telegram is the most practical channel — instant, mobile, free, no extra infra needed.

### What gets sent

When `escalate_to_human: true`, the bot sends a Telegram message to a configured chat:

```
🚨 Escalation — WayPoint AI

Reason: complaint
Session: a1b2c3d4
Time: 2025-03-12 14:32 UTC

User message:
"I'm very unhappy with the Rome trip, the hotel was terrible."

Bot answer:
"I'm sorry to hear that. I'm connecting you with a specialist (ref: a1b2c3d4)."

Confidence: 0.45
```

### New env variables

```
TELEGRAM_BOT_TOKEN=7123456789:AAF...   # from @BotFather
TELEGRAM_CHAT_ID=-1001234567890        # group chat ID (negative = group)
```

### Implementation

**File: `src/routing/telegram_notify.py`**

```python
async def notify_escalation(
    session_id: str,
    reason: str,
    user_message: str,
    bot_answer: str,
    confidence: float,
) -> None
```

- Uses `httpx.AsyncClient` to POST to `https://api.telegram.org/bot{token}/sendMessage`
- `parse_mode="HTML"` for formatting
- If `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID` is empty → skip silently (optional feature)
- Errors are caught and logged, never raise (must not break the chat response)

**Integration point: `src/routing/router.py`**

In `Router.route()`, after escalation is confirmed:
```python
if bot_response.escalate_to_human:
    try:
        await notify_escalation(...)
    except Exception:
        logger.warning("Telegram notification failed")
```

### Acceptance criteria
- [ ] Escalation message sent to Telegram within 2 seconds of the chat response
- [ ] No notification sent when `escalate_to_human: false`
- [ ] If `TELEGRAM_BOT_TOKEN` is not set, feature is silently disabled
- [ ] Telegram failure does not affect API response

---

## 6.2 Admin Dashboard (React)

### What it looks like

Protected by a password (admin token from `.env`). Single-page dashboard at `/admin`:

```
┌─────────────────────────────────────────────────────┐
│  WayPoint AI — Admin Dashboard          [Re-index]  │
├──────────┬───────────────────────────────────────────┤
│ Stats    │  Total: 142  │  Escalated: 12  │  Avg conf: 0.81 │
├──────────┴───────────────────────────────────────────┤
│  [All]  [Escalated only]         Search: [_______]  │
├─────────────────────────────────────────────────────┤
│ Time        │ Message (truncated)   │ Conf │ Escl   │
│ 14:32       │ I want to complain... │ 0.45 │  🔴    │
│ 14:28       │ What is the cancel... │ 0.92 │  ✅    │
│ 14:15       │ How much is Bali?...  │ 0.88 │  ✅    │
└─────────────────────────────────────────────────────┤
│  Click row → expand full conversation               │
└─────────────────────────────────────────────────────┘
```

### New backend endpoints

**File: `src/api/admin_routes.py`** — all require `Authorization: Bearer {ADMIN_TOKEN}`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/conversations` | Paginated conversation list. Query params: `page`, `limit`, `escalated_only`, `search` |
| `GET` | `/admin/conversations/{session_id}` | Full turn-by-turn history for one session |
| `GET` | `/admin/stats` | `{ total, escalated, avg_confidence, escalation_rate }` |

Response shape for `/admin/conversations`:
```json
{
  "total": 142,
  "page": 1,
  "items": [
    {
      "session_id": "a1b2c3d4...",
      "message": "I want to complain...",
      "answer": "I'm sorry to hear...",
      "confidence": 0.45,
      "escalated": true,
      "created_at": "2025-03-12T14:32:00Z"
    }
  ]
}
```

These read directly from the `conversations` SQLite table via `aiosqlite`.

### Frontend

**New files:**
- `frontend/src/pages/AdminPage.tsx` — main dashboard page
- `frontend/src/components/admin/ConversationTable.tsx` — sortable table
- `frontend/src/components/admin/ConversationDetail.tsx` — expanded row / modal
- `frontend/src/components/admin/StatsBar.tsx` — top stats strip
- `frontend/src/hooks/useAdmin.ts` — fetch wrapper with admin token header

**Routing:** Add React Router. `/` → ChatWidget, `/admin` → AdminPage (redirects to login if no token in `localStorage`).

**Login:** Simple token input form — stores token in `localStorage["admin_token"]`, no backend call needed (first real API call will fail with 401 if wrong).

**Re-index button:** Calls `POST /ingest` with admin token. Shows spinner → success/error toast.

### Acceptance criteria
- [ ] `/admin` redirects to login form if no token stored
- [ ] Wrong token → 401 → back to login
- [ ] Table shows all conversations, newest first
- [ ] "Escalated only" filter works
- [ ] Search filters by message content (frontend-side on loaded data)
- [ ] Click row → expands to show full answer + sources
- [ ] Re-index button calls `/ingest` and shows result
- [ ] Stats bar shows correct totals

---

## Dependencies to add

```
# requirements.txt — no new deps needed (httpx already included)

# frontend/package.json
react-router-dom        # routing for /admin page
```

---

## Implementation prompt

Use this prompt to implement Phase 6:

---

```
Read phases/phase_6_admin_and_notifications.md and implement it fully.

Part 1 — Telegram notifications:
- Create src/routing/telegram_notify.py with async notify_escalation()
- Use httpx.AsyncClient to POST to Telegram Bot API
- Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to src/config.py (empty string defaults)
- Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env.example
- Wire notify_escalation() into src/routing/router.py after escalation is confirmed
- Errors must be caught silently — never affect the API response

Part 2 — Admin API:
- Create src/api/admin_routes.py with GET /admin/conversations, GET /admin/conversations/{id}, GET /admin/stats
- All endpoints require HTTPBearer admin token (reuse existing auth pattern from /ingest)
- Read from SQLite (data/conversations.db) via aiosqlite
- Register the router in src/api/main.py

Part 3 — Admin frontend:
- Install react-router-dom in frontend/
- Add routing: / → ChatWidget, /admin → AdminPage
- Create frontend/src/pages/AdminPage.tsx
- Create frontend/src/components/admin/ConversationTable.tsx
- Create frontend/src/components/admin/StatsBar.tsx
- Create frontend/src/hooks/useAdmin.ts
- Admin token stored in localStorage, passed as Bearer header
- Re-index button calls POST /ingest

Write tests for the new backend endpoints in tests/test_admin.py.
Follow all conventions in CLAUDE.md.
```
