import { useCallback, useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "";
const TOKEN_KEY = "admin_token";

export function useAdminToken() {
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem(TOKEN_KEY)
  );

  const saveToken = useCallback((t: string) => {
    localStorage.setItem(TOKEN_KEY, t);
    setToken(t);
  }, []);

  const clearToken = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
  }, []);

  return { token, saveToken, clearToken };
}

function authHeaders(token: string) {
  return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
}

export function useAdminStats(token: string | null) {
  const [stats, setStats] = useState<null | {
    total_conversations: number;
    total_turns: number;
    escalated_count: number;
    avg_confidence: number;
  }>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_URL}/admin/stats`, { headers: authHeaders(token) });
      if (!res.ok) throw new Error(String(res.status));
      setStats(await res.json());
    } catch (e: unknown) {
      setError(String(e));
    }
  }, [token]);

  useEffect(() => { load(); }, [load]);
  return { stats, error, reload: load };
}

export function useConversations(
  token: string | null,
  page: number,
  escalatedOnly: boolean,
  search: string
) {
  const [data, setData] = useState<null | {
    items: ConversationSummary[];
    total: number;
    page: number;
    page_size: number;
  }>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    const params = new URLSearchParams({
      page: String(page),
      page_size: "20",
      escalated_only: String(escalatedOnly),
      ...(search ? { search } : {}),
    });
    try {
      const res = await fetch(`${API_URL}/admin/conversations?${params}`, {
        headers: authHeaders(token),
      });
      if (res.ok) setData(await res.json());
    } finally {
      setLoading(false);
    }
  }, [token, page, escalatedOnly, search]);

  useEffect(() => { load(); }, [load]);
  return { data, loading, reload: load };
}

export function useConversationDetail(token: string | null, sessionId: string | null) {
  const [detail, setDetail] = useState<null | ConversationDetail>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!token || !sessionId) { setDetail(null); return; }
    setLoading(true);
    fetch(`${API_URL}/admin/conversations/${sessionId}`, { headers: authHeaders(token) })
      .then((r) => r.json())
      .then(setDetail)
      .finally(() => setLoading(false));
  }, [token, sessionId]);

  return { detail, loading };
}

export interface ConversationSummary {
  session_id: string;
  first_seen: string;
  last_seen: string;
  turn_count: number;
  escalated: boolean;
  last_message: string;
}

export interface AdminTurn {
  timestamp: string;
  user_message: string;
  bot_answer: string;
  confidence: number | null;
  escalated: boolean;
}

export interface ConversationDetail {
  session_id: string;
  turns: AdminTurn[];
}
