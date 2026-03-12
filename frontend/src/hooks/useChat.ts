import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatResponse, Message } from "../types";

const API_URL = import.meta.env.VITE_API_URL ?? "";

async function createSession(): Promise<string> {
  const res = await fetch(`${API_URL}/chat/session`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to create session");
  const data = await res.json();
  return data.session_id as string;
}

async function sendMessage(
  sessionId: string,
  message: string
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const initRef = useRef(false);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    const stored = sessionStorage.getItem("chat_session_id");
    if (stored) {
      setSessionId(stored);
      return;
    }
    createSession()
      .then((id) => {
        sessionStorage.setItem("chat_session_id", id);
        setSessionId(id);
      })
      .catch(() => setError("Could not connect to the server."));
  }, []);

  const send = useCallback(
    async (text: string) => {
      if (!sessionId || !text.trim() || isLoading) return;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: text.trim(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setError(null);

      try {
        const response = await sendMessage(sessionId, text.trim());
        const botMsg: Message = {
          id: crypto.randomUUID(),
          role: "bot",
          content: response.answer,
          response,
        };
        setMessages((prev) => [...prev, botMsg]);
      } catch {
        setError("Something went wrong. Please try again.");
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, isLoading]
  );

  return { messages, isLoading, error, send, sessionId };
}
