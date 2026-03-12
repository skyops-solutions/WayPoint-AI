import { useChat } from "../hooks/useChat";
import { InputBar } from "./InputBar";
import { MessageList } from "./MessageList";

export function ChatWidget() {
  const { messages, isLoading, error, send } = useChat();

  return (
    <div className="flex h-screen flex-col bg-gray-100">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-3 shadow-sm">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-blue-600 text-white text-lg">
          ✈️
        </div>
        <div>
          <p className="text-sm font-semibold text-gray-900">Travel Assistant</p>
          <p className="text-xs text-gray-400">Ask about bookings, destinations & more</p>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Messages */}
      <MessageList messages={messages} isLoading={isLoading} />

      {/* Input */}
      <InputBar onSend={send} disabled={isLoading} />
    </div>
  );
}
