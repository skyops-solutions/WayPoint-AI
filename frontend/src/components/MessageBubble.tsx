import type { Message } from "../types";
import { BotResponseCard } from "./BotResponse";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-blue-600 px-4 py-2.5 text-sm text-white shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-2xl rounded-tl-sm bg-white px-4 py-3 shadow-sm border border-gray-100">
        {message.response ? (
          <BotResponseCard data={message.response} />
        ) : (
          <p className="text-sm text-gray-800">{message.content}</p>
        )}
      </div>
    </div>
  );
}
