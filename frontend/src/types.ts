export interface Source {
  doc: string;
  page: number;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  booking_link: string | null;
  related_services: string[];
  sources: Source[];
  confidence: number;
  escalate_to_human: boolean;
}

export interface Message {
  id: string;
  role: "user" | "bot";
  content: string;
  response?: ChatResponse;
}
