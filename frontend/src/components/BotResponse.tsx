import { useState } from "react";
import type { ChatResponse } from "../types";

interface Props {
  data: ChatResponse;
}

export function BotResponseCard({ data }: Props) {
  const [sourcesOpen, setSourcesOpen] = useState(false);

  return (
    <div className="space-y-2">
      {/* Escalation banner */}
      {data.escalate_to_human && (
        <div className="flex items-center gap-2 rounded-md bg-amber-50 border border-amber-300 px-3 py-2 text-sm text-amber-800">
          <span>🧑‍💼</span>
          <span>Speaking with a travel specialist shortly.</span>
        </div>
      )}

      {/* Answer */}
      <p className="text-sm text-gray-800 whitespace-pre-wrap">{data.answer}</p>

      {/* Book Now button */}
      {data.booking_link && (
        <a
          href={data.booking_link}
          target="_blank"
          rel="noreferrer"
          className="inline-block rounded-md bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
        >
          Book Now →
        </a>
      )}

      {/* Related services */}
      {data.related_services.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {data.related_services.map((svc) => (
            <span
              key={svc}
              className="rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-medium text-blue-700"
            >
              {svc}
            </span>
          ))}
        </div>
      )}

      {/* Sources — collapsible */}
      {data.sources.length > 0 && (
        <div>
          <button
            onClick={() => setSourcesOpen((o) => !o)}
            className="text-xs text-gray-400 hover:text-gray-600 underline"
          >
            {sourcesOpen ? "Hide sources" : `Sources (${data.sources.length})`}
          </button>
          {sourcesOpen && (
            <ul className="mt-1 space-y-0.5">
              {data.sources.map((s, i) => (
                <li key={i} className="text-xs text-gray-500">
                  📄 {s.doc} — p.{s.page}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
