import { useState } from "react";
import { AdminLogin } from "./AdminLogin";
import {
  useAdminToken,
  useAdminStats,
  useConversations,
  useConversationDetail,
  type ConversationSummary,
} from "../hooks/useAdmin";

const API_URL = import.meta.env.VITE_API_URL ?? "";

export function AdminPanel() {
  const { token, saveToken, clearToken } = useAdminToken();
  const [page, setPage] = useState(1);
  const [escalatedOnly, setEscalatedOnly] = useState(false);
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [reindexStatus, setReindexStatus] = useState<string | null>(null);

  const { stats, reload: reloadStats } = useAdminStats(token);
  const { data, loading, reload: reloadList } = useConversations(token, page, escalatedOnly, search);
  const { detail, loading: detailLoading } = useConversationDetail(token, expandedId);

  if (!token) return <AdminLogin onSuccess={saveToken} />;

  const totalPages = data ? Math.ceil(data.total / data.page_size) : 1;

  const handleReindex = async () => {
    setReindexStatus("Running...");
    try {
      const res = await fetch(`${API_URL}/ingest`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      const json = await res.json();
      setReindexStatus(
        res.ok
          ? `Done — ${json.docs_indexed} doc(s), ${json.chunks_indexed} chunk(s)`
          : `Error: ${json.detail}`
      );
      reloadStats();
    } catch {
      setReindexStatus("Failed to connect.");
    }
    setTimeout(() => setReindexStatus(null), 5000);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSearch(searchInput);
    setPage(1);
  };

  const toggleRow = (id: string) =>
    setExpandedId((prev) => (prev === id ? null : id));

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-900">WayPoint AI — Admin</h1>
          <p className="text-xs text-gray-400">Conversation Dashboard</p>
        </div>
        <div className="flex items-center gap-3">
          {reindexStatus && (
            <span className="text-xs text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
              {reindexStatus}
            </span>
          )}
          <button
            onClick={handleReindex}
            className="bg-blue-600 text-white text-sm px-4 py-1.5 rounded-lg hover:bg-blue-700"
          >
            Re-index Docs
          </button>
          <button
            onClick={clearToken}
            className="text-sm text-gray-500 hover:text-gray-800"
          >
            Sign out
          </button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-6 space-y-6">
        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="Conversations" value={stats.total_conversations} />
            <StatCard label="Total Turns" value={stats.total_turns} />
            <StatCard label="Escalated" value={stats.escalated_count} highlight />
            <StatCard label="Avg Confidence" value={`${(stats.avg_confidence * 100).toFixed(0)}%`} />
          </div>
        )}

        {/* Filters */}
        <div className="bg-white rounded-xl border border-gray-200 p-4 flex flex-wrap gap-4 items-center">
          <form onSubmit={handleSearch} className="flex gap-2 flex-1 min-w-[200px]">
            <input
              type="text"
              placeholder="Search messages..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="flex-1 border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button type="submit" className="bg-gray-800 text-white text-sm px-4 py-1.5 rounded-lg hover:bg-gray-700">
              Search
            </button>
            {search && (
              <button
                type="button"
                onClick={() => { setSearch(""); setSearchInput(""); setPage(1); }}
                className="text-sm text-gray-500 hover:text-gray-800"
              >
                Clear
              </button>
            )}
          </form>
          <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={escalatedOnly}
              onChange={(e) => { setEscalatedOnly(e.target.checked); setPage(1); }}
              className="rounded"
            />
            Escalated only
          </label>
          <button onClick={() => { reloadList(); reloadStats(); }} className="text-sm text-gray-500 hover:text-gray-800">
            Refresh
          </button>
        </div>

        {/* Table */}
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          {loading ? (
            <div className="p-8 text-center text-gray-400 text-sm">Loading...</div>
          ) : !data || data.items.length === 0 ? (
            <div className="p-8 text-center text-gray-400 text-sm">No conversations found.</div>
          ) : (
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Session</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Last message</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Turns</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Last seen</th>
                  <th className="text-left px-4 py-3 font-medium text-gray-500">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {data.items.map((row) => (
                  <>
                    <tr
                      key={row.session_id}
                      onClick={() => toggleRow(row.session_id)}
                      className="cursor-pointer hover:bg-gray-50 transition-colors"
                    >
                      <td className="px-4 py-3 font-mono text-xs text-gray-500">
                        {row.session_id.slice(0, 8)}...
                      </td>
                      <td className="px-4 py-3 text-gray-700 max-w-xs truncate">
                        {row.last_message}
                      </td>
                      <td className="px-4 py-3 text-gray-500">{row.turn_count}</td>
                      <td className="px-4 py-3 text-gray-500">
                        {new Date(row.last_seen).toLocaleString()}
                      </td>
                      <td className="px-4 py-3">
                        {row.escalated ? (
                          <span className="bg-red-100 text-red-700 text-xs px-2 py-0.5 rounded-full">Escalated</span>
                        ) : (
                          <span className="bg-green-100 text-green-700 text-xs px-2 py-0.5 rounded-full">OK</span>
                        )}
                      </td>
                    </tr>
                    {expandedId === row.session_id && (
                      <tr key={`${row.session_id}-detail`}>
                        <td colSpan={5} className="bg-gray-50 px-6 py-4">
                          {detailLoading ? (
                            <p className="text-sm text-gray-400">Loading...</p>
                          ) : detail ? (
                            <div className="space-y-3">
                              {detail.turns.map((turn, i) => (
                                <div key={i} className="space-y-1">
                                  <div className="flex items-center gap-2">
                                    <span className="text-xs font-medium text-gray-500 uppercase">User</span>
                                    <span className="text-xs text-gray-400">{new Date(turn.timestamp).toLocaleTimeString()}</span>
                                  </div>
                                  <p className="text-sm text-gray-800 bg-blue-50 rounded-lg px-3 py-2">
                                    {turn.user_message}
                                  </p>
                                  <div className="flex items-center gap-2">
                                    <span className="text-xs font-medium text-gray-500 uppercase">Bot</span>
                                    {turn.confidence !== null && (
                                      <span className="text-xs text-gray-400">
                                        conf: {(turn.confidence * 100).toFixed(0)}%
                                      </span>
                                    )}
                                    {turn.escalated && (
                                      <span className="bg-red-100 text-red-600 text-xs px-1.5 rounded">escalated</span>
                                    )}
                                  </div>
                                  <p className="text-sm text-gray-800 bg-white border border-gray-200 rounded-lg px-3 py-2">
                                    {turn.bot_answer}
                                  </p>
                                </div>
                              ))}
                            </div>
                          ) : null}
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination */}
        {data && totalPages > 1 && (
          <div className="flex items-center justify-between text-sm text-gray-500">
            <span>
              {data.total} conversations — page {page} of {totalPages}
            </span>
            <div className="flex gap-2">
              <button
                disabled={page === 1}
                onClick={() => setPage((p) => p - 1)}
                className="px-3 py-1 border rounded-lg disabled:opacity-40 hover:bg-gray-100"
              >
                Prev
              </button>
              <button
                disabled={page === totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="px-3 py-1 border rounded-lg disabled:opacity-40 hover:bg-gray-100"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, highlight = false }: { label: string; value: number | string; highlight?: boolean }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-2xl font-semibold ${highlight ? "text-red-600" : "text-gray-900"}`}>
        {value}
      </p>
    </div>
  );
}
