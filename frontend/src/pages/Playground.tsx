import { useRef, useState } from "react";

import { api } from "../api/client";
import type { ChatMessage } from "../types";

interface Turn extends ChatMessage {
  cached?: boolean;
  provider?: string;
  latencyMs?: number;
}

export default function Playground() {
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("llama3");
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(true);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function send() {
    const content = input.trim();
    if (!content || !apiKey.trim() || busy) return;

    setError(null);
    setInput("");
    const history = [...turns, { role: "user" as const, content }];
    setTurns([...history, { role: "assistant", content: "" }]);
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${api.baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey.trim()}` },
        body: JSON.stringify({
          model,
          messages: history.map(({ role, content }) => ({ role, content })),
          stream: streaming,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status}: ${body || res.statusText}`);
      }

      if (streaming && res.body) {
        await consumeStream(res.body, (delta, meta) =>
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = {
              ...last,
              content: last.content + delta,
              cached: meta?.cached ?? last.cached,
              provider: meta?.provider ?? last.provider,
            };
            return next;
          }),
        );
      } else {
        const data = await res.json();
        setTurns((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            role: "assistant",
            content: data.choices[0].message.content,
            cached: data.cached,
            provider: data.provider,
            latencyMs: data.latency_ms,
          };
          return next;
        });
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setError((err as Error).message);
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="max-w-3xl space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto_auto]">
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="Gateway API key (llmgw_...)"
          className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
        />
        <input
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="model"
          className="w-32 rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
        />
        <label className="flex items-center gap-2 whitespace-nowrap rounded border border-slate-800 px-3 py-2 text-sm text-slate-300">
          <input type="checkbox" checked={streaming} onChange={(e) => setStreaming(e.target.checked)} />
          Stream
        </label>
      </div>

      <div className="min-h-[16rem] space-y-3 rounded-lg border border-slate-800 bg-slate-900 p-4">
        {turns.length === 0 && (
          <p className="text-sm text-slate-500">
            Send a message through the gateway. It's checked against the semantic cache before
            (maybe) reaching the model -- send the same thing twice to see a cache hit.
          </p>
        )}
        {turns.map((turn, i) => (
          <div key={i} className={turn.role === "user" ? "text-right" : "text-left"}>
            <div
              className={
                "inline-block max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm " +
                (turn.role === "user" ? "bg-emerald-600 text-white" : "bg-slate-800 text-slate-100")
              }
            >
              {turn.content || (busy && i === turns.length - 1 ? "…" : "")}
            </div>
            {turn.role === "assistant" && (turn.provider || turn.latencyMs !== undefined) && (
              <div className="mt-1 text-xs text-slate-500">
                {turn.cached ? "cache hit" : `provider: ${turn.provider}`}
                {turn.latencyMs !== undefined && ` · ${turn.latencyMs.toFixed(0)}ms`}
              </div>
            )}
          </div>
        ))}
      </div>

      {error && <div className="text-sm text-red-300">{error}</div>}

      <div className="flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          rows={2}
          placeholder="Ask something..."
          className="flex-1 resize-none rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
        />
        {busy ? (
          <button
            onClick={() => abortRef.current?.abort()}
            className="rounded bg-slate-700 px-4 py-2 text-sm font-medium hover:bg-slate-600"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={send}
            disabled={!input.trim() || !apiKey.trim()}
            className="rounded bg-emerald-600 px-4 py-2 text-sm font-medium hover:bg-emerald-500 disabled:opacity-40"
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}

interface ChunkMeta {
  provider?: string;
  cached?: boolean;
}

async function consumeStream(
  body: ReadableStream<Uint8Array>,
  onDelta: (delta: string, meta: ChunkMeta | undefined) => void,
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      const line = rawEvent.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const payload = line.slice("data: ".length);
      if (payload === "[DONE]") return;

      try {
        const event = JSON.parse(payload);
        const delta: string | undefined = event.choices?.[0]?.delta?.content;
        onDelta(delta ?? "", { provider: event.provider, cached: event.cached });
      } catch {
        // ignore malformed keep-alive/partial lines
      }
    }
  }
}
