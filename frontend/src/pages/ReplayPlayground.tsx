import { useEffect, useRef, useState } from "react";

import { REPO_URL } from "../demo";
import recordingJson from "../replay/recordings.json";
import { KIND_LABEL, replayFrames, type Exchange, type Recording } from "../replay/replay";

const recording = recordingJson as unknown as Recording;

interface Played {
  exchange: Exchange;
  text: string;
  done: boolean;
}

function Chip({ children, tone }: { children: React.ReactNode; tone: "hit" | "miss" | "neutral" }) {
  const styles = {
    hit: "border-emerald-700 text-emerald-300",
    miss: "border-amber-700 text-amber-300",
    neutral: "border-slate-700 text-slate-300",
  }[tone];
  return <span className={`rounded border px-1.5 py-0.5 text-[11px] ${styles}`}>{children}</span>;
}

function ms(v: number | null): string {
  if (v == null) return "–";
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
}

export default function ReplayPlayground() {
  const [topicId, setTopicId] = useState(recording.topics[0].id);
  const [played, setPlayed] = useState<Played[]>([]);
  const timers = useRef<number[]>([]);

  const topic = recording.topics.find((t) => t.id === topicId)!;
  const busy = played.some((p) => !p.done);
  const nextIndex = played.length;

  function clearTimers() {
    timers.current.forEach((id) => window.clearTimeout(id));
    timers.current = [];
  }

  useEffect(() => clearTimers, []);

  function chooseTopic(id: string) {
    clearTimers();
    setTopicId(id);
    setPlayed([]);
  }

  function play(exchange: Exchange) {
    const slot = played.length;
    setPlayed((prev) => [...prev, { exchange, text: "", done: false }]);
    const update = (text: string, done: boolean) =>
      setPlayed((prev) => prev.map((p, i) => (i === slot ? { ...p, text, done } : p)));
    // Each chunk appears at the moment it arrived in the recording.
    for (const frame of replayFrames(exchange.events)) {
      timers.current.push(window.setTimeout(() => update(frame.text, false), frame.at_ms));
    }
    timers.current.push(window.setTimeout(() => update(exchange.events.map((e) => e.text).join(""), true),
                                          exchange.total_ms));
  }

  const p = recording.provenance;

  return (
    <div className="max-w-3xl space-y-4">
      <div className="rounded-lg border border-amber-800/60 bg-amber-950/30 p-3 text-xs text-amber-200">
        <strong className="font-semibold">Replay, not a live model.</strong> This public demo has no server. These are
        real requests that went through a real Synapse gateway, played back with their original timing. Answers come
        from <code>{p.gateway_model}</code>, a tiny model running on a laptop, so they can be wrong. What to watch is
        the gateway: a cache miss waits for the model, and a hit (even for a reworded question) comes back from the
        semantic cache in milliseconds.
      </div>

      <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Topic">
        {recording.topics.map((t) => (
          <button key={t.id} role="radio" aria-checked={t.id === topicId} onClick={() => chooseTopic(t.id)}
                  className={`rounded-md border px-3 py-1.5 text-xs ${t.id === topicId ? "border-slate-500 bg-slate-800 text-slate-50" : "border-slate-700 text-slate-400 hover:text-slate-200"}`}>
            {t.title}
          </button>
        ))}
      </div>

      <div className="min-h-[16rem] space-y-4 rounded-lg border border-slate-800 bg-slate-900 p-4">
        {played.length === 0 && (
          <p className="text-sm text-slate-500">Press the button below to send the first question through the gateway.</p>
        )}
        {played.map(({ exchange: e, text, done }, i) => (
          <div key={i} className="space-y-2">
            <div className="text-right">
              <span className="inline-block max-w-[85%] rounded-lg bg-slate-800 px-3 py-2 text-left text-sm text-slate-100">
                {e.prompt}
              </span>
            </div>
            <div>
              <div className="inline-block max-w-[85%] whitespace-pre-wrap rounded-lg border border-slate-800 px-3 py-2 text-sm text-slate-200">
                {text || <span className="text-slate-500">waiting for the first token…</span>}
              </div>
              {done && (
                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-slate-400">
                  <Chip tone={e.x_cache === "hit" ? "hit" : "miss"}>
                    {e.x_cache === "hit" ? "✓ cache hit" : "○ cache miss → model"}
                  </Chip>
                  <Chip tone="neutral">provider: {e.provider}</Chip>
                  <Chip tone="neutral">first token {ms(e.ttft_ms)}</Chip>
                  <Chip tone="neutral">total {ms(e.total_ms)}</Chip>
                  {e.usage && <Chip tone="neutral">{e.usage.completion_tokens} tokens</Chip>}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        {nextIndex < topic.exchanges.length ? (
          <button onClick={() => play(topic.exchanges[nextIndex])} disabled={busy}
                  className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50">
            {KIND_LABEL[topic.exchanges[nextIndex].kind]}
          </button>
        ) : (
          <button onClick={() => chooseTopic(topicId)}
                  className="rounded-md border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800">
            Start over
          </button>
        )}
        {nextIndex < topic.exchanges.length && (
          <span className="text-xs text-slate-500">
            Step {nextIndex + 1} of {topic.exchanges.length}: “{topic.exchanges[nextIndex].prompt}”
          </span>
        )}
      </div>

      <details className="text-xs text-slate-400">
        <summary className="cursor-pointer text-slate-300">How this was recorded</summary>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>Recorded {p.recorded_utc.replace("T", " ").replace("+00:00", " UTC")} with <code>scripts/record_replay.py</code> at
            commit <code>{p.git.commit?.slice(0, 7)}</code>.</li>
          <li>Backend: {p.backend}. Semantic cache embedder: {p.embedder_backend}.</li>
          <li>One warm-up request (“{p.warmup.prompt}”, {ms(p.warmup.ttft_ms)} to first token) was sent first and not
            recorded, so model loading isn't in the timings.</li>
          <li>To run the real thing: <a className="text-emerald-400 hover:underline" href={REPO_URL}>clone the repo</a>{" "}
            and <code>docker compose up</code>.</li>
        </ul>
      </details>
    </div>
  );
}
