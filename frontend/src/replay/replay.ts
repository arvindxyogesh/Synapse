// Types for recordings.json (written by scripts/record_replay.py) and the
// pure timing logic of the replay, kept separate so it can be unit-tested.

export interface RecordedEvent {
  t_ms: number; // time since the request was sent
  text: string;
}

export interface Exchange {
  kind: "first" | "repeat" | "reworded";
  prompt: string;
  x_cache: "hit" | "miss" | "bypass" | null;
  provider: string | null;
  ttft_ms: number | null;
  total_ms: number;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
  events: RecordedEvent[];
}

export interface Topic {
  id: string;
  title: string;
  exchanges: Exchange[];
}

export interface Recording {
  provenance: {
    recorded_utc: string;
    git: { commit: string | null; dirty: boolean };
    gateway_model: string;
    backend: string;
    embedder_backend: string | null;
    warmup: { prompt: string; recorded: boolean; ttft_ms: number | null; total_ms: number };
    note: string;
  };
  topics: Topic[];
}

export const KIND_LABEL: Record<Exchange["kind"], string> = {
  first: "Ask",
  repeat: "Ask the same again",
  reworded: "Ask it reworded",
};

export interface Frame {
  at_ms: number; // when to show it, relative to pressing "send"
  text: string; // full reply text visible at that moment
}

/** Cumulative text frames at their recorded times. Chunks recorded at the
 * same millisecond collapse into one frame. */
export function replayFrames(events: RecordedEvent[]): Frame[] {
  const frames: Frame[] = [];
  let text = "";
  for (const e of events) {
    text += e.text;
    const last = frames[frames.length - 1];
    if (last && last.at_ms === e.t_ms) last.text = text;
    else frames.push({ at_ms: e.t_ms, text });
  }
  return frames;
}
