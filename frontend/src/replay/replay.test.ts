import { describe, expect, it } from "vitest";

import recordingJson from "./recordings.json";
import { replayFrames, type Recording } from "./replay";

describe("replayFrames", () => {
  it("accumulates text at the recorded times", () => {
    expect(replayFrames([{ t_ms: 40, text: "Hel" }, { t_ms: 55, text: "lo" }, { t_ms: 90, text: "!" }])).toEqual([
      { at_ms: 40, text: "Hel" }, { at_ms: 55, text: "Hello" }, { at_ms: 90, text: "Hello!" },
    ]);
  });

  it("merges chunks recorded at the same moment and handles no events", () => {
    expect(replayFrames([{ t_ms: 10, text: "a" }, { t_ms: 10, text: "b" }])).toEqual([{ at_ms: 10, text: "ab" }]);
    expect(replayFrames([])).toEqual([]);
  });
});

describe("the recording shipped with the demo", () => {
  const rec = recordingJson as unknown as Recording;

  it("was made from committed code by a real backend, never the mock", () => {
    expect(rec.provenance.git.dirty).toBe(false);
    expect(rec.provenance.warmup.recorded).toBe(false);
    for (const t of rec.topics) for (const e of t.exchanges) expect(e.provider).not.toBe("mock");
  });

  it("shows a real miss first, and cache hits replay the original answer", () => {
    for (const t of rec.topics) {
      const [first, ...later] = t.exchanges;
      expect(first.kind).toBe("first");
      expect(first.x_cache).toBe("miss");
      const answer = first.events.map((e) => e.text).join("");
      for (const e of later.filter((x) => x.x_cache === "hit")) {
        expect(e.events.map((x) => x.text).join("")).toBe(answer);
      }
    }
  });
});
