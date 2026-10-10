#!/usr/bin/env python3
"""Record real gateway traffic for the static demo's replay Playground.

The public demo (GitHub Pages) has no server behind it, so its Playground
plays back requests recorded here, against a real running Synapse gateway,
with their original timing. Nothing is edited: each stream chunk is saved with
the time it arrived, along with the gateway's own verdict (x-cache, provider)
and token counts.

For each topic three requests are sent, in order: the question, the exact same
question again, and a reworded version -- so a viewer sees a cache miss, an
exact-match hit, and whatever the semantic cache actually decided for the
paraphrase (recorded as-is, hit or miss).

    python scripts/record_replay.py --gateway-url http://localhost:8000 --model qwen2.5-0.5b \\
        --out frontend/src/replay/recordings.json

Reads the gateway API key from SYNAPSE_API_KEY. Run against a gateway with an
empty cache, so the first ask of each topic is a genuine miss.

One warm-up request on an unrelated prompt is sent first and NOT recorded:
the very first request after startup also pays for loading the embedding
model and the LLM into memory (~8 s in the first attempt), which isn't what
a running gateway looks like. The provenance says so.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
WARMUP_PROMPT = "Say hello in one short sentence."

TOPICS = [
    {
        "id": "semantic-cache",
        "title": "What is a semantic cache?",
        "prompts": [
            ("first", "What is a semantic cache for an LLM API, in two sentences?"),
            ("repeat", "What is a semantic cache for an LLM API, in two sentences?"),
            ("reworded", "In two sentences, what is a semantic cache for an LLM API?"),
        ],
    },
    {
        "id": "quantization",
        "title": "What does quantization do?",
        "prompts": [
            ("first", "Briefly, what does 4-bit quantization do to a language model?"),
            ("repeat", "Briefly, what does 4-bit quantization do to a language model?"),
            ("reworded", "What does 4-bit quantization do to a language model? Keep it brief."),
        ],
    },
    {
        "id": "commit-messages",
        "title": "Writing commit messages",
        "prompts": [
            ("first", "Give three short tips for writing clear git commit messages."),
            ("repeat", "Give three short tips for writing clear git commit messages."),
            ("reworded", "Share three brief tips for clear git commit messages."),
        ],
    },
]


def stream_once(client: httpx.Client, url: str, key: str, model: str, prompt: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    events, usage, provider = [], None, None
    sent = time.perf_counter()
    with client.stream("POST", url, json=body, headers={"Authorization": f"Bearer {key}"}) as resp:
        resp.raise_for_status()
        x_cache = resp.headers.get("x-cache")
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            provider = event.get("provider", provider)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    events.append({"t_ms": round((time.perf_counter() - sent) * 1000, 1), "text": text})
    total_ms = round((time.perf_counter() - sent) * 1000, 1)
    return {
        "x_cache": x_cache,
        "provider": provider,
        "ttft_ms": events[0]["t_ms"] if events else None,
        "total_ms": total_ms,
        "usage": usage,
        "events": events,
    }


def git_commit(output: Path) -> dict:
    """Commit and dirty flag. The output file itself is excluded: a previous
    recording sitting there shouldn't mark this one's code as uncommitted."""
    def run(*cmd):
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None
    exclude = f":(exclude){output.resolve().relative_to(REPO)}"
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--", ".", exclude))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gateway-url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--backend-description", required=True,
                   help='recorded as provenance, e.g. "Ollama 0.40.1, qwen2.5:0.5b, MacBook (Apple M4)"')
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    key = os.environ.get("SYNAPSE_API_KEY")
    if not key:
        print("set SYNAPSE_API_KEY", file=sys.stderr)
        return 1

    base = args.gateway_url.rstrip("/")
    url = f"{base}/v1/chat/completions"
    with httpx.Client(timeout=300) as client:
        warmup = stream_once(client, url, key, args.model, WARMUP_PROMPT)
        print(f"warm-up (not recorded): ttft={warmup['ttft_ms']}ms total={warmup['total_ms']}ms", flush=True)
        topics = []
        for topic in TOPICS:
            exchanges = []
            for kind, prompt in topic["prompts"]:
                result = stream_once(client, url, key, args.model, prompt)
                if result["provider"] == "mock":
                    print("the mock provider answered -- refusing to record fake output", file=sys.stderr)
                    return 2
                print(f"{topic['id']:<16} {kind:<9} x-cache={result['x_cache']:<5} provider={result['provider']:<7} "
                      f"ttft={result['ttft_ms']}ms total={result['total_ms']}ms", flush=True)
                exchanges.append({"kind": kind, "prompt": prompt, **result})
            topics.append({"id": topic["id"], "title": topic["title"], "exchanges": exchanges})
        # The embedder only reports its backend after first use, so read it now.
        health = client.get(f"{base}/health").json()

    recording = {
        "provenance": {
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_commit(args.out),
            "gateway_model": args.model,
            "backend": args.backend_description,
            "embedder_backend": health.get("embedder_backend"),
            "client_platform": platform.platform(),
            "warmup": {"prompt": WARMUP_PROMPT, "recorded": False, "ttft_ms": warmup["ttft_ms"],
                       "total_ms": warmup["total_ms"]},
            "note": "Real requests to a real Synapse gateway, replayed with their original timing. "
                    "The model is a small laptop model, not the GPU-benchmarked one. One unrecorded "
                    "warm-up request was sent first, so model loading isn't in the timings.",
        },
        "topics": topics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recording, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
