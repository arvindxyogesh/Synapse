"""Instant-response stand-in for a model backend, for httpx_client_reuse.py."""

from fastapi import FastAPI

app = FastAPI()


@app.post("/v1/chat/completions")
def chat_completions():
    return {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
