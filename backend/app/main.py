from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, gateway, stats

# Schema is managed by Alembic migrations (see alembic/ and
# `alembic upgrade head`), not create_all-on-startup -- run migrations
# before starting the app (the Docker image's CMD does this automatically).

app = FastAPI(
    title="Synapse",
    description="A memory layer for your LLM calls: multi-provider gateway with semantic caching and observability.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Without this, browser JS can't read x-cache/x-provider off the
    # response (the streaming chat endpoint also echoes them inside the
    # SSE payload itself, so this is a belt-and-suspenders fix).
    expose_headers=["x-cache", "x-provider"],
)

app.include_router(gateway.router)
app.include_router(admin.router)
app.include_router(stats.router)


@app.get("/health")
def health():
    return {"status": "ok"}
