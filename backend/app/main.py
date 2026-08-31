from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, gateway, stats
from app.db import Base, engine

Base.metadata.create_all(bind=engine)

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
)

app.include_router(gateway.router)
app.include_router(admin.router)
app.include_router(stats.router)


@app.get("/health")
def health():
    return {"status": "ok"}
