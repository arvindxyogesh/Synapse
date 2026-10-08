from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import RequestLog
from app.pricing import estimate_cost_usd
from app.schemas import CacheThresholdState, ProviderBreakdown, RequestLogOut, StatsSummary, TimeseriesPoint
from app.threshold_controller import get_threshold_controller

router = APIRouter(prefix="/v1/stats", tags=["stats"])


@router.get("/summary", response_model=StatsSummary)
def summary(hours: int = Query(default=24, ge=1, le=24 * 30), db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = db.query(RequestLog).filter(RequestLog.created_at >= since).all()

    total = len(rows)
    if total == 0:
        return StatsSummary(
            total_requests=0, cache_hit_rate=0.0, total_cost_usd=0.0,
            cost_saved_usd=0.0, avg_latency_ms=0.0, p95_latency_ms=0.0,
        )

    cache_hits = sum(1 for r in rows if r.cached)
    total_cost = sum(r.cost_usd for r in rows)
    cost_saved = sum(estimate_cost_usd(r.model, r.prompt_tokens, r.completion_tokens) for r in rows if r.cached)
    # Latency stats cover successful responses only: a failed backend call
    # (status="error") has a latency too, but averaging it in would mix
    # "how fast do answers arrive" with "how fast do errors arrive".
    latencies = sorted(r.latency_ms for r in rows if r.status == "ok") or [0.0]
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]

    return StatsSummary(
        total_requests=total,
        cache_hit_rate=cache_hits / total,
        total_cost_usd=round(total_cost, 6),
        cost_saved_usd=round(cost_saved, 6),
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
    )


def _hour_bucket(ts: datetime) -> str:
    """'YYYY-MM-DDTHH:00:00' in UTC. Postgres returns timezone-aware values;
    SQLite returns naive ones that were stored as UTC."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:00:00")


@router.get("/timeseries", response_model=list[TimeseriesPoint])
def timeseries(hours: int = Query(default=24, ge=1, le=24 * 30), db: Session = Depends(get_db)):
    # Bucketed in Python rather than SQL: the previous SQL used SQLite's
    # strftime(), which doesn't exist in Postgres, so this endpoint failed
    # on the docker-compose stack (see DECISIONS.md D4).
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = db.query(RequestLog).filter(RequestLog.created_at >= since).all()

    buckets: dict[str, list[RequestLog]] = defaultdict(list)
    for r in rows:
        buckets[_hour_bucket(r.created_at)].append(r)

    points = []
    for bucket in sorted(buckets):
        bucket_rows = buckets[bucket]
        # As in /summary: latency averages successful responses only.
        ok_latencies = [r.latency_ms for r in bucket_rows if r.status == "ok"]
        points.append(
            TimeseriesPoint(
                bucket=bucket,
                requests=len(bucket_rows),
                cache_hits=sum(1 for r in bucket_rows if r.cached),
                cost_usd=round(sum(r.cost_usd for r in bucket_rows), 6),
                avg_latency_ms=round(sum(ok_latencies) / len(ok_latencies), 2) if ok_latencies else 0.0,
            )
        )
    return points


@router.get("/providers", response_model=list[ProviderBreakdown])
def provider_breakdown(hours: int = Query(default=24, ge=1, le=24 * 30), db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = (
        db.query(
            RequestLog.provider,
            func.count(RequestLog.id).label("requests"),
            func.sum(RequestLog.cost_usd).label("cost_usd"),
        )
        .filter(RequestLog.created_at >= since)
        .group_by(RequestLog.provider)
        .all()
    )
    return [
        ProviderBreakdown(provider=r.provider, requests=r.requests, cost_usd=round(r.cost_usd or 0.0, 6))
        for r in rows
    ]


@router.get("/requests", response_model=list[RequestLogOut])
def request_log(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(RequestLog)
        .order_by(RequestLog.created_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    return [
        RequestLogOut(
            id=r.id,
            provider=r.provider,
            model=r.model,
            cached=r.cached,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            cost_usd=r.cost_usd,
            latency_ms=round(r.latency_ms, 2),
            ttft_ms=round(r.ttft_ms, 2) if r.ttft_ms is not None else None,
            status=r.status,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@router.get("/cache-threshold", response_model=list[CacheThresholdState])
def cache_threshold_state():
    """Per-model adaptive cache-similarity threshold state -- see
    app/threshold_controller.py. Only models that have served at least one
    cache hit (so a controller state exists) are listed."""
    settings = get_settings()
    controller = get_threshold_controller()
    states = []
    for model in controller.all_models():
        state = controller.get_state(model)
        states.append(
            CacheThresholdState(
                model=model,
                threshold=round(state.threshold, 4),
                estimated_false_positive_rate=round(state.fp_rate_ewma, 4),
                verified_samples=state.verified_count,
                target_false_positive_rate=settings.target_false_positive_rate,
                last_direction=state.last_direction,
                last_adjusted_at=state.last_adjusted_at,
            )
        )
    return sorted(states, key=lambda s: s.model)
