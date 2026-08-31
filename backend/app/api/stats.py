from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Integer, cast, func
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import RequestLog
from app.pricing import estimate_cost_usd
from app.schemas import ProviderBreakdown, RequestLogOut, StatsSummary, TimeseriesPoint

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
    latencies = sorted(r.latency_ms for r in rows)
    avg_latency = sum(latencies) / total
    p95_latency = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]

    return StatsSummary(
        total_requests=total,
        cache_hit_rate=cache_hits / total,
        total_cost_usd=round(total_cost, 6),
        cost_saved_usd=round(cost_saved, 6),
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
    )


@router.get("/timeseries", response_model=list[TimeseriesPoint])
def timeseries(hours: int = Query(default=24, ge=1, le=24 * 30), db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    bucket_expr = func.strftime("%Y-%m-%dT%H:00:00", RequestLog.created_at)
    rows = (
        db.query(
            bucket_expr.label("bucket"),
            func.count(RequestLog.id).label("requests"),
            func.sum(cast(RequestLog.cached, Integer)).label("cache_hits"),
            func.sum(RequestLog.cost_usd).label("cost_usd"),
            func.avg(RequestLog.latency_ms).label("avg_latency_ms"),
        )
        .filter(RequestLog.created_at >= since)
        .group_by("bucket")
        .order_by("bucket")
        .all()
    )
    return [
        TimeseriesPoint(
            bucket=r.bucket,
            requests=r.requests,
            cache_hits=r.cache_hits or 0,
            cost_usd=round(r.cost_usd or 0.0, 6),
            avg_latency_ms=round(r.avg_latency_ms or 0.0, 2),
        )
        for r in rows
    ]


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
            status=r.status,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]
