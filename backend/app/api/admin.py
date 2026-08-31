from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import generate_api_key, require_admin_key
from app.db import get_db
from app.models import ApiKey
from app.ratelimit import RateLimiter, get_rate_limiter
from app.schemas import ApiKeyCreateRequest, ApiKeyCreateResponse, ApiKeyOut, ApiKeyUpdateRequest

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Depends(require_admin_key)])


def _to_out(record: ApiKey, limiter: RateLimiter) -> ApiKeyOut:
    return ApiKeyOut(
        id=record.id,
        name=record.name,
        key_prefix=record.key_prefix,
        revoked=record.revoked,
        rate_limit_per_minute=record.rate_limit_per_minute,
        monthly_quota_usd=record.monthly_quota_usd,
        quota_spent_usd=round(limiter.current_spend(record.id), 6),
    )


@router.post("/keys", response_model=ApiKeyCreateResponse)
def create_key(body: ApiKeyCreateRequest, db: Session = Depends(get_db)):
    full_key, key_hash, key_prefix = generate_api_key()
    record = ApiKey(
        name=body.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        rate_limit_per_minute=body.rate_limit_per_minute,
        monthly_quota_usd=body.monthly_quota_usd,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return ApiKeyCreateResponse(
        id=record.id,
        name=record.name,
        api_key=full_key,
        rate_limit_per_minute=record.rate_limit_per_minute,
        monthly_quota_usd=record.monthly_quota_usd,
    )


@router.get("/keys", response_model=list[ApiKeyOut])
def list_keys(db: Session = Depends(get_db), limiter: RateLimiter = Depends(get_rate_limiter)):
    records = db.query(ApiKey).order_by(ApiKey.created_at.desc()).all()
    return [_to_out(r, limiter) for r in records]


@router.post("/keys/{key_id}/revoke", response_model=ApiKeyOut)
def revoke_key(key_id: str, db: Session = Depends(get_db), limiter: RateLimiter = Depends(get_rate_limiter)):
    record = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not record:
        raise HTTPException(404, "API key not found")
    record.revoked = True
    db.commit()
    db.refresh(record)
    return _to_out(record, limiter)


@router.patch("/keys/{key_id}", response_model=ApiKeyOut)
def update_key(
    key_id: str,
    body: ApiKeyUpdateRequest,
    db: Session = Depends(get_db),
    limiter: RateLimiter = Depends(get_rate_limiter),
):
    """Update a key's rate limit / monthly quota. Pass a value to set it, or
    clear_rate_limit / clear_monthly_quota to remove it (unlimited)."""
    record = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not record:
        raise HTTPException(404, "API key not found")

    if body.clear_rate_limit:
        record.rate_limit_per_minute = None
    elif body.rate_limit_per_minute is not None:
        record.rate_limit_per_minute = body.rate_limit_per_minute

    if body.clear_monthly_quota:
        record.monthly_quota_usd = None
    elif body.monthly_quota_usd is not None:
        record.monthly_quota_usd = body.monthly_quota_usd

    db.commit()
    db.refresh(record)
    return _to_out(record, limiter)
