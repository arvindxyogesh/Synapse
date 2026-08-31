from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import generate_api_key, require_admin_key
from app.db import get_db
from app.models import ApiKey
from app.schemas import ApiKeyCreateRequest, ApiKeyCreateResponse, ApiKeyOut

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Depends(require_admin_key)])


@router.post("/keys", response_model=ApiKeyCreateResponse)
def create_key(body: ApiKeyCreateRequest, db: Session = Depends(get_db)):
    full_key, key_hash, key_prefix = generate_api_key()
    record = ApiKey(name=body.name, key_hash=key_hash, key_prefix=key_prefix)
    db.add(record)
    db.commit()
    db.refresh(record)
    return ApiKeyCreateResponse(id=record.id, name=record.name, api_key=full_key)


@router.get("/keys", response_model=list[ApiKeyOut])
def list_keys(db: Session = Depends(get_db)):
    return db.query(ApiKey).order_by(ApiKey.created_at.desc()).all()


@router.post("/keys/{key_id}/revoke", response_model=ApiKeyOut)
def revoke_key(key_id: str, db: Session = Depends(get_db)):
    record = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not record:
        raise HTTPException(404, "API key not found")
    record.revoked = True
    db.commit()
    db.refresh(record)
    return record
