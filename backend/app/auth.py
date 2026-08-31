import hashlib
import secrets

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import ApiKey

KEY_PREFIX = "llmgw"


def generate_api_key() -> tuple[str, str, str]:
    """Returns (full_key, key_hash, key_prefix). Only key_hash is stored."""
    secret = secrets.token_urlsafe(32)
    full_key = f"{KEY_PREFIX}_{secret}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[: len(KEY_PREFIX) + 9]
    return full_key, key_hash, key_prefix


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def require_api_key(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> ApiKey:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Bearer token")
    raw_key = authorization.removeprefix("Bearer ").strip()
    key_hash = hash_key(raw_key)
    api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash, ApiKey.revoked.is_(False)).first()
    if not api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or revoked API key")
    return api_key


def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid admin key")
