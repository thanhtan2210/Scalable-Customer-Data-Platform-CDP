from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
    APIKeyHeader,
)
from sqlalchemy.orm import Session
from jose import JWTError
from ..db.session import get_db
from ..db.models import User
from .security import decode_token
from .config import settings

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db),
) -> User:
    # 1. Try JWT Bearer authentication
    if credentials:
        try:
            payload = decode_token(credentials.credentials)
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token type",
                )
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token payload",
                )
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = db.query(User).filter(
            User.id == user_id,
            User.is_active == True,
        ).first()

        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found or inactive",
            )
        return user

    # 2. Backward compatibility fallback for X-API-Key service-to-service calls
    if api_key and api_key == settings.API_KEY:
        system_user = db.query(User).filter(User.email == "system@cdp.internal").first()
        if not system_user:
            system_user = User(
                id="system-service-user",
                email="system@cdp.internal",
                hashed_password="N/A",
                full_name="System API Key Service User",
                is_active=True,
                is_admin=True,
            )
            try:
                db.add(system_user)
                db.commit()
                db.refresh(system_user)
            except Exception:
                db.rollback()
                system_user = db.query(User).filter(User.email == "system@cdp.internal").first()
        return system_user

    # 3. No valid authentication credentials provided
    raise HTTPException(
        status_code=401,
        detail="Authorization header missing",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )
    return current_user
