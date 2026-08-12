from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ...db.session import get_db
from ...db.models import User
from ...core.security import (
    verify_password,
    hash_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from ...core.config import settings
from ...core.dependencies import get_current_user
from ..schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    RefreshTokenRequest,
    ChangePasswordRequest,
    TokenResponse,
    UserInfo,
)
from jose import JWTError

router = APIRouter(
    prefix="/auth",
    tags=["authentication"]
)


def _build_token_response(user: User) -> TokenResponse:
    token_data = {
        "sub": str(user.id),
        "email": user.email
    }
    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserInfo(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            is_admin=user.is_admin,
            created_at=user.created_at
        )
    )


@router.post("/register",
             response_model=TokenResponse,
             status_code=201)
def register(
    request: UserRegisterRequest,
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        full_name=request.full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _build_token_response(user)


@router.post("/login",
             response_model=TokenResponse)
def login(
    request: UserLoginRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.email == request.email,
        User.is_active == True
    ).first()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    user.last_login = datetime.utcnow()
    db.commit()
    return _build_token_response(user)


@router.post("/refresh",
             response_model=TokenResponse)
def refresh(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    try:
        payload = decode_token(request.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token"
            )
        user = db.query(User).filter(
            User.id == payload.get("sub"),
            User.is_active == True
        ).first()
        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )
        return _build_token_response(user)
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )


@router.get("/me",
            response_model=UserInfo)
def get_me(
    current_user: User = Depends(get_current_user)
):
    return UserInfo(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        is_admin=current_user.is_admin,
        created_at=current_user.created_at
    )


@router.post("/change-password")
def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="Current password incorrect"
        )
    current_user.hashed_password = hash_password(request.new_password)
    db.commit()
    return {"message": "Password changed successfully"}
