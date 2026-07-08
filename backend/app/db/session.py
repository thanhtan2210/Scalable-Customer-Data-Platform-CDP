from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from ..core.config import DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_RECYCLE, DB_POOL_TIMEOUT

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/churn_db")

if DATABASE_URL.startswith("postgresql"):
    engine = create_engine(
        DATABASE_URL,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_recycle=DB_POOL_RECYCLE,
        pool_timeout=DB_POOL_TIMEOUT,
        pool_pre_ping=True
    )
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
