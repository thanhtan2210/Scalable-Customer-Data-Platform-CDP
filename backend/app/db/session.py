from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import os
from ..core.config import settings, IS_PRODUCTION

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/churn_db")

# Production settings
if IS_PRODUCTION:
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,  # 30 phút
        pool_pre_ping=True  # detect stale connections
    )
else:
    # Development: SQLite hoặc simple pool
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
        if "sqlite" in DATABASE_URL else {}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
