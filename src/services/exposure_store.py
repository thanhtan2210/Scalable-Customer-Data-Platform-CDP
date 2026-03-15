import os
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
)
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

DATABASE_URL = os.environ.get('DATABASE_URL')
engine = None
metadata = MetaData()

exposures = Table(
    'exposures',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('customer_id', String(128), nullable=False, index=True),
    Column('ab_group', String(8), nullable=False),
    Column('event', String(64), nullable=False),
    Column('ts', Float, nullable=False),
)


def init_db(db_url: str = None):
    global engine
    db_url = db_url or DATABASE_URL
    if not db_url:
        return None
    try:
        engine = create_engine(db_url, future=True)
        metadata.create_all(engine)
        return engine
    except SQLAlchemyError:
        engine = None
        return None


def insert_exposure(customer_id: str, ab_group: str, event: str, ts: float = None):
    ts = ts or datetime.utcnow().timestamp()
    if engine is None:
        # try lazy init
        if not init_db():
            return False
    try:
        with engine.connect() as conn:
            conn.execute(
                exposures.insert().values(customer_id=customer_id,
                                          ab_group=ab_group, event=event, ts=ts)
            )
            conn.commit()
        return True
    except Exception:
        return False
