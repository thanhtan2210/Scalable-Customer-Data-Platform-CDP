from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import hashlib
import json
import time
import os

from ..services import exposure_store
from fastapi.responses import JSONResponse

app = FastAPI(title="AB Assignment & Logging Service")

DATA_DIR = Path(__file__).resolve().parents[2] / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def deterministic_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


class AssignRequest(BaseModel):
    customer_id: str
    ratio: float = 0.5


class ExposureEvent(BaseModel):
    customer_id: str
    ab_group: str
    event: str
    timestamp: float = None


@app.on_event("startup")
def startup():
    # initialize DB if DATABASE_URL provided
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        exposure_store.init_db(db_url)


@app.post("/assign")
def assign(req: AssignRequest):
    if not req.customer_id:
        raise HTTPException(status_code=400, detail="customer_id required")
    h = deterministic_hash(req.customer_id)
    r = (h % 1000000) / 1000000
    group = "A" if r < req.ratio else "B"
    return {"customer_id": req.customer_id, "ab_group": group}


@app.post("/log_exposure")
def log_exposure(ev: ExposureEvent):
    ev.timestamp = ev.timestamp or time.time()
    # If DATABASE_URL configured, write to DB; otherwise append to jsonl
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        ok = exposure_store.insert_exposure(
            ev.customer_id, ev.ab_group, ev.event, ev.timestamp
        )
        if not ok:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "detail": "failed to write exposure to DB"},
            )
        return {"status": "ok", "stored": "db"}

    out = DATA_DIR / "exposures.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev.dict()) + "\n")
    return {"status": "ok", "stored": "jsonl"}


@app.get("/health")
def health():
    return {"status": "ok"}
