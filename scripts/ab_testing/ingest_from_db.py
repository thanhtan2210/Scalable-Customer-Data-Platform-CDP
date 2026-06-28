"""Ingestion helper: export exposures from DB to CSV for downstream processing."""

import os
import csv
from pathlib import Path
from sqlalchemy import create_engine, text

OUT = Path("reports/ab_exposures_from_db.csv")


def run(db_url: str = None):
    db_url = db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL required")
        return
    eng = create_engine(db_url)
    q = text("SELECT customer_id, ab_group, event, ts FROM exposures ORDER BY ts")
    rows = []
    with eng.connect() as conn:
        res = conn.execute(q).mappings()
        for r in res:
            rows.append(
                {
                    "CustomerID": r["customer_id"],
                    "ab_group": r["ab_group"],
                    "event": r["event"],
                    "ts": r["ts"],
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["CustomerID", "ab_group", "event", "ts"])
        w.writeheader()
        w.writerows(rows)
    print("Exported", len(rows), "rows to", OUT)


if __name__ == "__main__":
    run()
