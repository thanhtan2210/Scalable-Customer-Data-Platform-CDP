"""Seed exposures into the exposures table for CI testing.

Reads `data/raw/cleaned_telco.csv` to obtain customer IDs and inserts a sample
of exposures into the DB using `src.services.exposure_store`.
"""

import os
import csv
import random
import hashlib
from pathlib import Path

from backend.app.core.services.exposure_store import init_db, insert_exposure

DATA_CSV = Path("data/raw/cleaned_telco.csv")


def deterministic_group(cid: str, ratio: float = 0.5) -> str:
    h = int(hashlib.sha256(str(cid).encode("utf-8")).hexdigest()[:16], 16)
    r = (h % 1000000) / 1000000
    return "A" if r < ratio else "B"


def main(db_url: str = None, n: int = 500):
    db_url = db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is required")
        return
    init_db(db_url)

    if not DATA_CSV.exists():
        print("Data file not found:", DATA_CSV)
        return

    ids = []
    with open(DATA_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row.get("CustomerID") or row.get("customerID")
            if cid:
                ids.append(cid)

    sample = random.sample(ids, min(n, len(ids)))
    for cid in sample:
        grp = deterministic_group(cid)
        insert_exposure(cid, grp, "exposed")

    print(f"Inserted {len(sample)} exposures into DB")


if __name__ == "__main__":
    main()
