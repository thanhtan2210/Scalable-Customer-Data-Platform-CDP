"""Initialize exposures DB (uses DATABASE_URL env var)."""

import os
from src.services.exposure_store import init_db


def main():
    db = os.environ.get("DATABASE_URL")
    if not db:
        print(
            "Set DATABASE_URL before running, e.g. postgresql://postgres:password@localhost:5432/abdb"
        )
        return
    e = init_db(db)
    if e:
        print("Initialized DB at", db)
    else:
        print("Failed to initialize DB with", db)


if __name__ == "__main__":
    main()
