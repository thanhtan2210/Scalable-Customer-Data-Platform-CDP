import csv
import random
from pathlib import Path

IN_CSV = Path("reports/ab_assignment.csv")
OUT_CSV = Path("reports/ab_outcomes.csv")


def main(churn_A=0.03, churn_B=0.024, sample_limit=None):
    if not IN_CSV.exists():
        print("Assignment CSV not found:", IN_CSV)
        return

    rows = []
    with open(IN_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if sample_limit:
        rows = rows[:sample_limit]

    out = []
    for r in rows:
        cid = r.get("CustomerID") or r.get("customerID")
        grp = r.get("ab_group")
        if not cid or not grp:
            continue
        p = churn_A if grp == "A" else churn_B
        churn = 1 if random.random() < p else 0
        out.append({"CustomerID": cid, "churn": churn})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["CustomerID", "churn"])
        w.writeheader()
        w.writerows(out)

    print(f"Wrote {len(out)} outcomes to {OUT_CSV}")


if __name__ == "__main__":
    main()
