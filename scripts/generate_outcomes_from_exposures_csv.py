"""Generate outcomes CSV from exposures CSV exported from DB.

Input: `reports/ab_exposures_from_db.csv` (CustomerID,ab_group,event,ts)
Output: `reports/ab_outcomes.csv` with columns CustomerID,churn (0/1)
"""
import csv
import random
from pathlib import Path

IN_CSV = Path('reports/ab_exposures_from_db.csv')
OUT_CSV = Path('reports/ab_outcomes.csv')


def main(churn_A=0.03, churn_B=0.024):
    if not IN_CSV.exists():
        print('Input exposures CSV not found:', IN_CSV)
        return

    seen = {}
    with open(IN_CSV, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row.get('CustomerID') or row.get('customer_id')
            grp = row.get('ab_group')
            if cid and grp and cid not in seen:
                seen[cid] = grp

    rows = []
    for cid, grp in seen.items():
        p = churn_A if grp == 'A' else churn_B
        churn = 1 if random.random() < p else 0
        rows.append({'CustomerID': cid, 'churn': churn})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['CustomerID', 'churn'])
        w.writeheader()
        w.writerows(rows)

    print(f'Wrote {len(rows)} outcomes to {OUT_CSV}')


if __name__ == '__main__':
    main()
