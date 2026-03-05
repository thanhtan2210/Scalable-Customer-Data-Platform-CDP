import json
import random
from pathlib import Path
import csv

EXPOSURES = Path('reports/exposures.jsonl')
OUTCOMES = Path('reports/ab_outcomes.csv')


def run(churn_A=0.03, churn_B=0.024):
    if not EXPOSURES.exists():
        print('No exposures file found at', EXPOSURES)
        return

    seen = {}
    with open(EXPOSURES, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get('customer_id') or obj.get('CustomerID')
            grp = obj.get('ab_group')
            if cid and grp and cid not in seen:
                seen[cid] = grp

    rows = []
    for cid, grp in seen.items():
        p = churn_A if grp == 'A' else churn_B
        churn = 1 if random.random() < p else 0
        rows.append({'CustomerID': cid, 'churn': churn})

    OUTCOMES.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTCOMES, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['CustomerID', 'churn'])
        w.writeheader()
        w.writerows(rows)

    print(f'Generated {len(rows)} outcomes to {OUTCOMES}')


if __name__ == '__main__':
    run()
