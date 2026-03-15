import requests
import csv
import random
import time
from pathlib import Path


# Discover service port written by start_ab_service.py, fallback to default
PORT_FILE = Path('reports/ab_service_port.txt')
if PORT_FILE.exists():
    port = PORT_FILE.read_text().strip()
    BASE = f'http://127.0.0.1:{port}'
else:
    BASE = 'http://localhost:8082'
ASSIGN_CSV = Path('reports/ab_assignment.csv')
OUTCOMES_CSV = Path('reports/ab_outcomes.csv')


def simulate(n=500, ratio=0.5, churn_rate_A=0.03, churn_rate_B=0.024):
    # ensure assignment exists
    if not ASSIGN_CSV.exists():
        print('Run: python scripts/ab_assign.py --input data/raw/cleaned_telco.csv --out reports/ab_assignment.csv --ratio 0.5')
        return

    # read assignments
    rows = []
    with open(ASSIGN_CSV, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # sample n customers
    sample = random.sample(rows, min(n, len(rows)))

    outcomes = []
    for r in sample:
        cid = r.get('CustomerID') or r.get('customerID')
        # hit assign endpoint (to simulate production deterministic assignment)
        try:
            resp = requests.post(
                BASE + '/assign', json={'customer_id': cid, 'ratio': ratio}, timeout=5)
            group = resp.json().get('ab_group')
        except Exception:
            # fallback: deterministic local assignment if service fails to respond with JSON
            import hashlib

            h = int(hashlib.sha256(str(cid).encode(
                'utf-8')).hexdigest()[:16], 16)
            rnum = (h % 1000000) / 1000000
            group = 'A' if rnum < ratio else 'B'
        # log exposure
        requests.post(BASE + '/log_exposure',
                      json={'customer_id': cid, 'ab_group': group, 'event': 'exposed'})
        # simulate churn outcome based on group probability
        p = churn_rate_A if group == 'A' else churn_rate_B
        churn = 1 if random.random() < p else 0
        outcomes.append({'CustomerID': cid, 'churn': churn})
        time.sleep(0.01)

    OUTCOMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTCOMES_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['CustomerID', 'churn'])
        w.writeheader()
        w.writerows(outcomes)
    print(f'Simulated {len(outcomes)} outcomes and wrote to {OUTCOMES_CSV}')


if __name__ == '__main__':
    simulate()
