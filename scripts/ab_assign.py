"""Assign customers to A/B groups.

Usage:
  & .\.venv\Scripts\Activate.ps1
  python scripts/ab_assign.py --input data/raw/cleaned_telco.csv --out reports/ab_assignment.csv --ratio 0.5
"""
import argparse
import hashlib
import pandas as pd
from pathlib import Path


def deterministic_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest()[:16], 16)


def assign(input_csv: Path, out_csv: Path, ratio: float = 0.5, strata: str = None):
    df = pd.read_csv(input_csv)
    if 'CustomerID' not in df.columns and 'customerID' not in df.columns:
        raise KeyError('CustomerID or customerID column not found')
    cid_col = 'CustomerID' if 'CustomerID' in df.columns else 'customerID'

    def assign_row(cid):
        h = deterministic_hash(str(cid))
        r = (h % 1000000) / 1000000
        return 'A' if r < ratio else 'B'

    df['ab_group'] = df[cid_col].apply(assign_row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[[cid_col, 'ab_group']].to_csv(out_csv, index=False)
    print(f'Assigned {len(df)} customers; saved to {out_csv}')


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', dest='out', required=True)
    p.add_argument('--ratio', type=float, default=0.5)
    p.add_argument('--strata', default=None)
    args = p.parse_args()
    assign(Path(args.input), Path(args.out), args.ratio, args.strata)


if __name__ == '__main__':
    cli()
