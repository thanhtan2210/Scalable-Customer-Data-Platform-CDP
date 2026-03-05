"""Analyze A/B experiment results.

Expect two CSVs:
- assignment CSV with columns: CustomerID, ab_group
- outcomes CSV with columns: CustomerID, churn (0/1)

Usage:
  python scripts/analyze_ab_results.py --assign reports/ab_assignment.csv --outcomes reports/ab_outcomes.csv --report reports/ab_analysis.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


def analyze(assign_path: Path, outcomes_path: Path):
    a = pd.read_csv(assign_path)
    o = pd.read_csv(outcomes_path)
    df = a.merge(o, on=['CustomerID', 'customerID'][0:1], how='left') if False else a.merge(
        o, left_on='CustomerID', right_on='CustomerID', how='left')
    # ensure churn column
    churn_col = 'churn' if 'churn' in df.columns else (
        'Churn' if 'Churn' in df.columns else 'churn')
    if churn_col not in df.columns:
        raise KeyError(
            'Outcome column churn or Churn not found in outcomes CSV')

    summary = {}
    groups = df['ab_group'].unique()
    counts = {}
    successes = {}
    for g in groups:
        sub = df[df['ab_group'] == g]
        n = len(sub)
        s = int(sub[churn_col].fillna(0).astype(int).sum())
        counts[g] = n
        successes[g] = s
        summary[f'{g}_rate'] = s / n if n > 0 else None

    # z-test: require two groups A and B
    if set(['A', 'B']).issubset(groups):
        count = [successes['A'], successes['B']]
        nobs = [counts['A'], counts['B']]
        stat, pval = proportions_ztest(count, nobs)
        summary['z_stat'] = float(stat)
        summary['p_value'] = float(pval)
        summary['delta'] = summary['A_rate'] - summary['B_rate']
    else:
        summary['note'] = 'Both A and B groups required for z-test'

    return summary


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--assign', required=True)
    p.add_argument('--outcomes', required=True)
    p.add_argument('--report', required=True)
    args = p.parse_args()
    s = analyze(Path(args.assign), Path(args.outcomes))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(s, f, indent=2)
    print('Saved report to', args.report)


if __name__ == '__main__':
    cli()
