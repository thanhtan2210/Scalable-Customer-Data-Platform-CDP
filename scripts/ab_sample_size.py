"""Sample size calculator for two-proportion test.

Usage:
  python scripts/ab_sample_size.py --p0 0.03 --mde 0.006 --alpha 0.05 --power 0.8
"""
import argparse
from statsmodels.stats.power import NormalIndPower


def calc(p0: float, mde: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0):
    # p0: baseline proportion (control churn rate)
    # mde: minimum detectable effect in absolute terms (e.g., 0.006 = 0.6%)
    p1 = p0 - mde
    # Instead use proportions_ztest effect size conversion from Cohen h
    from statsmodels.stats.proportion import proportion_effectsize

    h = proportion_effectsize(p1, p0)
    power_tool = NormalIndPower()
    n1 = power_tool.solve_power(
        effect_size=h, power=power, alpha=alpha, ratio=ratio)
    n2 = n1 * ratio
    return int(round(n1)), int(round(n2))


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--p0', type=float, required=True,
                   help='baseline proportion (e.g., 0.03)')
    p.add_argument('--mde', type=float, required=True,
                   help='minimum detectable effect (absolute)')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--power', type=float, default=0.8)
    p.add_argument('--ratio', type=float, default=1.0)
    args = p.parse_args()
    n1, n2 = calc(args.p0, args.mde, args.alpha, args.power, args.ratio)
    print(f'Required sample per group: {n1} (control), {n2} (treatment)')


if __name__ == '__main__':
    cli()
