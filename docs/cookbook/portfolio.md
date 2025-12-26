# Portfolio Optimization

Allocate investment across assets to maximize expected return subject to risk constraints.

## The Problem

Given assets with expected returns and risk categories, find the optimal allocation that:

- Maximizes expected portfolio return
- Limits exposure to high-risk assets
- Ensures diversification
- Maintains minimum allocation to stable assets

## Example

```python
from solvor import solve_lp

# Asset data: (name, expected_annual_return%, risk_category)
ASSETS = [
    ("US Large Cap", 8.0, "medium"),
    ("US Small Cap", 10.0, "high"),
    ("International", 7.0, "medium"),
    ("Emerging Markets", 12.0, "high"),
    ("US Bonds", 3.0, "low"),
    ("Intl Bonds", 2.5, "low"),
    ("REITs", 7.5, "medium"),
    ("Commodities", 5.0, "high"),
]

MAX_HIGH_RISK = 0.30
MAX_SINGLE_ASSET = 0.25
MIN_BONDS = 0.20

n = len(ASSETS)
returns = [asset[1] for asset in ASSETS]

# Objective: maximize return
c = [-r for r in returns]

A = []
b = []

# Weights sum to 1
A.append([1.0] * n)
b.append(1.0)
A.append([-1.0] * n)
b.append(-1.0)

# Max high-risk allocation
high_risk = [1.0 if asset[2] == "high" else 0.0 for asset in ASSETS]
A.append(high_risk)
b.append(MAX_HIGH_RISK)

# Max per asset
for i in range(n):
    row = [0.0] * n
    row[i] = 1.0
    A.append(row)
    b.append(MAX_SINGLE_ASSET)

# Min bonds allocation
bonds = [-1.0 if asset[2] == "low" else 0.0 for asset in ASSETS]
A.append(bonds)
b.append(-MIN_BONDS)

result = solve_lp(c=c, A=A, b=b)

if result.status.is_success:
    weights = result.solution
    print("Optimal portfolio:")
    total_return = 0
    for i, (name, ret, risk) in enumerate(ASSETS):
        w = weights[i]
        if w > 0.001:
            contrib = w * ret
            total_return += contrib
            print(f"  {name:<20}: {w*100:5.1f}%")
    print(f"\nExpected return: {total_return:.2f}%")
```

## See Also

- [Diet Problem](diet.md) - Another LP classic
- [Production Planning](production-planning.md)
