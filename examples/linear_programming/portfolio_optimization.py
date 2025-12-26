"""
Portfolio Optimization - Linear Programming Approximation

Allocate investment across assets to maximize expected return
subject to risk constraints. Simplified Markowitz model using LP.

The classic Markowitz model is quadratic (minimize variance).
This LP approximation uses:
- Maximize expected return
- Diversification constraints (max allocation per asset)
- Minimum return requirements

Source: Markowitz (1952) "Portfolio Selection"
        https://en.wikipedia.org/wiki/Modern_portfolio_theory
        Nobel Prize in Economics (1990)

Real-world considerations not modeled:
- Transaction costs
- Tax implications
- Liquidity constraints
- Short selling restrictions (we assume long-only)

Why LP: While true mean-variance is quadratic, many practical
constraints are linear. This shows the LP approach.
"""

from solvor.simplex import solve_lp

# Asset data: (name, expected_annual_return, risk_category)
# Returns are in percentage (e.g., 8.0 = 8%)
ASSETS = [
    ("US Large Cap", 8.0, "medium"),
    ("US Small Cap", 10.0, "high"),
    ("International Developed", 7.0, "medium"),
    ("Emerging Markets", 12.0, "high"),
    ("US Bonds", 3.0, "low"),
    ("International Bonds", 2.5, "low"),
    ("Real Estate (REITs)", 7.5, "medium"),
    ("Commodities", 5.0, "high"),
]

# Risk constraints
MAX_HIGH_RISK = 0.30  # Max 30% in high-risk assets
MAX_SINGLE_ASSET = 0.25  # Max 25% in any single asset
MIN_BONDS = 0.20  # Min 20% in bonds


def solve_portfolio(target_return: float = 6.0):
    """Optimize portfolio allocation with LP."""
    print("Portfolio Optimization")
    print("=" * 50)
    print()

    n = len(ASSETS)

    print("Available assets:")
    print(f"{'Asset':<25} {'Expected Return':>15} {'Risk':>10}")
    print("-" * 55)
    for name, ret, risk in ASSETS:
        print(f"{name:<25} {ret:>14.1f}% {risk:>10}")
    print()

    print("Constraints:")
    print(f"  Target minimum return: {target_return}%")
    print(f"  Max high-risk allocation: {MAX_HIGH_RISK * 100:.0f}%")
    print(f"  Max single asset: {MAX_SINGLE_ASSET * 100:.0f}%")
    print(f"  Min bonds allocation: {MIN_BONDS * 100:.0f}%")
    print()

    # Objective: maximize return (minimize negative return)
    returns = [asset[1] for asset in ASSETS]
    c = [-r for r in returns]  # Negate for minimization

    A = []
    b = []

    # Constraint: weights sum to 1
    A.append([1.0] * n)
    b.append(1.0)
    A.append([-1.0] * n)
    b.append(-1.0)

    # Constraint: minimum target return
    # Σ(return_i × w_i) ≥ target_return
    # → -Σ(return_i × w_i) ≤ -target_return
    A.append([-r for r in returns])
    b.append(-target_return)

    # Constraint: max high-risk allocation
    high_risk_mask = [1.0 if asset[2] == "high" else 0.0 for asset in ASSETS]
    A.append(high_risk_mask)
    b.append(MAX_HIGH_RISK)

    # Constraint: max per asset
    for i in range(n):
        row = [0.0] * n
        row[i] = 1.0
        A.append(row)
        b.append(MAX_SINGLE_ASSET)

    # Constraint: min bonds
    # Bonds are assets with "low" risk
    bonds_mask = [-1.0 if asset[2] == "low" else 0.0 for asset in ASSETS]
    A.append(bonds_mask)
    b.append(-MIN_BONDS)

    # Non-negativity (implicit in simplex)

    # Solve
    result = solve_lp(c=c, A=A, b=b)

    if result.ok:
        print("Optimal portfolio found!")
        print()

        weights = result.solution
        print("Allocation:")
        print(f"{'Asset':<25} {'Weight':>10} {'Contribution':>12}")
        print("-" * 50)

        total_return = 0
        for i, (name, ret, risk) in enumerate(ASSETS):
            w = weights[i]
            contrib = w * ret
            total_return += contrib
            if w > 0.001:  # Only show non-zero allocations
                print(f"{name:<25} {w * 100:>9.1f}% {contrib:>11.2f}%")

        print("-" * 50)
        print(f"{'Expected portfolio return:':<25} {total_return:>21.2f}%")
        print()

        # Verify constraints
        high_risk_total = sum(weights[i] for i in range(n) if ASSETS[i][2] == "high")
        bonds_total = sum(weights[i] for i in range(n) if ASSETS[i][2] == "low")
        max_weight = max(weights)

        print("Constraint verification:")
        print(f"  High-risk allocation: {high_risk_total * 100:.1f}% <= {MAX_HIGH_RISK * 100:.0f}% OK")
        print(f"  Bonds allocation: {bonds_total * 100:.1f}% >= {MIN_BONDS * 100:.0f}% OK")
        print(f"  Max single asset: {max_weight * 100:.1f}% <= {MAX_SINGLE_ASSET * 100:.0f}% OK")
        print(f"  Portfolio return: {total_return:.2f}% >= {target_return:.1f}% OK")

    else:
        print(f"No feasible portfolio found. Status: {result.status}")
        print("Try lowering the target return or relaxing constraints.")

    return result


if __name__ == "__main__":
    # Try different target returns
    print("=" * 60)
    print("Conservative portfolio (5% target)")
    print("=" * 60)
    solve_portfolio(target_return=5.0)

    print()
    print("=" * 60)
    print("Moderate portfolio (6% target)")
    print("=" * 60)
    solve_portfolio(target_return=6.0)

    print()
    print("=" * 60)
    print("Aggressive portfolio (7% target)")
    print("=" * 60)
    solve_portfolio(target_return=7.0)
