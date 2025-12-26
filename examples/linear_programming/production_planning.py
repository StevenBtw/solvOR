"""
Production Planning - Linear Programming

Maximize profit by deciding how many units of each product to manufacture,
subject to resource constraints (machine time, labor, materials).

Formulation:
    maximize    sum(profit[i] * x[i])         (total profit)
    subject to  sum(resource[i][j] * x[i]) <= capacity[j]  (resource limits)
                x[i] >= 0                      (non-negative production)

Why this solver:
    LP is exact and efficient for continuous optimization with linear constraints.
    Production planning with linear costs and constraints is a classic LP application.

Expected result:
    Optimal production mix maximizing profit within resource constraints.

Reference:
    Classic operations research textbook problem (Hillier & Lieberman)
"""

from solvor import solve_lp


def main():
    # Products with their profits and resource requirements
    # Format: (name, profit, machine_hours, labor_hours, material_units)
    products = [
        ("Widget A", 30, 2, 3, 4),
        ("Widget B", 40, 3, 2, 5),
        ("Widget C", 25, 1, 4, 3),
        ("Gadget X", 50, 4, 3, 6),
        ("Gadget Y", 35, 2, 5, 4),
    ]

    # Resource capacities per day
    resources = {
        "machine_hours": 100,
        "labor_hours": 120,
        "material_units": 150,
    }

    n_products = len(products)

    # Extract data
    names = [p[0] for p in products]
    profits = [p[1] for p in products]
    requirements = [p[2:] for p in products]  # requirements[product][resource]
    capacities = list(resources.values())

    # Build constraint matrix (A @ x <= b)
    A = []
    b = capacities
    for j in range(len(capacities)):
        row = [requirements[i][j] for i in range(n_products)]
        A.append(row)

    print("Production Planning - Linear Programming")
    print("=" * 55)
    print("\nProducts available:")
    print(f"{'Product':<12} {'Profit':<10} {'Machine':<10} {'Labor':<10} {'Material':<10}")
    print("-" * 52)
    for prod in products:
        print(f"{prod[0]:<12} ${prod[1]:<9} {prod[2]:<10} {prod[3]:<10} {prod[4]:<10}")

    print("\nResource capacities (per day):")
    for name, val in resources.items():
        print(f"  {name}: {val}")

    # Solve (maximize profit)
    result = solve_lp(profits, A, b, minimize=False)

    print("\n" + "=" * 55)
    print("OPTIMAL PRODUCTION PLAN")
    print("=" * 55)

    if result.ok:
        x = result.solution
        print(f"\nMaximum profit: ${result.objective:.2f}")
        print("\nProduction quantities:")

        total_profit = 0
        for i, (name, qty) in enumerate(zip(names, x)):
            if qty > 0.01:
                profit_contribution = profits[i] * qty
                total_profit += profit_contribution
                print(f"  {name:<12}: {qty:>6.2f} units  (profit: ${profit_contribution:.2f})")

        # Verify resource usage
        print("\nResource utilization:")
        resource_names = list(resources.keys())
        for j, name in enumerate(resource_names):
            used = sum(requirements[i][j] * x[i] for i in range(n_products))
            cap = capacities[j]
            pct = 100 * used / cap
            print(f"  {name:<16}: {used:>6.1f} / {cap:>6.1f}  ({pct:>5.1f}%)")
    else:
        print(f"No solution found: {result.status.name}")


if __name__ == "__main__":
    main()
