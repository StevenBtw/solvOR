"""
Diet Problem - Classic Linear Programming

Find the cheapest combination of foods that meets minimum nutritional requirements.
This is one of the earliest applications of LP, studied by Stigler in 1945.

Formulation:
    minimize    sum(cost[i] * x[i])           (total food cost)
    subject to  sum(nutrition[i][j] * x[i]) >= required[j]  (nutritional requirements)
                x[i] >= 0                     (non-negative quantities)

Since solve_lp uses A @ x <= b, we negate nutritional constraints:
    -sum(nutrition[i][j] * x[i]) <= -required[j]

Why this solver:
    LP is exact and efficient for continuous optimization with linear constraints.
    The diet problem is a textbook LP formulation.

Expected result:
    Optimal diet meeting all requirements at minimum cost.

Reference:
    Stigler, G.J. (1945) "The Cost of Subsistence"
    https://en.wikipedia.org/wiki/Stigler_diet
"""

from solvor import solve_lp


def main():
    # Foods with their costs per unit and nutritional content
    # Format: (name, cost, calories, protein_g, vitamin_c_mg, iron_mg)
    foods = [
        ("Bread", 2.0, 250, 8, 0, 2),
        ("Milk", 1.5, 150, 8, 2, 0.1),
        ("Eggs", 3.0, 155, 13, 0, 1.8),
        ("Chicken", 5.0, 165, 31, 0, 1.0),
        ("Beans", 1.0, 120, 8, 2, 2.5),
        ("Orange", 0.5, 60, 1, 70, 0.2),
        ("Spinach", 2.5, 25, 3, 30, 3.0),
    ]

    # Minimum daily requirements
    requirements = {
        "calories": 2000,
        "protein_g": 50,
        "vitamin_c_mg": 60,
        "iron_mg": 10,
    }

    n_foods = len(foods)
    n_nutrients = len(requirements)

    # Extract data
    names = [f[0] for f in foods]
    costs = [f[1] for f in foods]  # c vector
    nutrition = [f[2:] for f in foods]  # nutrition[food][nutrient]
    required = list(requirements.values())

    # Build constraint matrix (A @ x <= b means -nutrition @ x <= -required)
    A = []
    b = []
    for j in range(n_nutrients):
        # -sum(nutrition[i][j] * x[i]) <= -required[j]
        row = [-nutrition[i][j] for i in range(n_foods)]
        A.append(row)
        b.append(-required[j])

    print("Diet Problem - Linear Programming")
    print("=" * 50)
    print("\nFoods available:")
    print(f"{'Food':<12} {'Cost':<8} {'Cal':<8} {'Prot':<8} {'VitC':<8} {'Iron':<8}")
    print("-" * 52)
    for food in foods:
        print(f"{food[0]:<12} ${food[1]:<7.2f} {food[2]:<8} {food[3]:<8} {food[4]:<8} {food[5]:<8}")

    print("\nMinimum daily requirements:")
    for name, val in requirements.items():
        print(f"  {name}: {val}")

    # Solve
    result = solve_lp(costs, A, b)

    print("\n" + "=" * 50)
    print("OPTIMAL DIET")
    print("=" * 50)

    if result.ok:
        x = result.solution
        print(f"\nTotal cost: ${result.objective:.2f}")
        print("\nFood quantities (units):")

        for i, (name, qty) in enumerate(zip(names, x)):
            if qty > 0.01:  # Only show foods used
                print(f"  {name:<12}: {qty:.2f}")

        # Verify nutritional content
        print("\nNutrients provided:")
        nutrient_names = list(requirements.keys())
        for j, name in enumerate(nutrient_names):
            total = sum(nutrition[i][j] * x[i] for i in range(n_foods))
            req = required[j]
            status = "OK" if total >= req - 0.01 else "LOW"
            print(f"  {name:<12}: {total:>8.1f} (min: {req}) [{status}]")
    else:
        print(f"No solution found: {result.status.name}")


if __name__ == "__main__":
    main()
