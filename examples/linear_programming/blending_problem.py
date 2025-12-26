"""
Oil Blending Problem - Linear Programming

Classic refinery optimization: blend crude oils to produce gasoline
that meets octane and sulfur specifications at minimum cost.

Problem: Given different crude oil types with varying properties
(octane, sulfur content, cost), blend them to produce gasoline
that meets minimum octane and maximum sulfur requirements.

Source: Classic OR textbook problem
        Williams (2013) "Model Building in Mathematical Programming"
        https://en.wikipedia.org/wiki/Blending_problem

Formulation:
    Variables: x_i = fraction of crude i in blend (0 ≤ x_i ≤ 1)

    Minimize: Σ(cost_i × x_i)  (total cost per gallon)

    Subject to:
        Σ x_i = 1                           (fractions sum to 1)
        Σ(octane_i × x_i) ≥ min_octane     (octane requirement)
        Σ(sulfur_i × x_i) ≤ max_sulfur     (sulfur limit)

Why LP: Properties blend linearly by volume fraction.
No integer constraints needed, pure continuous optimization.
"""

from solvor.simplex import solve_lp

# Crude oil data
# Format: (name, octane_rating, sulfur_pct, cost_per_gallon)
CRUDE_OILS = [
    ("Light Sweet", 95, 0.2, 2.50),
    ("Medium", 88, 0.8, 2.10),
    ("Heavy Sour", 82, 1.5, 1.70),
    ("Ultra Light", 98, 0.1, 3.00),
]

# Gasoline specifications
MIN_OCTANE = 87  # Minimum octane rating
MAX_SULFUR = 0.5  # Maximum sulfur percentage


def solve_blending():
    """Solve oil blending problem with LP."""
    print("Oil Blending Problem")
    print("=" * 50)
    print()

    n = len(CRUDE_OILS)

    print("Available crude oils:")
    print(f"{'Name':<15} {'Octane':>8} {'Sulfur%':>8} {'Cost/gal':>10}")
    print("-" * 45)
    for name, octane, sulfur, cost in CRUDE_OILS:
        print(f"{name:<15} {octane:>8.0f} {sulfur:>8.1f} ${cost:>9.2f}")
    print()

    print("Gasoline requirements:")
    print(f"  Minimum octane: {MIN_OCTANE}")
    print(f"  Maximum sulfur: {MAX_SULFUR}%")
    print()

    # Build LP
    # Variables: x_0, x_1, ..., x_{n-1} (fraction of each crude)

    # Objective: minimize cost
    costs = [oil[3] for oil in CRUDE_OILS]

    # Constraints (in Ax ≤ b form)
    A = []
    b = []

    # Constraint 1: Σ x_i = 1 (fractions sum to 1)
    # Split into: Σ x_i ≤ 1 and -Σ x_i ≤ -1
    A.append([1.0] * n)
    b.append(1.0)
    A.append([-1.0] * n)
    b.append(-1.0)

    # Constraint 2: Σ(octane_i × x_i) ≥ MIN_OCTANE
    # Rewrite as: -Σ(octane_i × x_i) ≤ -MIN_OCTANE
    octanes = [oil[1] for oil in CRUDE_OILS]
    A.append([-o for o in octanes])
    b.append(-MIN_OCTANE)

    # Constraint 3: Σ(sulfur_i × x_i) ≤ MAX_SULFUR
    sulfurs = [oil[2] for oil in CRUDE_OILS]
    A.append(sulfurs)
    b.append(MAX_SULFUR)

    # Constraint 4: x_i ≥ 0 (non-negativity, implicit in simplex)
    # Constraint 5: x_i ≤ 1 (at most 100% of any crude)
    for i in range(n):
        row = [0.0] * n
        row[i] = 1.0
        A.append(row)
        b.append(1.0)

    # Solve
    result = solve_lp(c=costs, A=A, b=b)

    if result.ok:
        print("Optimal blend found!")
        print()

        fractions = result.solution
        print("Blend composition:")
        for i, (name, octane, sulfur, cost) in enumerate(CRUDE_OILS):
            pct = fractions[i] * 100
            if pct > 0.01:  # Only show non-zero components
                print(f"  {name}: {pct:.1f}%")

        print()

        # Verify properties
        blend_octane = sum(octanes[i] * fractions[i] for i in range(n))
        blend_sulfur = sum(sulfurs[i] * fractions[i] for i in range(n))
        blend_cost = result.objective

        print("Blend properties:")
        print(f"  Octane rating: {blend_octane:.1f} (min: {MIN_OCTANE}) OK")
        print(f"  Sulfur content: {blend_sulfur:.2f}% (max: {MAX_SULFUR}%) OK")
        print(f"  Cost per gallon: ${blend_cost:.3f}")

    else:
        print(f"No feasible blend found. Status: {result.status}")

    return result


if __name__ == "__main__":
    solve_blending()
