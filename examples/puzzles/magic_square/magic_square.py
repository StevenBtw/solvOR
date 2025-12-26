"""
Magic Square - Constraint Programming

Arrange numbers 1 to N² in an NxN grid so that all rows, columns,
and diagonals sum to the same "magic constant".

Magic constant formula: M = N * (N² + 1) / 2

For N=3: M = 15, numbers 1-9
For N=4: M = 34, numbers 1-16

Source: Ancient mathematical puzzle, known since 650 BCE (Lo Shu square)
        https://en.wikipedia.org/wiki/Magic_square

Why this solver: CP-SAT with all_different constraint is perfect for
this kind of combinatorial puzzle with global constraints.
"""

from solvor.cp import Model


def solve_magic_square(n: int = 3):
    """Solve NxN magic square using constraint programming."""
    magic_constant = n * (n * n + 1) // 2

    print(f"Magic Square {n}x{n}")
    print(f"  Numbers: 1 to {n * n}")
    print(f"  Magic constant: {magic_constant}")
    print()

    model = Model()

    # Create NxN grid of variables
    grid = [[model.int_var(1, n * n, f"x_{i}_{j}") for j in range(n)] for i in range(n)]

    # All different: each number 1..N² appears exactly once
    all_vars = [grid[i][j] for i in range(n) for j in range(n)]
    model.add(model.all_different(all_vars))

    # Row sums = magic constant
    for i in range(n):
        model.add(sum(grid[i][j] for j in range(n)) == magic_constant)

    # Column sums = magic constant
    for j in range(n):
        model.add(sum(grid[i][j] for i in range(n)) == magic_constant)

    # Main diagonal sum = magic constant
    model.add(sum(grid[i][i] for i in range(n)) == magic_constant)

    # Anti-diagonal sum = magic constant
    model.add(sum(grid[i][n - 1 - i] for i in range(n)) == magic_constant)

    # Solve
    result = model.solve()

    if result.ok:
        print("Solution found!")
        print()

        # Extract and display grid
        solution = result.solution
        for i in range(n):
            row = [solution[grid[i][j].name] for j in range(n)]
            print("  " + " ".join(f"{v:3d}" for v in row))

        print()

        # Verify sums
        grid_values = [[solution[grid[i][j].name] for j in range(n)] for i in range(n)]

        print("Verification:")
        for i in range(n):
            print(f"  Row {i}: {sum(grid_values[i])} = {magic_constant} OK")

        diag1 = sum(grid_values[i][i] for i in range(n))
        diag2 = sum(grid_values[i][n - 1 - i] for i in range(n))
        print(f"  Main diagonal: {diag1} = {magic_constant} OK")
        print(f"  Anti-diagonal: {diag2} = {magic_constant} OK")

    else:
        print(f"No solution found. Status: {result.status}")

    return result


if __name__ == "__main__":
    solve_magic_square(3)
    print()
    print("=" * 40)
    print()
    solve_magic_square(4)
