"""
N-Queens Problem

Place N queens on an NxN chessboard so that no two queens attack each other.
Queens attack horizontally, vertically, and diagonally.

Why this solver:
    Uses solvOR's CP-SAT Model for clean constraint modeling. The all_different
    constraint handles columns, while explicit constraints handle diagonals.

Expected results:
    N=4: 2 solutions, N=8: 92 solutions

Reference:
    https://en.wikipedia.org/wiki/Eight_queens_puzzle
"""

from solvor import Model


def solve_n_queens(n: int) -> list[int] | None:
    """Solve N-Queens using CP-SAT.

    Args:
        n: Board size (NxN)

    Returns:
        List of column positions for each row, or None if no solution
    """
    m = Model()

    # queens[i] = column position of queen in row i
    queens = [m.int_var(0, n - 1, f"q{i}") for i in range(n)]

    # All queens in different columns
    m.add(m.all_different(queens))

    # No two queens on same diagonal
    for i in range(n):
        for j in range(i + 1, n):
            m.add(queens[i] + i != queens[j] + j)  # Forward diagonal /
            m.add(queens[i] - i != queens[j] - j)  # Backward diagonal \

    result = m.solve()
    if result.solution:
        return [result.solution[f"q{i}"] for i in range(n)]
    return None


def print_board(solution: list[int]) -> None:
    """Print board representation."""
    n = len(solution)
    for row in range(n):
        line = ""
        for col in range(n):
            line += "Q " if solution[row] == col else ". "
        print(f"  {line.rstrip()}")


def main():
    print("N-Queens Problem (using solvOR CP-SAT)")
    print("=" * 40)

    # Solve 4-Queens
    print("\n4-Queens:")
    solution = solve_n_queens(4)
    if solution:
        print(f"  Solution: {solution}")
        print_board(solution)

    # Solve 8-Queens
    print("\n8-Queens:")
    solution = solve_n_queens(8)
    if solution:
        print(f"  Solution: {solution}")
        print_board(solution)

    # Solve 12-Queens
    print("\n12-Queens:")
    solution = solve_n_queens(12)
    if solution:
        print(f"  Solution: {solution}")
        print_board(solution)


if __name__ == "__main__":
    main()
