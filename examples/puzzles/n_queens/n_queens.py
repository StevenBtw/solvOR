"""
N-Queens Problem

Place N queens on an NxN chessboard so that no two queens attack each other.
Queens attack horizontally, vertically, and diagonally.

Formulation:
    Variables: q[i] = column position of queen in row i (0 to N-1)
    Constraints:
    - all_different(q)              (no two queens in same column)
    - all_different(q[i] + i)       (no two queens on same diagonal /)
    - all_different(q[i] - i)       (no two queens on same diagonal)

Why this solver:
    This classic backtracking implementation efficiently prunes the search
    space using column and diagonal tracking sets.

Expected results:
    N=4: 2 solutions, N=8: 92 solutions, N=12: 14,200 solutions

Reference:
    Classic combinatorial problem, see:
    https://en.wikipedia.org/wiki/Eight_queens_puzzle
"""


def solve_n_queens(n, find_all=False):
    """Solve N-Queens using backtracking.

    Args:
        n: Board size (NxN)
        find_all: If True, find all solutions; otherwise find first solution

    Returns:
        List of solutions, where each solution is a list of column positions
    """
    solutions = []

    def backtrack(row, placement, cols, diag1, diag2):
        if row == n:
            solutions.append(list(placement))
            return not find_all  # Return True to stop if only need one

        for col in range(n):
            d1 = row + col  # / diagonal
            d2 = row - col  # \ diagonal

            if col not in cols and d1 not in diag1 and d2 not in diag2:
                placement.append(col)
                cols.add(col)
                diag1.add(d1)
                diag2.add(d2)

                if backtrack(row + 1, placement, cols, diag1, diag2):
                    return True

                placement.pop()
                cols.remove(col)
                diag1.remove(d1)
                diag2.remove(d2)

        return False

    backtrack(0, [], set(), set(), set())
    return solutions


def str_board(solution):
    """Return string representation of board."""
    n = len(solution)
    lines = []
    for row in range(n):
        line = ""
        for col in range(n):
            if solution[row] == col:
                line += "Q "
            else:
                line += ". "
        lines.append(line.rstrip())
    return "\n".join(lines)


def main():
    print("N-Queens Problem")
    print("=" * 40)

    # Find all solutions for N=4 (only 2 solutions)
    print("\n4-Queens (all 2 solutions):")
    solutions = solve_n_queens(4, find_all=True)
    print(f"  Found {len(solutions)} solutions")
    for i, sol in enumerate(solutions):
        print(f"\n  Solution {i + 1}:")
        for line in str_board(sol).split("\n"):
            print(f"    {line}")

    # Find all solutions for N=8
    print("\n8-Queens:")
    solutions = solve_n_queens(8, find_all=True)
    print(f"  Found {len(solutions)} solutions (expected: 92)")
    print("\n  First solution:")
    for line in str_board(solutions[0]).split("\n"):
        print(f"    {line}")

    # Find one solution for N=12
    print("\n12-Queens:")
    solutions = solve_n_queens(12, find_all=False)
    print("  Found a solution:")
    for line in str_board(solutions[0]).split("\n"):
        print(f"    {line}")


if __name__ == "__main__":
    main()
