"""
Pentomino Tiling - Exact Cover with Dancing Links

Tile a rectangle with the 12 pentomino pieces (each made of 5 squares).
Classic combinatorial puzzle solved elegantly with Algorithm X (DLX).

The 12 pentominoes are named after letters they resemble:
F, I, L, N, P, T, U, V, W, X, Y, Z

Total area: 12 pieces × 5 squares = 60 squares
Common rectangles: 6×10, 5×12, 4×15, 3×20

Source: Solomon Golomb (1965) "Polyominoes: Puzzles, Patterns, Problems"
        Donald Knuth's "Dancing Links" paper (2000)
        https://en.wikipedia.org/wiki/Pentomino

Why this solver: DLX (Dancing Links) is the optimal algorithm for exact
cover problems. Each pentomino placement covers exactly 5 cells.

This example tiles a 6×10 rectangle (2339 solutions exist).
"""

from solvor.dlx import solve_exact_cover

# Define the 12 pentominoes as relative coordinates from a base cell
# Each piece includes all rotations and reflections
PENTOMINOES = {
    "F": [
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
        [(0, 2), (1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 1), (1, 2), (2, 0), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)],
    ],
    "I": [
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    ],
    "L": [
        [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)],
        [(0, 1), (1, 1), (2, 1), (3, 0), (3, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3)],
    ],
    "N": [
        [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)],
        [(0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)],
        [(0, 1), (1, 1), (2, 0), (2, 1), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, -1)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)],
        [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)],
    ],
    "P": [
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        [(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)],
    ],
    "U": [
        [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)],
    ],
    "V": [
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
        [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],
    ],
    "W": [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        [(0, 2), (1, 1), (1, 2), (2, 0), (2, 1)],
    ],
    "X": [
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    ],
    "Y": [
        [(0, 1), (1, 0), (1, 1), (2, 1), (3, 1)],
        [(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 0), (3, 0)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        [(0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
        [(0, 2), (1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (1, 0), (2, -1), (2, 0)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
    ],
}


def generate_placements(piece_name: str, height: int, width: int) -> list[list[int]]:
    """Generate all valid placements of a piece on the board."""
    placements = []
    orientations = PENTOMINOES[piece_name]

    for orientation in orientations:
        # Find bounding box
        min_r = min(r for r, c in orientation)
        min_c = min(c for r, c in orientation)

        # Normalize to (0, 0) origin
        normalized = [(r - min_r, c - min_c) for r, c in orientation]

        # Try all positions
        for start_r in range(height):
            for start_c in range(width):
                cells = [(start_r + r, start_c + c) for r, c in normalized]

                # Check if all cells are within bounds
                if all(0 <= r < height and 0 <= c < width for r, c in cells):
                    placements.append(cells)

    # Remove duplicates
    unique = []
    seen = set()
    for cells in placements:
        key = tuple(sorted(cells))
        if key not in seen:
            seen.add(key)
            unique.append(cells)

    return unique


def solve_pentomino(height: int = 6, width: int = 10, max_solutions: int = 1):
    """Solve pentomino tiling using exact cover."""
    print(f"Pentomino Tiling {height}×{width}")
    print(f"  Board area: {height * width} squares")
    print("  Pentomino area: 12 × 5 = 60 squares")
    print()

    if height * width != 60:
        print("Error: Board area must be 60 for pentomino tiling")
        return None

    # Build exact cover matrix
    # Columns: 12 piece indicators + 60 cell indicators
    pieces = list(PENTOMINOES.keys())
    n_pieces = len(pieces)
    n_cells = height * width
    n_cols = n_pieces + n_cells

    # Generate all placements for all pieces
    rows = []
    row_info = []  # (piece_name, cells) for each row

    for piece_idx, piece_name in enumerate(pieces):
        placements = generate_placements(piece_name, height, width)
        for cells in placements:
            row = [0] * n_cols
            row[piece_idx] = 1  # Mark piece used
            for r, c in cells:
                cell_idx = r * width + c
                row[n_pieces + cell_idx] = 1  # Mark cell covered
            rows.append(row)
            row_info.append((piece_name, cells))

    print(f"  Exact cover matrix: {len(rows)} rows × {n_cols} columns")
    print()

    # Solve
    result = solve_exact_cover(rows, max_solutions=max_solutions)

    if result.ok:
        solutions = result.solution if max_solutions > 1 else [result.solution]
        print(f"Found {len(solutions)} solution(s)")
        print()

        # Display first solution
        sol_rows = solutions[0]
        board = [["." for _ in range(width)] for _ in range(height)]

        for row_idx in sol_rows:
            piece_name, cells = row_info[row_idx]
            for r, c in cells:
                board[r][c] = piece_name

        print("Solution:")
        for row in board:
            print("  " + " ".join(row))

    else:
        print(f"No solution found. Status: {result.status}")

    return result


if __name__ == "__main__":
    solve_pentomino(6, 10, max_solutions=1)
