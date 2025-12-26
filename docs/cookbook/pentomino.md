# Pentomino Tiling

Tile a rectangle with the 12 pentomino pieces using exact cover (Dancing Links).

## The Problem

Pentominoes are shapes made of 5 connected squares. There are exactly 12 distinct pentominoes (named F, I, L, N, P, T, U, V, W, X, Y, Z).

**The puzzle:** Tile a rectangle of area 60 using each pentomino exactly once.

Common rectangles: 6x10, 5x12, 4x15, 3x20

**Fun fact:** The 6x10 rectangle has exactly 2,339 distinct solutions!

## Example

```python
from solvor import solve_exact_cover

# Define pentomino shapes as relative coordinates
PENTOMINOES = {
    "F": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)],
    "I": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    "X": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    # ... (define all 12 pieces with rotations)
}

def generate_placements(piece_coords, height, width):
    """Generate all valid placements of a piece on the board."""
    placements = []
    # Normalize coordinates
    min_r = min(r for r, c in piece_coords)
    min_c = min(c for r, c in piece_coords)
    normalized = [(r - min_r, c - min_c) for r, c in piece_coords]

    # Try all positions
    for start_r in range(height):
        for start_c in range(width):
            cells = [(start_r + r, start_c + c) for r, c in normalized]
            if all(0 <= r < height and 0 <= c < width for r, c in cells):
                placements.append(cells)
    return placements

def solve_pentomino(height=6, width=10):
    """Solve pentomino tiling using exact cover."""
    pieces = list(PENTOMINOES.keys())
    n_pieces = len(pieces)
    n_cells = height * width

    # Build exact cover matrix
    rows = []
    row_info = []

    for piece_idx, piece in enumerate(pieces):
        for cells in generate_placements(PENTOMINOES[piece], height, width):
            row = [0] * (n_pieces + n_cells)
            row[piece_idx] = 1  # Piece used
            for r, c in cells:
                row[n_pieces + r * width + c] = 1  # Cell covered
            rows.append(row)
            row_info.append((piece, cells))

    result = solve_exact_cover(rows)

    if result.status.is_success:
        board = [["." for _ in range(width)] for _ in range(height)]
        for row_idx in result.solution:
            piece, cells = row_info[row_idx]
            for r, c in cells:
                board[r][c] = piece
        for row in board:
            print(" ".join(row))

    return result

solve_pentomino(6, 10)
```

## How It Works

**Exact Cover Encoding:**

1. **Columns:** 12 piece indicators + 60 cell indicators = 72 columns
2. **Rows:** Each row represents one way to place one piece
3. **Solution:** Select rows that cover each column exactly once

**Why Dancing Links?** DLX exploits the sparse matrix structure. Each row has only 6 ones (1 piece + 5 cells), making it very efficient.

## See Also

- [Sudoku](sudoku.md) - Another constraint puzzle
- [N-Queens](n-queens.md) - Placement puzzle
