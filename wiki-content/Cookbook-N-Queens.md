# Cookbook: N-Queens

Place N queens on an N×N chessboard with no conflicts (no two queens attack each other).

## Example

```python
from solvor import Model

def solve_n_queens(n):
    m = Model()
    
    # queens[i] = column position of queen in row i
    queens = [m.int_var(0, n-1, f'q{i}') for i in range(n)]
    
    # All queens in different columns
    m.add(m.all_different(queens))
    
    # No two queens on same diagonal
    for i in range(n):
        for j in range(i+1, n):
            m.add(queens[i] + i != queens[j] + j)  # Forward diagonal
            m.add(queens[i] - i != queens[j] - j)  # Backward diagonal
    
    result = m.solve()
    return [result.solution[f'q{i}'] for i in range(n)] if result.solution else None

solution = solve_n_queens(8)
print(f"8-Queens solution: {solution}")
```

## See Also
- [`Model`](Model-(CP-SAT))
- [Cookbook: Sudoku](Cookbook-Sudoku)
