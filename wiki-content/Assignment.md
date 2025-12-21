# Assignment

You have workers and tasks. Each worker can do each task at some cost. What's the optimal one-to-one matching? This is the assignment problem, and it's one of the most studied problems in combinatorial optimization.

## Solvers in this Category

### [hungarian](hungarian)
The Hungarian algorithm for optimal assignment. O(n³) and the go-to for pure assignment problems. Elegant, optimal, and reasonably fast.

**Complexity:** O(n³)
**Guarantees:** Optimal assignment

### [solve_assignment](solve_assignment)
Assignment via min-cost flow reduction. Models assignment as a bipartite flow problem. Less efficient than Hungarian but conceptually simple.

**Complexity:** O(n³) via flow
**Guarantees:** Optimal assignment

## When to Use This Category

**Perfect for:**
- Worker-task assignment (minimize cost or maximize productivity)
- Job-machine assignment (minimize completion time)
- Server-request allocation (minimize latency or balance load)
- Matching problems where each entity pairs with exactly one other
- Any problem shaped like "assign n things to n slots one-to-one"

**The magic:** Polynomial-time optimal matching. What could be exponential brute force (n! possibilities) becomes O(n³) with clever algorithms.

## When NOT to Use This Category

- **One worker handles multiple tasks** → Use flow algorithms or MILP
- **Matching within a single group** (e.g., roommates, doubles partners) → Use stable matching or min-cost perfect matching
- **Constraints beyond costs** (availability, skills, preferences) → Use MILP or constraint programming
- **Online assignment** (tasks arrive over time) → Use online algorithms or greedy heuristics

## Comparing Solvers

| Solver | Implementation | Use When |
|--------|----------------|----------|
| `hungarian` | Specialized O(n³) algorithm | Pure assignment, best performance |
| `solve_assignment` | Min-cost flow reduction | Learning, or need flow-based extensions |

**Rule of thumb:** Use `hungarian` for assignment. Only use `solve_assignment` if you need to understand flow-based approaches or extend to more complex flow problems.

## Quick Example

```python
from solvor import hungarian, solve_assignment

# Cost matrix: rows = workers, columns = tasks
costs = [
    [10, 5, 13],  # Worker 0: Task A costs 10, B costs 5, C costs 13
    [3, 9, 18],   # Worker 1: Task A costs 3, B costs 9, C costs 18
    [10, 6, 12]   # Worker 2: Task A costs 10, B costs 6, C costs 12
]

# Hungarian algorithm
result = hungarian(costs)
print(result.solution)  # [1, 0, 2] -> Worker 0→Task B, 1→A, 2→C
print(result.objective) # 20 (5 + 3 + 12)

# Maximize instead of minimize
result = hungarian(costs, minimize=False)
print(result.solution)  # Assignment maximizing total value

# Same problem via flow
result = solve_assignment(costs)
print(result.solution)  # [1, 0, 2]
```

## Understanding the Result

`result.solution[i] = j` means "assign worker i to task j"

**Example:**
```python
solution = [1, 0, 2]
# Worker 0 → Task 1
# Worker 1 → Task 0
# Worker 2 → Task 2
```

For rectangular matrices (m workers, n tasks where m ≠ n):
- If m < n: Some tasks unassigned
- If m > n: Some workers unassigned (solution[i] = -1)

## Tips & Tricks

### Problem Formulation
- **Rows = workers, Columns = tasks** (the convention)
- **Minimize costs or maximize profits** - Just flip the flag
- **Rectangular matrices work** - m and n don't need to be equal 
- **Large costs for infeasibility** - Set cost[i][j] = 1e9 if worker i can't do task j

### Maximization Trick
To maximize instead of minimize:
1. Find max cost: `max_cost = max(max(row) for row in costs)`
2. Flip costs: `flipped[i][j] = max_cost - costs[i][j]`
3. Solve minimum assignment on flipped
4. Original maximum = m × max_cost - flipped result

Or just use `minimize=False` parameter.

### Performance
- **O(n³) is fast up to n=1000** - Beyond that, consider specialized implementations
- **Sparse matrices** - If most costs are infinity (infeasible), Hungarian still works but flow methods might be faster
- **Multiple runs** - Hungarian doesn't benefit much from warm-starting

## How Hungarian Works (Intuition)

Think of the cost matrix as a bipartite graph: workers on left, tasks on right, edge weights = costs.

1. **Subtract row minimums** - Each worker reduces their costs by their cheapest task
2. **Subtract column minimums** - Each task reduces remaining costs by cheapest worker
3. **Find maximum matching with zero-cost edges** - Assign using only zero edges if possible
4. **If matching incomplete, adjust costs** - Reveal new zero edges that enable better matching
5. **Repeat until full matching** - Terminates with optimal assignment

**Why it works:** The row/column reductions preserve optimality (shifting costs equally doesn't change relative preferences). The algorithm maintains "dual feasibility" while searching for "primal feasibility."

## Real-World Applications

### Software & Computing
- **Load Balancing:** Assign servers to requests (minimize latency or balance load)
- **Task Scheduling:** Assign jobs to machines (minimize completion time)
- **Resource Allocation:** Assign VMs to physical hosts

### Operations Research
- **Worker Assignment:** Assign employees to shifts or projects
- **Vehicle Routing (simplified):** Assign vehicles to routes
- **Facility Assignment:** Assign customers to facilities

### Academia
- **Course Assignment:** Assign students to courses or TAs to sections
- **Reviewer Assignment:** Assign papers to reviewers (maximize expertise match)
- **Roommate Assignment:** With additional constraints beyond costs

### Practical Example: Server-Request Assignment

```python
# 3 servers, 4 requests. Minimize total latency.
# Latency matrix (ms): rows = servers, cols = requests
latencies = [
    [10, 50, 20, 40],  # Server 0 latencies
    [60, 20, 40, 30],  # Server 1 latencies
    [30, 40, 10, 20],  # Server 2 latencies
]

result = hungarian(latencies)
# result.solution = [0, 1, 2, ?]
# Interpretation:
#   Server 0 handles Request 0 (10ms)
#   Server 1 handles Request 1 (20ms)
#   Server 2 handles Request 2 (10ms)
#   Request 3 unassigned (only 3 servers for 4 requests)

# Total latency = 10 + 20 + 10 = 40ms
print(result.objective)  # 40
```

## Extensions & Variations

### Variants Not Directly Supported
- **Stable Marriage:** Workers and tasks have preferences, find stable matching
- **Fair Assignment:** Maximize minimum assignment quality (max-min fairness)
- **Generalized Assignment:** Workers have capacities, can handle multiple tasks
- **Online Assignment:** Tasks arrive over time, decisions are irrevocable

For these, consider MILP formulations or specialized algorithms.

### Relationship to Other Problems
- **Bipartite Matching:** Unweighted version (just find any perfect matching)
- **Transportation Problem:** Assignment with supplies/demands (use min-cost flow)
- **Traveling Salesman:** Visit all cities (much harder, NP-hard)

## Common Gotchas

1. **Solution indices:** `solution[i]` is the task assigned to worker `i`, not worker assigned to task `i`
2. **Rectangular matrices:** If m ≠ n, some entities will be unassigned
3. **Infeasibility encoding:** Use large finite costs (1e9), not infinity (breaks numerics)
4. **Maximize vs minimize:** Double-check which direction you want

## See Also

- [Network Flow & MST](Network-Flow-&-MST) - Flow-based assignment and min-cost flow
- [Linear & Integer Programming](Linear-&-Integer-Programming) - MILP formulation of assignment
- [Cookbook: Assignment](Cookbook-Assignment) - Full worked example
