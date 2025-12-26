"""
Maximum Bipartite Matching - Max Flow Reduction

Match workers to jobs maximizing the number of assignments.
Each worker can do certain jobs, each job needs exactly one worker.

Classic reduction: Create source S, sink T
- S → each worker (capacity 1)
- Each worker → each job they can do (capacity 1)
- Each job → T (capacity 1)
Maximum flow = maximum matching

Source: König's theorem (1931), Ford-Fulkerson (1956)
        https://en.wikipedia.org/wiki/Maximum_bipartite_matching

Applications:
- Job assignment
- College admissions
- Kidney exchange
- Stable marriage variants

Why max flow: Elegant reduction that's O(VE) using Ford-Fulkerson.
For larger problems, Hopcroft-Karp is faster at O(E√V).
"""

from solvor.flow import max_flow

# Workers and the jobs they're qualified for
WORKERS = {
    "Alice": ["Frontend", "Backend", "Design"],
    "Bob": ["Backend", "DevOps", "Database"],
    "Carol": ["Frontend", "Design"],
    "Dave": ["DevOps", "Database", "Backend"],
    "Eve": ["Design", "Frontend", "QA"],
    "Frank": ["QA", "DevOps"],
}

JOBS = ["Frontend", "Backend", "Design", "DevOps", "Database", "QA"]


def solve_matching():
    """Solve bipartite matching using max flow."""
    print("Maximum Bipartite Matching")
    print("=" * 50)
    print()

    print("Workers and their qualifications:")
    for worker, jobs in WORKERS.items():
        print(f"  {worker}: {', '.join(jobs)}")
    print()

    print(f"Jobs to fill: {', '.join(JOBS)}")
    print()

    # Build flow network
    # Node naming: "S" (source), "T" (sink), "W_name" (workers), "J_name" (jobs)
    graph = {}

    # Source connections
    source = "S"
    graph[source] = [(f"W_{w}", 1, 0) for w in WORKERS]

    # Worker to job connections
    for worker, qualified_jobs in WORKERS.items():
        graph[f"W_{worker}"] = [(f"J_{job}", 1, 0) for job in qualified_jobs]

    # Job to sink connections
    for job in JOBS:
        graph[f"J_{job}"] = [("T", 1, 0)]

    sink = "T"

    # Solve max flow
    result = max_flow(graph, source, sink)

    print(f"Maximum matching size: {int(result.objective)}")
    print()

    # Extract matching from flow
    matching = []
    for (u, v), flow_val in result.solution.items():
        if flow_val > 0 and u.startswith("W_") and v.startswith("J_"):
            worker = u[2:]  # Remove "W_" prefix
            job = v[2:]  # Remove "J_" prefix
            matching.append((worker, job))

    print("Optimal assignment:")
    for worker, job in sorted(matching):
        print(f"  {worker} -> {job}")

    # Check for unassigned
    assigned_workers = {w for w, j in matching}
    assigned_jobs = {j for w, j in matching}

    unassigned_workers = set(WORKERS.keys()) - assigned_workers
    unassigned_jobs = set(JOBS) - assigned_jobs

    if unassigned_workers:
        print()
        print(f"Unassigned workers: {', '.join(unassigned_workers)}")

    if unassigned_jobs:
        print()
        print(f"Unfilled jobs: {', '.join(unassigned_jobs)}")

    return result


if __name__ == "__main__":
    solve_matching()
