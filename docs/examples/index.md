# Examples

A collection of worked examples, from minimal API demos to benchmark-grade implementations.

## Quick Start

The [`quick_examples/`](https://github.com/StevenBtw/solvOR/tree/main/examples/quick_examples) directory contains minimal examples (10-20 lines) for each solver:

```bash
cd examples/quick_examples
python lp_example.py
python dijkstra_example.py
python cp_example.py
```

## Categories

| Category | Description |
|----------|-------------|
| [Quick Examples](quick-examples.md) | Minimal API demos for every solver |
| [Classic Problems](classic.md) | TSP, knapsack, VRP, bin packing, job shop |
| [Puzzles](puzzles.md) | Sudoku, N-Queens, magic square, pentomino |
| [Real World](real-world.md) | Nurse scheduling, timetabling |

## Running Examples

All examples are self-contained:

```bash
python examples/puzzles/sudoku/sudoku_solver.py
python examples/classic/tsp/tsp_anneal.py
python examples/real_world/nurse_scheduling.py
```

Examples print their results and explain the problem being solved.

## Benchmark Data

All benchmark instances are from well-known open-source repositories:

- **TSPLIB**: Classic TSP instances
- **Solomon VRPTW**: Vehicle routing benchmarks
- **OR-Library**: Job shop scheduling
- **Falkenauer**: Bin packing instances
- **ITC 2007**: Timetabling competition
