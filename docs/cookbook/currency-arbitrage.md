# Currency Arbitrage

Detect arbitrage opportunities in currency exchange using negative cycle detection.

## The Problem

Given exchange rates between currencies, find if there's a sequence of trades that results in profit:

- Start with 1 USD
- Trade USD -> EUR -> GBP -> USD
- End with more than 1 USD

This is profit from nothing - arbitrage!

## Why It Works

By converting exchange rates to edge weights using logarithms, finding arbitrage becomes finding negative cycles:

- Multiplying rates: rate1 x rate2 x rate3
- Becomes adding logs: log(rate1) + log(rate2) + log(rate3)
- Profit means product > 1, so sum of logs > 0
- Negate logs: sum of negative logs < 0 = negative cycle!

## Example

```python
from math import log
from solvor import bellman_ford

def find_arbitrage(currencies, rates):
    """Detect currency arbitrage using Bellman-Ford."""
    n = len(currencies)
    currency_idx = {c: i for i, c in enumerate(currencies)}

    # Convert rates to edges with negative log weights
    edges = []
    for (src, dst), rate in rates.items():
        src_idx = currency_idx[src]
        dst_idx = currency_idx[dst]
        weight = -log(rate)
        edges.append((src_idx, dst_idx, weight))

    # Run Bellman-Ford
    result = bellman_ford(n, edges, start=0)

    if not result.status.is_success:
        print("Arbitrage detected!")
        return True
    return False

# Example with arbitrage opportunity
currencies = ["USD", "EUR", "GBP", "CHF"]
rates = {
    ("USD", "EUR"): 0.92,
    ("EUR", "USD"): 1.10,   # Slight inefficiency
    ("EUR", "GBP"): 0.86,
    ("GBP", "EUR"): 1.17,
    ("EUR", "CHF"): 0.97,
    ("CHF", "EUR"): 1.04,   # Arbitrage here
    ("CHF", "USD"): 1.13,
    ("USD", "CHF"): 0.89,
}

find_arbitrage(currencies, rates)
```

## Why Bellman-Ford?

Bellman-Ford detects negative cycles, which is exactly what arbitrage is after the log transformation. Dijkstra can't handle negative edges, and Floyd-Warshall finds shortest paths but doesn't directly report cycles.

The key insight: if `bellman_ford` returns a non-success status, there's a negative cycle somewhere in the graph, meaning arbitrage exists.

## See Also

- [Shortest Paths](../algorithms/graph/shortest-paths.md)
