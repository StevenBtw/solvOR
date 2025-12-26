"""
Currency Arbitrage Detection - Bellman-Ford Negative Cycle

Detect arbitrage opportunities in currency exchange rates.
An arbitrage exists when you can trade through a cycle of currencies
and end up with more money than you started.

Example: USD -> EUR -> GBP -> USD with profit

Mathematical formulation:
- Edge weight = -log(exchange_rate)
- Negative cycle in this graph = arbitrage opportunity

Source: Classic application of Bellman-Ford algorithm
        https://en.wikipedia.org/wiki/Arbitrage
        Cormen et al. "Introduction to Algorithms"

Why Bellman-Ford: Only algorithm that detects negative cycles.
Dijkstra fails with negative edges, Floyd-Warshall detects but doesn't
extract the cycle path.
"""

from math import log

from solvor.bellman_ford import bellman_ford

# Exchange rates (fictional, includes arbitrage opportunity)
# Format: (from_currency, to_currency, rate)
# rate = how many units of to_currency you get for 1 unit of from_currency
EXCHANGE_RATES = [
    ("USD", "EUR", 0.92),
    ("EUR", "USD", 1.10),  # Note: 0.92 * 1.10 = 1.012 (small profit)
    ("USD", "GBP", 0.79),
    ("GBP", "USD", 1.27),
    ("EUR", "GBP", 0.86),
    ("GBP", "EUR", 1.17),  # Note: 0.86 * 1.17 = 1.006 (small profit)
    ("USD", "JPY", 149.50),
    ("JPY", "USD", 0.0068),  # 149.50 * 0.0068 = 1.017 (profit!)
    ("EUR", "JPY", 162.30),
    ("JPY", "EUR", 0.0063),
    ("GBP", "JPY", 189.40),
    ("JPY", "GBP", 0.0054),
    # Add an obvious arbitrage cycle
    ("USD", "CHF", 0.88),
    ("CHF", "EUR", 1.08),
    ("EUR", "USD", 1.10),  # USD->CHF->EUR->USD: 0.88 * 1.08 * 1.10 = 1.045 (4.5% profit!)
]


def build_edges():
    """Build edge list with -log(rate) as weights for Bellman-Ford."""
    # Map currency names to integer indices
    currencies = sorted(set(c for src, dst, _ in EXCHANGE_RATES for c in [src, dst]))
    currency_to_idx = {c: i for i, c in enumerate(currencies)}

    edges = []
    for src, dst, rate in EXCHANGE_RATES:
        # Use -log(rate) so that multiplying rates becomes adding weights
        # A negative cycle means product of rates > 1 (profit)
        weight = -log(rate)
        edges.append((currency_to_idx[src], currency_to_idx[dst], weight))

    return edges, currencies, currency_to_idx


def main():
    print("Currency Arbitrage Detection")
    print("=" * 50)
    print()

    edges, currencies, currency_to_idx = build_edges()
    n_nodes = len(currencies)

    print(f"Currencies: {', '.join(currencies)}")
    print(f"Exchange rates: {len(EXCHANGE_RATES)}")
    print()

    # Run Bellman-Ford from USD to detect negative cycles
    print("Checking for arbitrage opportunities...")
    print()

    start_idx = currency_to_idx["USD"]
    result = bellman_ford(start_idx, edges, n_nodes)

    if result.status.name == "UNBOUNDED":
        print("Arbitrage detected! (Negative cycle found)")
        print()

        # Find profitable cycles manually
        print("Profitable 3-currency cycles:")
        for c1 in currencies:
            for c2 in currencies:
                for c3 in currencies:
                    if c1 != c2 and c2 != c3 and c3 != c1:
                        r1 = next((r for s, d, r in EXCHANGE_RATES if s == c1 and d == c2), None)
                        r2 = next((r for s, d, r in EXCHANGE_RATES if s == c2 and d == c3), None)
                        r3 = next((r for s, d, r in EXCHANGE_RATES if s == c3 and d == c1), None)

                        if r1 and r2 and r3:
                            product = r1 * r2 * r3
                            if product > 1.001:  # More than 0.1% profit
                                profit = (product - 1) * 100
                                print(f"  {c1} -> {c2} -> {c3} -> {c1}")
                                print(f"    Rates: {r1} x {r2} x {r3} = {product:.4f}")
                                print(f"    Profit: {profit:.2f}%")
                                print()

    else:
        print("No arbitrage opportunities found (no negative cycles).")
        print()

        # Show distances from USD
        print("Best exchange paths from USD:")
        distances = result.solution
        if distances:
            for i, currency in enumerate(currencies):
                if currency != "USD" and i in distances:
                    dist = distances[i]
                    # Convert back from -log to rate
                    implied_rate = 2.718281828 ** (-dist)
                    print(f"  USD -> {currency}: {implied_rate:.4f}")


if __name__ == "__main__":
    main()
