"""
PageRank Example

Find important nodes in a graph based on link structure.
"""

from solvor import pagerank

# Web pages linking to each other
links = {
    "home": ["about", "products", "contact"],
    "about": ["home", "team"],
    "products": ["home", "pricing"],
    "pricing": ["products", "contact"],
    "contact": ["home"],
    "team": ["about"],
}


def neighbors(page):
    return links.get(page, [])


result = pagerank(links.keys(), neighbors)

# Sort by rank
ranked = sorted(result.solution.items(), key=lambda x: -x[1])
print("Page rankings:")
for page, score in ranked:
    print(f"  {page}: {score:.3f}")
# home has highest rank (many pages link to it)
