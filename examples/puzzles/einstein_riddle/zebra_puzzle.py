"""
Einstein's Riddle (Zebra Puzzle) - SAT Encoding

Famous logic puzzle allegedly created by Einstein (though unverified).
"Who owns the zebra? Who drinks water?"

Five houses in a row, each with different:
- Nationality (Englishman, Spaniard, Ukrainian, Norwegian, Japanese)
- Color (Red, Green, Ivory, Yellow, Blue)
- Pet (Dog, Snails, Fox, Horse, Zebra)
- Drink (Coffee, Tea, Milk, OJ, Water)
- Cigarette (Old Gold, Kools, Chesterfields, Lucky Strike, Parliaments)

Source: Life International magazine (1962)
        https://en.wikipedia.org/wiki/Zebra_Puzzle

Clues:
1. The Englishman lives in the red house.
2. The Spaniard owns the dog.
3. Coffee is drunk in the green house.
4. The Ukrainian drinks tea.
5. The green house is immediately to the right of the ivory house.
6. The Old Gold smoker owns snails.
7. Kools are smoked in the yellow house.
8. Milk is drunk in the middle house.
9. The Norwegian lives in the first house.
10. The Chesterfields smoker lives next to the fox owner.
11. Kools are smoked next to the house where the horse is kept.
12. The Lucky Strike smoker drinks orange juice.
13. The Japanese smokes Parliaments.
14. The Norwegian lives next to the blue house.

Why SAT: This puzzle has "next to" constraints (adjacency) that require
encoding implications. We use direct SAT encoding with boolean variables
for each (attribute, house) pair.
"""

from itertools import combinations

from solvor.sat import solve_sat


def solve_zebra_puzzle():
    """Solve Einstein's riddle using SAT encoding."""
    print("Einstein's Riddle (Zebra Puzzle)")
    print("=" * 40)
    print()

    # Define attributes
    NATIONALITIES = ["Englishman", "Spaniard", "Ukrainian", "Norwegian", "Japanese"]
    COLORS = ["Red", "Green", "Ivory", "Yellow", "Blue"]
    PETS = ["Dog", "Snails", "Fox", "Horse", "Zebra"]
    DRINKS = ["Coffee", "Tea", "Milk", "OJ", "Water"]
    CIGARETTES = ["OldGold", "Kools", "Chesterfields", "LuckyStrike", "Parliaments"]

    # Create boolean variables: var[category][item][house] = SAT variable ID
    # Variable ID scheme: (category_idx * 5 * 5) + (item_idx * 5) + house_idx + 1
    def var_id(category_idx, item_idx, house_idx):
        return category_idx * 25 + item_idx * 5 + house_idx + 1

    categories = [NATIONALITIES, COLORS, PETS, DRINKS, CIGARETTES]
    cat_names = ["Nationality", "Color", "Pet", "Drink", "Cigarette"]

    clauses = []

    # Constraint type 1: Each item is in exactly one house
    for cat_idx, category in enumerate(categories):
        for item_idx in range(5):
            # At least one house
            clauses.append([var_id(cat_idx, item_idx, h) for h in range(5)])
            # At most one house
            for h1, h2 in combinations(range(5), 2):
                clauses.append([-var_id(cat_idx, item_idx, h1), -var_id(cat_idx, item_idx, h2)])

    # Constraint type 2: Each house has exactly one item per category
    for cat_idx, category in enumerate(categories):
        for house_idx in range(5):
            # At least one item
            clauses.append([var_id(cat_idx, i, house_idx) for i in range(5)])
            # At most one item
            for i1, i2 in combinations(range(5), 2):
                clauses.append([-var_id(cat_idx, i1, house_idx), -var_id(cat_idx, i2, house_idx)])

    # Helper: same house constraint (item1 in house h implies item2 in house h)
    def same_house(cat1_idx, item1_idx, cat2_idx, item2_idx):
        for h in range(5):
            # item1 in h -> item2 in h: -item1[h] OR item2[h]
            clauses.append([-var_id(cat1_idx, item1_idx, h), var_id(cat2_idx, item2_idx, h)])
            clauses.append([var_id(cat1_idx, item1_idx, h), -var_id(cat2_idx, item2_idx, h)])

    # Helper: item is in specific house
    def in_house(cat_idx, item_idx, house):
        clauses.append([var_id(cat_idx, item_idx, house - 1)])

    # Helper: next to constraint (|house1 - house2| = 1)
    def next_to(cat1_idx, item1_idx, cat2_idx, item2_idx):
        # For each house h1 of item1, item2 must be in h1-1 or h1+1
        for h1 in range(5):
            adjacent_options = []
            if h1 > 0:
                adjacent_options.append(var_id(cat2_idx, item2_idx, h1 - 1))
            if h1 < 4:
                adjacent_options.append(var_id(cat2_idx, item2_idx, h1 + 1))
            # item1[h1] -> (item2[h1-1] OR item2[h1+1])
            clauses.append([-var_id(cat1_idx, item1_idx, h1)] + adjacent_options)

    # Helper: immediately right of (house1 = house2 + 1)
    def right_of(cat1_idx, item1_idx, cat2_idx, item2_idx):
        # item1 is immediately to the right of item2
        # So if item2 is in house h, item1 must be in house h+1
        for h in range(5):
            if h < 4:
                # item2[h] -> item1[h+1]
                clauses.append([-var_id(cat2_idx, item2_idx, h), var_id(cat1_idx, item1_idx, h + 1)])
            else:
                # item2 can't be in house 5 (no room on right)
                clauses.append([-var_id(cat2_idx, item2_idx, h)])

    # Category indices
    NAT, COL, PET, DRK, CIG = 0, 1, 2, 3, 4

    # Item indices within categories
    nat_idx = {n: i for i, n in enumerate(NATIONALITIES)}
    col_idx = {c: i for i, c in enumerate(COLORS)}
    pet_idx = {p: i for i, p in enumerate(PETS)}
    drk_idx = {d: i for i, d in enumerate(DRINKS)}
    cig_idx = {c: i for i, c in enumerate(CIGARETTES)}

    # Clue 1: The Englishman lives in the red house
    same_house(NAT, nat_idx["Englishman"], COL, col_idx["Red"])

    # Clue 2: The Spaniard owns the dog
    same_house(NAT, nat_idx["Spaniard"], PET, pet_idx["Dog"])

    # Clue 3: Coffee is drunk in the green house
    same_house(DRK, drk_idx["Coffee"], COL, col_idx["Green"])

    # Clue 4: The Ukrainian drinks tea
    same_house(NAT, nat_idx["Ukrainian"], DRK, drk_idx["Tea"])

    # Clue 5: The green house is immediately to the right of the ivory house
    right_of(COL, col_idx["Green"], COL, col_idx["Ivory"])

    # Clue 6: The Old Gold smoker owns snails
    same_house(CIG, cig_idx["OldGold"], PET, pet_idx["Snails"])

    # Clue 7: Kools are smoked in the yellow house
    same_house(CIG, cig_idx["Kools"], COL, col_idx["Yellow"])

    # Clue 8: Milk is drunk in the middle house (house 3)
    in_house(DRK, drk_idx["Milk"], 3)

    # Clue 9: The Norwegian lives in the first house
    in_house(NAT, nat_idx["Norwegian"], 1)

    # Clue 10: The Chesterfields smoker lives next to the fox owner
    next_to(CIG, cig_idx["Chesterfields"], PET, pet_idx["Fox"])

    # Clue 11: Kools are smoked next to the house where the horse is kept
    next_to(CIG, cig_idx["Kools"], PET, pet_idx["Horse"])

    # Clue 12: The Lucky Strike smoker drinks orange juice
    same_house(CIG, cig_idx["LuckyStrike"], DRK, drk_idx["OJ"])

    # Clue 13: The Japanese smokes Parliaments
    same_house(NAT, nat_idx["Japanese"], CIG, cig_idx["Parliaments"])

    # Clue 14: The Norwegian lives next to the blue house
    next_to(NAT, nat_idx["Norwegian"], COL, col_idx["Blue"])

    # Solve
    result = solve_sat(clauses)

    if result.ok:
        sol = result.solution

        # Decode solution
        def get_house(cat_idx, item_idx):
            for h in range(5):
                if sol.get(var_id(cat_idx, item_idx, h), False):
                    return h + 1
            return None

        # Build house descriptions
        house_data = {h: {} for h in range(1, 6)}
        for cat_idx, (cat_name, category) in enumerate(zip(cat_names, categories)):
            for item_idx, item in enumerate(category):
                house = get_house(cat_idx, item_idx)
                if house:
                    house_data[house][cat_name] = item

        print("Solution:")
        print("-" * 70)
        print(f"{'House':<8} {'Nation':<12} {'Color':<8} {'Pet':<8} {'Drink':<8} {'Cigarette':<14}")
        print("-" * 70)

        for house in range(1, 6):
            data = house_data[house]
            print(
                f"{house:<8} {data.get('Nationality', '?'):<12} {data.get('Color', '?'):<8} "
                f"{data.get('Pet', '?'):<8} {data.get('Drink', '?'):<8} {data.get('Cigarette', '?'):<14}"
            )

        print("-" * 70)
        print()

        # Answer the questions
        zebra_house = get_house(PET, pet_idx["Zebra"])
        water_house = get_house(DRK, drk_idx["Water"])

        zebra_owner = house_data[zebra_house]["Nationality"]
        water_drinker = house_data[water_house]["Nationality"]

        print(f"Who owns the zebra? The {zebra_owner}!")
        print(f"Who drinks water? The {water_drinker}!")

    else:
        print(f"No solution found. Status: {result.status}")

    return result


if __name__ == "__main__":
    solve_zebra_puzzle()
