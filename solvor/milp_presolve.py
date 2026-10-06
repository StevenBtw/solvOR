"""
Root presolve for MILP rows and bounds.

Internal module used by solve_milp and MilpModel. Rules, applied per row as
rows arrive:

- empty rows are checked (0 sense b) and dropped;
- single-variable rows become variable bounds and are dropped, so identity
  rows like x_j <= 1 cost nothing;
- rows whose variables and coefficients are all integers are divided by the
  gcd of their coefficients and get an integral right-hand side. This keeps
  every integer solution and only cuts off fractional LP points (a row
  -x - y <= -3.5 becomes x + y >= 4 in effect);
- integer variables get integral bounds.
"""

from math import ceil, floor, gcd, inf, isclose, isfinite, isinf

__all__ = ["presolve_row", "round_integer_bounds"]


def presolve_row(
    row: dict[int, float],
    rhs: float,
    sense: str,
    int_set: set[int],
    lower: list[float],
    upper: list[float],
    eps: float,
) -> tuple[bool, tuple[dict[int, float], float] | None]:
    """Presolve one row; may tighten lower/upper in place.

    Returns (feasible, kept): kept is the row to add as (coefficients, rhs),
    or None when the row became bounds or was empty.
    """
    if isinf(rhs):
        # <= +inf and >= -inf can never bind; <= -inf, >= +inf and = +-inf can never hold
        return sense != "=" and (rhs > 0) == (sense == "<="), None

    if not row:
        violated = (sense == "<=" and rhs < -eps) or (sense == ">=" and rhs > eps) or (sense == "=" and abs(rhs) > eps)
        return not violated, None

    if len(row) == 1:
        ((j, a),) = row.items()
        bound = rhs / a
        if sense == "=" or (sense == "<=") == (a > 0):
            upper[j] = min(upper[j], bound)
        if sense == "=" or (sense == ">=") == (a > 0):
            lower[j] = max(lower[j], bound)
        if j in int_set:
            return round_integer_bounds([j], lower, upper, eps), None
        return lower[j] <= upper[j] + eps, None

    if all(j in int_set for j in row) and all(_is_exact_integer(a) for a in row.values()):
        tightened = _tighten_integer_row(row, rhs, sense, eps)
        if tightened is None:
            return False, None
        return True, tightened
    return True, (row, rhs)


def round_integer_bounds(int_vars, lower: list[float], upper: list[float], eps: float) -> bool:
    """Round integer variables' bounds inward; False if some lower bound passes its upper bound."""
    ok = True
    for j in int_vars:
        if lower[j] > -inf:
            lower[j] = float(ceil(lower[j] - eps))
        if upper[j] < inf:
            upper[j] = float(floor(upper[j] + eps))
        if lower[j] > upper[j] + eps:
            ok = False
    return ok


def _tighten_integer_row(
    row: dict[int, float], rhs: float, sense: str, eps: float
) -> tuple[dict[int, float], float] | None:
    """Divide an all-integer row by the gcd of its coefficients and make the rhs integral.

    Valid because the row's left-hand side is an integer at every integer point.
    Returns None when an equality row cannot hold (non-integral rhs after division).
    """
    g = 0
    for a in row.values():
        g = gcd(g, abs(round(a)))
    if g > 1:
        row = {j: a / g for j, a in row.items()}
        rhs = rhs / g
    if sense == "<=":
        return row, float(floor(rhs + eps))
    if sense == ">=":
        return row, float(ceil(rhs - eps))
    if abs(rhs) >= 1e15:
        return row, rhs  # too large to judge integrality: leave the row to the LP rather than call it impossible
    if not isclose(rhs, round(rhs), rel_tol=0.0, abs_tol=eps):
        return None
    return row, float(round(rhs))


def _is_exact_integer(value: float) -> bool:
    """Coefficients must be exactly integral: a near-integer one, tightened as if exact, cuts off valid points."""
    return isfinite(value) and abs(value) < 2**53 and value == round(value)
