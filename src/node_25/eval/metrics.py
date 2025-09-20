from math import sqrt
from statistics import mean, pstdev
from typing import List, Tuple

def mean_ci_95(xs: List[float]) -> Tuple[float, float]:
    """95% CI using normal approximation (fine for n>=10). Returns (mean, half_width)."""
    if not xs:
        return 0.0, 0.0
    m = mean(xs)
    s = pstdev(xs) if len(xs) > 1 else 0.0
    n = len(xs)
    half = 1.96 * (s / sqrt(n)) if n > 0 else 0.0
    return m, half
