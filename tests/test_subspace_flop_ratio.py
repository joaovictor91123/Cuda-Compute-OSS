"""Pin the subspace FLOP-ratio claim to the code (pure arithmetic, no GPU).

strategy/README.md and strategy/subspace.py both advertise the asymptotic FLOP
ratio of the smart multiply vs the exact O(N^3) product. The honest figure is
``4M/N`` -- ``3M/N`` is the core-only count that drops the transform's mandatory
per-call basis construction and therefore overstates the savings (see
subspace.py's module docstring: "~4M/N once the basis construction is counted,
not ~3M/N", and PRs #65 / #50 that added the basis term to flop_actual).

This test recomputes the ratio straight from ``subspace._flop_actual`` plus
``RandomizedSVDTransform.basis_flops`` so the documented number cannot silently
drift back to the pre-#65 ``3M/N``. It is pure arithmetic and runs on CPU/CI.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace
from strategy.transforms import RandomizedSVDTransform


def _full_ratio_coeff(n: int, m: int) -> float:
    """(flop_actual incl. basis) / flop_exact, expressed as a coefficient of M/N."""
    rsvd = RandomizedSVDTransform()
    flop_actual = subspace._flop_actual(n, m) + rsvd.basis_flops(n, m)
    flop_exact = 2.0 * n * n * n
    return (flop_actual / flop_exact) * (n / m)


def _core_only_coeff(n: int, m: int) -> float:
    """Same coefficient but EXCLUDING basis construction (the stale 3M/N figure)."""
    flop_exact = 2.0 * n * n * n
    return (subspace._flop_actual(n, m) / flop_exact) * (n / m)


def test_documented_flop_ratio_is_four_m_over_n():
    # As M/N -> 0 the leading term dominates and the coefficient converges to 4,
    # matching strategy/README.md and subspace.py ("~4M/N ... not ~3M/N").
    n, m = 2_000_000, 1000            # M/N = 5e-4
    coeff = _full_ratio_coeff(n, m)
    assert abs(coeff - 4.0) < 0.02, f"expected ~4M/N, got {coeff:.4f}*M/N"
    # ...and it is unambiguously the 4M/N figure, not the old 3M/N one.
    assert coeff > 3.5


def test_basis_construction_is_the_difference_between_4_and_3():
    # Dropping basis_flops gives exactly the 3M/N core-only figure the README
    # used to advertise -- documenting *why* 3M/N understates the real cost.
    n, m = 2_000_000, 1000
    assert abs(_core_only_coeff(n, m) - 3.0) < 0.02
    # The honest ratio is strictly larger because basis construction is not free.
    assert _full_ratio_coeff(n, m) > _core_only_coeff(n, m)


if __name__ == "__main__":
    test_documented_flop_ratio_is_four_m_over_n()
    test_basis_construction_is_the_difference_between_4_and_3()
    print("ok")
