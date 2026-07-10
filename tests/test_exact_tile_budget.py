"""CPU-only tests for subspace._exact_tile's device budget (issue #144).

multiply_exact's per-(row, k) working set is acc (T,n) + A panel (T,T) +
B panel (T,n) + the GEMM output matmul(Ar, Bk) (T,n) -- the product cannot alias
an operand, and is live while it is folded into acc. That is T*(3n + T), not the
T*(2n + T) the tile was originally solved from.

Pure arithmetic; no GPU needed.  Run:  python tests/test_exact_tile_budget.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace


class _FakeBackend:
    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free


ITEM = 4  # fp32


def _budget_elems(free, frac):
    return max(1, int(free * frac) // ITEM)


def _legacy_exact_tile(n, free, frac):
    """Pre-fix estimator: solved T^2 + 2nT = budget (acc + Ar + Bk only)."""
    b = _budget_elems(free, frac)
    return max(1, min(int((math.sqrt(4 * n * n + 4 * b) - 2 * n) / 2), n))


def test_exact_tile_fits_the_real_working_set():
    """acc + Ar + Bk + prod = T*(3n + T) must fit the budget."""
    for n, free, frac in [(8192, 1 << 30, 0.6), (4096, 256 * 1024**2, 0.3),
                          (1024, 32 * 1024**2, 0.5)]:
        T = subspace._exact_tile(n, _FakeBackend(free), ITEM, frac)
        assert T * (3 * n + T) <= _budget_elems(free, frac), (n, free, frac, T)


def test_legacy_tile_would_have_overshot():
    """Regression witness: the old T*(2n+T) model overshoots once the GEMM
    output is counted -- ~1.7-1.8x for T << n."""
    for n, free, frac in [(8192, 1 << 30, 0.6), (4096, 256 * 1024**2, 0.3)]:
        b = _budget_elems(free, frac)
        old = _legacy_exact_tile(n, free, frac)
        # Even ignoring the non-in-place copy, the product alone breaks budget.
        assert old * (3 * n + old) > b
        assert old * (3 * n + old) / b > 1.3


def test_exact_tile_is_not_larger_than_legacy():
    for n, free, frac in [(8192, 1 << 30, 0.6), (4096, 256 * 1024**2, 0.3)]:
        assert (subspace._exact_tile(n, _FakeBackend(free), ITEM, frac)
                <= _legacy_exact_tile(n, free, frac))


def test_exact_tile_clamped_to_n_and_at_least_one():
    # Huge budget -> capped at n (single tile).
    assert subspace._exact_tile(64, _FakeBackend(1 << 34), ITEM, 0.9) == 64
    # Tiny budget -> never 0 (a 0 tile makes range(0, n, 0) raise).
    assert subspace._exact_tile(8192, _FakeBackend(16), ITEM, 0.1) >= 1


def test_exact_tile_scales_with_free_memory():
    n = 4096
    small = subspace._exact_tile(n, _FakeBackend(32 * 1024**2), ITEM, 0.5)
    large = subspace._exact_tile(n, _FakeBackend(512 * 1024**2), ITEM, 0.5)
    assert small < large


def test_multiply_exact_accumulates_in_place():
    """`acc = acc + prod` doubles the (T,n) residency; the source must use +=."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "strategy", "subspace.py")).read()
    assert "acc = acc + backend.matmul(Ar, Bk)" not in src
    assert "acc += backend.matmul(Ar, Bk)" in src


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
