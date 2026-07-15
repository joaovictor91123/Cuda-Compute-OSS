"""CPU-only tests for the --verify host-RAM guard (issue #136).

`_verify` materializes A, B, the reference product and C as float64 in host RAM
(~4*n*n*8 bytes) and runs an O(n^3) CPU multiply. It must SKIP (not OOM) when
that working set would not fit safely. These drive `_verify` directly with a
stub backend, so no GPU is needed.

Run:  python tests/test_verify_guard.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul.config import Config as MatmulConfig
from matmul.runner import _verify as matmul_verify
from strategy.config import Config as StrategyConfig
from strategy.runner import _verify as strategy_verify


class _StubBackend:
    """Minimal stand-in exposing only the host_available_bytes() the guard reads."""

    def __init__(self, host_bytes):
        self._host_bytes = host_bytes

    def host_available_bytes(self):
        return self._host_bytes


def _small_operands(n=8):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    C = (A @ B).astype(np.float32)
    return A, B, C


def test_matmul_verify_runs_when_ram_is_ample():
    n = 8
    A, B, C = _small_operands(n)
    plenty = 4 * n * n * 8 * 100          # far above the 4*n^2*8 working set
    res = matmul_verify(A, B, C, n, MatmulConfig(dtype="fp32", verbose=False),
                        _StubBackend(plenty))
    assert "skipped" not in res
    assert res["ok"] is True and res["max_rel_err"] < 1e-4


def test_matmul_verify_skips_when_ram_is_tiny():
    n = 8
    A, B, C = _small_operands(n)
    res = matmul_verify(A, B, C, n, MatmulConfig(dtype="fp32", verbose=False),
                        _StubBackend(1))    # 1 byte of host RAM -> must skip, not crash
    assert res.get("skipped")
    assert "ok" not in res


def test_strategy_verify_runs_and_skips():
    n = 8
    A, B, C = _small_operands(n)
    cfg = StrategyConfig(dtype="fp32", verbose=False)
    ok = strategy_verify(A, B, C, n, cfg, _StubBackend(4 * n * n * 8 * 100))
    assert "skipped" not in ok and ok["max_rel_err"] < 1e-4
    assert ok["ok"] is True and "tol" in ok
    skipped = strategy_verify(A, B, C, n, cfg, _StubBackend(1))
    assert skipped.get("skipped")
    assert "ok" not in skipped


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
