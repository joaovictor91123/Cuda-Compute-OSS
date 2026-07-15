"""CPU-only tests for strategy CLI --verify exit-code handling.

Previously strategy._verify returned only max_rel_err and cli.main always exited
0, so a huge reconstruction error (e.g. --fill random) looked like success to
scripts/CI. Mirror matmul's contract: ok=False -> exit 1; skipped / pass -> 0.

Run:  python tests/test_strategy_cli_verify.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import cli
from strategy import runner as _runner


def _main_with_run_result(info, argv=("--quiet",)):
    orig = _runner.run
    _runner.run = lambda n, cfg, **kw: info
    try:
        return cli.main(list(argv))
    finally:
        _runner.run = orig


_BASE = {"mode": "subspace(M=8, transform=rsvd)", "seconds": 1.0, "gflops": 1.0}


def test_skipped_verify_exits_zero_not_keyerror():
    info = {**_BASE, "verify": {"skipped": "n=20000: float64 CPU reference too large"}}
    assert _main_with_run_result(info) == 0


def test_passing_verify_exits_zero():
    info = {**_BASE, "verify": {"ok": True, "max_rel_err": 1e-6, "tol": 1e-4}}
    assert _main_with_run_result(info) == 0


def test_failing_verify_exits_one():
    info = {**_BASE, "verify": {"ok": False, "max_rel_err": 1.0, "tol": 1e-4}}
    assert _main_with_run_result(info) == 1


def test_no_verify_key_exits_zero():
    assert _main_with_run_result(dict(_BASE)) == 0


if __name__ == "__main__":
    fns = [v for kk, v in sorted(globals().items()) if kk.startswith("test_")]
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
