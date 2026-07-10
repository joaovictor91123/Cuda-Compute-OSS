"""Regression test: less_vram_than_exact is 'unknown' (None) when VRAM is not measured.

Off CUDA, attention.benchmark cannot measure peak VRAM (_peak_bytes returns 0),
so the improvement report must say None ("unknown"), not a misleading False.

CPU-safe: skips cleanly when torch is not installed. Uses fp32 so it is
independent of the fp16 FFT path.
Run:  python tests/test_benchmark_vram.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    torch = None

if torch is not None:
    from attention.benchmark import run_once


def _skip_if_no_torch():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


def _cpu_result():
    return run_once(batch=1, heads=1, seq=64, dim=8, dtype="fp32",
                    window=16, mode="all", device="cpu")


def test_less_vram_is_unknown_on_cpu():
    if _skip_if_no_torch():
        return
    res = _cpu_result()
    # primary improvement block
    assert res["improvement"]["less_vram_than_exact"] is None
    # every candidate
    for name, c in res["candidates"].items():
        assert c["improvement"]["less_vram_than_exact"] is None, f"{name} not None"


def test_faster_than_exact_is_still_measured_bool():
    if _skip_if_no_torch():
        return
    res = _cpu_result()
    for name, c in res["candidates"].items():
        assert isinstance(c["improvement"]["faster_than_exact"], bool), f"{name}"


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
