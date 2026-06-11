"""
cco/isolate.py — run a submitted kernel in an ISOLATED subprocess so it cannot forge its score.

THE THREAT. The submitted kernel runs with full Python privileges. If it shares an interpreter
with the scorer, it can read the secret input seed straight from `sys.argv`, monkeypatch
`torch.allclose` / the comparison / the timing events / even the no-delegation trap, or execute
arbitrary code at import (a repo-root `sitecustomize.py`, `os.system`, ...). An in-process guard
cannot stop this — the attacker is already inside the interpreter.

THE FIX. Move every TRUSTED operation into the PARENT, and run only `kernel_fn` in a clean child:

    parent: generate inputs from the secret seed (NEVER sent to the child)
        │  serialize the input TENSORS (not the seed) to a temp file
        ▼
    child : python -E, clean cwd, kernel loaded by file path (no sys.path injection, no
            sitecustomize), runs kernel_fn under the dispatch trap, times it with CUDA events,
            and writes back the OUTPUT TENSORS it produced
        │
        ▼
    parent: validates those outputs against ITS OWN oracle (the child's comparison is never
            trusted), and bounds the reported timing by its own wall-clock.

What this CLOSES: in-process correctness forgery, seed-read, scorer-state monkeypatching, the
sitecustomize / PYTHON* / os.system paths affecting the verdict, OOM/crash (child non-zero exit =
FAIL), and garbage-on-replay of CORRECTNESS (the parent validates several distinct buffers, drawn
both before AND after the timed window).

Residual (documented, not yet closed): a kernel that is genuinely correct but under-reports its
CUDA-event timing from inside the child. The parent wall-clock tripwire rejects the impossible case
(claimed GPU time > wall time); full immunity needs parent-driven two-point wall-clock timing — a
follow-up. The child trap is defense-in-depth only; the load-bearing correctness check is the
parent's oracle.

Usage (needs a CUDA GPU + torch/triton; run in the Linux/WSL env):
    ~/cco-gpu/bin/python cco/isolate.py --self-test
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def _to_cpu(out):
    """Detach a kernel output (tensor or tuple/list of tensors) to CPU for cross-process transfer."""
    if isinstance(out, (tuple, list)):
        return type(out)(_to_cpu(o) for o in out)
    return out.detach().to("cpu")


def _to_cuda(out):
    if isinstance(out, (tuple, list)):
        return type(out)(_to_cuda(o) for o in out)
    return out.to("cuda")


# =====================================================================================
# CHILD — runs in the isolated subprocess. Untrusted-kernel territory; produces only the
# raw outputs + a (defense-in-depth) trap verdict + event timings. Makes NO judgement.
# =====================================================================================

def _child_main(job_path: str, out_path: str) -> int:
    import importlib.util

    import torch

    job = torch.load(job_path, weights_only=True)  # parent-written, but tensor-only is safe + strict
    warmup = int(job["warmup"])
    n_blocks = int(job["n_blocks"])
    rep = int(job["rep"])
    n_val_pre = int(job["n_val_pre"])

    buffers = [{k: v.to("cuda") for k, v in b.items()} for b in job["buffers"]]
    n_val = len(buffers)

    # Load the submission BY FILE PATH — this does not add its directory to sys.path, so a
    # sitecustomize next to it never auto-imports, and it cannot shadow stdlib/cco modules.
    spec = importlib.util.spec_from_file_location("cco_submission_kernel", job["kernel_path"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel_fn = mod.kernel_fn

    from cco.dispatch_trap import DelegationError, run_guarded  # defense-in-depth only

    delegation = None
    val_outputs = []
    event_block_us: list[float] = []

    try:
        # --- pre-timing validation calls (parent will check these against its oracle) ---
        for i in range(min(n_val_pre, n_val)):
            out = run_guarded(kernel_fn, buffers[i])
            val_outputs.append(_to_cpu(out))

        # --- warmup + timed reps (rotating buffers; outputs discarded here) ---
        for _ in range(warmup):
            kernel_fn(**buffers[0])
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _blk in range(n_blocks):
            start.record()
            for r in range(rep):
                kernel_fn(**buffers[r % n_val])
            end.record()
            torch.cuda.synchronize()
            event_block_us.append(start.elapsed_time(end) * 1000.0 / rep)

        # --- post-timing validation calls (AFTER the timed window, so a correct-then-garbage
        #     call-counting kernel cannot find a safe window) ---
        for i in range(n_val_pre, n_val):
            out = kernel_fn(**buffers[i])
            val_outputs.append(_to_cpu(out))
        torch.cuda.synchronize()
    except DelegationError as e:
        delegation = str(e)

    torch.save(
        {"val_outputs": val_outputs, "event_block_us": event_block_us, "delegation": delegation},
        out_path,
    )
    return 0


# =====================================================================================
# PARENT — trusted. Generates inputs from the secret seed, spawns the child, validates the
# returned outputs against the oracle, bounds the timing by wall-clock. Returns the same dict
# shape as benchmark.run_scored_sample so the blob/significance path is unchanged.
# =====================================================================================

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_isolated(kernel_path: str, config: dict, seed: int, compare_fn, *,
                 n_blocks: int = 30, warmup: int = 25, rep: int = 100,
                 n_val: int = 6, timeout_s: float = 600.0) -> dict:
    """Score `kernel_path` in an isolated subprocess; correctness is judged HERE against the oracle.

    `compare_fn(output, expected, atol, rtol, multi_output) -> {"match": bool, "max_abs_error": ..}`
    is supplied by the caller (benchmark._do_compare) so this module stays torch-light at import.
    """
    import time

    import torch

    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    multi = config.get("multi_output", False)
    dtype = config["test_dtypes"][0]
    tols = config["tolerances"].get(dtype, {"atol": 1e-2, "rtol": 1e-2})

    size_label, size = None, None
    for label, sz in config["test_sizes"]:
        if label == "large":
            size_label, size = label, sz
            break
    if size is None:
        size_label, size = config["test_sizes"][-1]

    base = {"size_label": size_label, "dtype": str(dtype), "n_blocks": n_blocks,
            "rep": rep, "warmup": warmup, "n_buffers": n_val, "isolated": True,
            "output_aliased_input": None}  # alias is subsumed by parent-side oracle validation

    # Inputs are generated from the SECRET seed in the parent and never handed the seed to the child.
    val_buffers = [gen_fn(size, dtype, "cuda", seed=seed + i) for i in range(n_val)]

    tmp = tempfile.mkdtemp(prefix="cco_isolate_")
    job_path = os.path.join(tmp, "job.pt")
    out_path = os.path.join(tmp, "out.pt")
    try:
        torch.save({
            "kernel_path": os.path.abspath(kernel_path),
            "buffers": [{k: v.detach().to("cpu") for k, v in b.items()} for b in val_buffers],
            "warmup": warmup, "n_blocks": n_blocks, "rep": rep,
            "n_val_pre": n_val // 2,
        }, job_path)

        boot = (f"import sys; sys.path.insert(0, {_repo_root()!r}); "
                f"from cco.isolate import _child_main; _child_main(sys.argv[1], sys.argv[2])")
        # -E ignores PYTHON* env; cwd is a clean temp dir so a repo-root sitecustomize is not on
        # sys.path[0]; PYTHON* stripped from env as belt-and-suspenders.
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        cmd = [sys.executable, "-E", "-c", boot, job_path, out_path]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout_s)
        child_wall_s = time.perf_counter() - t0

        if proc.returncode != 0 or not os.path.exists(out_path):
            tail = (proc.stderr or "")[-400:]
            return {**base, "correct": False, "max_abs_error": 0.0, "delegation": None,
                    "error": f"child exited {proc.returncode} (OOM/crash): {tail}",
                    "latencies_us": [], "median_us": 0.0, "mean_us": 0.0, "stdev_us": 0.0,
                    "child_wall_s": child_wall_s}

        # weights_only=True is SECURITY-CRITICAL: out.pt is written by the UNTRUSTED child, so it
        # must deserialize as tensors/basic-types only — never reconstruct arbitrary pickled classes
        # (that would hand the child a pickle-RCE into the trusted parent).
        res = torch.load(out_path, weights_only=True)
        delegation = res.get("delegation")
        latencies_us = list(res.get("event_block_us") or [])

        # --- the load-bearing check: validate the child's outputs against OUR oracle ---
        correct, worst_err = (delegation is None), 0.0
        if delegation is None:
            outs = res.get("val_outputs") or []
            if len(outs) < n_val:
                correct = False
            for i, out_cpu in enumerate(outs):
                out = _to_cuda(out_cpu)
                cmp = compare_fn(out, ref_fn(val_buffers[i]), tols["atol"], tols["rtol"], multi)
                correct = correct and bool(cmp["match"])
                worst_err = max(worst_err, cmp.get("max_abs_error", 0.0))

        # --- timing sanity tripwire (parent's trusted wall-clock) ---
        # Claimed GPU time across the timed window cannot exceed the child's wall time.
        claimed_gpu_s = sum(latencies_us) * rep / 1e6
        timing_inconsistent = bool(latencies_us) and claimed_gpu_s > child_wall_s
        if timing_inconsistent:
            correct = False  # impossible-fast claim ⇒ reject

        import statistics
        return {
            **base,
            "correct": bool(correct and delegation is None),
            "max_abs_error": worst_err,
            "delegation": delegation,
            "timing_inconsistent": timing_inconsistent,
            "child_wall_s": child_wall_s,
            "latencies_us": latencies_us,
            "median_us": statistics.median(latencies_us) if latencies_us else 0.0,
            "mean_us": statistics.fmean(latencies_us) if latencies_us else 0.0,
            "stdev_us": statistics.pstdev(latencies_us) if len(latencies_us) > 1 else 0.0,
        }
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# =====================================================================================
# Self-test (needs a CUDA GPU). Proves the parent verdict survives a kernel that tries to
# forge correctness in-process, read the seed, or delegate.
# =====================================================================================

_CLEAN = '''
import torch, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def kernel_fn(x, weight, eps=1e-6):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
'''

# Tries every in-process trick: monkeypatch torch.allclose, read --seed from argv, then return
# garbage. The PARENT must still rule it incorrect.
_FORGER = '''
import sys, torch
torch.allclose = lambda *a, **k: True          # forge any in-process comparison
torch.Tensor.allclose = lambda *a, **k: True
_seen_seed = any(a == "--seed" for a in sys.argv)   # try to read the secret seed (must fail)
KERNEL_TYPE = "rms_norm"
def kernel_fn(x, weight, eps=1e-6):
    return torch.empty_like(x)                  # fast garbage
'''

_DELEGATOR = '''
import torch, torch.nn.functional as F
KERNEL_TYPE = "rms_norm"
def kernel_fn(x, weight, eps=1e-6):
    return F.rms_norm(x, (x.shape[-1],), weight, eps)   # correct, but delegated
'''


def _self_test() -> int:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: isolate self-test needs CUDA")
        return 0

    def _gen(size, dtype, device, seed=42):
        torch.manual_seed(seed)
        M, N = size["M"], size["N"]
        return {"x": torch.randn(M, N, device=device, dtype=dtype),
                "weight": torch.randn(N, device=device, dtype=dtype)}

    def _ref(inp):
        x = inp["x"].float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x / rms * inp["weight"].float()).to(inp["x"].dtype)

    def _cmp(out, exp, atol, rtol, multi_output):
        ok = torch.allclose(out.float(), exp.float(), atol=atol, rtol=rtol)
        return {"match": bool(ok), "max_abs_error": (out.float() - exp.float()).abs().max().item()}

    config = {"input_generator": _gen, "reference_fn": _ref, "multi_output": False,
              "test_dtypes": [torch.float16], "tolerances": {torch.float16: {"atol": 1e-2, "rtol": 1e-2}},
              "test_sizes": [("large", {"M": 1024, "N": 1024})]}

    import shutil
    failures = 0

    def run(src):
        d = tempfile.mkdtemp(prefix="cco_isotest_")
        kp = os.path.join(d, "kernel.py")
        with open(kp, "w") as f:
            f.write(src)
        try:
            return run_isolated(kp, config, seed=123456, compare_fn=_cmp, n_blocks=5, rep=20, n_val=4)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def check(cond, label):
        nonlocal failures
        print(("ok    " if cond else "FAIL  ") + label)
        if not cond:
            failures += 1

    r = run(_CLEAN)
    check(r["correct"] and not r.get("delegation"), f"clean Triton kernel -> correct (median {r['median_us']:.1f}us)")
    check(len(r["latencies_us"]) == 5, "clean kernel produced a 5-block timing sample")

    r = run(_FORGER)
    check(not r["correct"],
          "in-process forger (patches torch.allclose, reads argv, returns garbage) -> REJECTED by oracle")

    r = run(_DELEGATOR)
    check(not r["correct"], "runtime delegator (F.rms_norm) -> REJECTED")
    check(bool(r.get("delegation")), "  ...and flagged as delegation by the child trap")

    print("-" * 60)
    print("SELF-TEST PASSED" if not failures else f"SELF-TEST FAILED: {failures} case(s)")
    return 1 if failures else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Isolated kernel scoring (CCO).")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--child", nargs=2, metavar=("JOB", "OUT"), help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        return _child_main(a.child[0], a.child[1])
    if a.self_test:
        return _self_test()
    p.error("pass --self-test (or import run_isolated)")


if __name__ == "__main__":
    raise SystemExit(main())
