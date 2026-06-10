"""CCO submission artifact — kernel.py  (THE ONE MUTABLE FILE).

This is the file you edit and submit; everything else in the repo is locked and byte-verified
against manifest.json at your PR HEAD. It must export exactly two names:

    KERNEL_TYPE = "<one of the 5 tracks>"   # selects the oracle/config/champion you compete on
    def kernel_fn(...): ...                  # your Triton kernel under test

The seed below is a COPY of the current `rms_norm` champion, so a fresh clone self-scores out of
the box:  `uv run benchmark.py`. To compete on a different track, REPLACE this file with your
kernel and set KERNEL_TYPE accordingly — start from `champions/<track>/kernel.py`. (The per-track
champions in `champions/` are the real baselines you must beat; this root copy is just a runnable
starting point so the harness has something to import.)

Rules (enforced mechanically by cco/guard_kernel.py + cco/dispatch_trap.py): Triton-only; no
delegation to torch.matmul / F.* / torch.ops.aten.* / the `@` operator / cuBLAS; no
get_inputs / get_flops / get_bytes. See CONTRIBUTING.md.
"""

import torch
import triton
import triton.language as tl

KERNEL_TYPE = "rms_norm"


@triton.jit
def _rms_norm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xm,
    stride_ym,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N
    rms = tl.sqrt(mean_sq + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x / rms) * w

    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.dim() == 2 and weight.dim() == 1 and weight.shape[0] == x.shape[1]
    M, N = x.shape

    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    _rms_norm_kernel[grid](
        x, weight, y,
        x.stride(0),
        y.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
