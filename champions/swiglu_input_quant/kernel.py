"""CCO champion baseline — swiglu_input_quant (Triton, multi-output).

Two independent outputs over x of shape (m, 2N):

  1. SwiGLU `out` (m, N): out = x0 * SiLU(x1), where x0 = x[:, :N], x1 = x[:, N:], and
     SiLU(x1) = x1 * sigmoid(x1). Computed in fp32, stored in the input dtype.
  2. Blockwise FP8 quant of the ORIGINAL x: x_fp8 (m, 2N) float8_e4m3fn + x_scale (n_blocks, m)
     fp32. For each 128-column block: row_max = |block|.amax(dim=1).clamp(min=1e-15);
     scale = row_max / 448.0 (448 = e4m3 max); q = x * (1/scale); cast to e4m3.

Implemented as two Triton kernels (SwiGLU; quant) — all math is in-kernel (the SiLU and the
quant reductions), so it is delegation-free. Naive but real; miners can fuse the two passes
(x is read twice here) and tile better. CCO artifact contract: KERNEL_TYPE + kernel_fn only.
"""

import torch
import triton
import triton.language as tl

KERNEL_TYPE = "swiglu_input_quant"

E4M3_MAX = 448.0
BLOCK_SIZE = 128  # FP8 quant block width (matches the reference)


@triton.jit
def _swiglu_kernel(X, OUT, N, N2, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    x0 = tl.load(X + row * N2 + cols, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(X + row * N2 + N + cols, mask=mask, other=0.0).to(tl.float32)
    out = x0 * (x1 * tl.sigmoid(x1))
    tl.store(OUT + row * N + cols, out.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _quant_kernel(X, X_FP8, X_SCALE, M, N2, BLOCK: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    j = tl.program_id(1)
    cols = j * BLOCK + tl.arange(0, BLOCK)
    block = tl.load(X + row * N2 + cols).to(tl.float32)

    row_max = tl.max(tl.abs(block), axis=0)
    row_max = tl.maximum(row_max, EPS)
    scale = row_max / 448.0
    q = block * (1.0 / scale)

    tl.store(X_FP8 + row * N2 + cols, q.to(X_FP8.dtype.element_ty))
    tl.store(X_SCALE + j * M + row, scale)


def kernel_fn(x: torch.Tensor):
    m, n2 = x.shape
    n = n2 // 2
    x = x.contiguous()

    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    x_fp8 = torch.empty((m, n2), dtype=torch.float8_e4m3fn, device=x.device)
    n_blocks = n2 // BLOCK_SIZE
    x_scale = torch.empty((n_blocks, m), dtype=torch.float32, device=x.device)

    BLOCK_N = 1024
    _swiglu_kernel[(m, triton.cdiv(n, BLOCK_N))](x, out, n, n2, BLOCK_N=BLOCK_N)
    _quant_kernel[(m, n_blocks)](x, x_fp8, x_scale, m, n2, BLOCK=BLOCK_SIZE, EPS=1e-15)

    return out, x_fp8, x_scale
