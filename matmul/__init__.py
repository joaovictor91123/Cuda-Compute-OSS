"""GPU matrix-multiplication system (square n x n, arbitrarily large n).

Quick API
---------
    import numpy as np
    from matmul import matmul

    A = np.random.rand(4096, 4096).astype("float32")
    B = np.random.rand(4096, 4096).astype("float32")
    C = matmul(A, B)            # GPU only (CUDA/MPS); raises if no GPU is present

For huge n that will not fit in RAM, use the disk-backed runner in
``matmul.runner`` or the CLI (``python -m matmul``).
"""
from __future__ import annotations

import numpy as np

from .config import Config, DTYPES
from .backend import Backend
from . import gemm, storage, runner

__all__ = ["Config", "Backend", "matmul", "DTYPES", "gemm", "storage", "runner"]


def matmul(A: np.ndarray, B: np.ndarray, out: np.ndarray | None = None,
           config: Config | None = None) -> np.ndarray:
    """Multiply two square in-memory matrices and return C = A @ B.

    A/B may be NumPy arrays or memmaps. If ``out`` is given it is written in
    place (and returned); otherwise a new array is allocated.
    """
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same size")
    if np.dtype(A.dtype) != np.dtype(B.dtype):
        raise ValueError(
            f"A and B must share a dtype; got {np.dtype(A.dtype)} and {np.dtype(B.dtype)}"
        )
    # Validate the element dtype up front -- raises on int/uint/bf16/... -- so an
    # unsupported input fails here with a clear message instead of deep inside
    # torch.bmm after the backend is already built. (Called unconditionally, even
    # when a config is supplied, so a bad A/B dtype never reaches the GPU.)
    dtype_name = _dtype_name(A.dtype)
    if config is None:
        cfg = Config(dtype=dtype_name, verbose=False)
    else:
        # Budgets (_fits_in_core / auto_tile) use cfg.item_bytes, but uploads use
        # the real A/B dtype. A mismatched config.dtype would under-budget (e.g.
        # fp32 arrays + Config(dtype="fp16") → 2x too small) and OOM instead of
        # tiling. Reject before the backend is built.
        if config.dtype != dtype_name:
            raise ValueError(
                f"config.dtype={config.dtype!r} does not match A/B dtype "
                f"{dtype_name!r} ({np.dtype(A.dtype)})"
            )
        cfg = config
    backend = Backend(cfg.device, cfg.verbose)
    C = out if out is not None else np.empty_like(A, dtype=cfg.np_dtype)
    gemm.multiply(A, B, C, backend, cfg)
    return C


def _dtype_name(dt) -> str:
    dt = np.dtype(dt)
    for name, npdt in DTYPES.items():
        if np.dtype(npdt) == dt:
            return name
    # Reject rather than silently mislabel an unsupported dtype as fp32, which
    # would either crash deep in torch.bmm (integers) or compute in a dtype the
    # caller did not ask for. bf16 is intentionally not exposed (see README).
    raise ValueError(
        f"unsupported dtype {dt}; matmul supports {list(DTYPES)}"
    )
