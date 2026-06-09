# CCO Champion Kernels — Detailed Explanation

This document explains the five **champion kernels** that seed the CCO competition: what each
one does, why it matters in AI, how it computes its result, and how it maps onto CUDA hardware.
It also answers: *why five? are these the most important? can we add more?*

A **champion** is the current best kernel for a track. Miners submit a `kernel.py` challenger;
if it beats the standing champion (faster, still correct, within VRAM) with statistical
significance, it becomes the new champion. Each champion below is a real, hand-written Triton
kernel — no delegation to `torch`/cuBLAS/cuDNN — validated against a PyTorch oracle through the
5-stage correctness harness on real hardware.

---

## 1. The big picture: these five *are* a transformer layer

Every modern LLM forward pass is, per layer, essentially:

```
        ┌─────────────────────── one transformer layer ───────────────────────┐
 x ──► RMSNorm ──► QKV projection ──► RoPE(Q,K) ──► Attention ──► out-proj ──► +x
                     (matmul)        (qkv_part_rope) (dsa_forward)   (matmul)
   └─► RMSNorm ──► MLP up (matmul) ──► SwiGLU (+FP8 quant) ──► MLP down (matmul) ──► +x
        (rms_norm)                      (swiglu_input_quant)
 ... × N layers ... ──► RMSNorm ──► LM head (matmul)
```

So the five champions are not arbitrary — they are the **building blocks of an LLM layer**:

| Champion | What it is in the model | GPU bottleneck regime |
|---|---|---|
| `rms_norm` | the normalization before attention & MLP | **memory-bound** |
| `matmul` | every linear projection (QKV, out, MLP up/down, LM head) | **compute-bound** (tensor cores) |
| `qkv_part_rope` | rotary position embedding on Q/K | **memory-bound** |
| `dsa_forward` | the attention itself | **compute-bound** (tensor cores) |
| `swiglu_input_quant` | the MLP gated activation (+ FP8 quant for the next matmul) | **memory-bound** |

The deeper reason there are exactly these five is **bottleneck coverage** (see §4): GPU kernels
fall into two fundamentally different optimization regimes — **memory-bandwidth-bound** and
**compute(tensor-core)-bound** — and these five deliberately span both, plus the spectrum
between them. Optimizing a memory-bound kernel (coalescing, vectorized loads, fusing passes) is
a completely different skill from optimizing a compute-bound one (tiling, tensor-core
utilization, software pipelining). Covering both is what makes the competition teach
*transferable* lessons.

### Measured on the dev GPU (RTX 5070 Ti, sm_120)

| Champion | Bottleneck | Throughput | Tensor cores? | vs. PyTorch ref |
|---|---|---|---|---|
| rms_norm | memory | ~1.1 TFLOPS (≈80% of HBM bandwidth) | no | 9.6× |
| matmul | compute | ~91.5 TFLOPS (≈99% of cuBLAS, fp16) | **yes** | ~1.0× |
| qkv_part_rope | memory | ~0.24 TFLOPS (elementwise) | no | 9.7× |
| swiglu_input_quant | memory | ~0.59 TFLOPS | no | 29.8× |
| dsa_forward | compute | ~101.8 TFLOPS | **yes** | 18× |

(The "vs PyTorch ref" speedups for the memory-bound kernels are large because their reference is
unfused eager PyTorch; the competition scores challengers against the *champion*, not the ref.)
The roofline **ridge point** on this GPU is ≈197 FLOP/byte (peak fp16 ÷ HBM bandwidth): kernels
with arithmetic intensity below it are memory-bound, above it compute-bound.

---

## 2. Per-kernel deep dives

Each section covers: **Role in AI · What it computes · Performance character · How the champion
computes it · CUDA hardware mapping · What miners optimize.**

### 2.1 `rms_norm` — RMS normalization

- **Role in AI.** RMSNorm is the normalization used by LLaMA, Mistral, Qwen, etc. (a cheaper
  replacement for LayerNorm — no mean subtraction). It runs **twice per layer** (pre-attention,
  pre-MLP) plus a final norm — so a 32-layer model runs it ~65 times per token per forward pass.
- **What it computes.** Per token row `x` (length `N` = hidden size), with a learned per-feature
  `weight`: `y = x / sqrt(mean(x²) + eps) * weight`. The reduction (`mean(x²)`) is over the
  hidden dimension.
- **Performance character.** **Memory-bound.** Arithmetic intensity ≈ a few FLOP/byte (well below
  the ~197 ridge): it reads `x` + `weight` and writes `y`, doing almost no arithmetic per byte.
  The champion hits ~80% of HBM bandwidth — near the hardware ceiling — so there's little headroom
  (an honest competition: small wins, not 10×).
- **How the champion computes it.** One Triton program **per row**. It loads the whole row
  (`BLOCK = next_pow2(N)`), upcasts to **fp32** (bf16 squares underflow), reduces sum-of-squares
  with `tl.sum`, computes `rms`, multiplies by `weight`, and stores in the input dtype.
- **CUDA hardware mapping.** Each row → one CUDA block; the sum-of-squares → an in-register/shared
  warp reduction; the row load/store must be **coalesced**. No tensor cores (no matmul).
- **What miners optimize.** Memory coalescing, **vectorized loads** (`float4` = 16 bytes/inst),
  multiple rows per block, fewer passes over the row, occupancy tuning.

### 2.2 `matmul` — general matrix multiply (GEMM)

- **Role in AI.** GEMM is the **workhorse of all deep learning**. In a transformer, every linear
  layer is a matmul: the Q/K/V projections, the attention output projection, the MLP up/down
  projections, and the final LM head. Matmuls dominate the FLOPs of training and inference.
- **What it computes.** `C = A @ B`, an `(M×K)·(K×N) → (M×N)` product, with **fp32 accumulation**
  even for fp16/bf16 inputs (to preserve accuracy over the K-length sum).
- **Performance character.** **Compute-bound.** Arithmetic intensity grows with K (hundreds of
  FLOP/byte), well above the ridge: the bottleneck is **tensor-core throughput**, not memory. The
  champion reaches ~99% of cuBLAS on fp16 (~91.5 TFLOPS here).
- **How the champion computes it.** A classic **tiled GEMM**: each program computes one
  `BLOCK_M×BLOCK_N` output tile, looping over K in `BLOCK_K` chunks and accumulating partial
  products via `tl.dot` into an fp32 accumulator; a **grouped block ordering** (`GROUP_M`)
  improves L2 reuse. fp32 inputs use `input_precision="ieee"` to meet the tight tolerance.
- **CUDA hardware mapping.** `tl.dot` compiles to **Tensor Core** instructions (HMMA): bf16/fp16
  multiply with fp32 accumulate. The tiling maps to the **memory hierarchy** — operand tiles
  stream HBM → L2 → shared memory; accumulators live in registers. This is *the* kernel that
  exercises tensor cores.
- **What miners optimize.** Tile sizes; `num_stages` (software-pipelined async prefetch);
  `num_warps`; block-swizzling for L2; split-K; persistent kernels; async copy / TMA on
  Hopper/Blackwell; warp specialization. (Note: this track is the strongest *delegation magnet* —
  beating cuBLAS honestly is hard — which is exactly why the no-delegation guard + runtime trap
  must be airtight here.)

### 2.3 `qkv_part_rope` — partial rotary position embedding

- **Role in AI.** RoPE encodes token position by **rotating** the query/key vectors by an
  angle that depends on position — giving the model relative-position awareness. "**Partial**"
  RoPE (used by e.g. DeepSeek-style attention) rotates only a slice (`rope_dim`) of each head and
  leaves the rest (`nope_dim`) untouched. It runs on Q and K before attention, every layer.
- **What it computes.** For the rope slice, pair `(x0, x1)` and apply a 2-D rotation per
  frequency `f` at sequence position `s`:
  `out0 = x0·cos(s,f) − x1·sin(s,f)`, `out1 = x0·sin(s,f) + x1·cos(s,f)`. The `nope` dims and the
  V head pass through unchanged.
- **Performance character.** **Memory-bound** (arithmetic intensity ≈0.34): pure elementwise
  read-modify-write of the QKV tensor.
- **How the champion computes it.** One Triton program **per (batch, seq, head) row**: copy the
  nope dims, then rotate-or-copy the rope dims via a per-head mask (Q/K rotate, V copies). The
  whole copy+rotate is in the kernel (no `torch.clone`), so it stays delegation-free.
- **CUDA hardware mapping.** Coalesced elementwise loads/stores; `cos`/`sin` read from a small
  precomputed table; no tensor cores. The interesting part is the **partial / per-head structure**
  (only some dims of some heads change).
- **What miners optimize.** Vectorized loads, fusing the copy and rotation, processing multiple
  heads/rows per block, avoiding a separate read of the nope region.

### 2.4 `swiglu_input_quant` — gated activation + FP8 quantization

- **Role in AI.** Two fused MLP operations. (a) **SwiGLU** is the gated activation in LLaMA-style
  MLPs: the up-projection produces two halves, and one gates the other through SiLU. (b) **FP8
  input quantization** converts the activations to **FP8 (e4m3)** with per-block scales so the
  *next* matmul can run on FP8 tensor cores (modern inference quantization). Fusing them avoids
  re-reading the activations from HBM.
- **What it computes (multi-output).** `out = x0 · SiLU(x1)` where `SiLU(x1)=x1·sigmoid(x1)`; plus
  `x_fp8` = blockwise FP8 quant of the original `x` (per 128-column block:
  `scale = |block|.max / 448`, `q = x / scale`, cast to e4m3) and the `x_scale` table.
- **Performance character.** **Memory-bound** (arithmetic intensity ≈0.17): reads `x`, writes
  three outputs.
- **How the champion computes it.** Two Triton kernels: a SwiGLU kernel (compute SiLU in fp32,
  multiply, store) and a blockwise-quant kernel (reduce the per-row max over a 128-column block,
  scale, cast to **fp8 e4m3**, store the fp8 tensor and the fp32 scale).
- **CUDA hardware mapping.** Elementwise compute + a per-block reduction; **FP8 storage** (the new
  8-bit inference format, 448 max for e4m3); no tensor cores in this kernel — but it *produces the
  FP8 operands* that feed a downstream FP8 tensor-core matmul.
- **What miners optimize.** Fusing the SwiGLU and quant passes (read `x` once), vectorization, the
  reduction, and the fp8 cast path. (The shipped correctness tolerance here is loose — 50% —
  reflecting FP8 error; Phase-2 hardening will tighten it.)

### 2.5 `dsa_forward` — attention (FlashAttention-style)

- **Role in AI.** Attention is **the core of the transformer** — every token attends to previous
  tokens. This track is "Dynamic Sparse Attention" with **GQA** (grouped-query attention: several
  query heads share one KV head, shrinking the KV cache) and variable-length sequences. For the
  competition's configs it reduces to **dense causal GQA self-attention** — the exact operation in
  every LLM.
- **What it computes (multi-output).** `out = softmax(scale · Q·Kᵀ, causal) · V` per head, with
  GQA head grouping, plus `lse` (log-sum-exp of the scores, used by the backward pass and for
  numerical stability).
- **Performance character.** **Compute-bound** — attention is two GEMMs back-to-back (`Q·Kᵀ` and
  `P·V`) plus a softmax. The champion reaches ~102 TFLOPS here.
- **How the champion computes it.** **FlashAttention-2**: tile the queries, stream over KV blocks
  with an **online softmax** (track a running max and rescale the accumulator) so the full
  `N×N` score matrix is **never materialized** — this is what makes long-context attention fit in
  memory. Causal masking + fp32 accumulation.
- **CUDA hardware mapping.** The two matmuls → **tensor cores** (`tl.dot`); K/V tiles → shared
  memory; the running softmax statistics → registers. This is the most sophisticated champion —
  it combines tensor-core GEMMs with a streaming reduction, the defining trick of flash attention.
- **What miners optimize.** Tile sizes (`BLOCK_M`/`BLOCK_N`), `num_stages` pipelining, splitting
  the diagonal (masked) block from the full off-diagonal blocks, num_warps, **exploiting the
  actual block-sparsity** (only compute selected KV blocks), KV-cache layout, FP8 attention.

> **Note on the dsa oracle.** The original reference was an `O(n²)` Python loop with a GPU sync per
> score (~40 s for the *tiny* config). CCO replaced it with a fast vectorized PyTorch oracle that
> computes the identical result (validated to ~2e-3 against the loop version) so the canonical
> rerun is feasible — a benchmark-design fix that fell out of building this champion.

---

## 3. How they integrate — CUDA × AI

**The AI side.** Chain the five and you have a working transformer layer (see §1). Speeding any of
them up *directly* speeds up real LLM training/inference: `matmul` and `dsa_forward` cut the bulk
of the FLOPs/latency; `rms_norm`, `qkv_part_rope`, `swiglu_input_quant` cut the memory traffic
between the big ops (and, fused well, eliminate whole HBM round-trips).

**The CUDA side.** The five split cleanly across the two things a GPU can be limited by:

- **Tensor-core (compute) bound — `matmul`, `dsa_forward`.** The art is keeping the tensor cores
  fed: tiling through shared memory, software-pipelined async copies, operand layouts, and (for
  attention) the flash-attention trick of never materializing the score matrix. These exercise
  HMMA/WGMMA instructions and the full HBM→L2→SMEM→register hierarchy.
- **Memory-bandwidth bound — `rms_norm`, `qkv_part_rope`, `swiglu_input_quant`.** The art is
  moving bytes efficiently: coalesced + vectorized loads, fusing multiple ops into one pass so
  data is read once, and maximizing occupancy to hide latency. Tensor cores are irrelevant; HBM
  bandwidth is the wall.

Triton is the integration layer: `@triton.jit` kernels compile to PTX/SASS, `tl.dot` lowers to
tensor-core instructions, and `tl.load`/`tl.store` lower to coalesced global-memory access — so a
miner writes Python-level Triton and gets CUDA-level performance, which the harness measures on
real hardware.

---

## 4. Why five? Are these the most important? Can we add more?

**Why five.** It is a deliberate, minimal set chosen for **bottleneck coverage**, not a top-5
popularity list:

- 2 clearly **memory-bound** (`rms_norm`, `swiglu_input_quant`),
- 1 clearly **compute-bound** (`matmul`),
- 2 spanning the middle / both regimes (`qkv_part_rope` memory-bound; `dsa_forward` compute-bound
  attention).

Because the two regimes demand different optimization techniques, covering both means a
cross-kernel optimization pattern discovered on one track can be tested for transfer to another —
which is the whole point of a knowledge-growing competition. Five is also a manageable surface for
v1: each track is its own king-of-the-hill ladder with its own label and emission share.

**Are they the most important?** They are the most important *representative primitives of LLM
inference* — each is a real, ubiquitous operation, and together they are a transformer layer. By
raw FLOPs, `matmul` and `dsa_forward` dominate; the other three are smaller but run constantly and
are latency-critical because they're memory-bound. So "most important" depends on the metric, but
as a *spanning* set they're well chosen.

**Can we add more? — yes, by design.** The framework is "inherit by changing data, not code": the
benchmark harness auto-discovers tracks. Adding a kernel is purely additive:

1. `references/<name>.py` — a pure-PyTorch oracle (the ground truth; may use high-level torch ops).
2. `kernel_configs/<name>.{toml,py}` — shapes, dtypes, tolerances, and the input generator
   (+ `flops_fn`/`bytes_fn` for the roofline).
3. `champions/<name>/kernel.py` — a seed champion (a real Triton kernel; the bar to beat).
4. a `cco-winner-<name>` entry in `cco.config.json`'s `label_multipliers`.

No harness code changes. **Candidate future tracks:** `layer_norm`, `softmax`, fused **QKV
projection**, **MoE** (expert routing + grouped GEMM), **flash-attention backward**, **paged /
KV-cache attention**, **int8/fp8 quantized GEMM**, `gemm+bias+gelu` fusion, collectives
(all-reduce), and vision ops (conv). Each new track adds a king-of-the-hill ladder and its own
slice of emissions — the natural way the competition grows over time.
