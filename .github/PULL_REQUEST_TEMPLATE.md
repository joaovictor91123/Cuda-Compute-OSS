<!-- Read CONTRIBUTING.md before opening this PR.
     ONLY the fenced JSON payload below, the acknowledgement checkboxes, and this body's hash
     carry authority. All other prose (title, this description, comments) is non-authoritative and
     is ignored by the automated gates. Editing the PR after the gates pass closes it — open a
     fresh PR instead. -->

## Optimization

<!-- One sentence describing your optimization. Human-readable only; not scored. -->

## Submission payload

<!-- Fill in every field. Schema: payload-schema.json. `signature` is your SN74 hotkey signing the
     message `<commit_sha>:<kernel_sha256>:<kernel_type>`. `claimed_speedup` is advisory only —
     the canonical rerun is authoritative. -->

```json
{
  "version": 1,
  "commit_sha": "<40-hex PR HEAD commit sha>",
  "kernel_type": "<rms_norm | matmul | qkv_part_rope | swiglu_input_quant | dsa_forward>",
  "kernel_sha256": "<sha256 of your kernel.py>",
  "hotkey": "<your SN74 SS58 hotkey>",
  "signature": "<hotkey signature over commit_sha:kernel_sha256:kernel_type>",
  "claimed_speedup": 1.00
}
```

## Acknowledgements

- [ ] I read [CONTRIBUTING.md](../CONTRIBUTING.md) and [DESIGN.md](../DESIGN.md).
- [ ] This PR changes **only** `kernel.py` — no other file is added, modified, or removed.
- [ ] `kernel.py` is a real **Triton** kernel: it does **not** delegate the computation to `torch.matmul`/`mm`/`bmm`, `torch.nn.functional.*`, the `@` operator, `torch.ops.aten.*`, cuBLAS/cuDNN, or inline CUDA-C (see CONTRIBUTING § No delegation).
- [ ] My SN74 hotkey is bound to this GitHub identity and the payload signature verifies.
- [ ] I self-scored locally and got `correctness: PASS` on the declared track.
