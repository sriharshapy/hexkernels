# FP16 Elementwise (Hadamard) Product (n=1000)

Implement:
```c
typedef __fp16 hvx_hf;
void candidate_kernel(const hvx_hf *a, const hvx_hf *b, hvx_hf *out, int n);
```

**Semantics:** `out[i] = (hvx_hf)((float)a[i] * (float)b[i])` for `i` in `[0, n)`.

This is a Hadamard (elementwise) product of two fp16 tensors — used for gating and
masking in modern neural networks (e.g., SiLU gate, attention masks).

**Bit-exact contract:** each output element is the single correctly-rounded IEEE fp16
multiply of `a[i]` and `b[i]`. No reductions. Scalar and HVX-float must agree
bit-for-bit on the `uint16` representation.

- `n=1000`; handle the tail (`n` may not be a multiple of 128).
- Arrays are 128-byte aligned (`HVX_ALIGN`).
- Use HVX FP16 intrinsics for vectorized throughput, e.g. `Q6_Vhf_vmpy_VhfVhf`.
- Include `<hexagon_types.h>` and `<hexagon_protos.h>` for HVX types/intrinsics.
