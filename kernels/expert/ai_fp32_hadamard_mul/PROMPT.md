# FP32 Elementwise (Hadamard) Product (n=1000)

Implement:
```c
void candidate_kernel(const float *a, const float *b, float *out, int n);
```

**Semantics:** `out[i] = a[i] * b[i]` for `i` in `[0, n)`.

This is a Hadamard (elementwise) product of two fp32 tensors — used for gating
and masking in modern neural networks (e.g., SiLU/SwiGLU gates, attention masks,
channel-wise feature scaling).

**Scalar reference:** each output element is the single correctly-rounded IEEE
fp32 multiply of `a[i]` and `b[i]`.  No reductions.

**Tolerance contract (not bit-exact):** HVX v68 has no correctly-rounded native
sf*sf vector multiply (`Q6_Vsf_vmpy_VsfVsf` fails to select at v68) and the
`qf32` round-trip (`Q6_Vqf32_vmpy_VsfVsf` + `Q6_Vsf_equals_Vqf32`) is not
bit-exact for sf*sf (~1e-7 relative error per multiply). The harness therefore
compares with `hvx_close_f32` (atol 1e-4, rtol 1e-3), well outside that error.

- `n=1000`; handle the tail (`n` may not be a multiple of 32/128).
- Arrays are 128-byte aligned (`HVX_ALIGN`).
- Use the `qf32` round-trip for vectorised throughput:
  `Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(va, vb))`.
- Include `<hexagon_types.h>` and `<hexagon_protos.h>` for HVX types/intrinsics.
