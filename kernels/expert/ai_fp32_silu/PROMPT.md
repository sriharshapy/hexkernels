# FP32 Pointwise SiLU / Swish Activation (n=1024)

Implement:
```c
void candidate_kernel(const float *x, float *out, int n);
```

## Semantics (PINNED — tolerance-checked, not bit-exact)

For each element `i` in `[0, n)`:
```c
out[i] = x[i] / (1.0f + expf(-x[i])); /* SiLU = x * sigmoid(x) */
```

All computation stays in fp32. Output is fp32 (no down-cast).

## Notes

- `n = 1024` — handle any tail (n may not be a multiple of 128).
- Inputs and output are 128-byte aligned (`HVX_ALIGN`).
- Harness poisons the output buffer to `0xA5A5A5A5` per element; a no-op will fail.
- `#include <math.h>` for `expf`.
- Use HVX float intrinsics for throughput where possible; `expf` may be computed per element.
- Inputs span `[-6, 6]` in eighth-integer steps — both negative and positive SiLU values are present.
- Compare uses `hvx_close_f32` (atol 1e-4, rtol 1e-3), not bit-exactness — HVX v68
 has no vector transcendental; a fast vectorized rational tanh approximation
 (Pade + Newton-Raphson reciprocal), used via sigmoid(x)=0.5*(1+tanh(x/2)), can
 beat scalar `expf` by an order of magnitude while staying within tolerance.
