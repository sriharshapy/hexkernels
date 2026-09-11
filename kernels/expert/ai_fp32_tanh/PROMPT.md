# FP32 Pointwise Tanh Activation (n=1024)

Implement:
```c
void candidate_kernel(const float *x, float *out, int n);
```

## Semantics (PINNED — tolerance-checked, not bit-exact)

For each element `i` in `[0, n)`:
```c
out[i] = tanhf(x[i]);   /* hyperbolic tangent, fp32 */
```

All computation stays in fp32. Output is fp32 (no down-cast).

## Notes

- `n = 1024` — handle any tail (n may not be a multiple of 128).
- Inputs and output are 128-byte aligned (`HVX_ALIGN`).
- Harness poisons the output buffer to `0xA5A5A5A5` per element; a no-op will fail.
- `#include <math.h>` for `tanhf`.
- Use HVX float intrinsics for throughput where possible; `tanhf` may be computed per element.
- Inputs span `[-6, 6]` in eighth-integer steps — both negative and positive tanh values are present.
- tanh output range is `(-1, 1)`; sigmoid or identity both fail the tolerance check.
- HVX v68 has no vector transcendental; a fast vectorized rational approximation
  (e.g. Pade + Newton-Raphson reciprocal) can beat scalar `tanhf` handily while
  staying within tolerance (`hvx_close_f32`: atol 1e-4, rtol 1e-3).
