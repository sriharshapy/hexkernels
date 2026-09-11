# FP16 Pointwise Sigmoid (n=1024)

Implement:
```c
void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n);
```

where `hvx_hf` is `__fp16`.

## Semantics (PINNED — tolerance-checked, not bit-exact)

For each element `i` in `[0, n)`:
```
float v = (float)x[i];
float s = 1.0f / (1.0f + expf(-v));
out[i] = (hvx_hf)s;   // single fp16 round at the very end
```

**All intermediates must remain fp32.  Cast to `hvx_hf` exactly once, at the final assignment.**

## Notes

- `n = 1024` — handle any tail (n may not be a multiple of 128-bit vector lanes).
- Inputs and output are 128-byte aligned (`HVX_ALIGN`).
- Harness poisons the output buffer to `0xA5A5` per element; a no-op will fail.
- Inputs span `[-6, 6]` quarter-integers covering near-zero, large-negative, large-positive.
- Use HVX intrinsics for throughput where possible; `expf` may be computed per element.
- `#include <math.h>` for `expf`.
- Compare uses `hvx_close_f16bits` (atol 4e-3, rtol 8e-3), not bit-exactness — HVX
  v68 has no native fp16<->fp32 vector conversion or vector transcendental, but
  computing sigmoid(x) = 0.5*(1+tanh(x/2)) via a vectorized rational tanh
  approximation on scalar-widened fp32 lanes can beat scalar `expf` by several
  times while staying within tolerance.
