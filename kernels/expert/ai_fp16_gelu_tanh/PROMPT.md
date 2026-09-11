# FP16 Pointwise GELU — tanh approximation (n=1024)

Implement:
```c
void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n);
```

where `hvx_hf` is `__fp16`.

## Semantics (PINNED — tolerance-checked, not bit-exact)

For each element `i` in `[0, n)`:
```
float v = (float)x[i];
float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
float g = 0.5f * v * (1.0f + tanhf(inner));
out[i] = (hvx_hf)g;   // single fp16 round at the very end
```

**All intermediates must remain fp32.  Cast to `hvx_hf` exactly once, at the final assignment.**

Constants are pinned: `0.7978845608f` (≈ sqrt(2/π)) and `0.044715f`.

## Notes

- `n = 1024` — handle any tail (n may not be a multiple of 128-bit vector lanes).
- Inputs and output are 128-byte aligned (`HVX_ALIGN`).
- Harness poisons the output buffer to `0xA5A5` per element; a no-op will fail.
- Use HVX intrinsics for throughput where possible; `tanhf` may be computed per element.
- `#include <math.h>` for `tanhf`.
- Compare uses `hvx_close_f16bits` (atol 4e-3, rtol 8e-3), not bit-exactness — HVX
  v68 has no native fp16<->fp32 vector conversion or vector transcendental, but a
  vectorized rational tanh approximation (Pade + Newton-Raphson reciprocal) run
  on scalar-widened fp32 lanes can beat scalar `tanhf` by several times while
  staying within tolerance.
