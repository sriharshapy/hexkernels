# fp16 32x32 matmul on the HMX matrix engine + fused GELU epilogue

Implement `candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n)`
for `n == 32` (`hvx_hf` is `__fp16`):

```
acc[i][j] = sum_k A[i*n+k] * B[k*n+j] (float32-accumulate, n=32)
m[i][j] = (hvx_hf)acc[i][j] (fp16-round)
out[i][j] = 0.5*m*(1 + tanh(0.7978845608*(m + 0.044715*m^3)))
```

- Pack `A` and `B` into the fp16 crouton layout `off(r,c)=(r/2)*64+c*2+(r&1)` (same
 layout for activation, weight, and output), run ONE HMX matmul
 (`Q6_mxclracc_hf`, then `Q6_activation_hf_mxmem_RR` +
 `Q6_weight_hf_mxmem_RR`, then `Q6_mxmem_AR_after_hf`), then read the crouton
 result back.
- GELU has no HVX vector transcendental — the epilogue (the tanh-approximation
 GELU) is necessarily scalar, applied per-element after the crouton unpack.
 The matmul itself must run on the matrix engine, not just the epilogue.
- Scalar VTCM accesses cost ~48 cycles each in timing mode — stage the crouton
 pack/unpack through cacheable buffers and move to/from VTCM in bulk 128B
 vector copies.
- Output is compared to a float32-accumulate scalar reference (through the SAME
 GELU formula) with an fp16 tolerance — HVX/HMX float arithmetic is non-IEEE
 (qf16), so bit-exactness is NOT required.
- A PLAIN SCALAR fp16 matmul + GELU (no HVX, no HMX) is the baseline; your HMX
 kernel must beat it by >=1.2x.

`the HVX headers`. Do NOT write `main`. Respond with a single complete C
code block.


## Available helper (HMX matrix engine)

You MAY `#include "hmx_helpers.h"` and call these sim-verified helpers instead of hand-writing
the crouton `mxmem` sequence. Each owns the crouton pack + `mxclracc`/`mxmem` matmul + VTCM
staging for `M,N,K` all multiples of 32; you do the epilogue (bias/relu/residual/requant/cast)
yourself in HVX or scalar.

int8 (A uint8 row-major MxK, B int8 row-major KxN):
```
void hmx_tile_matmul_i8(const uint8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);
    /* C = sx12((sum_k A*B)*17+8 >> 4): the HMX 0x40-config 12-bit requant field, sign-extended */
void hmx_tile_matmul_i8_field(const uint8_t *A, const int8_t *B, uint16_t *field, int M, int N, int K);
    /* the raw uint16 crouton field, before sign-extension */
```
fp16 (tolerance comparison, not bit-exact):
```
void hmx_tile_matmul_fp16(const __fp16 *A, const __fp16 *B, float *C, int M, int N, int K);
void hmx_tile_matmul_fp16_field(const __fp16 *A, const __fp16 *B, __fp16 *field, int M, int N, int K);
```
Using the helper is optional — a correct kernel by any means is accepted.
