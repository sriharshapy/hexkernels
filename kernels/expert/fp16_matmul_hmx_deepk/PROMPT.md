# fp16 32x32 matrix multiply with a deep K=128 reduction on the HMX matrix engine

Implement `candidate_kernel(const __fp16 *A, const __fp16 *B, __fp16 *out, int n, int k_dim)`
computing the row-major fp16 matrix product for `n == 32`, `k_dim == 128`:

```
out[i*n+j] = sum_k A[i*k_dim+k] * B[k*n+j]      (row-major, M=N=32, K=128)
```

- The harness has already enabled the HMX context before calling you; use VTCM
  scratch at `HVX_VTCM_BASE`.
- Output is compared to a float32-accumulate scalar reference (cast to `__fp16`)
  with an fp16 tolerance -- HVX/HMX float arithmetic is non-IEEE (qf16), so
  bit-exactness is NOT required, and this task's tolerance is a bit wider than
  the shallow K=32 sibling's (deeper reductions drift more).
- K=128 is 4x deeper than the single-tile fp16 sibling. Clear the HMX float
  accumulator ONCE (`Q6_mxclracc_hf`), then issue all four (activation, weight)
  load-matmul pairs -- one per 32-deep K-tile -- before the single store; each
  pair accumulates into the persistent accumulator.
- fp16 crouton layout (helper in `harness_common.h`): `off(r,c)=(r/2)*64+c*2+(r&1)`,
  same for activation, weight, and output, applied per K-tile.
- Scalar VTCM accesses are expensive in timing mode -- stage the crouton
  pack/unpack through cacheable buffers and move to/from VTCM in bulk 128B
  vector copies.
- A competent HVX qf16 row-broadcast matmul (over the full K=128 reduction) is
  the baseline; the HMX expert must beat it by >=1.2x.


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
