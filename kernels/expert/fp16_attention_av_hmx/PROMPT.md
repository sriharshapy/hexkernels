# fp16 attention A.V (scores x values) on the HMX matrix engine, deep-K reduction

Implement `candidate_kernel(const hvx_hf *P, const hvx_hf *V, hvx_hf *out, int n, int k_dim)`
computing the attention output matmul for `n == 32` (S == D == 32, single output
crouton tile) with a DEEP `k_dim == 128` reduction (128 keys/values):

```
out[i][j] = sum_k P[i*k_dim+k] * V[k*n+j]     (row-major, dtype __fp16;
                                                V is already in the generic
                                                B[K,N] matmul layout)
```

- The harness has already enabled the HMX context before calling you; you may use
  VTCM scratch at `HVX_VTCM_BASE`.
- Output is compared to a float-accumulate scalar reference (cast to `__fp16`)
  with an fp16 tolerance (HVX/HMX float arithmetic is non-IEEE qf16, NOT bit-exact).
- `k_dim=128` is accumulated over four 32-deep crouton K-tiles: clear the HMX float
  accumulator ONCE (`Q6_mxclracc_hf`), then issue all four `(activation, weight)`
  load-matmul pairs before the single store.
- fp16 crouton layout (helper `hvx_crouton_off(r,c)` in `harness_common.h`):
  `off(r,c) = (r/2)*64 + c*2 + (r&1)`, used for P, V, and the output, per K-tile.
- **V is already in the generic `B[K,N]` matmul layout** (row `k` is the values for
  key/value `k`) — no transpose or index swap needed when packing the weight tile.
- Scalar VTCM accesses are expensive in timing mode — stage the crouton pack/unpack
  through cacheable buffers and move to/from VTCM in bulk 128B vector copies.
- A competent HVX qf16 row-broadcast/AXPY matmul (splat `P[i][k]` across a vector,
  accumulate `P[i][k] * V[k][:]` in qf16) is the baseline; the HMX expert must beat
  it by >=1.2x — this margin is THIN for fp16 (int8 wins far more decisively).


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
