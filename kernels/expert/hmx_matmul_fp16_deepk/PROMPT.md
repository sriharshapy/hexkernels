# FP16 deep-K matrix multiply on the HMX matrix engine

Implement `candidate_kernel(const __fp16 *A, const __fp16 *B, __fp16 *out, int n,
int k_dim)` computing `out[i*n+j] = sum_k A[i*k_dim+k] * B[k*n+j]` for `n == 32`
and `k_dim == 128` (M=N=32, K=128), in FP16.

- The harness has already enabled the HMX context; use VTCM scratch at
  `HVX_VTCM_BASE` for crouton packing.
- K=128 is accumulated over four 32-deep crouton K-tiles: clear the HMX float
  accumulator once (`Q6_mxclracc_hf`), issue all four (activation, weight)
  load-matmul pairs (`Q6_activation_hf_mxmem_RR` + `Q6_weight_hf_mxmem_RR`, limit
  2047) into the persistent accumulator, then ONE store (`Q6_mxmem_AR_after_hf`).
- FP16 crouton offset `(r/2)*64 + c*2 + (r&1)` (`hvx_crouton_off`) for activation,
  weight, and output. Stage pack/unpack through cacheable buffers + bulk 128B copies.
- qf16 arithmetic is non-IEEE -> compared with an fp16 tolerance (NOT bit-exact).
- Deep-K amortizes the pack/unpack overhead; the HMX expert must beat the HVX
  qf16 baseline by >=1.2x.

Implement ONLY this function (Hexagon HVX C). Include `<hexagon_types.h>`,
`<hexagon_protos.h>` and `<hmx_hexagon_protos.h>`. Do NOT write `main()`. Respond
with a single complete C code block.


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
