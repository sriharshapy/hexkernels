# HMX int8 K-accumulation over 2 contraction tiles (K=64 into one 32x32 tile)

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n, int K)`
for `n == 32`, `K == 64`. Output is a single 32x32 tile, but the contraction
K = 64 = 2*32, so the HMX accumulator must persist across TWO matmul packets:

```
acc[i][j] = sum_{k=0}^{K-1} A[i*K+k] * B[k*n+j]      (A uint8 0..7, B int8 -3..3)
out[i][j] = sign_extend_12bit( (acc*17 + 8) >> 4 )    (0x40-config HMX requant, int32)
```

- A is row-major `[n][K]`, B is row-major `[K][n]`, out is `[n][n]`.
- Issue exactly ONE `mxclracc` (clear the accumulator once). Then for each of the
  `K/32` contraction tiles, pack the 32x32 `(activation, weight)` pair into VTCM
  crouton tiles and issue one `{ activation.ub=mxmem(...); weight.b=mxmem(...) }`
  matmul packet -- the packets accumulate into the SAME accumulator. After all
  K-tiles, load the `0x40` requant config and do ONE requant store, then unpack.
- Clearing inside the K-loop (a common bug) yields only the last tile's partial
  product -- do not do that.
- Stage crouton pack/unpack through cacheable buffers + bulk 128B vector copies.
  Layout helpers: `hvx_hmx_i8_act_off`, `hvx_hmx_i8_wgt_off`, `hvx_hmx_i8_out_off`.

Implement ONLY this function (Hexagon HVX C). Include `<hexagon_types.h>` and
`<hexagon_protos.h>`. Do NOT write `main()`. Respond with a single complete C
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
