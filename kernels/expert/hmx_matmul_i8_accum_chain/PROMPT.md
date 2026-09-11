# int8 32x32 matmul with chained K=64 accumulation on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out,
int n, int k_dim)` for `n == 32`, `k_dim == 64`:

```
acc[i][j] = sum_{k=0}^{k_dim-1} A[i*k_dim+k] * B[k*n+j]
out[i][j] = sign_extend_12bit((acc*17 + 8) >> 4)
```

`A` is `[M x K]` (uint8 0..7), `B` is `[K x N]` (int8 -3..3). K=64 exceeds one
32-deep crouton tile, so chain the accumulation: clear the HMX accumulator ONCE
with `mxclracc`, then issue BOTH 32-deep K-tile matmul packets so they accumulate
into the same accumulator, and do a SINGLE `0x40`-config requant store at the end.

- The harness has already enabled the HMX context; use VTCM scratch at
  `HVX_VTCM_BASE`. int8 activation in the HIGH byte of a crouton slot
  (`hvx_hmx_i8_act_off`), weight 4-deep packed (`hvx_hmx_i8_wgt_off`), output
  crouton read as uint16 (`hvx_hmx_i8_out_off`).
- Per K-tile: `{ activation.ub=mxmem(...); weight.b=mxmem(...) }`. After both
  tiles: `bias=mxmem(0x40-fill)` -> `mxmem(...):after.uh=acc:2x1` -> isync, then
  sign-extend the 12-bit field.
- Stage crouton pack/unpack through cacheable buffers + bulk 128B copies; the HMX
  expert must beat the HVX vrmpy baseline by >=1.2x.

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
